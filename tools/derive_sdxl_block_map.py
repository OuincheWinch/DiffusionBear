#!/usr/bin/env python3
"""Derive the SDXL kohya -> diffusers block map from the real UNet module tree.

The hand-written table this replaces was wrong: it mapped 442 Linear targets
onto paths that do not exist and produced 120 bogus middle-block entries.
Inference from SDXL's advertised block structure is not enough, so instead we
read the truth off a real diffusers UNet and zip it against the ordered slots
in Draw Things' own table.

Two ideas make it work:

1. A kohya slot is classified by aggregating *every* leaf module name under
   it, not by substring-matching one key. Slots 1.1 and 2.1 carry `proj_in` /
   `proj_out`, which contain neither "attn" nor "transformer_blocks" and are
   invisible to a single-key heuristic.
2. If DT and the real UNet expose the same number of slots in each class and
   direction, zip them in order. If the counts disagree, refuse -- a mismatch
   means the assumption is wrong, not that a partial mapping is acceptable.

    python tools/derive_sdxl_block_map.py \
        --config backend/data/models/juggernaut-xi/unet/config.json \
        --out backend/drawthings_sdxl_block_map.json

Requires torch + diffusers. The output is committed so the app and its tests do
not need either.
"""

import argparse
import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import drawthings_names as names  # noqa: E402

SLOT_RE = re.compile(
    r"^(?P<dir>input_blocks|output_blocks)\."
    r"(?P<a>\d+)\.(?P<b>\d+)\.(?P<leaf>.+)$"
)
# middle_block carries only two numeric levels: middle_block.1.<leaf>
MID_RE = re.compile(r"^(?:middle_block|mid_block)\.(?P<a>\d+)\.(?P<leaf>.+)$")

# Leaf names that only occur inside a resnet.
RESNET_LEAVES = {"in_layers.0", "in_layers.2", "emb_layers.1", "out_layers.0",
                 "out_layers.3", "skip_connection", "time_emb_proj"}
# Leaf names that only occur inside a transformer attention block.
ATTN_LEAVES = {"attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
               "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
               "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2", "norm1",
               "norm2", "norm3"}


def slot_key(key):
    """Return `(direction, a, b, leaf)` for a Draw Things UNet key, or None."""
    tail = key.split("diffusion_model.")[-1]
    match = SLOT_RE.match(tail)
    if match:
        direction = "down" if match.group("dir") == "input_blocks" else "up"
        return direction, int(match.group("a")), int(match.group("b")), match.group("leaf")
    match = MID_RE.match(tail)
    if match:
        return "mid", int(match.group("a")), 0, match.group("leaf")
    return None


def classify(leaves):
    """Classify a kohya slot by the union of the leaf module names under it."""
    if leaves & ATTN_LEAVES and leaves & {"proj_in", "proj_out", "attn1.to_q",
                                          "attn2.to_q", "ff.net.0.proj"}:
        return "attn"
    if leaves & RESNET_LEAVES:
        return "resnet"
    # Upsample is `conv`, downsample is `op`: Draw Things names them
    # differently in the two directions.
    if leaves and leaves <= {"conv", "op.conv", "op"}:
        return "sample"
    return "other"


def real_slots(unet, kinds=("Linear",)):
    """Ordered real diffusers slot prefixes, grouped by direction and class."""
    out = collections.defaultdict(list)
    for name, module in unet.named_modules():
        if module.__class__.__name__ not in kinds:
            continue
        parts = name.split(".")
        if ".upsamplers." in name or ".downsamplers." in name:
            # Upsample/Downsample convs are Conv2d, so the Linear-only walk
            # above never sees them; they are the `sample` class.
            slot = name.rsplit(".", 1)[0]  # drop the trailing .conv
            direction = "down" if name.startswith("down_blocks") else "up"
            if slot not in out[(direction, "sample")]:
                out[(direction, "sample")].append(slot)
            continue
        for kind in ("attentions", "resnets"):
            if kind in parts:
                i = parts.index(kind)
                slot = ".".join(parts[: i + 2])
                direction = "mid" if name.startswith("mid_block") else (
                    "down" if name.startswith("down_blocks") else "up")
                if slot not in out[(direction, kind[:-1])]:
                    out[(direction, kind[:-1])].append(slot)
                break
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to an SDXL unet config.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--table", default="UNetXLBase")
    args = parser.parse_args(argv)

    import torch
    from diffusers import UNet2DConditionModel

    config = json.loads(pathlib.Path(args.config).read_text())
    with torch.device("meta"):
        unet = UNet2DConditionModel.from_config(config)

    table = names.load_name_map()[args.table]

    # Group every DT key by its kohya slot, collecting the union of leaf names.
    slots = collections.defaultdict(set)
    for key in table:
        parsed = slot_key(key)
        if parsed is None:
            continue
        direction, a, b, leaf = parsed
        # `ff.net.0.proj` is one module; normalise weight/bias away.
        leaf = leaf.rsplit(".", 1)[0] if leaf.endswith((".weight", ".bias")) else leaf
        slots[(direction, a, b)].add(leaf)

    classified = {slot: classify(leaves) for slot, leaves in slots.items()}
    print("DT slots by class:")
    for kind in ("attn", "resnet", "sample", "other"):
        got = sorted(s for s, c in classified.items() if c == kind)
        print(f"  {kind:7} {len(got):>3}  {[f'{d}.{a}.{b}' for d, a, b in got][:12]}")

    real = real_slots(unet)
    real_sample = real_slots(unet, kinds=("Conv2d",))
    print("\nreal UNet slots:")
    for key, v in sorted(real.items()):
        print(f"  {key[0]:5} {key[1]:7} {len(v):>3}  {v[:6]}")

    mapping = {}
    problems = []
    for kind, real_kind in (("attn", "attention"), ("resnet", "resnet"), ("sample", "sample")):
        dt_slots = sorted(s for s, c in classified.items() if c == kind)
        for direction in ("down", "up", "mid"):
            mine = [s for s in dt_slots if s[0] == direction]
            source = real_sample if kind == "sample" else real
            theirs = [p for (d, k), group in source.items()
                      if d == direction and k == real_kind for p in group]
            if not mine and not theirs:
                continue
            if len(mine) != len(theirs):
                problems.append(
                    f"{direction}/{kind}: DT has {len(mine)} slots "
                    f"{[f'{a}.{b}' for _, a, b in mine]}, real UNet has {len(theirs)} {theirs}"
                )
                continue
            for slot, target in zip(mine, theirs):
                mapping[f"{slot[0]}.{slot[1]}.{slot[2]}"] = target

    if problems:
        print("\nREFUSING to emit a partial map:")
        for p in problems:
            print(f"  {p}")
        return 1

    out = pathlib.Path(args.out)
    out.write_text(json.dumps(mapping, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {out} -- {len(mapping)} slot mappings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())