"""Verify the kohya -> diffusers block map against a real SDXL UNet.

`drawthings_sdxl_block_map.json` is derived by `tools/derive_sdxl_block_map.py`
and never hand-written -- it was wrong twice before that. This is the check on
the derivation: it builds the real UNet module tree from a shipped
`config.json` and confirms every entry names a module that exists, and reports
every real `nn.Linear` the map does not cover.

The UNet is built under `init_empty_weights`, so no parameters are allocated
and nothing is downloaded -- only the module tree and shapes are read.

    python tools/verify_sdxl_block_map.py path/to/unet/config.json [--shapes]

With `--shapes`, also print the real weight shape of every covered `nn.Linear`.
That is what the LoRA converter checks a reassembled fused tensor against; a
fused tensor concatenated on the wrong axis still has the right shape, so the
shapes alone do not prove the axis -- they only rule a wrong one out cheaply.

Exits 1 if any map entry does not resolve to a real module.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import drawthings_names as names  # noqa: E402

# kohya key leaves a LoRA cannot target, and why. `in_layers`/`out_layers` are
# kohya spellings of what diffusers calls conv1/conv2; `emb_layers.1` is the
# timestep embedding, an Embedding rather than a Linear; `input_blocks.0.0` is
# the stem Conv2d. None of them carry an attention projection.
NOT_A_LINEAR_LEAVES = ("in_layers", "out_layers", "emb_layers", "conv_shortcut")

# Whether a row must land on an `nn.Linear` is `drawthings_names`' call, not a
# second list here: two lists drift, and this tool exists to check that layer
# rather than to assert it. `lora_targets` therefore reports what the converter
# would resolve, and the failures below are rows the converter *does* resolve to
# something that is not a Linear.


def build_unet(config_path):
    """Return a real diffusers SDXL UNet with no parameters allocated."""
    import torch  # noqa: F401  -- imported for the side effect of a clear error
    from accelerate import init_empty_weights
    from diffusers import UNet2DConditionModel

    config = json.loads(Path(config_path).read_text())
    with init_empty_weights():
        return UNet2DConditionModel.from_config(config)


def resolve(unet, dotted):
    """Resolve a dotted module path against a real module tree.

    Raises `AttributeError` or `IndexError` when the path does not exist --
    deliberately not caught and softened, because "this slot does not exist" is
    the exact finding this tool exists to report.
    """
    module = unet
    for part in dotted.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def linear_paths(unet):
    """Return every `nn.Linear` weight path in the tree, sorted."""
    import torch.nn as nn

    return sorted(
        f"{name}.weight"
        for name, module in unet.named_modules()
        if isinstance(module, nn.Linear)
    )


def check_block_map(unet):
    """Return `(resolved, missing)` for every entry in the committed block map."""
    resolved, missing = [], []
    for slot, target in sorted(names.load_block_map().items()):
        try:
            module = resolve(unet, target)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            missing.append((slot, target, f"{type(exc).__name__}: {exc}"))
            continue
        shape = getattr(module, "weight", None)
        shape = tuple(shape.shape) if shape is not None else None
        resolved.append((slot, target, type(module).__name__, shape))
    return resolved, missing


def lora_targets(unet, table="UNetXLBase", prefix="__unet__"):
    """Yield `(key, diffusers_path, module_type, shape, kind)` for key rows.

    Walks the harvested table the way the converter will, so a key whose block
    slot resolves but whose module is not a `Linear` shows up here instead of
    producing a LoRA that attaches to nothing. `kind` is "projection" for the
    leaves that must be `Linear`, "norm" for affine norms, and None for
    anything filtered out and not walked.
    """
    rows = names.load_name_map()[table]
    for key in sorted(rows):
        leaf = key.rsplit(".", 1)[0]
        if any(leaf.endswith(x) for x in NOT_A_LINEAR_LEAVES):
            continue
        if leaf.endswith("input_blocks.0.0"):
            continue  # the stem Conv2d; see the block map for why it is unmapped
        # Ask the resolver first: a row it refuses has no Linear by definition
        # and is not this tool's business.
        path = names.resolve_sd(prefix, *_layer_param(rows[key][0]))
        if path is None:
            continue
        kind = "projection" if names._has_linear(path) else "other"
        if path is None:
            yield key, None, "UNRESOLVED", None, kind
            continue
        try:
            module = resolve(unet, path)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            yield key, path, f"MISSING: {type(exc).__name__}", None, kind
            continue
        weight = getattr(module, "weight", None)
        yield key, path, type(module).__name__, (
            tuple(weight.shape) if weight is not None else None
        ), kind


def _layer_param(tid):
    _, layer, param = tid.split("-")
    return int(layer), int(param)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("-")]
    flags = {a for a in argv[1:] if a.startswith("-")}
    # No default. The previous default pointed into a second local checkout of
    # this repo, so the tool could not run on any machine that did not have it,
    # and the SDXL gate's block-map evidence rested on it. Required is honest:
    # this reads a real model's config and there is no portable location for one.
    if not args:
        print("usage: verify_sdxl_block_map.py <path/to/unet/config.json> "
              "[--shapes]\n"
              "  builds the real SDXL UNet from that config (no parameters "
              "allocated, nothing downloaded) and checks every block-map entry "
              "against it.", file=sys.stderr)
        return 2
    path = Path(args[0])
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 2

    print(f"config: {path}")
    unet = build_unet(path)
    print(f"built UNet with {sum(1 for _ in unet.modules())} modules "
          f"(no parameters allocated)\n")

    resolved, missing = check_block_map(unet)
    print(f"block map: {len(resolved)} entries resolve, {len(missing)} missing")
    for slot, target, why in missing:
        print(f"  MISSING {slot:10} -> {target}  ({why})")

    covered = {target for _slot, target, _kind, _shape in resolved}
    every = linear_paths(unet)

    targets = list(lora_targets(unet))
    projections = [t for t in targets if t[4] == "projection"]
    norms = [t for t in targets if t[4] != "projection"]
    bad = [t for t in projections if t[2] != "Linear"]

    print(f"\nUNetXLBase rows walked: {len(targets)} "
          f"({len(projections)} projections, {len(norms)} norms)")
    print(f"  {len(projections) - len(bad)}/{len(projections)} projections "
          f"land on an nn.Linear")
    for key, path, why, _shape, _kind in bad:
        print(f"  NOT-LINEAR  {why:22} {key} -> {path}")
    print(f"  {len(norms)} non-Linear rows were refused by the resolver and "
          f"skipped: {sorted({t[1].rsplit('.', 1)[-1] for t in norms})[:6]}")

    reachable = {
        f"{path}.weight"
        for path in covered
    }
    reachable |= {
        f"{t[1]}.weight" for t in targets if t[1] is not None
    }
    unreachable = [p for p in every if p not in reachable]
    print(f"\nreal nn.Linear weights in the tree: {len(every)}")
    print(f"  {len(every) - len(unreachable)} reachable from the harvested table "
          f"or a block-map slot")
    print(f"  {len(unreachable)} reachable from neither")
    # These are Draw Things' own omission, verified against the pinned revision
    # (060f99d): UNetXLBase contains zero `attn2.to_k` / `attn2.to_v` rows and
    # zero `time_emb_proj` rows, while the SD 1.x `UNet` table has 16 of the
    # first two. So a Draw Things SDXL LoRA cannot target SDXL cross-attention
    # k/v. Not a harvest loss -- do not "fix" it in the harvester.
    dt_omitted = [p for p in unreachable
                  if ".attn2.to_k." in p or ".attn2.to_v." in p
                  or "time_emb_proj" in p or "add_embedding" in p
                  or "time_embedding" in p]
    print(f"    of which {len(dt_omitted)} are Draw Things' own omission "
          f"(attn2.to_k / attn2.to_v / time_emb_proj / add_embedding / "
          f"time_embedding)")
    for p in [x for x in unreachable if x not in dt_omitted][:12]:
        print(f"    UNEXPECTED {p}")
    if len([x for x in unreachable if x not in dt_omitted]) > 12:
        print("    ... and more")

    if "--shapes" in flags:
        print("\ncovered Linear shapes:")
        for slot, target, kind, shape in resolved:
            print(f"  {slot:10} {target:44} {kind:12} {shape}")

    ok = not missing and not bad
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))