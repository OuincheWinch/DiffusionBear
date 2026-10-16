"""Pair, validate and reassemble the tensors of a Draw Things LoRA.

A Draw Things `.ckpt` LoRA stores one tensor per `t-<scope>-<layer>-<param>`
slot, `__up__` and `__down__`. Turning that into something mflux or the SDXL
engine will load takes three steps, in this order:

1. Resolve each slot to a target module path -- `drawthings_names.resolve_sd`
   for SD/SDXL, `resolve_flux` for FLUX. An unresolvable slot is a hard error
   naming the tensor, never a guess: a plausible-but-wrong key loads and then
   generates garbage.
2. Reassemble the slots that jointly form one target. Three shapes occur and
   they are not interchangeable; see `merge_pieces`.
3. Validate the `up`/`down` pair and its rank. A rank mismatch still loads and
   still generates, just noise.

Nothing here writes. `convert` in Task 7 does that, atomically.
"""

import os
from pathlib import Path

import numpy as np

import drawthings_names as names


class ConversionError(RuntimeError):
    """Raised when a Draw Things LoRA cannot be converted soundly."""


class Pair:
    """One converted LoRA: a target module path and its two factors.

    Orientation is kohya's and both engines' expectation:
    `down = (rank, in_features)`, `up = (out_features, rank)`. No transpose.
    """

    __slots__ = ("base", "down", "up")

    def __init__(self, base, down, up):
        self.base = base
        self.down = down
        self.up = up


def rank_of(pair):
    """Return the LoRA rank, taken from `down`."""
    return pair.down.shape[0]


# Reassembly modes. See `merge_pieces`.
MERGE_SINGLE = "single"
MERGE_SUM = "sum"
MERGE_BLOCK = "block"
MERGE_SHARED = "shared"


def merge_pieces(pieces, mode):
    """Combine `(up, down)` pieces into one factorisation.

    The mode is a property of the target weight, not of how many pieces there
    are, and all three shapes occur in a real file. Picking the wrong one
    yields an array of exactly the right shape holding the wrong numbers, which
    is Review Focus #1.

    `MERGE_SUM` -- the pieces share the output rows and split the input
    columns, so their products add. FLUX
    `single_transformer_blocks.*.proj_out`: DT computes
    `x_o(attn_out) + xFF(xOut)` with 3072- and 12288-wide factors, and mflux
    computes `proj_out(concat([attn_out, mlp_hidden]))` on one Linear(15360,
    3072). DT's two tensors are literally that Linear's two column blocks.

    `MERGE_BLOCK` -- the pieces split the output rows and share the input
    columns, so the result is block diagonal with the rank axis zero padded.
    FLUX AdaLN chunks, and SDXL `attn.in_proj_weight` whose slots carry
    different down matrices and so are independent LoRAs.

    `MERGE_SHARED` -- one weight split along axis 0 with the down matrix
    duplicated across the slots. SDXL `ff.net.0.proj` and relatives: measured,
    every slot's down is byte-identical to the others, so there is only one
    down and the ups stack.

    `MERGE_SINGLE` passes a lone piece through unchanged.
    """
    if mode == MERGE_SINGLE:
        if len(pieces) != 1:
            raise ConversionError(
                f"{MERGE_SINGLE} merge takes one piece, got {len(pieces)}")
        return Pair(None, pieces[0][1], pieces[0][0])

    if not pieces:
        raise ConversionError("no pieces to merge")

    ranks = {u.shape[1] for u, _ in pieces}
    if len(ranks) != 1:
        raise ConversionError(
            f"pieces disagree on rank: {sorted(ranks)}; a merged LoRA has one rank")
    rank = ranks.pop()

    if mode == MERGE_SHARED:
        widths = {d.shape[1] for _, d in pieces}
        if len(widths) != 1:
            raise ConversionError(
                f"shared-down merge needs one input width, got {sorted(widths)}")
        first = pieces[0][1]
        for _, d in pieces[1:]:
            if not np.array_equal(first, d):
                raise ConversionError(
                    "shared-down merge but the down matrices differ; picking one "
                    "of them would emit a LoRA that loads and generates nothing")
        return Pair(None, first, np.concatenate([u for u, _ in pieces], axis=0))

    if mode == MERGE_SUM:
        widths = {u.shape[0] for u, _ in pieces}
        if len(widths) != 1:
            raise ConversionError(
                f"sum merge needs one output width, got {sorted(widths)}")
        # The pieces split the input columns, so each down occupies its own
        # column block of the summed input width, and the ups sit side by side
        # along the rank axis. `np.concatenate` cannot do this because the
        # blocks have different widths, so the target is allocated and filled.
        offsets = []
        offset = 0
        for _, d in pieces:
            offsets.append(offset)
            offset += d.shape[1]
        down = np.zeros((rank * len(pieces), offset), dtype=np.float32)
        for i, ((_u, d), start) in enumerate(zip(pieces, offsets)):
            down[i * rank:(i + 1) * rank, start:start + d.shape[1]] = d
        return Pair(None, down,
                    np.concatenate([u for u, _ in pieces], axis=1))

    if mode == MERGE_BLOCK:
        widths = {d.shape[1] for _, d in pieces}
        if len(widths) != 1:
            raise ConversionError(
                f"block merge needs one input width, got {sorted(widths)}")
        # The pieces split the output rows, so each up occupies its own row
        # block of the stacked output height, with the rank axis zero padded.
        up = np.zeros((sum(u.shape[0] for u, _ in pieces), rank * len(pieces)),
                      dtype=np.float32)
        down = np.zeros((rank * len(pieces), widths.pop()), dtype=np.float32)
        offset = 0
        for i, (u, d) in enumerate(pieces):
            up[offset:offset + u.shape[0], i * rank:(i + 1) * rank] = u
            down[i * rank:(i + 1) * rank, :] = d
            offset += u.shape[0]
        return Pair(None, down, up)

    raise ConversionError(f"unknown merge mode {mode!r}")


def detect_merge(pieces):
    """Return the merge mode for a slot group whose mode is not fixed by a table.

    Used for SDXL, where the harvested tables say which ids share a weight but
    not how they factorise it. Measured on
    `add_detail_xl_lora_f16.ckpt`: all 70 fused UNet groups have byte-identical
    downs across their slots and all 32 `attn.in_proj_weight` groups have all
    different ones. No group is mixed. A mixed group is ambiguous and refused,
    because both readings are arithmetically valid and nothing in the data says
    which is right.
    """
    if len(pieces) == 1:
        return MERGE_SINGLE
    first = pieces[0][1]
    same = [np.array_equal(first, d) for _, d in pieces[1:]]
    if all(same):
        return MERGE_SHARED
    if not any(same):
        return MERGE_BLOCK
    raise ConversionError(
        "ambiguous fused slot: some down matrices match and some do not, so the "
        "group is neither one shared down nor independent pieces")


def _check_pair(base, down, up):
    """Validate one factorisation, naming the target in any refusal."""
    if down.ndim != 2 or up.ndim != 2:
        raise ConversionError(
            f"{base}: LoRA factors must be 2-D, got down{down.shape} up{up.shape}")
    if down.shape[0] != up.shape[1]:
        raise ConversionError(
            f"{base}: rank mismatch, down is {down.shape} and up is {up.shape}; "
            f"down.shape[0]={down.shape[0]} must equal up.shape[1]={up.shape[1]}")
    if down.shape[0] == 0:
        raise ConversionError(f"{base}: rank 0")


# SDXL output key prefixes. The SDXL engine loads its UNet and both text
# encoders separately, so the same diffusers path needs three different roots.
_SDXL_PREFIX = {
    "__unet__": "unet",
    "__unet_fixed__": "unet",
    "__text_model__": "text_encoder",
    "__te2__text_model__": "text_encoder_2",
}

def _collect(store, target_of, skippable=None, skipped=None):
    """Read every LoRA tensor once, grouped by its resolved target.

    `target_of(parsed, name)` returns `(group_key, mode, position, declared)`
    or `None` when the tensor has no destination. When `skippable(name, parsed)`
    returns True the tensor is recorded in `skipped` and conversion continues;
    every other undestined tensor is a hard error. `mode` may be `None`,
    meaning "decide from the data once the group is whole" -- used for
    SDXL, where the harvested tables say which ids share a weight but not how
    they factorise. `declared` is the slice count the resolver knows about,
    which can exceed the number of tensors the file actually supplies.

    Returns `{group_key: (mode, {position: (down_array, up_array)})}` with every
    position of every group present, or raises naming what is missing.
    """
    groups = {}
    for name in store.names():
        parsed = names.parse_name(name)
        if not parsed.is_lora:
            continue
        placement = target_of(parsed, name)
        if placement is None:
            if skippable is not None and skippable(name, parsed):
                skipped.append(name)
                continue
            raise ConversionError(
                f"no destination for tensor {name!r} "
                f"(scope {parsed.scope!r}, layer {parsed.layer}, param {parsed.param})")
        key, mode, position, declared = placement
        group = groups.setdefault(key, {"mode": mode, "slots": {}, "total": None})
        if group["mode"] != mode:
            raise ConversionError(
                f"{key}: merge mode {group['mode']!r} conflicts with {mode!r}")
        if declared is not None:
            if group["total"] is not None and group["total"] != declared:
                raise ConversionError(
                    f"{key}: slice count {group['total']} conflicts with {declared}")
            group["total"] = declared
        slot = group["slots"].setdefault(position, {})
        if parsed.part in slot:
            raise ConversionError(
                f"{key}: two {parsed.part} tensors at position {position}: "
                f"{slot[parsed.part]} and {name!r}")
        _shape, array = store.read(name)
        slot[parsed.part] = np.asarray(array, dtype=np.float32)
        slot[f"_{parsed.part}_name"] = name

    result = {}
    for key, group in groups.items():
        slots = group["slots"]
        for position, slot in sorted(slots.items()):
            if set(slot) & {"up", "down"} != {"up", "down"}:
                missing = sorted({"up", "down"} - set(slot))
                present = sorted(v for k, v in slot.items() if k.endswith("_name"))
                raise ConversionError(
                    f"{key}: no {'/'.join(missing)} tensor at position {position}"
                    + (f"; the tensors that are present are {present}" if present
                       else ""))
        result[key] = (group["mode"], slots, group["total"])
    return result


def _assemble(store_groups):
    """Turn collected groups into validated `Pair`s."""
    pairs = []
    for base, (mode, slots, _total) in store_groups.items():
        pieces = [(slots[p]["up"], slots[p]["down"]) for p in sorted(slots)]
        if mode is None:
            mode = detect_merge(pieces)
        merged = merge_pieces(pieces, mode)
        _check_pair(base, merged.down, merged.up)
        pairs.append(Pair(base, merged.down, merged.up))
    return pairs


# SDXL output key prefixes. The SDXL engine loads its UNet and both text
# encoders separately, so one diffusers path needs three different roots.

# FLUX keys need NO root prefix. mflux matches weight keys by exact string
# equality against `FluxLoRAMapping`'s patterns, which are already rooted at
# `transformer_blocks.{block}...` / `single_transformer_blocks.{block}...`.
# Prefixing them again matches nothing and mflux then reports "No LoRA layers
# were applied".


def _sdxl_placement(tables):
    """Return `{(prefix, tid): (output_base, position, total)}`.

    Each id is resolved on its own through `resolve_sd`, so a prefix whose target
    layout differs from the harvested key -- `__te2__text_model__`, which is
    open_clip spelling on the way to a HuggingFace `CLIPTextModelWithProjection`
    -- lands on the right module. `total` comes from `slot_of`, which reports 1
    for those independent projections so they are not merged into one
    triple-rank tensor.
    """
    placement = {}
    for prefix, table_list in names.PREFIX_TO_TABLES.items():
        root = _SDXL_PREFIX.get(prefix)
        if root is None:
            continue
        for tid in _ids_for(tables, prefix):
            layer, param = _split_tid(tid)
            found = names.slot_of(prefix, layer, param)
            if found is None:
                continue
            _path, position, total = found
            path = names.resolve_sd(prefix, layer, param)
            if path is None:
                continue
            placement[(prefix, tid)] = (f"{root}.{path}", position, total)
    return placement


def _ids_for(tables, prefix):
    """Yield every tensor id reachable from `prefix`."""
    for table in names.PREFIX_TO_TABLES[prefix]:
        for entries in names.group_fused({table: tables[table]})[table].values():
            for _position, tid in entries:
                yield tid


def _split_tid(tid):
    """Split `t-<scope>-<layer>-<param>` from the right.

    The scope is optional in the grammar, so splitting from the left would raise
    a bare `ValueError: too many values to unpack` on a scoped id instead of a
    `ConversionError` naming the tensor.
    """
    _t, layer, param = tid.rsplit("-", 2)
    return int(layer), int(param)


def _build_sdxl(store, tables):
    placement = _sdxl_placement(tables)

    def target_of(parsed, tensor_name):
        if parsed.prefix not in _SDXL_PREFIX:
            return None
        found = placement.get((parsed.prefix, f"t-{parsed.layer}-{parsed.param}"))
        if found is None:
            return None
        base, position, total = found
        # One id means one whole weight; several ids share one and their
        # factorisation is decided from the data.
        return base, (MERGE_SINGLE if total == 1 else None), position, total

    return _assemble(_collect(store, target_of))


def _build_flux(store, skipped):
    def target_of(parsed, tensor_name):
        target = names.resolve_flux(parsed.scope, parsed.layer, parsed.param)
        if target is None:
            return None
        key = target.module_path
        mode = MERGE_SINGLE
        if target.merged:
            mode = MERGE_SUM if target.merge_axis == 1 else MERGE_BLOCK
        return key, mode, target.slice_index, target.slice_count

    collected = _collect(store, target_of, skippable=_flux_skippable,
                         skipped=skipped)

    # A merged FLUX target may declare more slices than the file supplies:
    # mflux's `norm1.linear` is six chunks wide and DT only trains three of
    # them. The rest are zeros, which is what a LoRA that did not train them
    # means. Their shape is taken from a sibling slice -- the merge validators
    # require every piece of a group to share its width anyway.
    filled = {}
    for key, (mode, slots, declared) in collected.items():
        total = declared if declared is not None else max(slots) + 1
        if len(slots) < total:
            template = slots[min(slots)]
            for position in range(total):
                if position not in slots:
                    slots[position] = {
                        "up": np.zeros(template["up"].shape, dtype=np.float32),
                        "down": np.zeros(template["down"].shape, dtype=np.float32),
                    }
        filled[key] = (mode, slots, declared)
    return _assemble(filled)


def _flux_skippable(name, parsed):
    """True for a FLUX tensor whose target model has no module for it.

    `x_ada_ln_3/4/5` are the dual block's norm2 modulation. mflux's joint block
    has `norm2` as a non-affine LayerNorm with no Linear behind it, so there is
    nothing to attach to. Every real FLUX.1 LoRA carries these -- 57 of the 798
    pairs in `bitcoin_lora_flux1d_f16_lora_f16.ckpt` -- so refusing the whole
    file would make FLUX conversion useless, while silently dropping them would
    be the failure this project treats as worst. They are skipped and counted,
    and the count is reported in the conversion summary.
    """
    return parsed.scope in names.FLUX_UNADDRESSABLE


def build_pairs(store, base_model, report=None):
    """Return every converted `Pair`, or raise `ConversionError`.

    `base_model` is `"sdxl"` or `"flux"`. Nothing is written and nothing is
    guessed: any tensor without a destination, any incomplete or mismatched
    pair, and any ambiguous fused group stops the conversion and names the
    tensor.

    The one exception is FLUX's norm2 modulation, which has no module in the
    target model at all. Those tensors are skipped rather than refused, and
    when `report` is given they are listed under `report["skipped"]` so the
    caller can say so out loud.
    """
    skipped = []
    if base_model == "sdxl":
        pairs = _build_sdxl(store, names.load_name_map())
    elif base_model == "flux":
        pairs = _build_flux(store, skipped)
    else:
        raise ConversionError(
            f"unknown base model {base_model!r}; expected 'sdxl' or 'flux'")
    if not pairs:
        raise ConversionError("no LoRA tensors found")
    pairs.sort(key=lambda p: p.base)
    if report is not None:
        report["skipped"] = sorted(skipped)
        report["skipped_tensors"] = len(skipped)
    return pairs

# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

# The Draw Things model directory is read-only. `Store.open` already enforces
# that with a percent-escaped URI plus `PRAGMA query_only`, and the converter
# must never write there either -- a LoRA converter that silently overwrote a
# user's model store would be the worst thing this code could do.
DRAW_THINGS_MODEL_DIR = (
    Path.home() / "Library/Containers/com.liuliu.draw-things/Data/Documents/Models"
)
CONTAINERS_ROOT = Path.home() / "Library/Containers"
PROTECTED_ROOTS = (DRAW_THINGS_MODEL_DIR, CONTAINERS_ROOT)


def _protected_roots() -> tuple[Path, ...]:
    """The protected roots, resolved at CALL time.

    The module-level constants above are captured at import, which is a problem: the
    test suite replaces `Path.home()` process-wide and does not restore it, so from
    that point the captured roots point at a temp directory and `_check_output` stops
    protecting the real tree. That is not hypothetical -- it is how a Draw Things
    read-only test started passing for the wrong reason, and how `sources/drawthings.py`
    ended up with a `model_dir()` function specifically to avoid the same trap.

    These names stay defined for callers that import them, but anything that enforces
    the rule uses this function.
    """
    home = Path.home()
    roots = [
        home / "Library/Containers/com.liuliu.draw-things/Data/Documents/Models",
        home / "Library/Containers",
        home / ".diffusionbee",
    ]
    custom_db = os.environ.get("DIFFUSIONBEE_ROOT")
    if custom_db:
        roots.append(Path(custom_db))
    return tuple(roots)


def _check_output(out_path, allow_root=None):
    """Refuse an unsafe output path, before writing anything.

    The path is resolved first: a relative path or one containing `..` walks
    out of the directory it appears to name.

    A denylist alone only stops the paths it names, so `allow_root` inverts the
    test: when given, the output must be inside it. That is the asset-directory
    rule (Review Focus #4); a caller converting into a scratch directory has to
    say so explicitly rather than have it permitted by omission.

    The protected roots are checked FIRST and unconditionally. `allow_root` used to
    return early, before them, which meant a caller naming the Draw Things folder as
    its allowed root could write straight into the user's model library -- the one
    directory this app must never mutate. An allow-list says where a write is
    permitted; it cannot grant permission the denylist already refuses.
    """
    resolved = Path(out_path).resolve()
    for root in _protected_roots():
        root_resolved = root.resolve()
        if resolved == root_resolved or root_resolved in resolved.parents:
            if root_resolved == _protected_roots()[0].resolve():
                raise ConversionError(
                    f"refusing to write {resolved}: the Draw Things model "
                    "directory is read-only")
            raise ConversionError(
                f"refusing to write {resolved}: it is inside {root_resolved}")
    if allow_root is not None:
        root = Path(allow_root).resolve()
        if resolved != root and root not in resolved.parents:
            raise ConversionError(
                f"refusing to write {resolved}: it is outside the permitted "
                f"root {root}")
    return resolved


def _tensors_for(pairs, alpha=None):
    """Return the safetensors dict for `pairs`.

    Key spelling is kohya's and both engines' expectation:
    `<base>.lora_down.weight` is `(rank, in_features)` and
    `<base>.lora_up.weight` is `(out_features, rank)`. No transpose.

    `alpha` is omitted unless the caller supplies one. mflux falls back to the
    user scale and the SDXL engine defaults alpha to rank, both giving an
    effective scale of 1.0, which is what a Draw Things LoRA already is --
    DT stores no alpha, so emitting one would silently rescale the whole file.
    """
    tensors = {}
    for pair in pairs:
        for attribute, suffix in (("down", "lora_down"), ("up", "lora_up")):
            key = f"{pair.base}.{suffix}.weight"
            if key in tensors:
                raise ConversionError(
                    f"two pairs share the base {pair.base!r}; the converted "
                    "file would silently drop one of them")
            tensors[key] = np.ascontiguousarray(
                getattr(pair, attribute), dtype=np.float32)
        if alpha is not None:
            tensors[f"{pair.base}.alpha"] = np.array(
                [float(alpha)], dtype=np.float32)
    return tensors


def _save_file(tensors, path):
    from safetensors.numpy import save_file
    save_file(tensors, str(path))


def write_pairs(pairs, out_path, alpha=None, allow_root=None):
    """Write `pairs` to `out_path` atomically.

    Writes to a sibling temporary file and `os.replace`s it into place, so a
    reader never sees a partial file and a failure leaves nothing behind. The
    output path is refused up front if it is inside a protected tree, and must
    be inside `allow_root` when one is given.
    """
    resolved = _check_output(out_path, allow_root)
    tensors = _tensors_for(pairs, alpha=alpha)
    tmp = resolved.with_name(resolved.name + ".tmp")
    try:
        _save_file(tensors, tmp)
        os.replace(tmp, resolved)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        "pairs": len(pairs),
        "tensors": len(tensors),
        "bytes": resolved.stat().st_size,
    }


def convert(ckpt_path, out_path, base_model, alpha=None, allow_root=None):
    """Convert a Draw Things `.ckpt` LoRA to safetensors at `out_path`.

    The Draw Things directory is only ever read. Returns
    `{"tensors": int, "pairs": int, "bytes": int, "skipped_tensors": int}` --
    `skipped_tensors` is non-zero only for FLUX's norm2 modulation, which has no
    module in the target model, and is reported rather than dropped quietly.
    """
    from drawthings_codec import open_store

    if base_model not in ("sdxl", "flux"):
        raise ConversionError(
            f"unknown base model {base_model!r}; expected 'sdxl' or 'flux'")
    report = {}
    with open_store(ckpt_path) as store:
        pairs = build_pairs(store, base_model, report=report)
    summary = write_pairs(pairs, out_path, alpha=alpha, allow_root=allow_root)
    summary["skipped_tensors"] = report.get("skipped_tensors", 0)
    return summary
