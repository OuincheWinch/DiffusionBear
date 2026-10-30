# Shared Qwen 3.5 (Qwen3_5Model, GatedDeltaNet hybrid) MLX support for the
# MOSS engine text encoder.
#
# WHY A LOCAL PORT (do not delete without re-reading this):
#   The official `moss-image-sft/text_encoder` is a bare Qwen3_5Model (no
#   vision tower used at runtime, no lm_head). Its keys are checkpointed under
#   a `language_model.` prefix while mlx-lm's `TextModel` expects `model.*`,
#   and the distributed weights include the full VLM (visual.*) plus MTP
#   layers that the text-only path must never load. `mlx_lm.convert` cannot
#   bridge that, so this module owns the mapping/quantisation/load cycle:
#
#   1. `text_weights()` -- strip `language_model.`, prepend `model.`, drop
#      visual/mtp, and apply mlx-lm's `TextModel.sanitize` fixes.
#   2. `QUANT_PREDICATE` -- 4-bit affine g64 on the big linears + embeddings
#      ONLY. `linear_attn.in_proj_*`/`out_proj`/`conv1d`/`A_log`/`dt_bias`
#      and every RMSNorm stay bf16: the GDN delta-rule mixes scale/offset
#      semantics inside one projection, so quantising it shifts state
#      trajectories far more than a same-size quantise elsewhere.
#   3. `load_mlx_text_model()` -- build, quantise, load, leave lazy.
#
# Fork/local-package note (see AGENTS.md "Forked packages"): this pins
# behaviour against mlx-lm 0.31.x `models/qwen3_5.py`; re-verify after any
# mlx-lm upgrade (pair the norms shift and the linear/full layer split).
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


_SRC_PREFIX = "language_model."
_DST_PREFIX = "model."
_EXCLUDE_MARKERS = ("visual.", "mtp.")


def text_weights(src_dir: Path):
    """Memory-mapped text-only weight dict keyed for mlx-lm TextModel."""
    weights = {}
    files: set[str] = set()
    import json

    with open(src_dir / "model.safetensors.index.json") as fh:
        idx = json.load(fh)["weight_map"]
    for key, fname in idx.items():
        if key.startswith(_SRC_PREFIX):
            files.add(fname)
    for fname in sorted(files):
        weights.update(mx.load(str(src_dir / fname)))
    mapped = {}
    for key, value in weights.items():
        if not key.startswith(_SRC_PREFIX):
            continue
        if any(key.startswith(_SRC_PREFIX + m) for m in _EXCLUDE_MARKERS):
            continue
        mapped[_DST_PREFIX + key[len(_SRC_PREFIX):]] = value
    return mapped


_QUANT_EXCLUDES = (
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
    "out_proj", "conv1d", "a_log", "dt_bias",
)


def quant_predicate(path: str, module, config=None) -> bool:
    """g64 q4 subset: big linears + embeddings; GDN projections stay bf16."""
    if config is None:
        pass  # nn.quantize calls with (path, module); mlx-lm adds config.
    from mlx.nn.layers.base import Module

    p = path.lower()
    cls = type(module).__name__
    if cls not in ("Linear", "Embedding"):
        return False
    if "linear_attn" in p and any(m in p for m in _QUANT_EXCLUDES):
        return False
    if cls == "Linear" and min(module.weight.shape) < 2048:
        return False
    return True


def build_text_model(config: dict):
    from mlx_lm.models import qwen3_5

    # The converted config nests the source text_config under "text_config";
    # TextModelArgs fields live there, so read that dict directly.
    text_cfg = config.get("text_config") or config
    return qwen3_5.TextModel(qwen3_5.TextModelArgs.from_dict(text_cfg))


def load_mlx_text_model(dir_path: Path):
    """Instantiate the (possibly quantised) TextModel from a converted dir.

    Returns the TextModel itself: callers drive `.model(...)` (the post-norm
    hidden states) and never need this module's lm_head.
    """
    import json

    with open(dir_path / "config.json") as fh:
        config = json.load(fh)
    model = build_text_model(config)
    quant = config.get("quantization")
    if quant:
        nn.quantize(
            model, bits=quant["bits"], group_size=quant["group_size"],
            class_predicate=quant_predicate)
    # Weights were saved as tree_flatten(model.model.*) — the TextModel wraps
    # the Qwen3_5TextModel, so load into the inner module directly.
    model.model.load_weights(list(mx.load(str(dir_path / "model.safetensors")).items()))
    return model


def convert_text_encoder(src_dir: Path, out_dir: Path) -> None:
    """One-time conversion: bf16 HF text-only checkpoint -> MLX q4g64 dir."""
    import json

    config = {
        "model_type": "qwen3_5",
        "quantization": {"bits": 4, "group_size": 64},
    }
    with open(src_dir / "config.json") as fh:
        src_config = json.load(fh)
    config["text_config"] = src_config.get("text_config", src_config)
    weights = text_weights(Path(src_dir))
    weights = build_text_model(config).sanitize(weights)
    model = build_text_model(config)
    model.load_weights(list(weights.items()), strict=False)
    # Quantise the already-loaded bf16 module weights in place, the
    # transformer's own quantitative recipe lives with the model, not the file.
    nn.quantize(model, bits=config["quantization"]["bits"],
                group_size=config["quantization"]["group_size"],
                class_predicate=quant_predicate)
    del weights
    # Materialise once: saving touches the same arrays; no reason to keep the
    # raw (mapped) dict alive afterwards.
    import mlx.utils as mlx_utils

    flat = mlx_utils.tree_flatten(model.model.parameters())
    total = sum(v.nbytes for _, v in flat)
    print("param materialised", f"{total / 1e9:.2f} GB bytes", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ~5.5GB: one file keeps the on-disk layout identical to single-shard
    # mlx-lm repos and saves a shard-index edge case.
    mx.save_safetensors(str(out_dir / "model.safetensors"), dict(flat))
    del flat, model
    for name in ("tokenizer.json", "tokenizer_config.json"):
        token_file = src_dir.parent / "tokenizer" / name
        if token_file.exists():
            import shutil
            shutil.copyfile(token_file, out_dir / name)
    with open(out_dir / "config.json", "w") as fh:
        json.dump(config, fh, indent=1)
    print("CONVERTED ->", out_dir, flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    convert_text_encoder(Path(args.src), Path(args.out))
