import gc
import json
import os
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .transformer import MOSSImageTransformer

_TE_SUFFIX = "qwen3_5-te-q8g64"


def _shards(path: Path):
    for index_name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        index = path / index_name
        if index.is_file():
            weight_map = json.loads(index.read_text("utf-8")).get("weight_map") or {}
            resolved = sorted(set(path / name for name in weight_map.values()))
            if all(p.is_file() for p in resolved):
                return resolved
    return sorted(path.glob("*.safetensors"))


def _safetensors_has_scales(path: Path) -> bool:
    """Inspect only the safetensors JSON header to see if weights are pre-quantized."""
    try:
        with path.open("rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            if header_len <= 0 or header_len > 64 * 1024 * 1024:
                return False
            header = json.loads(fh.read(header_len).decode("utf-8"))
        return any(str(k).endswith(".scales") for k in header)
    except Exception:
        return False


def _build_transformer(t_dir: Path) -> MOSSImageTransformer:
    cfg = {}
    cfg_path = t_dir / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text("utf-8"))
    return MOSSImageTransformer(
        dim=cfg.get("dim", 3072),
        ffn_dim=cfg.get("ffn_dim", 8192),
        num_heads=cfg.get("num_heads", 24),
        num_layers=cfg.get("num_layers", 42),
        text_dim=cfg.get("text_dim", 4096),
        freq_dim=cfg.get("freq_dim", 256),
        latent_channels=cfg.get("latent_channels", 128),
        rope_base=cfg.get("rope_base", 10000.0),
    )


def _prune_bf16_transformer_files(t_dir: Path, bf16_shards: list[Path], keep_path: Path) -> None:
    keep_resolved = keep_path.resolve()
    for shard in bf16_shards:
        if shard.is_file() and shard.resolve() != keep_resolved:
            shard.unlink()
    for index_name in ("diffusion_pytorch_model.safetensors.index.json", "model.safetensors.index.json"):
        idx = t_dir / index_name
        if idx.is_file():
            idx.unlink()


def ensure_moss_transformer_quantized(
    root: Path,
    *,
    bits: int = 4,
    group_size: int = 64,
    prune_bf16: bool = False,
    progress_cb=None,
) -> Path | None:
    """Quantize BF16/Diffusers MOSS transformer shards one shard at a time into an MLX cache file."""
    t_dir = root / "transformer" if (root / "transformer").is_dir() else root
    if not t_dir.is_dir():
        return None
    q_path = t_dir / f"model-q{int(bits)}-g{int(group_size)}.safetensors"
    source_shards = [p for p in _shards(t_dir) if not p.name.startswith("model-q") and p.is_file()]
    if q_path.is_file() and q_path.stat().st_size > 0 and _safetensors_has_scales(q_path):
        if prune_bf16 and source_shards:
            _prune_bf16_transformer_files(t_dir, source_shards, q_path)
        return q_path
    if not source_shards:
        return q_path if q_path.is_file() else None
    if any(_safetensors_has_scales(s) for s in source_shards):
        return source_shards[0]

    # Build lightweight skeleton solely to discover QuantizedLinear targets and expected keys.
    t_ref = _build_transformer(t_dir)
    nn.quantize(
        t_ref,
        bits=int(bits),
        group_size=int(group_size),
        class_predicate=lambda _, mod: (
            isinstance(mod, nn.Linear)
            and hasattr(mod, "weight")
            and mod.weight.shape[-1] % int(group_size) == 0
        ),
    )
    quant_modules = {p for p, m in t_ref.named_modules() if isinstance(m, nn.QuantizedLinear)}
    expected_keys = {k for k, _ in tree_flatten(t_ref.parameters())}
    del t_ref
    gc.collect()
    mx.clear_cache()

    out_weights = {}
    total_shards = len(source_shards)
    for idx, shard in enumerate(source_shards):
        raw = mx.load(str(shard))
        for raw_key, tensor in raw.items():
            key = _map_transformer_key(raw_key)
            if key.endswith(".weight") and key[:-7] in quant_modules:
                mod_path = key[:-7]
                wq, scales, biases = mx.quantize(tensor, group_size=int(group_size), bits=int(bits))
                mx.eval(wq, scales, biases)
                out_weights[key] = wq
                out_weights[f"{mod_path}.scales"] = scales
                out_weights[f"{mod_path}.biases"] = biases
            elif key in expected_keys:
                mx.eval(tensor)
                out_weights[key] = tensor
        del raw
        gc.collect()
        mx.clear_cache()
        if progress_cb is not None:
            progress_cb(idx + 1, total_shards)

    missing_keys = sorted(expected_keys - set(out_weights.keys()))
    if missing_keys:
        raise RuntimeError(
            f"MOSS transformer quantization incomplete ({len(missing_keys)} missing keys, e.g. {missing_keys[:5]})"
        )

    part = t_dir / f".model-q{int(bits)}-g{int(group_size)}.{os.getpid()}.safetensors"
    try:
        mx.save_safetensors(str(part), out_weights)
        os.replace(part, q_path)
    finally:
        if part.exists():
            part.unlink()
        del out_weights
        gc.collect()
        mx.clear_cache()

    if prune_bf16:
        _prune_bf16_transformer_files(t_dir, source_shards, q_path)
    return q_path


def _map_transformer_key(key: str) -> str:
    # q4 mflux saves use "layers.N.proj.q.*"; the MLX module is "proj_q".
    if key.startswith("layers."):
        parts = key.split(".")
        if len(parts) >= 5 and parts[2] == "proj":
            layer, sub, tail = parts[1], parts[3], parts[4:]
            return ".".join(["layers", layer, f"proj_{sub}", *tail])
        return key
    # mflux save uses "time_embedding.0.*"; module fields are time_embedding_0.
    for idx in ("0", "2"):
        prefix = f"time_embedding.{idx}."
        if key.startswith(prefix):
            return "time_embedding_" + idx + "." + key[len(prefix):]
    return key


def load_moss_transformer(root: Path, *, quantize: int = 4) -> MOSSImageTransformer:
    t_dir = root / "transformer" if (root / "transformer").is_dir() else root
    transformer = _build_transformer(t_dir)

    q_cached = t_dir / f"model-q{int(quantize)}-g64.safetensors"
    if q_cached.is_file() and q_cached.stat().st_size > 0:
        shards = [q_cached]
    else:
        shards = _shards(t_dir)
        if shards and int(quantize) in (4, 8) and not any(_safetensors_has_scales(s) for s in shards):
            quantized = ensure_moss_transformer_quantized(root, bits=int(quantize), group_size=64)
            if quantized is not None and quantized.is_file():
                shards = [quantized]

    raw = {}
    for shard in shards:
        raw.update(mx.load(str(shard)))
    mapped = {_map_transformer_key(k): v for k, v in raw.items()}
    del raw

    quant_paths = {key[:-7] for key in mapped if key.endswith(".scales")}
    if quant_paths:
        nn.quantize(
            transformer,
            bits=int(quantize),
            group_size=64,
            class_predicate=lambda path, module: path in quant_paths,
        )
    transformer.update(tree_unflatten(list(mapped.items())), strict=False)
    expected_keys = {k for k, _ in tree_flatten(transformer.parameters())}
    missing_keys = sorted(expected_keys - set(mapped.keys()))
    if missing_keys:
        raise RuntimeError(
            f"MOSS transformer weights incomplete ({len(missing_keys)} missing keys, e.g. {missing_keys[:5]})"
        )
    mx.eval(transformer.parameters())
    del mapped
    gc.collect()
    mx.clear_cache()
    return transformer


def _mlx_te_dir(root: Path) -> Path:
    candidates = [
        root.parent / _TE_SUFFIX,
        root / _TE_SUFFIX,
        Path(__file__).resolve().parents[1] / "data" / "models" / _TE_SUFFIX,
    ]
    try:
        import app_settings
        candidates.append(Path(app_settings.ASSET_DIR) / "models" / _TE_SUFFIX)
    except Exception:
        pass
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(
        f"MOSS Q4 requires the MLX Qwen text encoder at {root.parent / _TE_SUFFIX}"
    )


from moss_paths import find_shared_moss_tokenizer_dir


def _moss_tokenizer_dir(root: Path) -> Path:
    direct = root / "tokenizer"
    if (direct / "chat_template.jinja").is_file():
        return direct
    shared = find_shared_moss_tokenizer_dir(exclude_dir=None)
    if shared is not None:
        return shared
    raise FileNotFoundError("MOSS tokenizer not found; expected data/models/moss-tokenizer")


def load_moss_text_encoder(root: Path):
    from transformers import AutoTokenizer
    from moss_qwen35_mlx import load_mlx_text_model

    q8_dir = _mlx_te_dir(root)
    model = load_mlx_text_model(q8_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(_moss_tokenizer_dir(root)), local_files_only=True,
    )
    return model, tokenizer


def encode_moss_prompt(model, tokenizer, text: str, max_length: int = 4096) -> mx.array:
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    encoded = tokenizer(
        formatted,
        return_tensors="np",
        truncation=True,
        max_length=max_length,
    )
    ids = mx.array(encoded["input_ids"])
    out = model.model(ids)
    hidden = out[0] if isinstance(out, (tuple, list)) else out
    if hidden.ndim == 2:
        hidden = hidden[None, :, :]
    hidden = hidden.astype(mx.bfloat16)
    mx.eval(hidden)
    return hidden


def release_text_encoder(model=None):
    if model is not None:
        del model
    gc.collect()
    mx.clear_cache()
