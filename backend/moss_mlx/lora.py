import gc
from pathlib import Path

import mlx.core as mx
from mlx import nn

_ATTRS = {
    "q": "proj_q",
    "k": "proj_k",
    "v": "proj_v",
    "o": "proj_o",
    "ffn_up_gate": "proj_ffn_up_gate",
    "ffn_down": "proj_ffn_down",
}


class MOSSQ4LoRALinear(nn.Module):
    """LoRA residual around an existing Q4 QuantizedLinear."""

    def __init__(self, base: nn.Module, down: mx.array, up: mx.array, scale: float):
        super().__init__()
        self.linear = base
        self.scale = float(scale)
        self.lora_A = mx.transpose(down)   # (in, r)
        self.lora_B = mx.transpose(up)     # (r, out)

    def __call__(self, x):
        return self.linear(x) + self.scale * mx.matmul(mx.matmul(x, self.lora_A), self.lora_B)


def apply_moss_lora(transformer: nn.Module, lora_path: str | Path, scale: float = 1.0) -> int:
    path = Path(lora_path)
    if not path.is_file():
        raise FileNotFoundError(f"MOSS LoRA not found: {path}")
    raw = mx.load(str(path))
    rank = 128
    meta_rank = None
    try:
        import struct, json
        with path.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            meta = json.loads(fh.read(n)).get("__metadata__") or {}
        meta_rank = int(meta.get("rank", 0) or 0) or None
    except Exception:
        meta_rank = None
    pending = {}
    for key, value in raw.items():
        prefix = "diffusion_model."
        if not key.startswith(prefix):
            continue
        key = key[len(prefix):]
        if key.startswith("layers.") and ".proj." in key and key.endswith(".lora_down.weight"):
            head, _ = key.rsplit(".lora_down.weight", 1)
            layer_s, sub = head.split(".")[1], head.split(".")[3]
            pending.setdefault((int(layer_s), sub), {})["down"] = value
        elif key.startswith("layers.") and ".proj." in key and key.endswith(".lora_up.weight"):
            head, _ = key.rsplit(".lora_up.weight", 1)
            layer_s, sub = head.split(".")[1], head.split(".")[3]
            pending.setdefault((int(layer_s), sub), {})["up"] = value
        elif key.startswith("layers.") and ".proj." in key and key.endswith(".alpha"):
            head, _ = key.rsplit(".alpha", 1)
            layer_s, sub = head.split(".")[1], head.split(".")[3]
            pending.setdefault((int(layer_s), sub), {})["alpha"] = value
    applied = 0
    for (layer_idx, sub), parts in pending.items():
        if "down" not in parts or "up" not in parts:
            continue
        attr = _ATTRS.get(sub)
        if attr is None:
            continue
        block = transformer.layers[layer_idx]
        base = getattr(block, attr)
        alpha = float(parts.get("alpha", meta_rank or parts["down"].shape[0]))
        r = int(parts["down"].shape[0])
        adapter_scale = float(scale) * alpha / max(1, r)
        setattr(block, attr, MOSSQ4LoRALinear(base, parts["down"], parts["up"], adapter_scale))
        applied += 1
    mx.eval(transformer.parameters())
    del raw
    gc.collect()
    mx.clear_cache()
    return applied


def apply_dmd_time_modules(transformer: nn.Module, adapter_dir: str | Path) -> None:
    """Replace time projection weights/biases with the DMD fp32 tensors."""
    path = Path(adapter_dir) / "time_embedding.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"DMD time embedding missing: {path}")
    state = mx.load(str(path))

    def linear(name: str, in_dim: int, out_dim: int) -> nn.Linear:
        layer = nn.Linear(in_dim, out_dim, bias=True)
        layer.weight = state[f"{name}.weight"]
        layer.bias = state[f"{name}.bias"]
        return layer

    transformer.time_embedding_0 = linear("time_embedding.0", 256, 3072)
    transformer.time_embedding_2 = linear("time_embedding.2", 3072, 3072)
    if "shared_block_modulation.bias" in state:
        transformer.shared_block_modulation.bias = state["shared_block_modulation.bias"]
    if "video_head_modulation.bias" in state:
        transformer.video_head_modulation.bias = state["video_head_modulation.bias"]
    mx.eval(transformer.parameters())
    del state
