"""Pure MLX implementation of TAESD (Tiny AutoEncoder for SDXL) by Ollin Boer Bohan.
Runs in ~0.5s on Apple Silicon Metal with zero PyTorch runtime dependency.
"""
import gc
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

class MLXTinyBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv_0 = nn.Conv2d(c, c, 3, padding=1)
        self.conv_2 = nn.Conv2d(c, c, 3, padding=1)
        self.conv_4 = nn.Conv2d(c, c, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        y = nn.relu(self.conv_0(x))
        y = nn.relu(self.conv_2(y))
        y = self.conv_4(y)
        return nn.relu(y + x)

class MLXTAESDDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_in = nn.Conv2d(4, 64, 3, padding=1)
        self.b0_0 = MLXTinyBlock(64); self.b0_1 = MLXTinyBlock(64); self.b0_2 = MLXTinyBlock(64)
        self.up0 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c0 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.b1_0 = MLXTinyBlock(64); self.b1_1 = MLXTinyBlock(64); self.b1_2 = MLXTinyBlock(64)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.b2_0 = MLXTinyBlock(64); self.b2_1 = MLXTinyBlock(64); self.b2_2 = MLXTinyBlock(64)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.b3_0 = MLXTinyBlock(64)
        self.c_out = nn.Conv2d(64, 3, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.tanh(x / 3.0) * 3.0
        x = nn.relu(self.c_in(x))
        x = self.b0_0(x); x = self.b0_1(x); x = self.b0_2(x)
        x = self.c0(self.up0(x))
        x = self.b1_0(x); x = self.b1_1(x); x = self.b1_2(x)
        x = self.c1(self.up1(x))
        x = self.b2_0(x); x = self.b2_1(x); x = self.b2_2(x)
        x = self.c2(self.up2(x))
        x = self.b3_0(x)
        x = self.c_out(x)
        return x

import threading

_CACHED_TAESD = None
_COMPILED_TAESD = None
_CACHED_TAESD_PATH = None
_TAESD_LOCK = threading.Lock()

_TAESD_PT_TO_MLX_MAP = {
    "c_in.weight": "decoder.layers.0.weight", "c_in.bias": "decoder.layers.0.bias",
    "b0_0.conv_0.weight": "decoder.layers.2.conv.0.weight", "b0_0.conv_0.bias": "decoder.layers.2.conv.0.bias",
    "b0_0.conv_2.weight": "decoder.layers.2.conv.2.weight", "b0_0.conv_2.bias": "decoder.layers.2.conv.2.bias",
    "b0_0.conv_4.weight": "decoder.layers.2.conv.4.weight", "b0_0.conv_4.bias": "decoder.layers.2.conv.4.bias",
    "b0_1.conv_0.weight": "decoder.layers.3.conv.0.weight", "b0_1.conv_0.bias": "decoder.layers.3.conv.0.bias",
    "b0_1.conv_2.weight": "decoder.layers.3.conv.2.weight", "b0_1.conv_2.bias": "decoder.layers.3.conv.2.bias",
    "b0_1.conv_4.weight": "decoder.layers.3.conv.4.weight", "b0_1.conv_4.bias": "decoder.layers.3.conv.4.bias",
    "b0_2.conv_0.weight": "decoder.layers.4.conv.0.weight", "b0_2.conv_0.bias": "decoder.layers.4.conv.0.bias",
    "b0_2.conv_2.weight": "decoder.layers.4.conv.2.weight", "b0_2.conv_2.bias": "decoder.layers.4.conv.2.bias",
    "b0_2.conv_4.weight": "decoder.layers.4.conv.4.weight", "b0_2.conv_4.bias": "decoder.layers.4.conv.4.bias",
    "c0.weight": "decoder.layers.6.weight",
    "b1_0.conv_0.weight": "decoder.layers.7.conv.0.weight", "b1_0.conv_0.bias": "decoder.layers.7.conv.0.bias",
    "b1_0.conv_2.weight": "decoder.layers.7.conv.2.weight", "b1_0.conv_2.bias": "decoder.layers.7.conv.2.bias",
    "b1_0.conv_4.weight": "decoder.layers.7.conv.4.weight", "b1_0.conv_4.bias": "decoder.layers.7.conv.4.bias",
    "b1_1.conv_0.weight": "decoder.layers.8.conv.0.weight", "b1_1.conv_0.bias": "decoder.layers.8.conv.0.bias",
    "b1_1.conv_2.weight": "decoder.layers.8.conv.2.weight", "b1_1.conv_2.bias": "decoder.layers.8.conv.2.bias",
    "b1_1.conv_4.weight": "decoder.layers.8.conv.4.weight", "b1_1.conv_4.bias": "decoder.layers.8.conv.4.bias",
    "b1_2.conv_0.weight": "decoder.layers.9.conv.0.weight", "b1_2.conv_0.bias": "decoder.layers.9.conv.0.bias",
    "b1_2.conv_2.weight": "decoder.layers.9.conv.2.weight", "b1_2.conv_2.bias": "decoder.layers.9.conv.2.bias",
    "b1_2.conv_4.weight": "decoder.layers.9.conv.4.weight", "b1_2.conv_4.bias": "decoder.layers.9.conv.4.bias",
    "c1.weight": "decoder.layers.11.weight",
    "b2_0.conv_0.weight": "decoder.layers.12.conv.0.weight", "b2_0.conv_0.bias": "decoder.layers.12.conv.0.bias",
    "b2_0.conv_2.weight": "decoder.layers.12.conv.2.weight", "b2_0.conv_2.bias": "decoder.layers.12.conv.2.bias",
    "b2_0.conv_4.weight": "decoder.layers.12.conv.4.weight", "b2_0.conv_4.bias": "decoder.layers.12.conv.4.bias",
    "b2_1.conv_0.weight": "decoder.layers.13.conv.0.weight", "b2_1.conv_0.bias": "decoder.layers.13.conv.0.bias",
    "b2_1.conv_2.weight": "decoder.layers.13.conv.2.weight", "b2_1.conv_2.bias": "decoder.layers.13.conv.2.bias",
    "b2_1.conv_4.weight": "decoder.layers.13.conv.4.weight", "b2_1.conv_4.bias": "decoder.layers.13.conv.4.bias",
    "b2_2.conv_0.weight": "decoder.layers.14.conv.0.weight", "b2_2.conv_0.bias": "decoder.layers.14.conv.0.bias",
    "b2_2.conv_2.weight": "decoder.layers.14.conv.2.weight", "b2_2.conv_2.bias": "decoder.layers.14.conv.2.bias",
    "b2_2.conv_4.weight": "decoder.layers.14.conv.4.weight", "b2_2.conv_4.bias": "decoder.layers.14.conv.4.bias",
    "c2.weight": "decoder.layers.16.weight",
    "b3_0.conv_0.weight": "decoder.layers.17.conv.0.weight", "b3_0.conv_0.bias": "decoder.layers.17.conv.0.bias",
    "b3_0.conv_2.weight": "decoder.layers.17.conv.2.weight", "b3_0.conv_2.bias": "decoder.layers.17.conv.2.bias",
    "b3_0.conv_4.weight": "decoder.layers.17.conv.4.weight", "b3_0.conv_4.bias": "decoder.layers.17.conv.4.bias",
    "c_out.weight": "decoder.layers.18.weight", "c_out.bias": "decoder.layers.18.bias",
}


def _default_taesd_path() -> Path:
    """Where the TAESD weights live.

    This used to be `Path(__file__).parent / "data" / "models" / "taesdxl"`, which
    is correct in the repo checkout and WRONG in the installed app: inside the signed
    bundle `__file__` is Contents/Resources/backend, so the path resolved to
    Contents/Resources/backend/data/models/taesdxl -- inside the signature. The
    weights are in the model store, so the lookup missed and every SDXL run with
    fast_vae fell back to the full VAE inside an `except` that only printed to
    stderr. The ~0.5s decode was dead in the shipped app and nothing said so.

    ASSET_DIR is the store, and it already honours the `store_path` relocation, so
    this follows the weights wherever the owner put them.
    """
    import app_settings

    return Path(app_settings.ASSET_DIR) / "models" / "taesdxl" / "diffusion_pytorch_model.safetensors"


def get_taesd_decoder(model_path: str | Path | None = None) -> callable:
    global _CACHED_TAESD, _COMPILED_TAESD, _CACHED_TAESD_PATH
    with _TAESD_LOCK:
        if model_path is None:
            model_path = _default_taesd_path()
        model_path = Path(model_path).expanduser().resolve()
        if _COMPILED_TAESD is not None and _CACHED_TAESD_PATH == model_path:
            return _COMPILED_TAESD
        if _COMPILED_TAESD is not None:
            _CACHED_TAESD = None
            _COMPILED_TAESD = None
            _CACHED_TAESD_PATH = None
            gc.collect()
        if not model_path.exists():
            raise FileNotFoundError(f"TAESD model weights not found at {model_path}")

        model = MLXTAESDDecoder()

        def p2m(t):
            arr = mx.array(t)
            return arr.transpose(0, 2, 3, 1) if arr.ndim == 4 else arr

        weights = {}
        with safe_open(str(model_path), framework="numpy") as f:
            for mlx_k, pt_k in _TAESD_PT_TO_MLX_MAP.items():
                weights[mlx_k] = p2m(f.get_tensor(pt_k))

        tree = {}
        for k, v in weights.items():
            parts = k.split(".")
            d = tree
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v

        def update_module(mod, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    update_module(getattr(mod, k), v)
                else:
                    setattr(mod, k, v)

        update_module(model, tree)
        _CACHED_TAESD = model
        _COMPILED_TAESD = mx.compile(model)
        _CACHED_TAESD_PATH = model_path
        return _COMPILED_TAESD


def clear_taesd_cache():
    global _CACHED_TAESD, _COMPILED_TAESD, _CACHED_TAESD_PATH
    with _TAESD_LOCK:
        _CACHED_TAESD = None
        _COMPILED_TAESD = None
        _CACHED_TAESD_PATH = None
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass
