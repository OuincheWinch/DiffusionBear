"""Optional FLUX.2-klein 9B pruned 5.1B MLX Q4-G64 text encoder manager.

Existing installations keep using their already-downloaded full 9B text encoder
with zero forced downloads. Users on 8 GB / 16 GB Macs can opt in from
Parameters -> Engine to download and toggle `Ouinche/flux2-klein-9b-qwen3-te-pruned-5.1b-mlx-q4g64`
(2.57 GiB vs 9.50 GiB BF16).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import app_settings

SMALL_TE_REPO = "Ouinche/flux2-klein-9b-qwen3-te-pruned-5.1b-mlx-q4g64"
SMALL_TE_DIR_NAME = "flux2-klein-9b-te-pruned-q4"
SMALL_TE_SIZE_GB = 2.57
SMALL_TE_REQUIRED_FILES = ("config.json", "pruning_metadata.json", "model.safetensors")
SMALL_TE_ALL_FILES = (
    "config.json",
    "pruning_metadata.json",
    "quantization_config.json",
    "model.safetensors",
    "LICENSE.md",
    "NOTICE",
)

_STATE_LOCK = threading.Lock()
_DOWNLOADING = False
_LAST_ERROR: str | None = None


def small_te_dir() -> Path:
    return Path(app_settings.ASSET_DIR) / "models" / SMALL_TE_DIR_NAME


def is_installed(te_dir: Path | None = None) -> bool:
    root = te_dir or small_te_dir()
    if not root.is_dir():
        return False
    for name in SMALL_TE_REQUIRED_FILES:
        p = root / name
        if not p.is_file() or p.stat().st_size <= 0:
            return False
    return True


def is_enabled_and_installed(te_dir: Path | None = None) -> bool:
    """True only when user opted in AND the local Q4 pruned encoder is present."""
    if not bool(app_settings.get_setting("flux2_9b_small_te")):
        return False
    return is_installed(te_dir)


def status_dict(te_dir: Path | None = None) -> dict:
    installed = is_installed(te_dir)
    with _STATE_LOCK:
        downloading = _DOWNLOADING
        error = _LAST_ERROR
    return {
        "installed": installed,
        "enabled": bool(app_settings.get_setting("flux2_9b_small_te")) and installed,
        "downloading": downloading,
        "error": error,
        "repo": SMALL_TE_REPO,
        "size_gb": SMALL_TE_SIZE_GB,
    }


def load_local_pruned_q4_encoder(te_dir: Path | None = None):
    """Load the locally installed Q4-G64 pruned 5.1B encoder without any network call."""
    import mlx.core as mx
    import mlx.nn as nn
    from mflux.models.flux2.model.flux2_text_encoder.pruned_text_encoder import Flux2PrunedTextEncoder
    from mflux.models.flux2.weights.flux2_weight_definition import Qwen3SmallTextEncoderWeightDefinition

    root = te_dir or small_te_dir()
    if not is_installed(root):
        raise FileNotFoundError(f"FLUX.2-9B light text encoder is not installed at {root}")

    cfg = json.loads((root / "config.json").read_text("utf-8"))
    metadata = json.loads((root / "pruning_metadata.json").read_text("utf-8"))
    Flux2PrunedTextEncoder.validate_config(cfg)
    Flux2PrunedTextEncoder.validate_metadata(metadata)

    encoder = Flux2PrunedTextEncoder.build_encoder(metadata)
    nn.quantize(
        encoder,
        bits=4,
        group_size=64,
        class_predicate=Qwen3SmallTextEncoderWeightDefinition.quantization_predicate,
    )
    weights = mx.load(str(root / "model.safetensors"))
    encoder.load_weights(list(weights.items()), strict=False)
    mx.eval(encoder.parameters())
    return Flux2PrunedTextEncoder(encoder=encoder, metadata=metadata)


def _download_worker() -> None:
    global _DOWNLOADING, _LAST_ERROR
    try:
        from huggingface_hub import hf_hub_download

        dest = small_te_dir()
        dest.mkdir(parents=True, exist_ok=True)
        for filename in SMALL_TE_ALL_FILES:
            hf_hub_download(repo_id=SMALL_TE_REPO, filename=filename, local_dir=str(dest))
        if not is_installed(dest):
            raise RuntimeError("Downloaded FLUX.2-9B light text encoder is incomplete")
        app_settings.update_settings({"flux2_9b_small_te": True})
        with _STATE_LOCK:
            _LAST_ERROR = None
    except Exception as exc:
        with _STATE_LOCK:
            _LAST_ERROR = str(exc)
    finally:
        with _STATE_LOCK:
            _DOWNLOADING = False


def start_download() -> dict:
    """Start an explicit user-triggered download of the optional 2.57 GB Q4 encoder."""
    global _DOWNLOADING, _LAST_ERROR
    if is_installed():
        app_settings.update_settings({"flux2_9b_small_te": True})
        return status_dict()
    with _STATE_LOCK:
        if _DOWNLOADING:
            return status_dict()
        _DOWNLOADING = True
        _LAST_ERROR = None
    t = threading.Thread(target=_download_worker, name="flux2-small-te-dl", daemon=True)
    t.start()
    return status_dict()
