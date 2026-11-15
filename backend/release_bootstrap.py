"""Startup bootstrap for release assets and derived model artifacts."""
from __future__ import annotations

import os
from pathlib import Path

MOSS_Q4_REPO = "mflux-community/moss-image-sft-mflux-q4"
MOSS_Q4_DIRNAME = "moss-image-sft"


def _models_root() -> Path:
    import app_settings

    return Path(app_settings.ASSET_DIR) / "models"


def moss_q4_ready(target: Path) -> bool:
    return (
        (target / "transformer").is_dir()
        and any((target / "transformer").glob("*.safetensors"))
        and (target / "vae").is_dir()
        and any((target / "vae").glob("*.safetensors"))
    )


def ensure_moss_q4_pack(models_dir: Path | None = None) -> Path | None:
    if os.environ.get("DIFFUSIONBEAR_SKIP_MOSS_Q4_BOOTSTRAP") == "1":
        return None
    base = Path(models_dir) if models_dir is not None else _models_root()
    target = base / MOSS_Q4_DIRNAME
    if moss_q4_ready(target):
        return target
    if os.environ.get("DIFFUSIONBEAR_AUTO_DOWNLOAD_MOSS_Q4", "0") != "1":
        return None
    try:
        from huggingface_hub import snapshot_download

        print(f"[bootstrap] downloading {MOSS_Q4_REPO} -> {target}", flush=True)
        snapshot_download(
            repo_id=MOSS_Q4_REPO,
            local_dir=str(target),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"],
            ignore_patterns=["text_encoder/*", "text_encoder/**"],
        )
        return target if moss_q4_ready(target) else None
    except Exception as exc:
        print(f"[bootstrap] MOSS Q4 download failed: {exc}", flush=True)
        return None


def prequantize_sdxl_all(models_dir: Path | None = None) -> None:
    if os.environ.get("DIFFUSIONBEAR_SKIP_SDXL_BOOTSTRAP") == "1":
        return
    root = (Path(models_dir) if models_dir is not None else _models_root()).expanduser().resolve()
    targets = sorted(path.parent.parent for path in root.glob("*/unet/config.json"))
    if not targets:
        return
    try:
        import mlx_sdxl_engine as sx

        for model_dir in targets:
            engine = sx.MLXSDXLEngine(model_dir=model_dir.resolve(), quantize_unet=4)
            engine.load_unet()
            engine._release_unet()
        print("[bootstrap] SDXL prequantize complete", flush=True)
    except Exception as exc:
        print(f"[bootstrap] SDXL prequantize failed: {exc}", flush=True)


def bootstrap_release_assets() -> None:
    if os.environ.get("DIFFUSIONBEAR_SKIP_RELEASE_BOOTSTRAP") == "1":
        return
    if os.environ.get("DIFFUSIONBEAR_AUTO_DOWNLOAD_MOSS_Q4", "0") == "1":
        ensure_moss_q4_pack()
    if os.environ.get("DIFFUSIONBEAR_STARTUP_PREQUANTIZE_SDXL", "0") == "1":
        prequantize_sdxl_all()
