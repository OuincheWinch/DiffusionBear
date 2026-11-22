"""Mini-VAE weights (TAEF1/TAEF2/Krea2 and TAESD-XL), fetched with the model, never at render time.

``mlx_taef.from_pretrained`` downloads from Hugging Face on first use, which put a
~5 s network call (and an offline failure) inside the first fast-VAE decode, and the
SDXL TAESD weights were never installed at all. Here the weights are fetched once
when a model is installed (or at startup for models already installed), pinned and
sha256-verified, into the model store; the render path only reads local files and
falls back to the full VAE when they are missing.

All four upstream weight sets are MIT-licensed: madebyollin/taef1, madebyollin/taef2,
madebyollin/taesdxl and ionden/taew2.1.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
from pathlib import Path

import app_settings

TAESDXL = {
    "repo": "madebyollin/taesdxl",
    "revision": "b20258aaef75ef61e659c1e0f14f251cf0ad153e",
    "filename": "diffusion_pytorch_model.safetensors",
    "sha256": "ff4824aca94dd6111e0340fa749347fb74101060d9712cb5ef1ca8f1cf17502f",
}

TAEF_VARIANTS = ("taef1", "taef2", "krea2")

_lock = threading.Lock()
_inflight: set[str] = set()


def mini_vae_dir() -> Path:
    return Path(app_settings.ASSET_DIR) / "models" / "mini-vae"


def _legacy_taef_dir() -> Path:
    return Path.home() / ".cache" / "mlx-taef"


def _kernel_class(variant: str):
    from mlx_taef import TAEF1, TAEF2, Krea2

    return {"taef1": TAEF1, "taef2": TAEF2, "krea2": Krea2}[variant]


def local_decoder_path(variant: str) -> Path | None:
    """Converted MLX decoder weights on disk, or None. Never touches the network."""
    if variant == "taesdxl":
        from taesd_mlx import _default_taesd_path

        path = _default_taesd_path()
        return path if path.is_file() else None
    name = f"{_kernel_class(variant)._kernel.source.cache_key(role='decoder')}.mlx.safetensors"
    for root in (mini_vae_dir(), _legacy_taef_dir()):
        path = root / "converted" / name
        if path.is_file():
            return path
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_taesdxl() -> Path:
    from huggingface_hub import hf_hub_download
    from taesd_mlx import _default_taesd_path

    src = Path(hf_hub_download(
        repo_id=TAESDXL["repo"], filename=TAESDXL["filename"], revision=TAESDXL["revision"],
    ))
    if _sha256(src) != TAESDXL["sha256"]:
        raise ValueError(f"{TAESDXL['repo']}/{TAESDXL['filename']}: sha256 mismatch")
    dest = _default_taesd_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dest


def _fetch_taef(variant: str) -> Path:
    import mlx_taef.download as taef_download

    # get_or_convert pins the revision and verifies the upstream sha256; it reads
    # CACHE_ROOT at call time, so this sends the converted file to the model store.
    taef_download.CACHE_ROOT = mini_vae_dir()
    return taef_download.get_or_convert(_kernel_class(variant)._kernel, role="decoder")


def ensure(variant: str) -> Path:
    """Local path to the variant's decoder weights, downloading them if needed."""
    path = local_decoder_path(variant)
    if path is not None:
        return path
    return _fetch_taesdxl() if variant == "taesdxl" else _fetch_taef(variant)


def variants_for_model(model_id: str, info: dict | None = None) -> tuple[str, ...]:
    info = info or {}
    if info.get("engine") == "sdxl":
        return ("taesdxl",)
    key = f"{model_id} {info.get('family') or ''}".lower()
    if "flux2" in key or "flux.2" in key or "klein" in key:
        return ("taef2",)
    if "krea" in key or "anima" in key:
        return ("krea2",)
    if "z-image" in key or "zimage" in key or "flux1" in key or "flux-1" in key or "schnell" in key:
        return ("taef1",)
    return ()


def _prefetch(variants: list[str]) -> None:
    for variant in variants:
        try:
            ensure(variant)
        except Exception as exc:
            print(f"[mini-vae] could not fetch {variant}: {exc}", flush=True)
        finally:
            with _lock:
                _inflight.discard(variant)


def prefetch_async(variants) -> threading.Thread | None:
    """Fetch missing variants in the background; no-op for those present or in flight."""
    todo = []
    with _lock:
        for variant in dict.fromkeys(variants):
            if variant in _inflight or local_decoder_path(variant) is not None:
                continue
            _inflight.add(variant)
            todo.append(variant)
    if not todo:
        return None
    thread = threading.Thread(target=_prefetch, args=(todo,), daemon=True, name="mini-vae-prefetch")
    thread.start()
    return thread


def prefetch_for_model(model_id: str, info: dict | None = None) -> threading.Thread | None:
    return prefetch_async(variants_for_model(model_id, info))


def prefetch_for_installed_models() -> None:
    import generator

    variants: list[str] = []
    for model_id, info in generator.MODELS.items():
        try:
            if generator.resolve_local_model_path(model_id) is None:
                continue
        except Exception:
            continue
        variants.extend(variants_for_model(model_id, info))
    prefetch_async(variants)
