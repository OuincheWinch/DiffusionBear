import json
import threading
import time
import uuid
from pathlib import Path
from PIL import Image

import app_settings
from image_meta import atomic_write_json, save_image_with_metadata, extract_image_metadata, _artist_fallback

# Derive from app_settings rather than re-deriving backend/data here: this was a
# second hardcoded copy, so it ignored MLX_DIFFUSION_DATA_DIR and would have written
# upscaled images into the signed .app bundle instead of the user's store.
DATA_DIR = app_settings.DATA_DIR
GENERATED_DIR = DATA_DIR / "generated"
MAX_UPSCALE_PIXELS = 64 * 1024 * 1024
_UPSCALE_SLOTS = threading.BoundedSemaphore(1)
_THUMBNAIL_SLOTS = threading.BoundedSemaphore(1)


def upscale_image(image_id: str, scale: int = 2) -> dict:
    with _UPSCALE_SLOTS:
        return _upscale_image(image_id, scale)


def _upscale_image(image_id: str, scale: int = 2) -> dict:
    """Upscale image using fast Lanczos resampling with unsharp masking."""
    src_img = GENERATED_DIR / f"{image_id}.png"
    if not src_img.exists():
        src_img = GENERATED_DIR / f"{image_id}.jpeg"
    if not src_img.exists():
        src_img = GENERATED_DIR / f"{image_id}.jpg"
    src_json = GENERATED_DIR / f"{image_id}.json"
    if not src_img.exists():
        raise FileNotFoundError(f"image not found: {image_id}")

    parent_meta = {}
    if src_json.exists():
        try:
            parent_meta = json.loads(src_json.read_text())
        except Exception:
            pass

    scale = max(2, min(4, int(scale)))
    t0 = time.time()
    actual_method = f"lanczos_{scale}x"

    with Image.open(src_img) as img:
        w, h = img.size
        output_pixels = w * h * scale * scale
        if output_pixels > MAX_UPSCALE_PIXELS:
            raise ValueError(
                f"upscale output {w * scale}x{h * scale} exceeds "
                f"{MAX_UPSCALE_PIXELS} pixel limit"
            )
        new_w, new_h = w * scale, h * scale
        img = img.convert("RGB")

        # High quality Lanczos resampling
        upscaled = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        # Unsharp mask for crisp high-frequency edge definition
        from PIL import ImageFilter
        upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=3))

    new_id = uuid.uuid4().hex
    fmt = parent_meta.get("format", "png")
    if fmt not in ("png", "jpeg", "jpg"):
        fmt = "jpeg" if src_img.suffix.lower() in (".jpeg", ".jpg") else "png"
    dest_path = GENERATED_DIR / f"{new_id}.{fmt}"
    elapsed = round(time.time() - t0, 2)
    new_w, new_h = upscaled.size

    tags = list(dict.fromkeys((parent_meta.get("tags") or []) + ["upscaled", actual_method]))
    meta = dict(parent_meta)
    meta.update({
        "id": new_id,
        "width": new_w,
        "height": new_h,
        "upscaled_from": image_id,
        "upscale_factor": scale,
        "upscale_method": actual_method,
        "generation_time": elapsed,
        "created_at": time.time(),
        "software": "MLX-DIFFUSION",
        "generator": "MLX-DIFFUSION",
        "artist": _artist_fallback(parent_meta.get("artist")),
        "file": dest_path.name,
        "format": fmt,
        "tags": tags,
    })

    stealth = bool(parent_meta.get("stealth", False))
    save_image_with_metadata(
        image=upscaled,
        dest_path=dest_path,
        meta=meta,
        output_format=fmt,
        stealth=stealth,
    )
    try:
        upscaled.close()
    except Exception:
        pass
    atomic_write_json(GENERATED_DIR / f"{new_id}.json", meta)
    try:
        thumbnail_path(new_id)
    except Exception:
        pass
    return meta


def thumbnail_path(image_id: str, force_recreate: bool = False) -> Path:
    with _THUMBNAIL_SLOTS:
        return _thumbnail_path(image_id, force_recreate)


def _thumbnail_path(image_id: str, force_recreate: bool = False) -> Path:
    thumb = GENERATED_DIR / f"{image_id}_thumb.png"
    src = GENERATED_DIR / f"{image_id}.png"
    if not src.exists():
        src = GENERATED_DIR / f"{image_id}.jpeg"
    if not src.exists():
        src = GENERATED_DIR / f"{image_id}.jpg"

    needs_create = force_recreate or not thumb.exists()

    if needs_create and src.exists():
        meta = None
        json_path = GENERATED_DIR / f"{image_id}.json"
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        if not meta:
            meta = extract_image_metadata(src)

        try:
            with Image.open(src) as img:
                img = img.convert("RGB")
                img.thumbnail((512, 512))
                stealth = bool(meta.get("stealth", False)) if meta else False
                if meta and not stealth:
                    save_image_with_metadata(
                        image=img,
                        dest_path=thumb,
                        meta=meta,
                        output_format="png",
                        stealth=False,
                    )
                else:
                    save_image_with_metadata(
                        image=img,
                        dest_path=thumb,
                        meta=meta or {},
                        output_format="png",
                        stealth=True,
                    )
        except Exception:
            pass
    return thumb
