#!/usr/bin/env python3
"""
Retroactively re-embeds Civitai-compliant metadata into images in backend/data/generated.
- Ensures Generator Software is "DiffusionBear"
- Ensures Artist is the configured artist (data/settings.json "artist_name") or "MLX-DIFFUSION"
- Ensures Model and Model hash are in the parameters text chunk
- Ensures Civitai resources and Hashes are embedded
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app_settings import metadata_artist  # noqa: E402
from image_meta import atomic_write_json
from generator import (  # noqa: E402
    DATA_DIR,
    GENERATED_DIR,
    MODELS,
    build_generation_metadata_text,
    build_image_exif,
    build_pnginfo,
    save_image_with_metadata,
    _enrich_loras_with_registry,
    get_model_info,
)


def process_images(hours: float = 48.0, process_all: bool = False):
    now = time.time()
    cutoff = 0 if process_all else (now - hours * 3600)

    json_files = sorted(GENERATED_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    targets = [p for p in json_files if p.stat().st_mtime >= cutoff]

    print(f"Found {len(targets)} JSON sidecars to inspect (cutoff: {'ALL' if process_all else f'{hours}h'})...")

    fixed_count = 0
    skipped_count = 0
    error_count = 0

    for jpath in targets:
        try:
            meta = json.loads(jpath.read_text())
        except Exception as e:
            print(f"[-] Error reading {jpath.name}: {e}")
            error_count += 1
            continue

        img_name = meta.get("file") or f"{jpath.stem}.png"
        img_path = GENERATED_DIR / img_name
        if not img_path.exists():
            alt_path = GENERATED_DIR / f"{jpath.stem}.png"
            if alt_path.exists():
                img_path = alt_path
            else:
                skipped_count += 1
                continue

        # Ensure enriched fields
        meta["software"] = "MLX-DIFFUSION"
        meta["generator"] = "MLX-DIFFUSION"
        # Preserve a previously personalized artist; only purge the legacy domain / gaps.
        cur_artist = str(meta.get("artist") or "")
        if not cur_artist or cur_artist == "www.ouinche.com":
            meta["artist"] = metadata_artist()

        model_name = str(meta.get("model", ""))
        minfo = (get_model_info(model_name) if model_name else None) or {}
        if minfo.get("civitai_version_id"):
            meta["modelVersionId"] = minfo["civitai_version_id"]
        if minfo.get("civitai_model_id"):
            meta["modelId"] = minfo["civitai_model_id"]
        if minfo.get("civitai_model_name"):
            meta["model_label"] = minfo["civitai_model_name"]
        if minfo.get("civitai_version_name"):
            meta["modelVersionName"] = minfo["civitai_version_name"]
        if minfo.get("sha256"):
            meta["sha256"] = minfo["sha256"]
        if minfo.get("ecosystem"):
            meta["ecosystem"] = minfo["ecosystem"]

        if meta.get("guidance") is None:
            meta["guidance"] = 1.0

        raw_sampler = meta.get("sampler") or "Euler"
        if "flowmatch" in raw_sampler.lower():
            meta["sampler"] = "Euler"

        # Re-enrich LoRAs
        if meta.get("loras"):
            meta["loras"] = _enrich_loras_with_registry(meta["loras"])

        fmt = meta.get("format", "png")
        if fmt not in ("png", "jpeg", "jpg"):
            fmt = "jpeg" if img_path.suffix.lower() in (".jpeg", ".jpg") else "png"

        stealth = bool(meta.get("stealth", False))

        try:
            with Image.open(img_path) as img:
                img_copy = img.copy()

            save_image_with_metadata(
                image=img_copy,
                dest_path=img_path,
                meta=meta,
                output_format=fmt,
                stealth=stealth,
            )
            atomic_write_json(jpath, meta)

            # Also embed full Civitai metadata into thumbnail
            thumb_path = GENERATED_DIR / f"{jpath.stem}_thumb.png"
            try:
                timg = img_copy.convert("RGB")
                timg.thumbnail((512, 512))
                save_image_with_metadata(
                    image=timg,
                    dest_path=thumb_path,
                    meta=meta,
                    output_format="png",
                    stealth=stealth,
                )
            except Exception as te:
                print(f"[-] Warning: Error updating thumb for {jpath.stem}: {te}")

            fixed_count += 1
        except Exception as e:
            print(f"[-] Error updating {img_path.name}: {e}")
            error_count += 1

    print(f"\nDone! Fixed: {fixed_count}, Skipped (no image file): {skipped_count}, Errors: {error_count}")


def main():
    parser = argparse.ArgumentParser(description="Re-embed Civitai-compliant metadata into generated images")
    parser.add_argument("--hours", type=float, default=48.0, help="Hours to look back (default: 48)")
    parser.add_argument("--all", action="store_true", help="Process all images regardless of age")
    args = parser.parse_args()

    process_images(hours=args.hours, process_all=args.all)


if __name__ == "__main__":
    main()
