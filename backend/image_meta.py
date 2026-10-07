import datetime
import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo

import app_settings


_EXPORTABLE_PNG_FIELDS = (
    "prompt",
    "negative_prompt",
    "sampler",
    "width",
    "height",
    "steps",
    "guidance",
    "seed",
    "quantization",
    "generation_time",
    "load_time",
    "created_at",
    "format",
    "fast_vae",
    "cache_interval",
    "reference_strength",
    "upscale_factor",
    "upscale_method",
    "modelVersionName",
    "modelVersionId",
    "modelId",
)


def _artist_fallback(artist_value) -> str:
    return artist_value or app_settings.metadata_artist()


def _is_local_path(value) -> bool:
    text = str(value or "").strip()
    return text.startswith(("/", "~", "./", "../", "file://")) or bool(
        re.match(r"^[A-Za-z]:[\\/]", text)
    )


def _public_name(value, fallback: str) -> str:
    text = str(value or "").strip()
    return fallback if not text or _is_local_path(text) else text


def _fsync_parent(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_write_json(
    path: Path,
    data,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
) -> None:
    path = Path(path)
    payload = json.dumps(
        data,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=f".{path.name}.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_parent(path.parent)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def compute_aspect_ratio(w: int, h: int) -> str:
    """Compute standard aspect ratio string matching Civitai conventions."""
    if not w or not h:
        return "1:1"
    r = w / float(h)
    known = [
        (1.0, "1:1"),
        (1.5, "3:2"),
        (2.0 / 3.0, "2:3"),
        (16.0 / 9.0, "16:9"),
        (9.0 / 16.0, "9:16"),
        (4.0 / 3.0, "4:3"),
        (3.0 / 4.0, "3:4"),
        (21.0 / 9.0, "21:9"),
        (9.0 / 21.0, "9:21"),
        (4.0 / 5.0, "4:5"),
        (5.0 / 4.0, "5:4"),
    ]
    for val, name in known:
        if abs(r - val) < 0.08:
            return name
    g = math.gcd(int(w), int(h))
    return f"{w // g}:{h // g}"


def to_latin1_clean(text: str) -> str:
    """Sanitize strings to valid latin-1 to guarantee Pillow writes standard tEXt chunks,
    preventing fallback to iTXt chunks which fail in browser-based image parsers."""
    if not text:
        return ""
    replacements = {
        "\u2018": "'",   # ‘ left single quotation mark
        "\u2019": "'",   # ’ right single quotation mark
        "\u201a": "'",   # ‚ single low-9 quotation mark
        "\u201b": "'",   # ‛ single high-reversed-9 quotation mark
        "\u201c": '"',   # “ left double quotation mark
        "\u201d": '"',   # ” right double quotation mark
        "\u201e": '"',   # „ double low-9 quotation mark
        "\u201f": '"',   # ‟ double high-reversed-9 quotation mark
        "\u2032": "'",   # ′ prime
        "\u2033": '"',   # ″ double prime
        "\u2013": "-",   # – en dash
        "\u2014": "-",   # — em dash
        "\u2015": "-",   # ― horizontal bar
        "\u2026": "...", # … horizontal ellipsis
        "\u00a0": " ",   # non-breaking space
        "\u200b": "",    # zero-width space
        "\u200c": "",    # zero-width non-joiner
        "\u200d": "",    # zero-width joiner
        "\ufeff": "",    # zero-width no-break space (BOM)
        "\u2022": "*",   # • bullet
        "«": '"',
        "»": '"',
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_generation_metadata_text(meta: dict) -> str:
    """Format generation metadata following the universal Civitai / Automatic1111 / WebUI standard."""
    raw_prompt = meta.get("prompt", "").strip()
    prompt = to_latin1_clean(raw_prompt)

    # Determine model information
    model_name = str(meta.get("model", ""))
    minfo = {}
    if model_name:
        try:
            from generator import get_model_info
            minfo = get_model_info(model_name) or {}
        except Exception:
            minfo = {}

    model_civitai_name = _public_name(
        minfo.get("civitai_model_name")
        or meta.get("model_label")
        or minfo.get("label")
        or model_name,
        "MLX-Model",
    )
    model_version_name = _public_name(
        minfo.get("civitai_version_name") or meta.get("modelVersionName"),
        "v1.0",
    )
    chk_vid = minfo.get("civitai_version_id") or meta.get("modelVersionId")
    chk_mid = minfo.get("civitai_model_id") or meta.get("modelId")
    chk_hash = (minfo.get("sha256", "")[:10] if minfo.get("sha256") else "") or (str(chk_vid) if chk_vid else "")
    if not chk_hash:
        chk_hash = hashlib.sha256(model_civitai_name.encode("utf-8")).hexdigest()[:10].upper()

    hashes_map = {}
    if chk_hash:
        hashes_map["model"] = str(chk_hash)

    lora_hash_parts = []
    civitai_resources = [{
        "type": "checkpoint",
        "modelName": model_civitai_name,
        "modelVersionName": model_version_name,
    }]
    if chk_vid:
        civitai_resources[0]["modelVersionId"] = chk_vid
    if chk_mid:
        civitai_resources[0]["modelId"] = chk_mid

    used_tag_names = set()
    for index, lora in enumerate(meta.get("loras", []), 1):
        if not isinstance(lora, dict):
            continue
        raw_name = _public_name(
            lora.get("civitai_model_name") or lora.get("name"),
            f"lora_{index}",
        )
        base_tag_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_name).strip("_") or f"lora_{index}"
        tag_name = base_tag_name
        suffix = 2
        while tag_name in used_tag_names:
            tag_name = f"{base_tag_name}_{suffix}"
            suffix += 1
        used_tag_names.add(tag_name)
        scale = float(lora.get("scale", 1.0))
        if f"<lora:{tag_name}" not in prompt and f"<lora:{raw_name}" not in prompt:
            prompt = f"{prompt}, <lora:{tag_name}:{scale}>" if prompt else f"<lora:{tag_name}:{scale}>"

        l_name = _public_name(lora.get("civitai_model_name") or raw_name, "LoRA")
        l_vname = _public_name(
            lora.get("civitai_version_name") or lora.get("modelVersionName"),
            "v1.0",
        )
        l_vid = lora.get("modelVersionId") or lora.get("civitai_version_id")
        l_mid = lora.get("modelId") or lora.get("civitai_model_id")
        l_hash = (
            (lora.get("sha256")[:10] if lora.get("sha256") else "")
            or (str(l_vid) if l_vid else "")
        )

        if l_hash:
            hashes_map[f"lora:{tag_name}"] = str(l_hash)
            lora_hash_parts.append(f"{tag_name}: {l_hash}")

        res_entry = {
            "type": "lora",
            "weight": scale,
            "modelName": l_name,
            "modelVersionName": l_vname,
        }
        if l_vid:
            res_entry["modelVersionId"] = l_vid
        if l_mid:
            res_entry["modelId"] = l_mid
        civitai_resources.append(res_entry)

    lines = [prompt]
    neg_prompt = to_latin1_clean(str(meta.get("negative_prompt", "")).strip())
    if neg_prompt:
        lines.append(f"Negative prompt: {neg_prompt}")

    is_sdxl = "sdxl" in model_name.lower() or meta.get("engine") == "sdxl" or minfo.get("engine") == "sdxl"
    clip_skip = 2 if is_sdxl else 1
    w = int(meta.get("width", 1024))
    h = int(meta.get("height", 1024))

    created_at = meta.get("created_at") or time.time()
    if isinstance(created_at, (int, float)):
        iso_date = datetime.datetime.fromtimestamp(created_at, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        iso_date = str(created_at)

    raw_sampler = meta.get("sampler") or ("euler_trailing" if is_sdxl else "Euler")
    if "flowmatch" in raw_sampler.lower():
        civitai_sampler = "Euler"
    else:
        civitai_sampler = raw_sampler

    cfg_val = meta.get("guidance")
    if cfg_val is None:
        cfg_scale = 1.0
    else:
        try:
            cfg_scale = float(cfg_val)
        except (ValueError, TypeError):
            cfg_scale = 1.0

    param_parts = []
    param_parts.append(f"Steps: {meta.get('steps', 4)}")
    param_parts.append(f"Sampler: {civitai_sampler}")
    param_parts.append(f"CFG scale: {cfg_scale}")
    param_parts.append(f"Seed: {meta.get('seed', 0)}")
    param_parts.append(f"Size: {w}x{h}")
    param_parts.append(f"Model: {model_civitai_name}")
    if chk_hash:
        param_parts.append(f"Model hash: {chk_hash}")
    param_parts.append(f"Clip skip: {clip_skip}")
    param_parts.append(f"Created Date: {iso_date}")
    gen_time = meta.get("generation_time")
    if gen_time is not None:
        try:
            param_parts.append(f"Generation Time: {float(gen_time):.2f}s")
        except (TypeError, ValueError):
            pass
    param_parts.append("Software: DiffusionBear")
    param_parts.append("Generator: DiffusionBear")

    if hashes_map:
        param_parts.append(f"Hashes: {json.dumps(hashes_map, separators=(',', ':'))}")
    if lora_hash_parts:
        param_parts.append(f'Lora hashes: "{", ".join(lora_hash_parts)}"')

    param_parts.append(f"Civitai resources: {json.dumps(civitai_resources, separators=(',', ':'))}")
    if meta.get("fast_vae"):
        param_parts.append("Fast VAE: enabled")

    ref_strength = meta.get("reference_strength")
    if ref_strength is not None:
        try:
            param_parts.append(f"Denoising strength: {float(ref_strength):.3f}")
        except (TypeError, ValueError):
            param_parts.append(f"Denoising strength: {ref_strength}")

    aspect_val = compute_aspect_ratio(w, h)
    if minfo.get("ecosystem"):
        ecosystem = minfo["ecosystem"]
    elif is_sdxl:
        ecosystem = "SDXL"
    elif "z-image" in model_name.lower() or "zimage" in model_name.lower() or "zit" in model_name.lower():
        ecosystem = "ZImageTurbo"
    elif "krea" in model_name.lower():
        ecosystem = "Krea 2"
    elif "flux2" in model_name.lower() or "klein" in model_name.lower():
        ecosystem = "FLUX.2"
    elif "flux" in model_name.lower() and "mflux" not in model_name.lower():
        ecosystem = "FLUX.1"
    else:
        ecosystem = "SDXL"

    civitai_meta_resources = []
    for r in civitai_resources:
        if "modelVersionId" not in r and "modelId" not in r:
            continue
        entry = {
            "type": "Checkpoint" if r["type"] == "checkpoint" else "LORA",
            "strength": r.get("weight", 1.0) if r["type"] == "lora" else 1,
        }
        if "modelVersionId" in r:
            entry["modelVersionId"] = r["modelVersionId"]
        if "modelId" in r:
            entry["modelId"] = r["modelId"]
        civitai_meta_resources.append(entry)

    civitai_meta = {
        "workflow": "txt2img",
        "priority": "low",
        "outputFormat": str(meta.get("format", "png")).lower(),
        "ecosystem": ecosystem,
        "aspectRatio": {
            "value": aspect_val,
            "width": w,
            "height": h,
        },
        "prompt": prompt,
        "negativePrompt": neg_prompt,
        "sampler": civitai_sampler,
        "cfgScale": cfg_scale,
        "steps": meta.get("steps", 4),
        "clipSkip": clip_skip,
        "seed": meta.get("seed", 0),
        "enhancedCompatibility": False,
        "quantity": 1,
        "resources": civitai_meta_resources,
        "software": "DiffusionBear",
        "generator": "DiffusionBear",
        "artist": _artist_fallback(meta.get("artist")),
    }
    if ref_strength is not None:
        try:
            civitai_meta["denoisingStrength"] = float(ref_strength)
        except (TypeError, ValueError):
            pass
    param_parts.append(f"Civitai metadata: {json.dumps(civitai_meta, separators=(',', ':'))}")

    lines.append(", ".join(param_parts))
    return to_latin1_clean("\n".join(lines))


def build_image_exif(meta: dict, metadata_text: str | None = None, image_size: tuple[int, int] | None = None) -> Image.Exif:
    """Construct Pillow Exif object matching Civitai/MODEL.jpg IFD0 and Sub-IFD 34665 standard."""
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = 1
    exif[ExifTags.Base.ResolutionUnit] = 2
    exif[ExifTags.Base.XResolution] = 72.0
    exif[ExifTags.Base.YResolution] = 72.0
    exif[ExifTags.Base.Software] = "DiffusionBear"
    exif[ExifTags.Base.Artist] = to_latin1_clean(_artist_fallback(meta.get("artist")))

    created_at = meta.get("created_at") or time.time()
    try:
        dt = datetime.datetime.fromtimestamp(float(created_at), tz=datetime.timezone.utc)
        date_str = dt.strftime("%Y:%m:%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        date_str = None
    if date_str:
        exif[ExifTags.Base.DateTime] = date_str
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        exif_ifd[ExifTags.Base.DateTimeOriginal] = date_str
        exif_ifd[ExifTags.Base.DateTimeDigitized] = date_str

    if metadata_text:
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
        exif_ifd[36864] = b"0210"  # ExifVersion 2.1
        exif_ifd[40960] = b"0100"  # FlashPixVersion 1.0
        exif_ifd[40961] = 65535    # ColorSpace Uncalibrated
        if image_size:
            w, h = int(image_size[0]), int(image_size[1])
        else:
            w = int(meta.get("width", 1024))
            h = int(meta.get("height", 1024))
        exif_ifd[40962] = w        # ExifImageWidth
        exif_ifd[40963] = h        # ExifImageHeight
        clean_text = to_latin1_clean(metadata_text)
        exif_ifd[ExifTags.Base.UserComment] = b"UNICODE\x00\xfe\xff" + clean_text.encode("utf-16be")
    return exif


def build_pnginfo(meta: dict, metadata_text: str | None = None) -> PngInfo:
    """Construct PngInfo with parameters chunk and individual keys, strictly using latin-1
    to guarantee standard tEXt chunks readable by browser-based parsers."""
    info = PngInfo()
    if metadata_text:
        info.add_text("parameters", to_latin1_clean(metadata_text))
    info.add_text("Software", "DiffusionBear")
    info.add_text("software", "DiffusionBear")
    info.add_text("Generator", "DiffusionBear")
    info.add_text("generator", "DiffusionBear")
    artist_clean = to_latin1_clean(_artist_fallback(meta.get("artist")))
    info.add_text("Artist", artist_clean)
    info.add_text("artist", artist_clean)
    public_model = _public_name(
        meta.get("model_label") or meta.get("model"),
        "MLX-Model",
    )
    info.add_text("model", to_latin1_clean(public_model))
    for key in _EXPORTABLE_PNG_FIELDS:
        if key not in meta or meta[key] is None:
            continue
        value = meta[key]
        val_str = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        info.add_text(key, to_latin1_clean(val_str))
    return info


def save_image_with_metadata(
    image: Image.Image,
    dest_path: Path,
    meta: dict,
    output_format: str = "png",
    stealth: bool = False,
    quality: int = 95,
    attempts: int = 4,
):
    """Save an image atomically with embedded EXIF (UserComment, Artist, Software)
    and PNG text chunks (parameters), or completely clean if stealth=True."""
    is_jpeg = str(output_format).lower() in ("jpeg", "jpg")

    if stealth:
        exif = None
        pnginfo = None
    else:
        metadata_text = build_generation_metadata_text(meta)
        exif = build_image_exif(meta, metadata_text, image_size=image.size)
        pnginfo = build_pnginfo(meta, metadata_text) if not is_jpeg else None

    dest_path = Path(dest_path)
    target_dir = dest_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{output_format.lower()}.tmp"
    last_err = None
    for i in range(attempts):
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=str(target_dir), suffix=suffix)
            with os.fdopen(fd, "wb") as f:
                if is_jpeg:
                    if image.mode in ("RGBA", "LA", "P"):
                        image = image.convert("RGB")
                    if exif:
                        image.save(f, format="JPEG", exif=exif, quality=quality)
                    else:
                        image.save(f, format="JPEG", quality=quality)
                else:
                    save_kwargs = {"format": "PNG"}
                    if pnginfo:
                        save_kwargs["pnginfo"] = pnginfo
                    if exif:
                        save_kwargs["exif"] = exif
                    image.save(f, **save_kwargs)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(tmp_path, dest_path)
                _fsync_parent(target_dir)
            except Exception:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                raise
            return
        except PermissionError as e:
            last_err = e
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            time.sleep(1.5 * (i + 1))
        except Exception:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise
    raise RuntimeError(
        f"Failed to save image to {dest_path} after {attempts} attempts "
        f"(external drive asleep/disconnected or macOS blocked access): {last_err}"
    ) from last_err


def extract_image_metadata(img_path_or_file) -> dict:
    """Extract and parse generation metadata from an image (EXIF UserComment or PNG parameters)."""
    with Image.open(img_path_or_file) as img:
        artist = None
        software = None
        raw_text = None
        try:
            exif = img.getexif()
            if exif:
                artist = exif.get(ExifTags.Base.Artist)
                software = exif.get(ExifTags.Base.Software)
                sub = exif.get_ifd(ExifTags.IFD.Exif)
                uc = sub.get(ExifTags.Base.UserComment)
                if uc and len(uc) > 8:
                    hdr = uc[:8]
                    if hdr.startswith(b"UNICODE"):
                        payload = uc[8:]
                        if payload.startswith(b"\xfe\xff"):
                            payload = payload[2:]
                            encoding = "utf-16be"
                        elif payload.startswith(b"\xff\xfe"):
                            payload = payload[2:]
                            encoding = "utf-16le"
                        else:
                            encoding = "utf-16be"
                        try:
                            raw_text = payload.decode(encoding)
                        except UnicodeDecodeError:
                            raw_text = payload.decode("utf-16le" if encoding == "utf-16be" else "utf-16be", errors="replace")
                    elif hdr.startswith(b"ASCII"):
                        raw_text = uc[8:].decode("latin1", errors="replace")
                    else:
                        raw_text = uc.decode("utf-8", errors="replace")
        except Exception:
            pass

        if not raw_text and hasattr(img, "info"):
            raw_text = img.info.get("parameters")

        if raw_text and raw_text.startswith("\ufeff"):
            raw_text = raw_text[1:]

        if not raw_text:
            return {
                "has_metadata": False,
"artist": _artist_fallback(artist),
                "software": software or "DiffusionBear",
                "generator": "DiffusionBear",
            }

        res = {
            "has_metadata": True,
            "raw_text": raw_text,
            "artist": _artist_fallback(artist),
            "software": software or "DiffusionBear",
            "generator": "DiffusionBear",
            "prompt": "",
            "negative_prompt": "",
            "params": {},
            "civitai_resources": None,
            "civitai_metadata": None,
            "model": None,
            "loras": [],
        }

        lines = raw_text.split("\n")
        neg_idx = -1
        param_idx = -1
        for idx, line in enumerate(lines):
            if line.startswith("Negative prompt:"):
                neg_idx = idx
            elif line.startswith("Steps:"):
                param_idx = idx

        if neg_idx != -1:
            res["prompt"] = "\n".join(lines[:neg_idx]).strip()
            if param_idx != -1:
                res["negative_prompt"] = "\n".join(lines[neg_idx:param_idx]).replace("Negative prompt:", "").strip()
                param_line = "\n".join(lines[param_idx:]).strip()
            else:
                res["negative_prompt"] = "\n".join(lines[neg_idx:]).replace("Negative prompt:", "").strip()
                param_line = ""
        elif param_idx != -1:
            res["prompt"] = "\n".join(lines[:param_idx]).strip()
            param_line = "\n".join(lines[param_idx:]).strip()
        else:
            res["prompt"] = raw_text.strip()
            param_line = ""

        if param_line:
            res["param_line"] = param_line
            cm_match = re.search(r"Civitai metadata:\s*(\{.*\})", param_line)
            if cm_match:
                raw_cm = cm_match.group(1)
                depth = 0
                end_pos = -1
                for idx, ch in enumerate(raw_cm):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end_pos = idx + 1
                            break
                if end_pos != -1:
                    try:
                        res["civitai_metadata"] = json.loads(raw_cm[:end_pos])
                    except Exception:
                        pass
                    param_line = param_line.replace(f"Civitai metadata: {raw_cm[:end_pos]}", "")

            cr_match = re.search(r"Civitai resources:\s*(\[.*\])", param_line)
            if cr_match:
                raw_cr = cr_match.group(1)
                depth = 0
                end_pos = -1
                for idx, ch in enumerate(raw_cr):
                    if ch == '[':
                        depth += 1
                    elif ch == ']':
                        depth -= 1
                        if depth == 0:
                            end_pos = idx + 1
                            break
                if end_pos != -1:
                    try:
                        res["civitai_resources"] = json.loads(raw_cr[:end_pos])
                    except Exception:
                        pass
                    param_line = param_line.replace(f"Civitai resources: {raw_cr[:end_pos]}", "")

            for part in param_line.split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" in part:
                    k, v = part.split(":", 1)
                    res["params"][k.strip()] = v.strip()

        if res.get("civitai_resources"):
            for r in res["civitai_resources"]:
                if r.get("type") == "checkpoint":
                    res["model"] = r.get("modelName")
                elif r.get("type") == "lora":
                    res["loras"].append({
                        "name": r.get("modelName"),
                        "scale": r.get("weight", 1.0),
                        "modelVersionId": r.get("modelVersionId"),
                    })
        elif "Model" in res.get("params", {}):
            res["model"] = res["params"]["Model"]

        if not res["loras"] and "<lora:" in res["prompt"]:
            lora_tags = re.findall(r"<lora:([^:>]+)(?::([^>]+))?>", res["prompt"])
            for l_name, l_scale in lora_tags:
                try:
                    s_val = float(l_scale) if l_scale else 1.0
                except ValueError:
                    s_val = 1.0
                res["loras"].append({"name": l_name, "scale": s_val})

        res["resources"] = res.get("civitai_resources") or []
        return res
