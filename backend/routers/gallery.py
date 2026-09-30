import asyncio
import json
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

import generator
import app_settings
from prompt_enhancer import EnhancementCancelled, enhance_prompt, list_engine_system_prompts
from state import (
    GENERATED_DIR,
    _gallery_lock,
    _IMAGE_MUTATION_LOCK,
    _image_is_deleted,
    _mark_image_deleted,
    _unmark_image_deleted,
    GALLERY_INDEX,
    ensure_gallery_index,
    _validate_image_id,
    _atomic_write_text,
    _read_loras,
)

router = APIRouter(tags=["gallery"])

# Image bytes are written once under a unique id and never mutated, but
# delete_image() unlinks them permanently. A long `immutable` window would keep a
# deleted image visible in any browser that already fetched it, so this trades a
# little staleness for the refetch win: fresh for an hour (covers all tab
# switching and reloads), then served stale while revalidating, so a delete
# surfaces within about an hour instead of never.
_IMAGE_BYTES_HEADERS = {
    "Cache-Control": "public, max-age=3600, stale-while-revalidate=86400",
}


def _matches_lora_filter(item: dict, lora_filter: str) -> bool:
    if not lora_filter or lora_filter.strip().lower() in ("", "all", "*"):
        return True
    loras = item.get("loras")
    if isinstance(loras, (dict, str)):
        loras = [loras]
    elif not isinstance(loras, list):
        loras = []
    clean = lora_filter.strip().lower()
    if clean in ("__none__", "none", "no-lora", "no_lora", "without-lora", "without_lora"):
        return len(loras) == 0
    if clean in ("__any__", "any", "has-lora", "has_lora", "with-lora", "with_lora"):
        return len(loras) > 0
    for lora in loras:
        if isinstance(lora, dict):
            name = (lora.get("name") or "").strip().lower()
            path = (lora.get("path") or "").strip().lower()
            stem = Path(path).stem.lower() if path else ""
            filename = Path(path).name.lower() if path else ""
            version_id = str(lora.get("modelVersionId") or lora.get("civitai_version_id") or "").lower()
            source_id = str(lora.get("modelId") or lora.get("civitai_model_id") or "").lower()
            if clean in (name, stem, filename, version_id, source_id) or clean in name or (stem and clean in stem):
                return True
        elif isinstance(lora, str):
            value = lora.strip().lower()
            stem = Path(value).stem.lower()
            if clean in (value, stem) or clean in value:
                return True
    return False


def _load_image_meta(image_id: str) -> dict:
    _validate_image_id(image_id)
    path = (GENERATED_DIR / f"{image_id}.json").resolve()
    if not path.is_relative_to(GENERATED_DIR.resolve()) or not path.is_file():
        raise HTTPException(404, "not found")
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("metadata is too large")
        data = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as e:
        raise HTTPException(500, "image metadata is invalid") from e
    if not isinstance(data, dict) or data.get("id") != image_id:
        raise HTTPException(500, "image metadata does not match its id")
    return data


@router.get("/api/gallery")
def gallery(
    query: str = "",
    tags: str = "",
    sort: str = "newest",
    page: int = 1,
    limit: int = 24,
    model: str = "",
    lora: str = "",
):
    if len(query) > 200 or len(tags) > 1000 or len(model) > 120 or len(lora) > 4096:
        raise HTTPException(400, "gallery filter is too long")
    # Self-heal an index that came up empty because the scan could not read the
    # data directory at startup. Without this the gallery stays empty until the app
    # is restarted, which is indistinguishable from having no images.
    ensure_gallery_index()
    if sort not in ("newest", "oldest"):
        raise HTTPException(400, "invalid sort order")
    if model and model not in generator.MODELS:
        raise HTTPException(400, "unknown model")
    page = max(1, page)
    limit = min(max(1, limit), 100)
    with _gallery_lock:
        items = [dict(item) for item in GALLERY_INDEX.values() if isinstance(item, dict)]
    if query:
        clean_query = query.lower().strip()
        items = [item for item in items if clean_query in str(item.get("prompt", "")).lower() or clean_query in str(item.get("seed", ""))]
    tag_list = list(dict.fromkeys(tag.strip().lower() for tag in tags.split(",") if tag.strip()))
    if any(len(tag) > 64 for tag in tag_list):
        raise HTTPException(400, "tag filter is too long")
    if tag_list:
        items = [item for item in items if all(tag in item.get("tags", []) for tag in tag_list)]
    if model:
        repo = generator.MODELS[model].get("repo")
        items = [item for item in items if item.get("model", generator.DEFAULT_MODEL) == repo]
    if lora:
        items = [item for item in items if _matches_lora_filter(item, lora)]
    items.sort(key=lambda item: item.get("created_at", 0), reverse=(sort == "newest"))
    start = (page - 1) * limit
    return {"total": len(items), "page": page, "items": items[start:start + limit]}


@router.get("/api/gallery/loras")
def gallery_loras(model: str = ""):
    if len(model) > 120 or (model and model not in generator.MODELS):
        raise HTTPException(400, "unknown model")
    with _gallery_lock:
        items = [dict(item) for item in GALLERY_INDEX.values() if isinstance(item, dict)]
    if model:
        repo = generator.MODELS[model].get("repo")
        items = [item for item in items if item.get("model", generator.DEFAULT_MODEL) == repo]
    registry = _read_loras()
    by_name = {str(entry.get("name", "")).lower(): entry for entry in registry}
    by_stem = {Path(str(entry.get("path", ""))).stem.lower(): entry for entry in registry}
    counts = Counter()
    details = {}
    with_lora = 0
    without_lora = 0
    for item in items:
        loras = item.get("loras")
        if isinstance(loras, (dict, str)):
            loras = [loras]
        elif not isinstance(loras, list):
            loras = []
        if not loras:
            without_lora += 1
            continue
        with_lora += 1
        seen = set()
        for lora in loras:
            if isinstance(lora, dict):
                raw_name = str(lora.get("name") or (Path(lora["path"]).stem if lora.get("path") else "") or "Unknown").strip()
            else:
                raw_name = Path(str(lora)).stem.strip()
            match = by_name.get(raw_name.lower()) or by_stem.get(raw_name.lower())
            canonical = match.get("name") if match else raw_name
            if canonical.lower() in seen:
                continue
            seen.add(canonical.lower())
            counts[canonical] += 1
            details.setdefault(canonical, {"name": canonical, "base_model": match.get("base_model") if match else None})
    return {
        "total": len(items),
        "with_lora": with_lora,
        "without_lora": without_lora,
        "loras": [{"name": name, "count": count, "base_model": details[name]["base_model"]} for name, count in counts.most_common()],
    }


@router.get("/api/images/{image_id}")
def image_meta(image_id: str):
    _validate_image_id(image_id)
    with _gallery_lock:
        indexed = GALLERY_INDEX.get(image_id)
        if isinstance(indexed, dict) and indexed.get("id") == image_id:
            return dict(indexed)
    data = _load_image_meta(image_id)
    with _gallery_lock:
        GALLERY_INDEX[image_id] = data
    return data


@router.get("/api/images/{image_id}/file")
def image_file(image_id: str, thumb: bool = False):
    _validate_image_id(image_id)
    if _image_is_deleted(image_id):
        raise HTTPException(404, "not found")
    generated_root = GENERATED_DIR.resolve()

    if thumb:
        # Fast path: the thumb is already on disk, so no mutation lock is needed.
        # A 24-cell gallery page used to serialize here on the global RLock.
        cached_thumb = GENERATED_DIR / f"{image_id}_thumb.png"
        if not cached_thumb.is_file():
            with _IMAGE_MUTATION_LOCK:
                if _image_is_deleted(image_id):
                    raise HTTPException(404, "not found")
                try:
                    cached_thumb = generator.thumbnail_path(image_id)
                except (OSError, ValueError):
                    cached_thumb = None
        if cached_thumb is not None:
            thumbnail = cached_thumb.resolve()
            if thumbnail.is_relative_to(generated_root) and thumbnail.is_file():
                return FileResponse(
                    thumbnail,
                    media_type="image/png",
                    headers=_IMAGE_BYTES_HEADERS | {
                        "Content-Disposition": f'inline; filename="{image_id}_thumb.png"',
                        "Access-Control-Expose-Headers": "Content-Disposition",
                    },
                )
        for suffix, media_type in ((".png", "image/png"), (".jpeg", "image/jpeg"), (".jpg", "image/jpeg")):
            path = (GENERATED_DIR / f"{image_id}{suffix}").resolve()
            if path.is_relative_to(generated_root) and path.is_file():
                return FileResponse(
                    path,
                    media_type=media_type,
                    headers=_IMAGE_BYTES_HEADERS | {
                        "Content-Disposition": f'inline; filename="{path.name}"',
                        "Access-Control-Expose-Headers": "Content-Disposition",
                    },
                )
        raise HTTPException(404, "not found")

    for suffix, media_type in ((".png", "image/png"), (".jpeg", "image/jpeg"), (".jpg", "image/jpeg")):
        path = (GENERATED_DIR / f"{image_id}{suffix}").resolve()
        if path.is_relative_to(generated_root) and path.is_file():
            return FileResponse(
                path,
                media_type=media_type,
                headers=_IMAGE_BYTES_HEADERS | {
                    "Content-Disposition": f'inline; filename="{path.name}"',
                    "Access-Control-Expose-Headers": "Content-Disposition",
                },
            )
    raise HTTPException(404, "not found")


@router.post("/api/images/{image_id}/reveal")
def reveal_image(image_id: str):
    _validate_image_id(image_id)
    if sys.platform != "darwin":
        raise HTTPException(400, "reveal is only available on macOS")
    generated_root = GENERATED_DIR.resolve()
    for suffix in (".png", ".jpeg", ".jpg"):
        path = (GENERATED_DIR / f"{image_id}{suffix}").resolve()
        if path.is_relative_to(generated_root) and path.is_file():
            try:
                subprocess.run(["open", "-R", str(path)], check=False, timeout=10)
            except (OSError, subprocess.SubprocessError) as e:
                raise HTTPException(500, "failed to reveal image") from e
            return {"ok": True, "path": str(path)}
    raise HTTPException(404, "not found")


class TagsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tags: list[str] = Field(max_length=32)


@router.post("/api/images/{image_id}/tags")
def set_tags(image_id: str, req: TagsRequest):
    if any(not isinstance(tag, str) or not tag.strip() or len(tag.strip()) > 64 for tag in req.tags):
        raise HTTPException(400, "invalid tag")
    with _IMAGE_MUTATION_LOCK:
        if _image_is_deleted(image_id):
            raise HTTPException(404, "not found")
        metadata = _load_image_meta(image_id)
        metadata["tags"] = sorted({tag.strip().lower() for tag in req.tags if tag.strip()})
        path = (GENERATED_DIR / f"{image_id}.json").resolve()
        _atomic_write_text(path, json.dumps(metadata, indent=2, allow_nan=False))
        with _gallery_lock:
            GALLERY_INDEX[image_id] = metadata
    return metadata


class UpscaleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scale: int = Field(default=2, ge=2, le=4)


class FillRequest(BaseModel):
    """A generative fill: regenerate the masked region of an existing image.

    The mask arrives as a base64 PNG data URL rather than a stored file. It is a
    transient instruction, not an artifact: writing it into generated/ would add a
    non-image to the gallery and to the storage manifest for every fill ever done.
    """

    model_config = ConfigDict(extra="forbid")

    # White = regenerate. Sent by the browser as a PNG data URL.
    mask: str = Field(min_length=16, max_length=24 * 1024 * 1024)
    prompt: str = Field(min_length=1, max_length=4000)
    model: str = Field(default="z-image-turbo", max_length=80)
    seed: int | None = None
    steps: int | None = Field(default=None, ge=1, le=50)
    width: int | None = Field(default=None, ge=64, le=4096)
    height: int | None = Field(default=None, ge=64, le=4096)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    loras: list[dict] | None = None

    @field_validator("prompt")
    @classmethod
    def _strip_prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("a prompt describing what to add is required")
        return v.strip()

    @field_validator("model")
    @classmethod
    def _known_engine(cls, v: str) -> str:
        import fill as fill_mod
        if v not in fill_mod.FILL_ENGINES:
            raise ValueError(
                f"{v} cannot be used for fill; allowed: "
                f"{', '.join(sorted(fill_mod.FILL_ENGINES))}"
            )
        return v


class PromptEnhanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=100_000)
    model: str = Field(default="flux2-klein-4b", min_length=1, max_length=120)
    loras: list[dict] = Field(default_factory=list, max_length=16)
    format: Literal["text", "json"] = "text"


@router.post("/api/prompt/enhance")
async def enhance_prompt_route(req: PromptEnhanceRequest, request: Request):
    model_info = generator.get_model_info(req.model)
    if model_info is None:
        raise HTTPException(400, "unknown model")
    for lora in req.loras:
        if not isinstance(lora.get("name", ""), str) or len(str(lora.get("name", ""))) > 200:
            raise HTTPException(400, "invalid LoRA metadata")
        if not isinstance(lora.get("path", ""), str) or len(str(lora.get("path", ""))) > 4096:
            raise HTTPException(400, "invalid LoRA metadata")
    cancel_event = threading.Event()
    result = {}

    def worker():
        try:
            result["res"] = enhance_prompt(
                req.prompt,
                engine=model_info["id"],
                loras=req.loras,
                output_format=req.format,
                cancel_event=cancel_event,
            )
        except EnhancementCancelled:
            result["cancelled"] = True

    task = asyncio.get_running_loop().run_in_executor(None, worker)
    while not task.done():
        if await request.is_disconnected():
            print("[prompt_enhancer] client disconnected — cancelling enhancement", file=sys.stderr)
            cancel_event.set()
            break
        await asyncio.sleep(0.2)
    try:
        await task
    except EnhancementCancelled:
        return {"cancelled": True, "original": req.prompt}
    except Exception as e:
        raise HTTPException(500, "prompt enhancement failed") from e
    if result.get("cancelled"):
        return {"cancelled": True, "original": req.prompt}
    response = result.get("res") or {}
    if response.get("error"):
        raise HTTPException(500, "prompt enhancement failed")
    return response


class SystemPromptUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    engine_key: str = Field(min_length=1, max_length=32)
    instructions: str = Field(max_length=8000)
    mode: Literal["text", "json"] = "text"


@router.get("/api/prompt/enhancer/system-prompts")
def list_system_prompts():
    try:
        return {"engines": list_engine_system_prompts()}
    except Exception as e:
        raise HTTPException(500, "failed to load enhancer profiles") from e


@router.post("/api/prompt/enhancer/system-prompts")
def update_system_prompt(req: SystemPromptUpdate):
    if req.engine_key not in app_settings.PROMPT_ENHANCER_KEYS:
        raise HTTPException(400, f"unknown engine '{req.engine_key}'")
    settings_key = "prompt_enhancer_json" if req.mode == "json" else "prompt_enhancer"
    try:
        app_settings.update_settings({settings_key: {req.engine_key: req.instructions}})
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"engines": list_engine_system_prompts()}


@router.post("/api/images/{image_id}/upscale")
def upscale(image_id: str, req: UpscaleRequest):
    _validate_image_id(image_id)
    _load_image_meta(image_id)
    try:
        metadata = generator.upscale_image(image_id, scale=req.scale)
        with _gallery_lock:
            GALLERY_INDEX[metadata["id"]] = metadata
        return metadata
    except FileNotFoundError as e:
        raise HTTPException(404, "image not found") from e
    except Exception as e:
        raise HTTPException(500, "upscale failed") from e


@router.post("/api/images/{image_id}/fill")
def fill(image_id: str, req: FillRequest):
    """Regenerate the masked region of an image.

    Blocking, like /upscale, and serialised on a single slot. That is a deliberate
    limitation rather than an oversight: the generation queue is typed to
    GenerateRequest, and a fill carries an image id and a mask rather than a prompt
    and a sampler, so joining that queue means reshaping it. A fill is also a
    30-280s render, so this request holds a connection for its whole duration --
    acceptable for a local desktop app talking to its own loopback backend, and the
    thing to revisit if fills ever need cancelling or queueing behind each other.

    A semaphore rather than the global generation lock, so a fill does not wedge
    text-to-image behind it for the duration of a composite.
    """
    _validate_image_id(image_id)
    _load_image_meta(image_id)
    import fill as fill_mod

    # The engine allowlist is enforced in exactly one place: fill_image(), before it
    # touches the engine. An earlier version also checked it here, which was dead
    # code twice over -- the request model rejects a disallowed engine with 422
    # before the handler runs, and the image lookup runs first regardless. No test
    # could observe the handler's own check, so removing it changed nothing
    # observable; keeping it would have been untested code implying a guarantee.
    try:
        with fill_mod._FILL_SLOT:
            metadata = fill_mod.fill_image(
                image_id,
                mask_b64=req.mask,
                prompt=req.prompt,
                model=req.model,
                seed=req.seed,
                steps=req.steps,
                width=req.width,
                height=req.height,
                guidance=req.guidance,
                loras=req.loras,
            )
    except fill_mod.FillError as e:
        # A FillError is a request the user can correct -- wrong dimensions, empty
        # mask, no prompt. 400 with the reason, not a generic 500, so the UI can
        # say what is actually wrong.
        #
        # FillError subclasses ValueError, so the two handlers below used to be
        # redundant: the bare `except ValueError` shadowed this one, which meant
        # this branch had no behaviour a test could observe. They are now one
        # handler, and the 400-on-ValueError behaviour is asserted directly.
        raise HTTPException(400, str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(404, "image not found") from e
    except Exception as e:
        # The response must not carry str(e) -- an engine or filesystem error can
        # hold a local path -- but swallowing it in the log as well made this
        # undiagnosable: a 500 read only "fill failed: ValueError" with nothing on
        # disk to explain it, and I ended up reproducing a *successful* fill twice
        # trying to guess the cause. The traceback goes to the log; the client gets
        # the class name.
        import traceback
        print(f"[fill] {image_id} failed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(500, f"fill failed: {type(e).__name__}") from e

    with _gallery_lock:
        GALLERY_INDEX[metadata["id"]] = metadata
    try:
        generator.thumbnail_path(metadata["id"])
    except Exception:
        # A missing thumbnail is cosmetic; the file route builds one on demand.
        pass
    return metadata


@router.get("/api/fill/engines")
def fill_engines():
    """Engines a fill can use, with the default and the reason each is offered.

    Server-authoritative so the picker cannot offer something the backend will
    refuse. Kept in step with the registry by test_fill.
    """
    import fill as fill_mod
    import generator as gen_mod

    engines = []
    for engine_id in sorted(fill_mod.FILL_ENGINES):
        info = gen_mod.MODELS.get(engine_id, {})
        engines.append({
            "id": engine_id,
            "label": info.get("label", engine_id),
            "is_default": engine_id == fill_mod.DEFAULT_FILL_ENGINE,
            "default_steps": info.get("default_steps"),
            "supports_loras": bool(info.get("supports_loras")),
        })
    return {"engines": engines, "default": fill_mod.DEFAULT_FILL_ENGINE}


@router.delete("/api/images/{image_id}")
def delete_image(image_id: str):
    _validate_image_id(image_id)
    with _IMAGE_MUTATION_LOCK:
        if _image_is_deleted(image_id):
            raise HTTPException(404, "not found")
        _mark_image_deleted(image_id)
        generated_root = GENERATED_DIR.resolve()
        removed = False
        errors = []
        for suffix in (".png", ".jpeg", ".jpg", ".json"):
            path = (GENERATED_DIR / f"{image_id}.{suffix}").resolve()
            if not path.is_relative_to(generated_root):
                continue
            try:
                path.unlink()
                removed = True
            except FileNotFoundError:
                continue
            except OSError as e:
                errors.append(str(e))
        thumbnail = (GENERATED_DIR / f"{image_id}_thumb.png").resolve()
        if thumbnail.is_relative_to(generated_root):
            try:
                thumbnail.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                errors.append(str(e))
        with _gallery_lock:
            GALLERY_INDEX.pop(image_id, None)
        if errors or not removed:
            _unmark_image_deleted(image_id)
        if errors:
            raise HTTPException(500, "image could not be completely deleted")
        if not removed:
            raise HTTPException(404, "not found")
    return {"deleted": image_id}
