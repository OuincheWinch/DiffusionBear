"""Model inventory, installation and onboarding.

Split out of routers/loras.py (C7): model listing, inventory, licence
refresh, uninstallation, local installation and the custom-model onboard
route lived there beside the LoRA registry, giving model management three
homes and no owner. LoRA endpoints stay in routers/loras.py; anything under
/api/models/* lives here.
"""
from pathlib import Path
import time
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import generator
import app_settings
import custom_models
import state

router = APIRouter(tags=["models"])


def _has_incomplete_files(path: Path) -> bool:
    try:
        return any(
            candidate.name.endswith((".incomplete", ".part"))
            for candidate in path.rglob("*")
            if candidate.is_file()
        )
    except (OSError, ValueError):
        return True


def _complete_weight_directory(path: Path) -> bool:
    if not path.is_dir() or _has_incomplete_files(path):
        return False
    index_files = list(path.rglob("*.safetensors.index.json"))
    if index_files:
        for index_file in index_files:
            try:
                data = json.loads(index_file.read_text("utf-8"))
                weight_map = data.get("weight_map") if isinstance(data, dict) else None
                if not isinstance(weight_map, dict) or not weight_map:
                    return False
                if any(not (index_file.parent / str(filename)).is_file() for filename in weight_map.values()):
                    return False
            except (OSError, UnicodeError, json.JSONDecodeError):
                return False
        return True
    return any(path.rglob("*.safetensors"))


def _complete_sdxl_directory(path: Path) -> bool:
    model_index = path / "model_index.json"
    if not model_index.is_file() or _has_incomplete_files(path):
        return False
    try:
        data = json.loads(model_index.read_text("utf-8"))
        components = [
            key
            for key, value in data.items()
            if isinstance(value, list)
            and len(value) >= 2
            and isinstance(value[0], str)
            and isinstance(value[1], str)
        ]
        if not components:
            return False
        for component in components:
            if not (path / component).is_dir():
                return False
        return _complete_weight_directory(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def _complete_hf_snapshot(repo: str) -> bool:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo.replace('/', '--')}"
    snapshots = cache_dir / "snapshots"
    if not snapshots.is_dir():
        return False
    try:
        candidates = sorted((path for path in snapshots.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        return False
    for candidate in candidates:
        try:
            if _complete_weight_directory(candidate) and any(candidate.rglob("*.json")):
                return True
        except OSError:
            continue
    return False


_fully_cached_ttl_s = 5.0
_fully_cached_cache: dict = {}


def _model_is_fully_cached(model_id: str, minfo: dict | None = None, local_path: Path | None = None) -> bool:
    # /api/models calls this 9 times per request and the frontend calls that 3-6
    # times per page load, so a single failed directory walk was enough to make
    # the UI report an installed model as missing and offer a full re-download.
    # Short TTL keeps the answer stable across that burst.
    key = (model_id, str(local_path) if local_path else None)
    now = time.monotonic()
    cached = _fully_cached_cache.get(key)
    if cached is not None and now - cached[0] < _fully_cached_ttl_s:
        return cached[1]
    result = _model_is_fully_cached_uncached(model_id, minfo, local_path)
    _fully_cached_cache[key] = (now, result)
    return result


def _model_is_fully_cached_uncached(model_id: str, minfo: dict | None = None, local_path: Path | None = None) -> bool:
    minfo = minfo or generator.get_model_info(model_id)
    if not minfo:
        return False
    local = local_path or generator.resolve_local_model_path(model_id)
    if local is None:
        # Custom entries own their directory via local_path, not via the
        # settings override resolve_local_model_path reads: without this a
        # verified local-source custom always reads as not-installed and the
        # listing strips the path the Generate button needs.
        entry_path = (minfo or {}).get("local_path")
        if entry_path:
            try:
                local = generator.resolve_local_model_path(model_id, entry_path)
            except (OSError, ValueError):
                local = None
    if local is not None:
        if minfo.get("engine") == "sdxl":
            return _complete_sdxl_directory(local)
        try:
            return not _has_incomplete_files(local) and any(local.rglob("*.safetensors"))
        except OSError:
            return False
    if minfo.get("engine") == "sdxl":
        model_dir = minfo.get("model_dir")
        return bool(model_dir and _complete_sdxl_directory(Path(model_dir)))
    if model_id == "krea2-turbo":
        path = generator.ASSET_DIR / "models" / "krea2-turbo-q4"
        try:
            return path.is_dir() and not _has_incomplete_files(path) and any(path.rglob("*.safetensors"))
        except OSError:
            return False
    # For other models (FLUX, etc.), check the model directory in ASSET_DIR/models first
    model_dir = generator.ASSET_DIR / "models" / model_id
    if model_dir.is_dir():
        try:
            return not _has_incomplete_files(model_dir) and any(model_dir.rglob("*.safetensors"))
        except OSError:
            return False
    # Fall back to HF cache check
    repo = generator.model_download_repo(model_id, minfo) or ""
    return "/" in repo and _complete_hf_snapshot(repo)


def _sort_models_for_picker(items: list[dict]) -> list[dict]:
    """Installed models first, then alphabetical by label.

    Sorted here rather than in each view because one list feeds the Generate
    dropdown, the Models tab and the defaults section, and sorting per view is
    how they drift apart.

    Installed before not-installed because the list is long and the model you
    can actually run is nearly always the one you want. Alphabetical within
    each group so the order does not jump around when a download starts.
    Adopted local models sort by their display name like any other, so a hand
    installed checkpoint appears in its natural place rather than at the end.
    """
    def key(entry: dict) -> tuple:
        label = str(entry.get("label") or entry.get("id") or "")
        return (0 if entry.get("installed") else 1, label.casefold(), label)

    return sorted(items, key=key)



@router.get("/api/models")
def list_models():
    items = []
    models = list(generator.MODELS.values())
    for entry in custom_models.read_custom_models():
        if entry.get("id") in generator.MODELS:
            continue
        minfo = generator.get_model_info(entry["id"])
        if minfo is not None:
            models.append(minfo)
    for model in models:
        entry = app_settings.apply_model_defaults(model["id"], model)
        entry["local_path_configured"] = bool(app_settings.configured_model_local_path(model["id"]))
        # User-declared samplers ride the model's own sampler list so the
        # generation page offers them without a separate picker; the engine
        # still validates and still owns the fallback.
        try:
            import samplers_store
            extra = samplers_store.custom_ids_for_engine(str(model.get("engine") or "").lower())
            if extra:
                entry["samplers"] = list(entry.get("samplers") or []) + [
                    sid for sid in extra if sid not in (entry.get("samplers") or [])
                ]
        except Exception:
            pass
        installed = _model_is_fully_cached(model["id"], model)
        entry["installed"] = installed
        if not installed:
            entry.pop("local_path", None)
        entry["download_repo"] = generator.model_download_repo(model["id"], model)
        entry["disk_usage_bytes"] = generator.model_disk_usage(model["id"], model)
        if entry.get("is_distilled"):
            # The steps slider caps at this and shows a "distilled" hint, so the
            # UI can mirror the engine's silent clamp instead of promising steps
            # the run will never take (see generator._effective_steps).
            entry["max_steps"] = entry.get("max_steps") or generator._DISTILLED_MAX_STEPS
        items.append(entry)
    return _sort_models_for_picker(items)


@router.get("/api/models/inventory")
def list_model_inventory(include_registered: bool = False):
    """One list of every model the app can see, each entry marked by origin.

    The Models tab used to render the built-in registry and the on-disk detection
    list as two unrelated things, so a repo downloaded from the browser was invisible
    beside a built-in model -- it was in neither list by id. This merges them and tags
    each entry `origin: registry | detected`, so the frontend can label them without
    comparing ids itself.

    `include_registered` is the same escape hatch `/api/hf/models/detected` has: by
    default a directory the registry already owns is not repeated as a leftover, which
    is what made juggernaut-xl-lightning look like an unknown stray.
    """
    from routers.downloads import _detected_model_rows

    items = []
    seen = set()

    cached = app_settings.get_setting("model_licenses") or {}
    for model in generator.MODELS.values():
        entry = app_settings.apply_model_defaults(model["id"], model)
        entry["origin"] = "registry"
        entry["name"] = model["id"]
        # The licence is a CACHE of what Hugging Face last told us, not a record of
        # fact. Absent until the user refreshes it -- the column is not guessed at.
        entry["license"] = cached.get(model["id"])
        items.append(entry)
        seen.add(model["id"])

    for row in _detected_model_rows(include_registered=include_registered):
        name = row["name"]
        # Whether a row belongs to the registry is a property the scan already computed
        # and reported as `registered`.
        #
        # This used to build `known` from `_detected_model_rows(include_registered=True)`
        # and ask whether each row of the False list was in it. Those two are in a subset
        # relation -- one `continue` is the only difference -- so every row matched and
        # every row was dropped, and the endpoint returned zero detected entries by
        # default. The invisibility it exists to remove was still there.
        if not include_registered and row.get("registered"):
            # Already listed above under its model id; repeating it under a directory
            # name is how one model came to appear twice in the same list.
            continue
        if name in seen:
            continue
        row["origin"] = "detected"
        row.setdefault("license", None)
        items.append(row)
        seen.add(name)

    return {"items": items}


@router.post("/api/models/refresh-licenses")
def refresh_model_licenses():
    """Fetch licence blocks for the registry models.

    Separate from `POST /api/loras/refresh-licenses` on purpose: LoRAs come from
    Civitai and key on `civitai_model_id`, models come from Hugging Face and key on a
    repo id. The button on this table used to call the LoRA one, so it refreshed a
    different registry, found nothing to change, and appeared to do nothing here.
    """
    return state.refresh_model_licenses()


@router.delete("/api/models/{model_id}")
def remove_model(model_id: str, force: bool = False):
    if model_id not in generator.MODELS:
        raise HTTPException(404, "unknown model")
    ok, reason = generator.uninstall_model(model_id, force=force)
    if not ok:
        raise HTTPException(409, reason or "model is in use")
    return {"status": "unlinked" if reason and "unlinked" in reason else "removed", "model_id": model_id}


class LocalInstallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=120)
    path: str = Field(min_length=1, max_length=4096)


@router.get("/api/models/local-sources")
def list_local_sources(model_id: str | None = None):
    if model_id is not None and not generator.get_model_info(model_id):
        raise HTTPException(404, "unknown model")
    return {"sources": generator.list_local_model_sources(model_id)}


@router.post("/api/models/install-local")
def install_local_model(req: LocalInstallRequest):
    minfo = generator.get_model_info(req.model_id)
    if not minfo:
        raise HTTPException(404, "unknown model")
    try:
        expanded_path = Path(req.path).expanduser()
    except (OSError, RuntimeError) as e:
        raise HTTPException(400, "invalid model path") from e
    if "\x00" in req.path or not expanded_path.is_absolute():
        raise HTTPException(400, "model path must be absolute")
    ok, reason = generator.validate_local_model_dir(req.model_id, req.path)
    if not ok:
        raise HTTPException(400, reason or "invalid model directory")
    resolved = generator.resolve_local_model_path(req.model_id, req.path)
    if resolved is None or not resolved.is_dir():
        raise HTTPException(400, "path does not exist")
    if not _model_is_fully_cached(req.model_id, minfo, resolved):
        raise HTTPException(400, "model directory is incomplete")
    try:
        app_settings.update_settings({"model_paths": {req.model_id: str(resolved.resolve())}})
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "installed", "model_id": req.model_id, "local_path": str(resolved.resolve())}


class OnboardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = Field(default=None)
    label: str | None = Field(default=None)
    family: str = Field()
    variant: str | None = Field(default=None)
    local_path: str = Field()
    convert_if_single_file: bool = Field(default=False)
    is_distilled: bool | None = Field(default=None)


@router.post("/api/models/onboard")
def onboard_model(req: OnboardRequest):
    declaration = {
        "id": req.id,
        "label": req.label,
        "family": req.family,
        "variant": req.variant,
        "local_path": req.local_path,
        "source": "local",
    }
    if req.is_distilled is not None:
        declaration["is_distilled"] = req.is_distilled
    try:
        candidate_dir = Path(req.local_path).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        raise HTTPException(400, "invalid model path") from e
    target_dir = candidate_dir.parent if candidate_dir.is_file() else candidate_dir
    is_single_sdxl = (
        candidate_dir.is_file() and candidate_dir.suffix.lower() == ".safetensors"
    ) or (
        candidate_dir.is_dir()
        and not (candidate_dir / "model_index.json").is_file()
        and any(candidate_dir.glob("*.safetensors"))
    )
    if req.family == "sdxl" and req.convert_if_single_file and is_single_sdxl:
        from routers.downloads import start_sdxl_conversion, ConversionRequest
        convert_res = start_sdxl_conversion(ConversionRequest(
            install_name=target_dir.name,
            custom_id=req.id,
            label=req.label,
            local_path=str(candidate_dir),
            register_on_finish=True,
        ))
        return {
            "status": "converting",
            "task_id": convert_res.get("task_id"),
            "install_name": target_dir.name,
        }
    ok, reason, entry = custom_models.verify_and_register(declaration)
    if not ok:
        raise HTTPException(400, reason or "invalid model directory")
    return entry

