from pathlib import Path
import time
import json
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import generator
import app_settings
import civitai_service
from state import (
    LORA_FILES_DIR,
    SDXL_LORA_DIR,
    _loras_lock,
    _read_loras,
    _write_loras,
    _validate_lora_path,
    _inspect_safetensors,
    sync_lora_entry_with_civitai,
    _upsert_lora_entries,
    _remove_lora_entries,
    JOBS,
    _JOBS_LOCK,
)

router = APIRouter(tags=["loras"])
_BASE_MODELS = {"sdxl", "flux2", "krea2", "z-image"}


def _has_incomplete_files(path: Path) -> bool:
    try:
        return any(
            candidate.name.endswith((".incomplete", ".part"))
            for candidate in path.rglob("*")
            if candidate.is_file() and ".cache" not in candidate.relative_to(path).parts
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
    for model in generator.MODELS.values():
        entry = app_settings.apply_model_defaults(model["id"], model)
        entry["local_path_configured"] = bool(app_settings.configured_model_local_path(model["id"]))
        installed = _model_is_fully_cached(model["id"], model)
        entry["installed"] = installed
        if not installed:
            entry.pop("local_path", None)
        entry["download_repo"] = generator.model_download_repo(model["id"], model)
        entry["disk_usage_bytes"] = generator.model_disk_usage(model["id"], model)
        items.append(entry)
    return _sort_models_for_picker(items)


@router.delete("/api/models/{model_id}")
def remove_model(model_id: str):
    if model_id not in generator.MODELS:
        raise HTTPException(404, "unknown model")
    ok, reason = generator.uninstall_model(model_id)
    if not ok:
        raise HTTPException(409, reason or "model is in use")
    return {"status": "unlinked" if reason and "unlinked" in reason else "removed", "model_id": model_id}


class LocalInstallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=120)
    path: str = Field(min_length=1, max_length=4096)


@router.get("/api/models/local-sources")
def list_local_sources(model_id: str | None = None):
    if model_id is not None and model_id not in generator.MODELS:
        raise HTTPException(404, "unknown model")
    return {"sources": generator.list_local_model_sources(model_id)}


@router.post("/api/models/install-local")
def install_local_model(req: LocalInstallRequest):
    if req.model_id not in generator.MODELS:
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
    if not _model_is_fully_cached(req.model_id, generator.MODELS[req.model_id], resolved):
        raise HTTPException(400, "model directory is incomplete")
    try:
        app_settings.update_settings({"model_paths": {req.model_id: str(resolved.resolve())}})
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "installed", "model_id": req.model_id, "local_path": str(resolved.resolve())}


@router.get("/api/loras")
def list_loras():
    return _read_loras()


@router.post("/api/loras/sync-civitai")
def sync_all_loras_civitai():
    snapshot = _read_loras()
    updates = []
    for entry in snapshot:
        candidate = dict(entry)
        if sync_lora_entry_with_civitai(candidate, force=False):
            updates.append(candidate)
    if updates:
        with _loras_lock:
            current = _read_loras()
            current_keys = {(entry.get("name"), entry.get("path")) for entry in current}
            changed = 0
            for update in updates:
                key = (update.get("name"), update.get("path"))
                if key not in current_keys:
                    continue
                for index, entry in enumerate(current):
                    if (entry.get("name"), entry.get("path")) == key:
                        current[index] = {**entry, **update}
                        changed += 1
                        break
            if changed:
                _write_loras(current)
    return {"status": "ok", "total": len(snapshot), "updated": len(updates), "loras": _read_loras()}


class LoRASaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=200)
    path: str = Field(min_length=1, max_length=4096)
    scale: float = Field(default=1.0, ge=0.0, le=10.0)
    base_model: str | None = Field(default=None, max_length=32)
    triggers: list[str] = Field(default_factory=list, max_length=16)
    sha256: str | None = Field(default=None, min_length=64, max_length=64, pattern=r"^[A-Fa-f0-9]{64}$")


@router.post("/api/loras")
def save_lora(req: LoRASaveRequest):
    name = req.name.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,199}", name):
        raise HTTPException(400, "invalid LoRA name")
    path = Path(req.path).expanduser()
    if not path.is_absolute():
        raise HTTPException(400, "registered LoRA path must be absolute")
    err = _validate_lora_path(req.path)
    if err:
        raise HTTPException(400, err)
    if req.base_model is not None and req.base_model not in _BASE_MODELS:
        raise HTTPException(400, "invalid base model")
    if any(not isinstance(trigger, str) or not trigger.strip() or len(trigger) > 200 or any(ord(char) < 32 or ord(char) == 127 for char in trigger) for trigger in req.triggers):
        raise HTTPException(400, "invalid LoRA trigger")
    loras = _read_loras()
    existing = next((entry for entry in loras if entry.get("name") == name or entry.get("path") == req.path), {})
    entry_data = req.model_dump(exclude_unset=True)
    entry_data["name"] = name
    entry = {**existing, **entry_data}
    detected_base, detected_triggers = _inspect_safetensors(path)
    if not entry.get("base_model") and detected_base:
        entry["base_model"] = detected_base
    entry.setdefault("base_model", "sdxl" if "sdxl" in req.path.lower() else "flux2")
    if not entry.get("triggers") and detected_triggers:
        entry["triggers"] = detected_triggers
    entry.setdefault("triggers", [])
    actual_sha256 = civitai_service.compute_file_sha256(path)
    if req.sha256 and req.sha256.upper() != actual_sha256:
        raise HTTPException(400, "LoRA SHA-256 does not match the file")
    entry["sha256"] = actual_sha256
    sync_lora_entry_with_civitai(entry)
    _upsert_lora_entries([entry])
    return entry


@router.delete("/api/loras/{name}")
def delete_lora(name: str):
    if not isinstance(name, str) or not name.strip() or len(name) > 200:
        raise HTTPException(400, "invalid LoRA name")
    name = name.strip()
    with _loras_lock:
        loras = _read_loras()
        target = next((entry for entry in loras if entry.get("name") == name), None)
        if not target:
            raise HTTPException(404, "not found")
        raw_path = target.get("path", "")
        path = Path(raw_path).expanduser().resolve() if raw_path else None
        if path is not None:
            with _JOBS_LOCK:
                for job in JOBS.values():
                    if job.get("status") not in ("queued", "generating"):
                        continue
                    request = job.get("request")
                    active_loras = getattr(request, "loras", []) if request is not None else []
                    for lora in active_loras:
                        active_path = lora.get("path") if isinstance(lora, dict) else getattr(lora, "path", "")
                        if active_path and Path(str(active_path)).expanduser().resolve() == path:
                            raise HTTPException(409, "LoRA is used by an active generation")
        _remove_lora_entries(names={name})
        if path is not None:
            generator._drop_mflux_pipeline()
            generator._kill_sdxl_daemon()
            roots = (LORA_FILES_DIR.resolve(), SDXL_LORA_DIR.resolve())
            if any(path.is_relative_to(root) for root in roots) and path.exists() and path.is_file():
                try:
                    path.unlink()
                except OSError as e:
                    try:
                        _upsert_lora_entries([target])
                    except Exception:
                        pass
                    raise HTTPException(500, "LoRA file could not be deleted") from e
    return {"deleted": name}
