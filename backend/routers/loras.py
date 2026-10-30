from pathlib import Path
import time
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import generator
import civitai_service
import lora_compat
import state
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




@router.get("/api/loras")
def list_loras():
    # No licence fetch here, deliberately. This route is polled by the frontend, so a
    # network call would make a render depend on Civitai's uptime. Licences arrive
    # through the explicit action below.
    return _read_loras()


@router.get("/api/loras/{name}/compatibility")
def lora_compatibility(name: str, model: str):
    """Can `name` be applied to `model`? Answered before a generation, not during one.

    The alternative is the message this replaces, which arrives after the weights are
    loaded: "44 of the 144 layers ... could not be applied ... and 39 more". True,
    late, and unactionable. This returns which block types the adapter names and the
    model lacks, which is a thing the user can act on.

    Cached per (adapter, model) because it is polled by the LoRA picker and re-reading a
    165 MB header on every render would be silly. The verdict cannot go stale within a
    session: it depends only on the file's keys and the model's weight index.
    """
    entry = next((e for e in _read_loras() if e.get("name") == name), None)
    if entry is None:
        raise HTTPException(404, f"LoRA not found: {name}")
    path = entry.get("path")
    if not path or not Path(path).is_file():
        raise HTTPException(400, f"LoRA file is missing: {name}")
    return lora_compat.check(path, model)


@router.post("/api/loras/refresh-licenses")
def refresh_loras_licenses():
    """Fetch licence blocks for entries that name a Civitai model.

    An explicit POST, not a GET, and not a side effect of listing: a user pressing
    "refresh" is asking for network calls and should see them counted.
    """
    return state.refresh_licenses()


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
    # Optional and all-or-nothing: a licence a source supplied, stored verbatim. It is
    # never synthesised here -- if the importer had nothing to store, this stays absent
    # rather than becoming an empty block that draws a badge saying nothing.
    license: dict | None = Field(default=None)


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
