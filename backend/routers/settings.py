from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import generator
import app_settings
from state import (
    DOWNLOAD_TASKS,
    JOBS,
    _DOWNLOAD_LOCK,
    _JOBS_LOCK,
    _MODEL_DOWNLOAD_LOCK,
    MODEL_DOWNLOAD_TASKS,
)

router = APIRouter(tags=["settings"])

HF_HUB_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artist_name: str | None = Field(default=None, max_length=200)
    default_output_format: str | None = Field(default=None, pattern=r"^(png|jpeg)$")
    default_stealth: bool | None = None
    default_fast_vae: bool | None = None
    default_sampler: str | None = Field(default=None, max_length=80)
    default_cache_interval: int | None = Field(default=None, ge=1, le=10)
    model_defaults: dict | None = None
    # UI language. "auto" follows the OS locale; a concrete code pins it. Stored
    # with the rest of the settings so the choice survives a reinstall, and
    # validated here rather than trusted from the client.
    language: str | None = Field(
        default=None, pattern=r"^(auto|en|fr|de|it|es|zh|ja|pt|ko)$"
    )
    # Interface theme. Same contract as `language`: "auto" follows the OS, and the
    # header toggle pins a concrete value. Pattern rather than trusting the client,
    # and case-sensitive so the stored value cannot differ from what was chosen.
    theme: str | None = Field(default=None, pattern=r"^(auto|light|dark)$")
    # Active enhancer backend. Same closed-set contract as theme: the picker
    # offers exactly these two, and anything else is refused at the schema
    # layer before the settings validator ever sees it.
    enhancer_model: str | None = Field(default=None, pattern=r"^(qwen15|qwen05)$")


@router.get("/api/settings")
def get_settings():
    return app_settings.get_settings()


@router.post("/api/settings")
def update_settings(req: SettingsUpdate):
    updates = {key: value for key, value in req.model_dump().items() if value is not None}
    try:
        return app_settings.update_settings(updates)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))


@router.get("/api/engine/status")
def engine_status():
    return generator.get_engine_status()


class EngineConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    memory_wired_limit_gb: float | None = Field(default=None, ge=0, le=128)
    memory_krea_wired_limit_gb: float | None = Field(default=None, ge=0, le=128)
    idle_kill_s_mflux: int | None = Field(default=None, ge=0, le=86400)
    idle_kill_s_sdxl: int | None = Field(default=None, ge=0, le=86400)
    idle_kill_s_qwen: int | None = Field(default=None, ge=0, le=86400)


@router.post("/api/engine/config")
def update_engine_config(req: EngineConfigRequest):
    updates = {key: value for key, value in req.model_dump().items() if value is not None}
    if not updates:
        raise HTTPException(400, "nothing to update")
    try:
        app_settings.update_settings(updates)
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))
    generator.rearm_engine_watchdogs()
    return generator.get_engine_status()


def _hf_cache_payload() -> dict:
    repos = []
    total = 0
    error = None
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        for repo in info.repos:
            size = int(getattr(repo, "size_on_disk", 0) or 0)
            total += size
            repos.append({
                "repo_id": repo.repo_id,
                "repo_type": getattr(repo, "repo_type", "model"),
                "size_bytes": size,
                "n_revisions": len(getattr(repo, "revisions", []) or []),
                "n_files": int(getattr(repo, "nb_files", 0) or 0),
                "last_accessed": getattr(repo, "last_accessed", None),
            })
        repos.sort(key=lambda item: item["size_bytes"], reverse=True)
    except Exception as e:
        error = str(e)
    return {
        "root": str(HF_HUB_CACHE),
        "exists": HF_HUB_CACHE.exists(),
        "total_bytes": total,
        "repos": repos,
        "error": error,
    }


@router.get("/api/hf/cache")
def hf_cache():
    return _hf_cache_payload()


class CacheClearRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_id: str | None = Field(default=None, min_length=3, max_length=193, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


@router.post("/api/hf/cache/clear")
def clear_hf_cache(req: CacheClearRequest):
    before = _hf_cache_payload()
    if not req.repo_id:
        raise HTTPException(400, "repo_id is required")
    if ".." in req.repo_id:
        raise HTTPException(400, "invalid repo_id")
    target = next((repo for repo in before["repos"] if repo["repo_id"] == req.repo_id), None)
    if target is None:
        raise HTTPException(404, f"{req.repo_id} not found in HF cache")
    with generator._model_maintenance_lock:
        with _JOBS_LOCK:
            with _DOWNLOAD_LOCK:
                with _MODEL_DOWNLOAD_LOCK:
                    for job in JOBS.values():
                        if job.get("status") not in ("queued", "generating"):
                            continue
                        request = job.get("request")
                        model_id = getattr(request, "model", "") if request is not None else ""
                        info = generator.get_model_info(model_id)
                        if info and generator.model_download_repo(info["id"], info) == req.repo_id:
                            raise HTTPException(409, f"{req.repo_id} is in use by an active generation")
                    for task in DOWNLOAD_TASKS.values():
                        if task.get("repo_id") == req.repo_id and task.get("status") == "downloading" and task.get("worker_active"):
                            raise HTTPException(409, f"{req.repo_id} is currently downloading")
                    for task in MODEL_DOWNLOAD_TASKS.values():
                        if task.get("repo_id") == req.repo_id and task.get("status") == "downloading" and task.get("worker_active"):
                            raise HTTPException(409, f"{req.repo_id} is currently downloading")
                    freed = target["size_bytes"]
                    if not generator._delete_hf_repo_cache(req.repo_id):
                        raise HTTPException(500, f"failed to remove {req.repo_id} from HF cache")
    return {"status": "cleared", "repo_id": req.repo_id, "freed_bytes": freed}


class PickPathRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str = Field(default="folder", pattern=r"^(folder|file)$")
    title: str | None = Field(default=None, max_length=200)
    extensions: list[str] | None = Field(default=None)


@router.post("/api/system/pick-path")
def pick_path(req: PickPathRequest):
    import sys
    import subprocess

    if sys.platform != "darwin":
        raise HTTPException(400, "Native path picker is only available on macOS")

    prompt = req.title or ("Select a folder" if req.kind == "folder" else "Select a file")
    clean_prompt = prompt.replace('"', '\\"').replace("\n", " ").replace("\r", " ")

    if req.kind == "folder":
        as_code = f'choose folder with prompt "{clean_prompt}"'
    else:
        if req.extensions:
            valid_exts = [ext.strip().lstrip(".") for ext in req.extensions if ext.strip().lstrip(".").isalnum()]
            if valid_exts:
                types_str = "{" + ", ".join(f'"{ext}"' for ext in valid_exts) + "}"
                as_code = f'choose file with prompt "{clean_prompt}" of type {types_str}'
            else:
                as_code = f'choose file with prompt "{clean_prompt}"'
        else:
            as_code = f'choose file with prompt "{clean_prompt}"'

    full_script = f'POSIX path of ({as_code})'

    try:
        proc = subprocess.run(
            ["osascript", "-e", full_script],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {"cancelled": True, "path": None, "reason": "timeout"}
    except (OSError, subprocess.SubprocessError) as e:
        raise HTTPException(500, f"Failed to open native picker: {e}")

    if proc.returncode != 0:
        if "User cancelled" in proc.stderr or "-128" in proc.stderr:
            return {"cancelled": True, "path": None}
        raise HTTPException(500, f"Picker failed: {proc.stderr.strip()}")

    selected_path = proc.stdout.strip()
    if selected_path.endswith("/") and len(selected_path) > 1:
        selected_path = selected_path.rstrip("/")

    return {"cancelled": False, "path": selected_path}
