import hashlib
import fnmatch
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from email.message import Message
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit, urlunsplit

from fastapi import APIRouter, HTTPException
from huggingface_hub import HfApi
from pydantic import BaseModel, ConfigDict, Field

import civitai_service
import generator
import hf_service
import hf_browse
import civitai_browse
import sdxl_convert
import sources
from .models import _model_is_fully_cached
from state import (
    LORA_FILES_DIR,
    SDXL_LORA_DIR,
    DOWNLOAD_TASKS,
    _DOWNLOAD_LOCK,
    MODEL_DOWNLOAD_TASKS,
    _MODEL_DOWNLOAD_LOCK,
    MAX_LORA_UPLOAD_BYTES,
    _inspect_safetensors,
    _read_loras,
    _remove_lora_entries,
    _sanitize_filename,
    _upsert_lora_entries,
)

router = APIRouter(tags=["downloads"])
_BASE_MODELS = {"sdxl", "flux2", "krea2", "z-image"}
_CIVITAI_HOSTS = {"civitai.com", "www.civitai.com", "civitai.red", "www.civitai.red"}


class CivitaiImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    url_or_id: str = Field(min_length=1, max_length=8192)
    name: str | None = Field(default=None, max_length=200)
    api_key: str | None = Field(default=None, max_length=8192)


class HFImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    url_or_repo: str = Field(min_length=1, max_length=4096)
    name: str | None = Field(default=None, max_length=200)
    token: str | None = Field(default=None, max_length=8192)


class DirectUrlImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    url: str = Field(min_length=1, max_length=8192)
    name: str | None = Field(default=None, max_length=200)
    triggers: list[str] | None = Field(default=None, max_length=16)
    base_model: Literal["sdxl", "flux2", "krea2", "z-image"] | None = None


class UnifiedDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    url_or_id: str = Field(min_length=1, max_length=8192)
    name: str | None = Field(default=None, max_length=200)
    token: str | None = Field(default=None, max_length=8192)
    api_key: str | None = Field(default=None, max_length=8192)
    triggers: list[str] | None = Field(default=None, max_length=16)
    base_model: Literal["sdxl", "flux2", "krea2", "z-image"] | None = None


def make_download_progress_cb(task_id: str):
    def on_progress(progress_data):
        with _DOWNLOAD_LOCK:
            task = DOWNLOAD_TASKS.get(task_id)
            if not task or task.get("status") != "downloading":
                return
            if isinstance(progress_data, dict):
                task["status_text"] = str(progress_data.get("status", task.get("status_text", "")))[:500]
                for key in ("progress", "downloaded_bytes", "total_bytes", "speed_mb_s"):
                    if key in progress_data:
                        task[key] = progress_data[key]
            else:
                task["status_text"] = str(progress_data)[:500]
    return on_progress


def _fail_task(task_id: str, error: Exception | str, auth: bool = False, auth_msg: str = ""):
    with _DOWNLOAD_LOCK:
        task = DOWNLOAD_TASKS.get(task_id)
        if not task:
            return False
        if task.get("status") != "downloading":
            if task.get("status") == "cancelled":
                task["worker_active"] = False
            return False
        message = str(error)
        cancelled = isinstance(error, civitai_service.DownloadCancelled) or "cancelled by user" in message.lower()
        if auth:
            task["status"] = "error"
            task["worker_active"] = False
            task["error"] = f"AUTH_REQUIRED: {message}"[:2000]
            task["status_text"] = auth_msg or "Authentication required"
        else:
            task["status"] = "cancelled" if cancelled else "error"
            task["worker_active"] = False
            task["error"] = message[:2000]
            task["status_text"] = "Download cancelled" if cancelled else f"Download failed: {message}"[:500]
        task["finished_at"] = time.time()
        task.pop("cancel_event", None)
        return True


def _finish_task(task_id: str, entry: dict, status_text: str):
    with _DOWNLOAD_LOCK:
        task = DOWNLOAD_TASKS.get(task_id)
        if not task:
            return False
        if task.get("status") != "downloading":
            if task.get("status") == "cancelled":
                task["worker_active"] = False
            return False
        task["status"] = "done"
        task["worker_active"] = False
        task["progress"] = 1.0
        task["finished_at"] = time.time()
        task["status_text"] = status_text[:500]
        task["result"] = entry
        if "base_model" in entry:
            task["base_model"] = entry["base_model"]
        task.pop("cancel_event", None)
        return True


def _cleanup_registered_entry(entry: dict | None, fallback_path: Path | None = None):
    if entry:
        name = entry.get("name")
        path = entry.get("path")
        _remove_lora_entries(names={name} if isinstance(name, str) else None, paths={path} if isinstance(path, str) else None)
        remaining_paths = {item.get("path") for item in _read_loras()}
        if isinstance(path, str) and path not in remaining_paths:
            try:
                resolved = Path(path).expanduser().resolve()
                roots = (LORA_FILES_DIR.resolve(), SDXL_LORA_DIR.resolve())
                if any(resolved.is_relative_to(root) for root in roots) and resolved.is_file():
                    resolved.unlink()
            except OSError:
                pass
    elif fallback_path is not None:
        try:
            resolved = Path(fallback_path).expanduser().resolve()
            roots = (LORA_FILES_DIR.resolve(), SDXL_LORA_DIR.resolve())
            if any(resolved.is_relative_to(root) for root in roots) and resolved.is_file():
                resolved.unlink()
        except OSError:
            pass


def _check_cancelled(cancel_event: threading.Event):
    if cancel_event.is_set():
        raise civitai_service.DownloadCancelled("Download cancelled by user")


def _display_name(value: object, fallback: str) -> str:
    text = str(value or fallback).strip()
    text = "".join("_" if ord(char) < 32 or ord(char) == 127 or char in "/\\" else char for char in text)
    text = text.strip()[:200] or fallback[:200]
    return text


def _run_civitai_download(task_id: str, req: CivitaiImportRequest, version_id: int, cancel_event: threading.Event):
    entry = None
    dest_path: Path | None = None
    registered = False
    try:
        meta = civitai_service.extract_civitai_metadata(
            civitai_service.fetch_by_version_id(version_id, api_key=req.api_key) or {}
        )
        target_dir = SDXL_LORA_DIR if meta["base_model"] == "sdxl" else LORA_FILES_DIR
        dest_path, enriched_meta = civitai_service.download_civitai_lora(
            version_id=version_id,
            dest_dir=target_dir,
            custom_name=req.name,
            api_key=req.api_key,
            progress_cb=make_download_progress_cb(task_id),
            cancel_event=cancel_event,
        )
        _check_cancelled(cancel_event)
        display_name = _display_name(req.name or enriched_meta.get("civitai_model_name") or dest_path.stem, "Civitai LoRA")
        entry = {
            "name": display_name,
            "path": str(dest_path.resolve()),
            "scale": 1.0,
            "triggers": enriched_meta.get("triggers", []),
            "base_model": enriched_meta.get("base_model", "sdxl"),
            "sha256": enriched_meta.get("sha256"),
            "source": "civitai",
            "civitai_version_id": enriched_meta.get("civitai_version_id") or version_id,
            "civitai_model_id": enriched_meta.get("civitai_model_id"),
            "civitai_model_name": enriched_meta.get("civitai_model_name"),
            "civitai_version_name": enriched_meta.get("civitai_version_name"),
        }
        _upsert_lora_entries([entry])
        registered = True
        if not _finish_task(task_id, entry, f"Installed {entry['name']} successfully!"):
            _cleanup_registered_entry(entry, fallback_path=dest_path)
    except civitai_service.CivitaiAuthError as e:
        if not registered:
            _cleanup_registered_entry(entry, fallback_path=dest_path)
        _fail_task(task_id, e, auth=True, auth_msg="Authentication required")
    except Exception as e:
        if not registered:
            _cleanup_registered_entry(entry, fallback_path=dest_path)
        _fail_task(task_id, e)


def _run_hf_download(task_id: str, req: HFImportRequest, meta: dict, cancel_event: threading.Event):
    target_dir = SDXL_LORA_DIR if meta["base_model"] == "sdxl" else LORA_FILES_DIR
    entry = None
    dest_path: Path | None = None
    registered = False
    try:
        dest_path, enriched_meta = hf_service.download_hf_lora(
            repo_id=meta["repo_id"],
            filename=meta["filename"],
            dest_dir=target_dir,
            revision=meta.get("revision", "main"),
            custom_name=req.name,
            token=req.token,
            progress_cb=make_download_progress_cb(task_id),
            cancel_event=cancel_event,
        )
        _check_cancelled(cancel_event)
        detected_base, detected_triggers = _inspect_safetensors(dest_path)
        triggers = list(enriched_meta.get("triggers", []))
        triggers.extend(trigger for trigger in detected_triggers if trigger not in triggers)
        display_name = _display_name(req.name or enriched_meta.get("model_name") or dest_path.stem, "Hugging Face LoRA")
        entry = {
            "name": display_name,
            "path": str(dest_path.resolve()),
            "scale": 1.0,
            "triggers": triggers[:16],
            "base_model": detected_base or enriched_meta.get("base_model") or "sdxl",
            "sha256": enriched_meta.get("sha256"),
            "source": "huggingface",
            "hf_repo_id": enriched_meta.get("repo_id"),
            "hf_filename": enriched_meta.get("filename"),
            "hf_revision": enriched_meta.get("revision"),
            "hf_resolved_revision": enriched_meta.get("resolved_revision"),
            "hf_commit_sha": enriched_meta.get("commit_sha"),
        }
        if enriched_meta.get("sha256"):
            try:
                civitai_data = civitai_service.fetch_by_hash(enriched_meta["sha256"])
                if civitai_data:
                    metadata = civitai_service.extract_civitai_metadata(civitai_data)
                    for key in ("civitai_version_id", "civitai_model_id", "civitai_model_name", "civitai_version_name"):
                        if metadata.get(key):
                            entry[key] = metadata[key]
            except Exception:
                pass
        _check_cancelled(cancel_event)
        _upsert_lora_entries([entry])
        registered = True
        if not _finish_task(task_id, entry, f"Installed {entry['name']} successfully!"):
            _cleanup_registered_entry(entry, fallback_path=dest_path)
    except hf_service.HFAuthError as e:
        if not registered:
            _cleanup_registered_entry(entry, fallback_path=dest_path)
        _fail_task(task_id, e, auth=True, auth_msg="Hugging Face authentication required")
    except Exception as e:
        if not registered:
            _cleanup_registered_entry(entry, fallback_path=dest_path)
        _fail_task(task_id, e)


def _start_hf_download_task(url_or_repo: str, custom_name: str | None = None, token: str | None = None) -> dict:
    parsed = hf_service.parse_hf_input(url_or_repo)
    if not parsed:
        raise HTTPException(400, "Invalid Hugging Face URL or Repo ID")
    try:
        meta = hf_service.fetch_hf_metadata(
            repo_id=parsed["repo_id"],
            filename=parsed.get("filename"),
            revision=parsed.get("revision", "main"),
            token=token,
        )
    except hf_service.HFAuthError as e:
        raise HTTPException(401, str(e))
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(400, str(e))
    identity = (meta["repo_id"], meta["filename"], meta.get("resolved_revision") or meta.get("revision"))
    task_id = uuid.uuid4().hex[:12]
    cancel_event = threading.Event()
    task = {
        "id": task_id,
        "source": "huggingface",
        "repo_id": meta["repo_id"],
        "filename": meta["filename"],
        "revision": meta.get("revision"),
        "resolved_revision": meta.get("resolved_revision"),
        "identity": identity,
        "model_name": _display_name(custom_name or meta.get("model_name") or meta["repo_id"], "Hugging Face LoRA"),
        "base_model": meta.get("base_model", "sdxl"),
        "status": "downloading",
        "progress": 0.05,
        "downloaded_bytes": 0,
        "total_bytes": meta.get("expected_bytes") or 0,
        "speed_mb_s": 0.0,
        "status_text": f"Connecting to Hugging Face for {_display_name(meta.get('model_name'), 'LoRA')}...",
        "started_at": time.time(),
        "finished_at": None,
        "error": None,
        "result": None,
        "worker_active": True,
        "cancel_event": cancel_event,
    }
    with _DOWNLOAD_LOCK:
        for existing_id, existing in DOWNLOAD_TASKS.items():
            if existing.get("worker_active") and existing.get("source") == "huggingface" and existing.get("identity") == identity:
                return {
                    "task_id": existing_id,
                    "status": existing.get("status", "downloading"),
                    "model_name": existing.get("model_name"),
                    "base_model": existing.get("base_model"),
                    "source": "huggingface",
                    "already_running": True,
                }
        DOWNLOAD_TASKS[task_id] = task
    req = HFImportRequest(url_or_repo=url_or_repo, name=custom_name, token=token)
    threading.Thread(target=_run_hf_download, args=(task_id, req, meta, cancel_event), daemon=True).start()
    return {"task_id": task_id, "status": "downloading", "model_name": task["model_name"], "base_model": task["base_model"], "source": "huggingface"}


def _content_disposition_filename(value: str) -> str | None:
    if not value:
        return None
    message = Message()
    message["Content-Disposition"] = value
    try:
        return message.get_filename()
    except Exception:
        return None


def _public_url(value: str) -> str:
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _run_direct_url_download(task_id: str, req: DirectUrlImportRequest, cancel_event: threading.Event):
    url = req.url.strip()
    fd = None
    tmp_path = None
    final_path = None
    entry = None
    registered = False
    downloaded = 0
    total_bytes = 0
    deadline = time.monotonic() + civitai_service.download_deadline_seconds()
    try:
        _check_cancelled(cancel_event)
        on_progress = make_download_progress_cb(task_id)
        on_progress({"status": f"Connecting to {urlsplit(url).hostname}...", "progress": 0.05, "downloaded_bytes": 0, "total_bytes": 0, "speed_mb_s": 0.0})
        response, final_url = civitai_service.open_public_https_stream(
            url,
            headers={"User-Agent": civitai_service.DEFAULT_USER_AGENT, "Accept": "application/octet-stream"},
            cancel_event=cancel_event,
            deadline=deadline,
        )
        with response:
            if response.status_code != 200:
                raise RuntimeError(f"Download failed with HTTP {response.status_code}")
            content_type = str(response.headers.get("Content-Type", "")).lower()
            if content_type.startswith(("text/html", "application/json", "text/xml")):
                raise ValueError("Download server returned a non-binary response")
            detected_name = _content_disposition_filename(str(response.headers.get("Content-Disposition", "")))
            if not detected_name:
                detected_name = Path(unquote(urlsplit(final_url).path)).name
            if not detected_name and req.name:
                detected_name = f"{req.name}.safetensors"
            detected_name = _sanitize_filename(detected_name or f"direct_lora_{task_id}.safetensors", f"direct_lora_{task_id}.safetensors")
            if not detected_name.lower().endswith(".safetensors"):
                raise ValueError("Downloaded filename is not a .safetensors file")
            total_bytes = civitai_service._content_length(response.headers)
            if total_bytes > MAX_LORA_UPLOAD_BYTES:
                raise ValueError("download exceeds the 8 GB limit")
            fd, tmp_name = tempfile.mkstemp(dir=str(LORA_FILES_DIR), prefix=".direct-", suffix=".download")
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "wb") as output:
                fd = None
                start_time = time.monotonic()
                last_speed_time = start_time
                last_speed_bytes = 0
                current_speed_mb = 0.0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    _check_cancelled(cancel_event)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Download deadline exceeded")
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > MAX_LORA_UPLOAD_BYTES:
                        raise ValueError("download exceeds the 8 GB limit")
                    output.write(chunk)
                    now = time.monotonic()
                    dt = now - last_speed_time
                    if dt >= 0.5:
                        current_speed_mb = ((downloaded - last_speed_bytes) / (1 << 20)) / dt
                        last_speed_time = now
                        last_speed_bytes = downloaded
                    fraction = min(0.95, downloaded / total_bytes) if total_bytes else 0.5
                    status = (
                        f"Downloading: {downloaded >> 20}MB / {total_bytes >> 20}MB ({fraction * 100:.0f}%) - {current_speed_mb:.1f} MB/s"
                        if total_bytes
                        else f"Downloading: {downloaded >> 20}MB - {current_speed_mb:.1f} MB/s"
                    )
                    on_progress({"status": status, "progress": fraction, "downloaded_bytes": downloaded, "total_bytes": total_bytes, "speed_mb_s": current_speed_mb})
                output.flush()
                os.fsync(output.fileno())
        if total_bytes and downloaded != total_bytes:
            raise ValueError("Download ended before the expected byte count")
        if not civitai_service.is_valid_safetensors(tmp_path):
            raise ValueError("Downloaded file is not a valid .safetensors archive")
        _check_cancelled(cancel_event)
        on_progress({"status": "Inspecting model architecture...", "progress": 0.96, "downloaded_bytes": downloaded, "total_bytes": total_bytes, "speed_mb_s": 0.0})
        detected_base, detected_triggers = _inspect_safetensors(tmp_path)
        final_base = req.base_model or detected_base or "sdxl"
        final_triggers = req.triggers if req.triggers is not None else detected_triggers
        if final_base not in _BASE_MODELS:
            raise ValueError("invalid LoRA base model")
        if any(not isinstance(trigger, str) or not trigger.strip() or len(trigger) > 200 or any(ord(char) < 32 or ord(char) == 127 for char in trigger) for trigger in final_triggers):
            raise ValueError("invalid LoRA trigger")
        target_dir = SDXL_LORA_DIR if final_base == "sdxl" else LORA_FILES_DIR
        target_root = target_dir.resolve()
        final_path = (target_root / detected_name).resolve()
        if not final_path.is_relative_to(target_root):
            raise ValueError("invalid LoRA destination")
        if final_path.exists():
            final_path = (target_root / f"{final_path.stem}_{task_id}.safetensors").resolve()
            if not final_path.is_relative_to(target_root):
                raise ValueError("invalid LoRA destination")
        shutil.move(str(tmp_path), str(final_path))
        tmp_path = None
        try:
            os.chmod(final_path, 0o600)
        except OSError:
            pass
        on_progress({"status": "Computing SHA-256 digest...", "progress": 0.98, "downloaded_bytes": downloaded, "total_bytes": total_bytes, "speed_mb_s": 0.0})
        sha = civitai_service.compute_file_sha256(final_path, cancel_event=cancel_event)
        _check_cancelled(cancel_event)
        display_name = _display_name(req.name or final_path.stem, "Direct LoRA")
        entry = {
            "name": display_name,
            "path": str(final_path),
            "scale": 1.0,
            "triggers": final_triggers,
            "base_model": final_base,
            "sha256": sha,
            "source": "direct_url",
            "url": _public_url(url),
        }
        try:
            civitai_data = civitai_service.fetch_by_hash(sha)
            if civitai_data:
                metadata = civitai_service.extract_civitai_metadata(civitai_data)
                for key in ("civitai_version_id", "civitai_model_id", "civitai_model_name", "civitai_version_name"):
                    if metadata.get(key):
                        entry[key] = metadata[key]
                if not entry.get("triggers") and metadata.get("triggers"):
                    entry["triggers"] = metadata["triggers"]
        except Exception:
            pass
        _check_cancelled(cancel_event)
        _upsert_lora_entries([entry])
        registered = True
        if not _finish_task(task_id, entry, f"Installed {entry['name']} successfully!"):
            _cleanup_registered_entry(entry)
    except Exception as e:
        if not registered:
            _cleanup_registered_entry(entry)
            if final_path is not None:
                try:
                    resolved = final_path.resolve()
                    if any(resolved.is_relative_to(root) for root in (LORA_FILES_DIR.resolve(), SDXL_LORA_DIR.resolve())) and resolved.is_file():
                        resolved.unlink()
                except OSError:
                    pass
        _fail_task(task_id, e)
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _start_civitai_download_task(req: CivitaiImportRequest):
    cleaned = req.url_or_id.strip()
    if hf_service.parse_hf_input(cleaned):
        return _start_hf_download_task(cleaned, custom_name=req.name, token=req.api_key)
    try:
        version_id = civitai_service.parse_civitai_input(cleaned, api_key=req.api_key)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if not version_id:
        raise HTTPException(400, "Invalid Civitai or Hugging Face URL / ID")
    try:
        meta = civitai_service.extract_civitai_metadata(
            civitai_service.fetch_by_version_id(version_id, api_key=req.api_key) or {}
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    task_id = uuid.uuid4().hex[:12]
    cancel_event = threading.Event()
    task = {
        "id": task_id,
        "source": "civitai",
        "version_id": version_id,
        "identity": ("civitai", version_id),
        "model_name": _display_name(req.name or meta.get("civitai_model_name") or f"LoRA #{version_id}", f"LoRA #{version_id}"),
        "base_model": meta.get("base_model", "sdxl"),
        "status": "downloading",
        "progress": 0.05,
        "downloaded_bytes": 0,
        "total_bytes": meta.get("expected_bytes") or 0,
        "speed_mb_s": 0.0,
        "status_text": f"Connecting to Civitai for {_display_name(meta.get('civitai_model_name'), 'LoRA')}...",
        "started_at": time.time(),
        "finished_at": None,
        "error": None,
        "result": None,
        "worker_active": True,
        "cancel_event": cancel_event,
    }
    with _DOWNLOAD_LOCK:
        for existing_id, existing in DOWNLOAD_TASKS.items():
            if existing.get("worker_active") and existing.get("source") == "civitai" and existing.get("version_id") == version_id:
                return {"task_id": existing_id, "status": existing.get("status", "downloading"), "model_name": existing.get("model_name"), "base_model": existing.get("base_model"), "source": "civitai", "already_running": True}
        DOWNLOAD_TASKS[task_id] = task
    threading.Thread(target=_run_civitai_download, args=(task_id, req, version_id, cancel_event), daemon=True).start()
    return {"task_id": task_id, "status": "downloading", "model_name": task["model_name"], "base_model": task["base_model"], "source": "civitai"}


def _start_direct_url_download_task(req: DirectUrlImportRequest):
    url = req.url.strip()
    if hf_service.parse_hf_input(url):
        return _start_hf_download_task(url, custom_name=req.name)
    try:
        validated_url = civitai_service.validate_public_https_url(url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    url_identity = hashlib.sha256(validated_url.encode("utf-8")).hexdigest()
    task_id = uuid.uuid4().hex[:12]
    cancel_event = threading.Event()
    path_stem = Path(urlsplit(validated_url).path).stem
    safe_stem = civitai_service._safe_filename(path_stem, f"direct_lora_{task_id}.safetensors").removesuffix(".safetensors")[:80]
    task = {
        "id": task_id,
        "source": "direct_url",
        "url": _public_url(validated_url),
        "identity": ("direct_url", url_identity),
        "model_name": _display_name(req.name or safe_stem, f"Direct LoRA #{task_id}"),
        "base_model": req.base_model or "sdxl",
        "status": "downloading",
        "progress": 0.05,
        "downloaded_bytes": 0,
        "total_bytes": 0,
        "speed_mb_s": 0.0,
        "status_text": f"Connecting to {urlsplit(validated_url).hostname}...",
        "started_at": time.time(),
        "finished_at": None,
        "error": None,
        "result": None,
        "worker_active": True,
        "cancel_event": cancel_event,
    }
    with _DOWNLOAD_LOCK:
        for existing_id, existing in DOWNLOAD_TASKS.items():
            if existing.get("worker_active") and existing.get("identity") == task["identity"]:
                return {"task_id": existing_id, "status": existing.get("status", "downloading"), "model_name": existing.get("model_name"), "base_model": existing.get("base_model"), "source": "direct_url", "already_running": True}
        DOWNLOAD_TASKS[task_id] = task
    threading.Thread(target=_run_direct_url_download, args=(task_id, req, cancel_event), daemon=True).start()
    return {"task_id": task_id, "status": "downloading", "model_name": task["model_name"], "base_model": task["base_model"], "source": "direct_url"}


def _validate_import_options(name: str | None, triggers: list[str] | None):
    if name is not None and (not name.strip() or any(ord(char) < 32 or ord(char) == 127 for char in name)):
        raise HTTPException(400, "invalid download name")
    if triggers is not None and any(not isinstance(trigger, str) or not trigger.strip() or len(trigger) > 200 for trigger in triggers):
        raise HTTPException(400, "invalid download trigger")


@router.post("/api/loras/download")
def download_lora_unified(req: UnifiedDownloadRequest):
    _validate_import_options(req.name, req.triggers)
    raw = req.url_or_id.strip()
    if hf_service.parse_hf_input(raw):
        return _start_hf_download_task(raw, custom_name=req.name, token=req.token)
    try:
        host = (urlsplit(raw).hostname or "").lower() if raw.lower().startswith(("http://", "https://")) else ""
    except ValueError:
        host = ""
    if host in _CIVITAI_HOSTS or (raw.isdigit() and len(raw) <= 10):
        return _start_civitai_download_task(CivitaiImportRequest(url_or_id=raw, name=req.name, api_key=req.api_key or req.token))
    if raw.lower().startswith(("http://", "https://")):
        return _start_direct_url_download_task(DirectUrlImportRequest(url=raw, name=req.name, triggers=req.triggers, base_model=req.base_model))
    if civitai_service.parse_civitai_input(raw):
        return _start_civitai_download_task(CivitaiImportRequest(url_or_id=raw, name=req.name, api_key=req.api_key or req.token))
    raise HTTPException(400, "Could not identify download source")


@router.get("/api/loras/downloads")
def get_lora_downloads():
    hidden_keys = {"cancel_event", "worker_active", "identity", "url", "repo_id", "filename", "revision", "resolved_revision", "_last_progress_at", "_speed_t", "_speed_b"}
    with _DOWNLOAD_LOCK:
        now = time.time()
        for task_id in [
            task_id
            for task_id, task in DOWNLOAD_TASKS.items()
            if task.get("status") in ("done", "error", "cancelled") and not task.get("worker_active") and task.get("finished_at") and now - task["finished_at"] > 600
        ]:
            DOWNLOAD_TASKS.pop(task_id, None)
        result = [{key: value for key, value in task.items() if key not in hidden_keys} for task in DOWNLOAD_TASKS.values()]
    result.sort(key=lambda item: item.get("started_at", 0), reverse=True)
    return result


@router.delete("/api/loras/downloads/{task_id}")
def cancel_lora_download(task_id: str):
    with _DOWNLOAD_LOCK:
        task = DOWNLOAD_TASKS.get(task_id)
        if not task:
            raise HTTPException(404, "Download task not found")
        if task.get("status") == "downloading":
            cancel_event = task.get("cancel_event")
            if cancel_event:
                cancel_event.set()
            task["status"] = "cancelled"
            task["status_text"] = "Download cancelled by user"
            task["finished_at"] = time.time()
            task.pop("cancel_event", None)
        return {"status": "ok", "task_id": task_id}


class ModelDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1, max_length=120)
    # Added for the Models tab. The registry (generator.MODELS) used to be the only way in,
    # so the HF browser could list a repo it could not install. These two let an arbitrary
    # `org/name` be fetched into ASSET_DIR/models/<install_name>; model_id then becomes just
    # the task label. Both are optional so existing callers are unaffected.
    repo_id: str | None = Field(default=None, max_length=200)
    install_name: str | None = Field(default=None, max_length=120)
    # Task 5: unlisted-model declaration. Required exactly when the repo is
    # outside generator.MODELS (the route synthesizes minfo); registry-known
    # repos ignore these fields entirely. The worker tail turns them into the
    # verify_and_register declaration on download completion.
    family: str | None = Field(default=None, max_length=20)
    variant: str | None = Field(default=None, max_length=20)


class _ModelDownloadTqdm:
    def __init__(self, task_id: str, cancel_event: threading.Event):
        self._task_id = task_id
        self._cancel_event = cancel_event

    def __call__(self, *args, **kwargs):
        from tqdm import tqdm

        return _ReportingTqdm(tqdm, self._task_id, self._cancel_event, *args, **kwargs)


class _ReportingTqdm:
    def __init__(self, base, task_id, cancel_event, *args, **kwargs):
        self._base = base(*args, **kwargs)
        self._task_id = task_id
        self._cancel_event = cancel_event

    def __getattr__(self, item):
        return getattr(self._base, item)

    def __enter__(self):
        self._base.__enter__()
        return self

    def __exit__(self, *exc):
        if self._cancel_event.is_set():
            raise civitai_service.DownloadCancelled("Download cancelled by user")
        return self._base.__exit__(*exc)

    def update(self, n=1):
        if self._cancel_event.is_set():
            raise civitai_service.DownloadCancelled("Download cancelled by user")
        with _MODEL_DOWNLOAD_LOCK:
            task = MODEL_DOWNLOAD_TASKS.get(self._task_id)
            if task and task.get("status") == "downloading":
                total = task.get("total_bytes") or 0
                # Clamp to the known total. tqdm is constructed once per file, so its
                # cumulative counts get summed across files and can overshoot: a 99.6MB
                # repo reported 194.3MB downloaded against a 99.7MB total, which made
                # every progress bar read over 100%. The exact upstream double-report is
                # not worth chasing here; clamping keeps the displayed number true.
                bumped = task.get("downloaded_bytes", 0) + int(n)
                task["downloaded_bytes"] = min(bumped, total) if total else bumped
                task["progress"] = min(0.99, task["downloaded_bytes"] / total) if total else 0.5
                # Heartbeat for the stall watchdog. huggingface_hub can stop writing
                # without raising, so liveness has to be inferred from byte movement.
                task["_last_progress_at"] = time.monotonic()
                now = time.monotonic()
                elapsed = now - task.get("_speed_t", now)
                if elapsed >= 0.4:
                    task["speed_mb_s"] = ((task["downloaded_bytes"] - task.get("_speed_b", 0)) / (1 << 20)) / elapsed
                    task["_speed_t"] = now
                    task["_speed_b"] = task["downloaded_bytes"]
                if total:
                    task["status_text"] = f"Downloading {task['downloaded_bytes'] >> 20}/{total >> 20} MB ({task['progress'] * 100:.0f}%)"
        return self._base.update(n)


# huggingface_hub gives up on nothing: a stalled transfer leaves the task in
# "downloading" for ever, which is what a fresh flux2-klein-4b install did -- one
# 2 GB shard sat at 575 MB with zero bytes of progress while the UI read "99%".
# No timeout means no error, so the user gets a bar that never moves and no
# explanation. This bounds the silence: past the limit the task fails with a
# message that says what happened and what to do.
# Base stall limit, scaled by file size: small files get base, large files get more time.
_DOWNLOAD_STALL_LIMIT_BASE_S = float(os.environ.get("DIFFUSIONBEAR_DOWNLOAD_STALL_BASE_SECONDS", "180"))
# Per-MB allowance: additional seconds per MB of the current file being downloaded
_DOWNLOAD_STALL_MB_PER_S = float(os.environ.get("DIFFUSIONBEAR_DOWNLOAD_STALL_MB_PER_S", "0.5"))
# Absolute cap so a single 10GB file doesn't wait forever
_DOWNLOAD_STALL_MAX_S = float(os.environ.get("DIFFUSIONBEAR_DOWNLOAD_STALL_MAX_SECONDS", "3600"))

# Backward compatibility: tests and old code expect this name
_DOWNLOAD_STALL_LIMIT_S = _DOWNLOAD_STALL_LIMIT_BASE_S

# Allow tests to override the base limit via the old name
def _get_stall_base() -> float:
    return _DOWNLOAD_STALL_LIMIT_BASE_S


def _set_stall_base(val: float) -> None:
    global _DOWNLOAD_STALL_LIMIT_BASE_S, _DOWNLOAD_STALL_LIMIT_S
    _DOWNLOAD_STALL_LIMIT_BASE_S = val
    _DOWNLOAD_STALL_LIMIT_S = val


def _estimate_current_file_mb(task: dict) -> float:
    """Estimate the size of the file currently being downloaded."""
    total = task.get("total_bytes") or 0
    done = task.get("downloaded_bytes") or 0
    # Heuristic: assume files are roughly equal size, estimate remaining / files_left
    # But we don't know file count here. Use a simple approach:
    # If we have total and progress, estimate current file as total / (10 * progress + 1)
    # Better: use speed to estimate - if speed is known, current file ~ speed * time_per_file
    # Simplest: cap at total size, minimum 100MB
    return max(100.0, (total - done) / (1 << 20))


def _stall_limit_for_task(task: dict) -> float:
    """Dynamic stall limit based on file size."""
    # Test mode: if base limit is set very low (< 10s), use it directly
    # This allows tests to force a specific timeout without file-size scaling
    if _DOWNLOAD_STALL_LIMIT_BASE_S < 10:
        return _DOWNLOAD_STALL_LIMIT_BASE_S
    current_file_mb = _estimate_current_file_mb(task)
    limit = _DOWNLOAD_STALL_LIMIT_BASE_S + (current_file_mb * _DOWNLOAD_STALL_MB_PER_S)
    return min(limit, _DOWNLOAD_STALL_MAX_S)


_INITIAL_CONNECT_TIMEOUT = 180.0


def _stalled_model_task_ids() -> list[str]:
    """Download tasks whose byte counter has not moved within the limit."""
    now = time.monotonic()
    out = []
    with _MODEL_DOWNLOAD_LOCK:
        for task_id, task in MODEL_DOWNLOAD_TASKS.items():
            if task.get("status") != "downloading" or not task.get("worker_active"):
                continue
            if task.get("phase") == "copying":
                # Snapshot copy to destination is I/O bound on local disk, not network download
                continue
            last = task.get("_last_progress_at")
            if last is None:
                continue
            total = task.get("total_bytes") or 0
            done = task.get("downloaded_bytes") or 0
            # Only meaningful once some bytes exist: a slow start on a big repo is
            # not a stall, and the very first file can take a while to appear.
            if done <= 0:
                continue
            limit = _stall_limit_for_task(task)
            if now - last > limit and (total == 0 or done < total):
                out.append(task_id)
    return out


def _enforce_download_stall_watchdog() -> None:
    for task_id in _stalled_model_task_ids():
        with _MODEL_DOWNLOAD_LOCK:
            task = MODEL_DOWNLOAD_TASKS.get(task_id, {})
            label = task.get("model_name") or task.get("model_id") or "model"
            got = (task.get("downloaded_bytes") or 0) >> 20
            total = (task.get("total_bytes") or 0) >> 20
            limit = _stall_limit_for_task(task)
        _fail_model_task(
            task_id,
            f"stalled: no data received for {limit:.0f}s while "
            f"downloading {label} ({got} MB of {total} MB). The transfer hung rather "
            f"than failed. Check the network, then start the download again -- it "
            f"resumes from what already arrived.",
        )

def _fail_model_task(task_id: str, error: Exception | str):
    with _MODEL_DOWNLOAD_LOCK:
        task = MODEL_DOWNLOAD_TASKS.get(task_id)
        if not task:
            return False
        if task.get("status") != "downloading":
            if task.get("status") == "cancelled":
                task["worker_active"] = False
            return False
        message = str(error)
        cancelled = isinstance(error, civitai_service.DownloadCancelled) or "cancelled by user" in message.lower()
        task["status"] = "cancelled" if cancelled else "error"
        task["worker_active"] = False
        task["error"] = message[:2000]
        task["status_text"] = "Download cancelled" if cancelled else f"Download failed: {message}"[:500]
        task["finished_at"] = time.time()
        task.pop("cancel_event", None)
        return True


def _finish_model_task(task_id: str, status_text: str):
    with _MODEL_DOWNLOAD_LOCK:
        task = MODEL_DOWNLOAD_TASKS.get(task_id)
        if not task:
            return False
        if task.get("status") != "downloading":
            if task.get("status") == "cancelled":
                task["worker_active"] = False
            return False
        task["status"] = "done"
        task["worker_active"] = False
        task["progress"] = 1.0
        task["finished_at"] = time.time()
        task["status_text"] = status_text[:500]
        task["result"] = {"model_id": task.get("model_id")}
        task.pop("cancel_event", None)
        return True


def _valid_model_filename(value: object) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 1024 or value.startswith("/") or "\\" in value or "\x00" in value:
        return None
    normalized = value
    parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    return normalized


def _incomplete_paths(target_dir: Path | None, repo: str) -> set[Path]:
    root = target_dir / ".cache" / "huggingface" / "download" if target_dir else Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo.replace('/', '--')}" / "blobs"
    try:
        return {
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.name.endswith((".incomplete", ".part"))
        } if root.is_dir() else set()
    except OSError:
        return set()


def _snapshot_files(root: Path | None) -> set[Path]:
    if root is None:
        return set()
    try:
        return {path for path in root.rglob("*") if path.is_file()}
    except OSError:
        return set()


def _drop_orphan_partials(target_dir: Path | None) -> tuple[int, int]:
    """Delete partial downloads no future run can resume. Returns (files, bytes).

    huggingface_hub uses a fresh temp name per download attempt, so a partial
    left behind by a killed run is never resumed -- yet it fails every later
    layout verification forever (seen: an 18-day-old 13 MB orphan failing a
    fresh 28-file download). Anything still partial after this run's own files
    completed cannot be live state: a failed file raises before this point.
    `.lock`/`.metadata` sidecars are left alone; only data partials go.
    """
    if target_dir is None:
        return 0, 0
    root = target_dir / ".cache" / "huggingface" / "download"
    if not root.is_dir():
        return 0, 0
    dropped = 0
    reclaimed = 0
    try:
        candidates = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.name.endswith((".incomplete", ".part"))
        )
    except OSError:
        return 0, 0
    for path in candidates:
        try:
            reclaimed += path.stat().st_size
            path.unlink()
            dropped += 1
        except OSError:
            continue
    return dropped, reclaimed


def _link_tokenizer_to_target(target_dir: Path, shared_tok: Path, is_qwen: bool = False) -> None:
    try:
        tok_dir = target_dir / "tokenizer"
        if not (tok_dir.exists() or tok_dir.is_symlink()):
            tok_dir.symlink_to(shared_tok)
        if is_qwen:
            proc_dir = target_dir / "processor"
            if not (proc_dir.exists() or proc_dir.is_symlink()):
                proc_dir.symlink_to(shared_tok)
    except OSError as err:
        print(f"[downloads] WARN could not symlink tokenizer: {err}", flush=True)


def _register_downloaded_qwen_tokenizer(target_dir: Path) -> None:
    """Register newly downloaded Qwen/FLUX tokenizer into central models/tokenizers/qwen2."""
    try:
        central_tok = generator.ASSET_DIR / "models" / "tokenizers" / "qwen2"
        if not (central_tok / "tokenizer_config.json").exists():
            source_tok = None
            for cand in (target_dir / "tokenizer", target_dir / "text_encoder", target_dir / "processor", target_dir):
                if (cand / "tokenizer_config.json").exists():
                    source_tok = cand
                    break
            if source_tok:
                central_tok.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                if not (central_tok.exists() or central_tok.is_symlink()):
                    central_tok.symlink_to(source_tok)
    except Exception as err:
        print(f"[downloads] WARN failed to register downloaded tokenizer: {err}", flush=True)


def _run_model_download(
    task_id: str,
    model_id: str,
    minfo: dict,
    cancel_event: threading.Event,
    repo_override: str | None = None,
    install_dir: Path | None = None,
):
    target_dir = None
    te_target = None
    te_before_files = set()
    repo = ""
    before_incomplete = set()
    before_files = set()
    cleanup_root = None
    try:
        _check_cancelled(cancel_event)
        if repo_override:
            repo = repo_override
        else:
            candidate_repo = generator.model_download_repo(model_id, minfo)
            if hf_service._valid_repo_id(candidate_repo) is None or str(candidate_repo).startswith("local:"):
                _fail_model_task(task_id, f"{minfo.get('label', model_id)} has no downloadable Hugging Face repository")
                return
            repo = str(candidate_repo)
        if hf_service._valid_repo_id(repo) is None:
            _fail_model_task(task_id, f"Invalid Hugging Face repository: {repo}")
            return
        if install_dir is not None:
            # HF-browser downloads: a plain directory under ASSET_DIR/models, not the
            # HF cache and not a registry model_dir. Containment is checked against
            # models/, NOT ASSET_DIR: safe_install_name permits dots, so ".."
            # resolves to ASSET_DIR and passes an ASSET_DIR-level check, which once
            # allowed model files into the asset root. The non-empty-parts rule
            # also rejects "." (models/ itself). Same invariant as
            # _resolve_inside_models, which guards the route above; this is the
            # last line before mkdir, so no future caller can bypass it by
            # building an install_dir another way.
            target_dir = install_dir
            models_root = (generator.ASSET_DIR / "models").resolve()
            try:
                inside = target_dir.resolve().relative_to(models_root)
            except ValueError:
                raise ValueError("install path must stay inside the models directory")
            if not inside.parts:
                raise ValueError("install path must stay inside the models directory")
            target_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(target_dir, 0o700)
            except OSError:
                pass
        elif minfo.get("engine") == "sdxl":
            configured_dir = minfo.get("model_dir")
            if not configured_dir:
                raise ValueError("model download directory is not configured")
            target_dir = Path(configured_dir)
            target_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(target_dir, 0o700)
            except OSError:
                pass
        cleanup_root = target_dir if target_dir is not None else generator._hf_repo_cache_dir(repo)
        before_files = _snapshot_files(cleanup_root)
        before_incomplete = _incomplete_paths(target_dir, repo)
        token = hf_service.get_hf_token()
        info = HfApi(token=token).model_info(repo, files_metadata=True, token=token)
        revision = str(getattr(info, "sha", "") or "")
        if not re.fullmatch(r"[A-Fa-f0-9]{40,64}", revision):
            raise ValueError("Hugging Face did not return a revision identity")
        files = []
        total = 0
        for sibling in getattr(info, "siblings", []) or []:
            filename = _valid_model_filename(getattr(sibling, "rfilename", None))
            if not filename:
                continue
            size = int(getattr(sibling, "size", 0) or 0)
            if size < 0:
                raise ValueError("invalid Hugging Face file size")
            files.append((filename, size))
            total += size
        if not files or len(files) > 10000:
            raise ValueError("Hugging Face model file list is invalid")

        # De-duplicate tokenizer download: if model belongs to Qwen / FLUX shared family
        # and a compatible tokenizer is already installed elsewhere, skip downloading tokenizer files.
        is_qwen_flux = generator.is_qwen_flux_tokenizer_model(model_id, minfo, repo)
        shared_tok = None
        if is_qwen_flux:
            shared_tok = generator._find_local_qwen_tokenizer_path(exclude_dir=target_dir)

        tokenizer_files_skipped = []
        if shared_tok is not None:
            filtered_files = []
            for filename, size in files:
                if generator.is_tokenizer_sibling(filename):
                    tokenizer_files_skipped.append((filename, size))
                    total -= size
                else:
                    filtered_files.append((filename, size))
            if tokenizer_files_skipped:
                files = filtered_files
                saved_kb = sum(s for _, s in tokenizer_files_skipped) // 1024
                print(
                    f"[downloads] Found compatible tokenizer already installed at {shared_tok}. "
                    f"Skipping download of {len(tokenizer_files_skipped)} tokenizer files ({saved_kb} KB) for {model_id}.",
                    flush=True,
                )

        # MOSS text-encoder substitution (see generator MODELS for the rationale):
        # upstream bf16 text_encoder/ is stripped from the download and our
        # pre-converted MLX q8-g64 encoder is fetched into a sibling directory
        # instead, which moss_engine._mlx_te_dir auto-detects. The engine's
        # snappy path uses the MLX encoder; the torch bf16 fallback needs the
        # full upstream snapshot, not this install.
        te_mlx_repo = minfo.get("te_mlx_repo")
        te_mlx_sibling = minfo.get("te_mlx_sibling")
        if te_mlx_repo and te_mlx_sibling and target_dir is not None:
            if hf_service._valid_repo_id(te_mlx_repo) is None:
                raise ValueError(f"Invalid Hugging Face text-encoder repository: {te_mlx_repo}")
            te_target = target_dir.parent / te_mlx_sibling
            te_before_files = _snapshot_files(te_target) if te_target.is_dir() else set()
            kept, te_files_skipped = [], []
            for filename, size in files:
                if filename.startswith("text_encoder/"):
                    te_files_skipped.append((filename, size))
                    total -= size
                else:
                    kept.append((filename, size))
            if te_files_skipped:
                files = kept
                saved_mb = sum(s for _, s in te_files_skipped) // (1 << 20)
                print(
                    f"[downloads] MOSS {model_id}: skipping upstream bf16 text_encoder "
                    f"({saved_mb} MB, {len(te_files_skipped)} files); a pre-converted "
                    f"MLX encoder from {te_mlx_repo} is fetched into {te_mlx_sibling} instead.",
                    flush=True,
                )
            te_info = HfApi(token=token).model_info(te_mlx_repo, files_metadata=True, token=token)
            def _te_size(sibling) -> int:
                name = getattr(sibling, "rfilename", "")
                if any(fnmatch.fnmatch(name, p) for p in ("*.json", "*.safetensors")):
                    return int(getattr(sibling, "size", 0) or 0)
                return 0

            total += sum(_te_size(sibling) for sibling in getattr(te_info, "siblings", []) or [])

        # Pre-flight disk space check: require 1.5x total size for safety margin
        try:
            stat = os.statvfs(target_dir if target_dir else generator.ASSET_DIR)
            free_bytes = stat.f_bavail * stat.f_frsize
            required = total * 3 // 2  # 1.5x for temp files, extraction, etc.
            if free_bytes < required:
                raise ValueError(
                    f"Insufficient disk space: need {required >> 30} GB (1.5x model size), "
                    f"have {free_bytes >> 30} GB free"
                )
        except OSError:
            pass  # Best effort; if statvfs fails, proceed anyway
        with _MODEL_DOWNLOAD_LOCK:
            task = MODEL_DOWNLOAD_TASKS.get(task_id)
            if task and task.get("status") == "downloading":
                task["total_bytes"] = total
                task["revision"] = revision
                task["status_text"] = f"Downloading {minfo.get('label', model_id)} weights..."
        from huggingface_hub import hf_hub_download

        # For SDXL: download to HF cache first (so files persist on failure), then copy to target_dir
        # For non-SDXL: download directly to target_dir via local_dir
        use_cache = minfo.get("engine") == "sdxl" and target_dir is not None

        for index, (filename, _) in enumerate(files, start=1):
            _check_cancelled(cancel_event)
            with _MODEL_DOWNLOAD_LOCK:
                task = MODEL_DOWNLOAD_TASKS.get(task_id)
                if task and task.get("status") == "downloading":
                    task["status_text"] = f"Downloading {minfo.get('label', model_id)} weights: {filename} ({index}/{len(files)})"
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                revision=revision,
                token=token,
                local_dir=None if use_cache else (str(target_dir) if target_dir else None),
                resume_download=True,
                tqdm_class=_ModelDownloadTqdm(task_id, cancel_event),
            )

        # For Qwen / FLUX models: link existing shared tokenizer if skipped, or register newly downloaded tokenizer
        if is_qwen_flux and target_dir is not None:
            if shared_tok is not None:
                _link_tokenizer_to_target(target_dir, shared_tok, is_qwen=("qwen" in str(model_id).lower()))
            else:
                generator._ensure_flux_tokenizer_available(target_dir)
                _register_downloaded_qwen_tokenizer(target_dir)

        # MOSS MLX text encoder: shipped separately (Ouinche repo) next to the
        # model dir, so the engine's sibling auto-detect finds it with no
        # second download at first generation. Skipped entirely when a
        # previous install already put the files there (shared sibling).
        if te_target is not None:
            with _MODEL_DOWNLOAD_LOCK:
                task = MODEL_DOWNLOAD_TASKS.get(task_id)
                if task and task.get("status") == "downloading":
                    task["status_text"] = f"Downloading MLX text encoder from {te_mlx_repo}..."
                    task["_last_progress_at"] = time.monotonic()
            _check_cancelled(cancel_event)
            te_target.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(te_target, 0o700)
            except OSError:
                pass
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=te_mlx_repo,
                local_dir=str(te_target),
                allow_patterns=["*.json", "*.safetensors"],
                token=token,
                resume_download=True,
                tqdm_class=_ModelDownloadTqdm(task_id, cancel_event),
            )
            print(f"[downloads] MLX text encoder installed at {te_target}", flush=True)

        # For SDXL: copy from HF cache snapshot to target_dir
        if use_cache:
            cache_dir = generator._hf_repo_cache_dir(repo)
            snapshot = generator._newest_hf_snapshot(cache_dir)
            if snapshot is None or not snapshot.is_dir():
                raise ValueError(f"Download completed but no valid snapshot found in HF cache for {repo}")
            import shutil
            print(f"[downloads] Copying {snapshot} -> {target_dir}", flush=True)
            with _MODEL_DOWNLOAD_LOCK:
                task = MODEL_DOWNLOAD_TASKS.get(task_id)
                if task:
                    task["phase"] = "copying"
                    task["status_text"] = f"Installing {minfo.get('label', model_id)} from cache..."
                    task["_last_progress_at"] = time.monotonic()
            # Remove any existing target_dir contents
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            target_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            # Copy entire snapshot
            for item in snapshot.iterdir():
                dest = target_dir / item.name
                with _MODEL_DOWNLOAD_LOCK:
                    task = MODEL_DOWNLOAD_TASKS.get(task_id)
                    if task:
                        task["_last_progress_at"] = time.monotonic()
                        task["status_text"] = f"Installing {item.name}..."
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
                with _MODEL_DOWNLOAD_LOCK:
                    task = MODEL_DOWNLOAD_TASKS.get(task_id)
                    if task:
                        task["_last_progress_at"] = time.monotonic()
            try:
                os.chmod(target_dir, 0o700)
            except OSError:
                pass
            with _MODEL_DOWNLOAD_LOCK:
                task = MODEL_DOWNLOAD_TASKS.get(task_id)
                if task:
                    task.pop("phase", None)

        # An SDXL engine can only load a diffusers directory. Verifying HERE means a
        # partial download is reported as a failure with a reason, instead of being
        # marked installed and then failing at generation with a traceback -- which is
        # what happened on a second machine, where an interrupted 9.5 GB shard left
        # an empty unet/ behind an INSTALLED badge.
        #
        # Orphaned partials from killed runs are dropped first: each attempt uses a
        # fresh temp name, so they are never resumed and would otherwise fail this
        # verification on every retry forever.
        if target_dir is not None:
            dropped, freed = _drop_orphan_partials(target_dir)
            if dropped:
                print(
                    f"[downloads] dropped {dropped} orphaned partial download(s), "
                    f"reclaimed {freed // (1 << 20)} MB",
                    flush=True,
                )
        if is_sdxl_engine(model_id):
            ok, why = sdxl_layout_report(target_dir)
            if not ok:
                # Check if it's the .bin files issue - auto-convert
                if "holds .bin files" in why and target_dir is not None:
                    print(f"[downloads] Detected .bin files in {model_id}, auto-converting to .safetensors...", flush=True)
                    try:
                        import sdxl_convert
                        def conv_progress(frac, stage):
                            with _MODEL_DOWNLOAD_LOCK:
                                task = MODEL_DOWNLOAD_TASKS.get(task_id)
                                if task and task.get("status") == "downloading":
                                    task["status_text"] = f"Converting {model_id}: {stage} ({frac*100:.0f}%)"
                                    task["progress"] = max(0.95, frac)
                        sdxl_convert.convert_bin_to_safetensors(
                            target_dir,
                            cancel_event=cancel_event,
                            on_progress=conv_progress,
                        )
                        # Re-check layout after conversion
                        ok, why = sdxl_layout_report(target_dir)
                        if not ok:
                            raise ValueError(why)
                        print(f"[downloads] Conversion successful for {model_id}", flush=True)
                    except Exception as conv_err:
                        raise ValueError(f"Auto-conversion from .bin to .safetensors failed: {conv_err}")
                else:
                    raise ValueError(why)

        # Auto-quantize Krea 2 text encoder on download completion & prune 7.5GB BF16 shards
        is_krea = model_id == "krea2-turbo" or (minfo and minfo.get("family") == "krea")
        if is_krea and target_dir is not None:
            try:
                def _krea_conv_cb(text: str):
                    with _MODEL_DOWNLOAD_LOCK:
                        t = MODEL_DOWNLOAD_TASKS.get(task_id)
                        if t and t.get("status") == "downloading":
                            t["status_text"] = text
                generator.ensure_krea_text_encoder_quantized(target_dir, task_progress_cb=_krea_conv_cb)
            except Exception as _k_exc:
                print(f"[downloads] WARN Krea 2 post-download text-encoder quantization failed: {_k_exc}", flush=True)

        # Task 5 tail: an unlisted download (the route stashed a
        # declaration) verifies and registers through the same
        # verify_and_register the onboard route uses -- same function, not a
        # copy. No declaration: old behavior exactly. A refusal fails the
        # task with the reason instead of finishing: never mark installed
        # what did not verify.
        with _MODEL_DOWNLOAD_LOCK:
            _tail_task = MODEL_DOWNLOAD_TASKS.get(task_id)
            _tail_declaration = dict(_tail_task.get("declaration")) if _tail_task and _tail_task.get("declaration") is not None else None
        if _tail_declaration is not None:
            if target_dir is not None:
                _tail_declaration["local_path"] = str(target_dir)
            import custom_models
            _tail_ok, _tail_reason, _tail_entry = custom_models.verify_and_register(_tail_declaration)
            if not _tail_ok:
                _fail_model_task(task_id, _tail_reason or "onboard verification failed")
                return
        if not _finish_model_task(task_id, f"{minfo.get('label', model_id)} installed successfully!"):
            for path in _snapshot_files(cleanup_root) - before_files:
                try:
                    path.unlink()
                except OSError:
                    pass
            return
        # Auto-register HF Browser downloads so they appear in Generate dropdown
        if install_dir is not None:
            # Identity marker for the auto-bind scan: an unbound install must
            # remain discoverable even if this settings write ever fails or is
            # undone (manifest/config tokens also work; this is the cheapest).
            try:
                (target_dir / "DIFFUSIONBEAR_MODEL_ID").write_text(
                    model_id, encoding="utf-8")
            except OSError as _marker_err:
                print(f"[downloads] WARN identity marker write failed: {_marker_err}", flush=True)
            try:
                import app_settings
                app_settings.update_settings({"model_paths": {model_id: str(install_dir.resolve())}})
            except Exception as e:
                print(f"[downloads] auto-register failed: {e}", flush=True)
    except Exception as e:
        if repo:
            # For SDXL with cache strategy: keep HF cache files (don't delete on failure)
            # Only clean up target_dir partials, not the HF cache
            use_cache = minfo.get("engine") == "sdxl" and target_dir is not None
            if not use_cache:
                for path in _snapshot_files(cleanup_root) - before_files:
                    try:
                        path.unlink()
                    except OSError:
                        pass
            # Same for the MOSS MLX text-encoder sibling, but never touch files
            # a previous install already had (te_before_files snapshot).
            if te_target is not None:
                for path in _snapshot_files(te_target) - te_before_files:
                    try:
                        path.unlink()
                    except OSError:
                        pass
            # Always clean up incomplete files in target_dir
            for path in _incomplete_paths(target_dir, repo) - before_incomplete:
                try:
                    path.unlink()
                except OSError:
                    pass
            # Also drop orphaned partials in target_dir (failed downloads leave them)
            if target_dir is not None:
                _drop_orphan_partials(target_dir)
        _fail_model_task(task_id, e)


def _unlisted_enum_refusal(family: str, variant: str | None) -> str | None:
    """Family/variant enum gate for unlisted-download POSTs.

    The worker tail enforces these through custom_models.verify_and_register,
    which runs minutes and gigabytes after the user clicked. Refuse here with
    the same strings custom_models.validate_entry uses on the onboard route
    so a typo fails fast, before any download starts.
    """
    if family not in ("sdxl", "flux2", "flux3", "krea", "zit", "qwen"):
        return "invalid family: must be one of sdxl|flux2|flux3|krea|zit|qwen"
    if family == "flux2":
        if variant not in ("4b", "9b"):
            return "invalid variant: flux2 requires one of 4b|9b"
    elif family == "qwen":
        if variant not in ("base", "turbo"):
            return "invalid variant: qwen requires one of base|turbo"
    elif variant is not None:
        return "invalid variant: must be absent unless family is flux2 or qwen"
    return None


@router.post("/api/models/download")
def download_model(req: ModelDownloadRequest):
    model_id = req.model_id.strip()
    minfo = generator.MODELS.get(model_id)
    install_dir = None

    if req.repo_id:
        # HF-browser path: any repo the Models tab listed, not just a registry entry.
        repo_id = hf_service._valid_repo_id(req.repo_id)
        if repo_id is None:
            raise HTTPException(400, "Invalid Hugging Face repository id")
        name = hf_browse.safe_install_name(repo_id)
        if req.install_name:
            name = hf_browse.safe_install_name(req.install_name)
        unlisted = generator.MODELS.get(model_id) is None
        declaration = None
        if unlisted:
            # The declaration is required exactly here (minfo is synthesized,
            # not registry): the worker tail registers the download through
            # verify_and_register. Registry-known repos ignore family/variant.
            family = (req.family or "").strip()
            if not family:
                raise HTTPException(400, "family is required for repos outside the model registry (one of sdxl|flux2|flux3|krea|zit|qwen)")
            refusal = _unlisted_enum_refusal(family, req.variant)
            if refusal is not None:
                raise HTTPException(400, refusal)
            declaration = {
                "id": None,
                "label": (req.model_id or "").strip() or None,
                "family": family,
                "variant": req.variant,
                "source": "hf",
                "hf_repo": repo_id,
            }
        if minfo is None:
            minfo = {"id": model_id, "label": name, "engine": "mflux"}
        # Contained against models/, not ASSET_DIR: safe_install_name permits
        # dots, so ".." survives it and ASSET_DIR/models/.. IS ASSET_DIR.
        # _resolve_inside_models also rejects "." (models/ itself) via the
        # non-empty-parts rule. Raises 400 here, before any task, thread or
        # directory exists.
        install_dir = _resolve_inside_models(name)
        if not install_dir.exists():
            with generator._model_maintenance_lock:
                with _MODEL_DOWNLOAD_LOCK:
                    for task_id, task in MODEL_DOWNLOAD_TASKS.items():
                        if task.get("repo_id") == repo_id and task.get("worker_active"):
                            return {"task_id": task_id, "status": task.get("status", "downloading"), "model_name": task.get("model_name"), "already_running": True}
            task_id = uuid.uuid4().hex[:12]
            cancel_event = threading.Event()
            task = {
                "id": task_id,
                "source": "model",
                "model_id": model_id,
                "repo_id": repo_id,
                "model_name": minfo.get("label", model_id),
                "engine": minfo.get("engine", "mflux"),
                "status": "downloading",
                "progress": 0.0,
                "downloaded_bytes": 0,
                "total_bytes": 0,
                "speed_mb_s": 0.0,
                "status_text": f"Preparing download of {minfo.get('label', model_id)}...",
                "started_at": time.time(),
                "_last_progress_at": time.monotonic(),
                "finished_at": None,
                "error": None,
                "result": None,
                "install_dir": str(install_dir),
                "worker_active": True,
                "cancel_event": cancel_event,
                "declaration": declaration,
            }
            with _MODEL_DOWNLOAD_LOCK:
                for existing_id, existing in MODEL_DOWNLOAD_TASKS.items():
                    if existing.get("repo_id") == repo_id and existing.get("worker_active"):
                        return {"task_id": existing_id, "status": existing.get("status", "downloading"), "model_name": existing.get("model_name"), "already_running": True}
                MODEL_DOWNLOAD_TASKS[task_id] = task
            threading.Thread(
                target=_run_model_download,
                args=(task_id, model_id, minfo, cancel_event),
                kwargs={"repo_override": repo_id, "install_dir": install_dir},
                daemon=True,
            ).start()
            return {"task_id": task_id, "status": "downloading", "model_name": task["model_name"], "source": "model"}
        # Fall through: the directory exists, so treat it like an already-installed model.
        return {"status": "already_installed", "model_id": model_id, "model_name": minfo.get("label", model_id)}

    if minfo is None:
        raise HTTPException(404, f"Unknown model: {model_id}")
    repo_id = generator.model_download_repo(model_id, minfo)
    if hf_service._valid_repo_id(repo_id) is None or str(repo_id).startswith("local:"):
        raise HTTPException(400, f"{minfo.get('label', model_id)} has no downloadable Hugging Face repository")
    with generator._model_maintenance_lock:
        with _MODEL_DOWNLOAD_LOCK:
            for task_id, task in MODEL_DOWNLOAD_TASKS.items():
                if task.get("repo_id") == repo_id and task.get("worker_active"):
                    return {"task_id": task_id, "status": task.get("status", "downloading"), "model_name": task.get("model_name"), "already_running": True}
        if generator.is_model_cached(model_id):
            return {"status": "already_installed", "model_id": model_id, "model_name": minfo.get("label", model_id)}
        task_id = uuid.uuid4().hex[:12]
        cancel_event = threading.Event()
        install_dir = generator.ASSET_DIR / "models" / model_id
        task = {
            "id": task_id,
            "source": "model",
            "model_id": model_id,
            "repo_id": repo_id,
            "model_name": minfo.get("label", model_id),
            "engine": minfo.get("engine", "mflux"),
            "status": "downloading",
            "progress": 0.0,
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "speed_mb_s": 0.0,
            "status_text": f"Preparing download of {minfo.get('label', model_id)}...",
            "started_at": time.time(),
            "_last_progress_at": time.monotonic(),
            "finished_at": None,
            "error": None,
            "result": None,
            "install_dir": str(install_dir),
            "worker_active": True,
            "cancel_event": cancel_event,
        }
        with _MODEL_DOWNLOAD_LOCK:
            for existing_id, existing in MODEL_DOWNLOAD_TASKS.items():
                if existing.get("repo_id") == repo_id and existing.get("worker_active"):
                    return {"task_id": existing_id, "status": existing.get("status", "downloading"), "model_name": existing.get("model_name"), "already_running": True}
            MODEL_DOWNLOAD_TASKS[task_id] = task
        threading.Thread(target=_run_model_download, args=(task_id, model_id, minfo, cancel_event), kwargs={"install_dir": install_dir}, daemon=True).start()
        return {"task_id": task_id, "status": "downloading", "model_name": task["model_name"], "source": "model"}



class CivitaiModelDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: int = Field(ge=1)
    model_version_id: int = Field(ge=1)
    name: str | None = Field(default=None, max_length=120)
    token: str | None = Field(default=None, max_length=8192)
    # Task 5: unlisted-model declaration. This route serves versions no
    # registry entry claims, so family is always required here (value
    # verified by the worker tail, not the route).
    family: str | None = Field(default=None, max_length=20)
    variant: str | None = Field(default=None, max_length=20)


def _run_civitai_model_download(task_id: str, version_id: int, name: str, token: str | None,
                                cancel_event: threading.Event):
    """Stream a Civitai model file into ASSET_DIR/models/<name>/.

    Civitai's /api/download/models/{versionId} answers with a 302 to a pre-signed URL on
    third-party storage. civitai_service.open_public_https_stream already follows that
    redirect, re-validates the host as public on every hop, and -- importantly -- strips
    the Authorization header once the redirect leaves civitai.com, so the token is never
    handed to the storage provider.
    """
    target_dir = generator.ASSET_DIR / "models" / name
    try:
        _check_cancelled(cancel_event)
        auth = civitai_service._valid_token(token) or civitai_service.get_civitai_api_key()
        headers = {"Authorization": f"Bearer {auth}"} if auth else {}
        url = f"{civitai_browse.API_BASE.replace('/api/v1', '')}/api/download/models/{version_id}?type=Model&format=SafeTensor"
        with _MODEL_DOWNLOAD_LOCK:
            task = MODEL_DOWNLOAD_TASKS.get(task_id)
            if task and task.get("status") == "downloading":
                task["status_text"] = f"Downloading {name} from Civitai..."
        response, final_url = civitai_service.open_public_https_stream(
            url, headers=headers, auth_hosts=civitai_service._CIVITAI_AUTH_HOSTS,
            cancel_event=cancel_event, deadline=civitai_service.download_deadline_seconds(),
        )
        with response:
            disposition = response.headers.get("Content-Disposition") or ""
            filename = civitai_service._safe_filename(
                _content_disposition_filename(disposition),
                fallback=f"{name}.safetensors",
            )
            target_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            destination = target_dir / filename
            expected = int(response.headers.get("Content-Length") or 0)
            with _MODEL_DOWNLOAD_LOCK:
                task = MODEL_DOWNLOAD_TASKS.get(task_id)
                if task and task.get("status") == "downloading" and expected:
                    task["total_bytes"] = expected
            written = 0
            started = time.monotonic()
            last = started
            # Atomic download: stream into a same-directory temp file opened ONCE
            # in "wb", validate it, then os.replace() onto the destination. The
            # old loop opened "ab" per chunk: a retry appended onto a cancelled
            # partial (corrupt safetensors kept as "installed"), and cancel or
            # error left the partial behind for the next attempt to append to.
            # Same-directory temp keeps the replace atomic (no cross-filesystem
            # EXDEV), "wb" truncates any temp a crashed run left behind, and
            # the ".part" suffix is what the existing orphan sweeps already
            # reclaim, so even a kill -9 cannot leave worse than that.
            tmp_destination = destination.with_name(destination.name + ".part")
            try:
                with tmp_destination.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if cancel_event.is_set():
                            raise civitai_service.DownloadCancelled("Download cancelled by user")
                        if not chunk:
                            continue
                        fh.write(chunk)
                        written += len(chunk)
                        now = time.monotonic()
                        if now - last >= 0.4:
                            with _MODEL_DOWNLOAD_LOCK:
                                task = MODEL_DOWNLOAD_TASKS.get(task_id)
                                if task and task.get("status") == "downloading":
                                    task["downloaded_bytes"] = min(written, expected) if expected else written
                                    task["progress"] = min(0.99, task["downloaded_bytes"] / expected) if expected else 0.5
                                    task["speed_mb_s"] = (task["downloaded_bytes"] / (1 << 20)) / max(now - started, 1e-6)
                            last = now
                if not civitai_service.is_valid_safetensors(tmp_destination):
                    raise ValueError("downloaded file is not a valid safetensors checkpoint")
                os.replace(tmp_destination, destination)
            finally:
                tmp_destination.unlink(missing_ok=True)
        # Task 5 tail: same verify+register as the HF path (same function,
        # not a copy). No declaration: old behavior exactly. A refusal fails
        # the task with the reason instead of finishing.
        with _MODEL_DOWNLOAD_LOCK:
            _tail_task = MODEL_DOWNLOAD_TASKS.get(task_id)
            _tail_declaration = dict(_tail_task.get("declaration")) if _tail_task and _tail_task.get("declaration") is not None else None

        # If the model is SDXL and currently a single-file checkpoint, convert it to
        # the diffusers layout before registering so the engine can load it.
        if _tail_declaration and _tail_declaration.get("family") == "sdxl":
            if not (target_dir / "model_index.json").is_file() and any(target_dir.glob("*.safetensors")):
                if not sdxl_convert.has_diffusers():
                    _fail_model_task(task_id, sdxl_convert.missing_dependency_message())
                    return
                with _MODEL_DOWNLOAD_LOCK:
                    task = MODEL_DOWNLOAD_TASKS.get(task_id)
                    if task and task.get("status") == "downloading":
                        task["status_text"] = f"Converting {name} to diffusers layout..."
                        task["progress"] = 0.0

                def _on_convert_progress(fraction: float, stage: str) -> None:
                    with _MODEL_DOWNLOAD_LOCK:
                        t = MODEL_DOWNLOAD_TASKS.get(task_id)
                        if t and t.get("status") == "downloading":
                            t["progress"] = max(0.0, min(0.99, fraction))
                            t["status_text"] = f"Converting {name}: {stage or 'working'}"

                source_ckpt = destination if destination.is_file() else next(target_dir.glob("*.safetensors"))
                sdxl_convert.convert(source_ckpt, target_dir, cancel_event=cancel_event, on_progress=_on_convert_progress, purge_source=True)

        if _tail_declaration is not None:
            _tail_declaration["local_path"] = str(target_dir)
            import custom_models
            _tail_ok, _tail_reason, _tail_entry = custom_models.verify_and_register(_tail_declaration)
            if not _tail_ok:
                _fail_model_task(task_id, _tail_reason or "onboard verification failed")
                return
        if target_dir is not None and generator.is_qwen_flux_tokenizer_model(name, {}, ""):
            generator._ensure_flux_tokenizer_available(target_dir)
        _finish_model_task(task_id, f"{name} installed successfully!")
    except civitai_service.DownloadCancelled:
        _fail_model_task(task_id, "Download cancelled by user")
    except Exception as exc:
        _fail_model_task(task_id, exc)



class ConversionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    install_name: str = Field(min_length=1, max_length=120)
    custom_id: str | None = Field(default=None)
    label: str | None = Field(default=None)
    local_path: str | None = Field(default=None)
    register_on_finish: bool = Field(default=True)


def _run_sdxl_conversion(task_id: str, install_name: str, cancel_event: threading.Event,
                         custom_id: str | None = None, label: str | None = None,
                         local_path: str | None = None, register_on_finish: bool = True):
    """Convert a downloaded single-file checkpoint into a diffusers directory.

    Runs the real work in a subprocess (see sdxl_convert) so a hung or crashing
    conversion cannot take the backend down. On success the directory lands where the SDXL
    engine can find it, and only then is the task marked done -- a half-written diffusers
    folder would read as "installed" and then fail at generation time.
    """
    if local_path:
        root = Path(local_path).expanduser().resolve()
    else:
        root = generator.ASSET_DIR / "models" / install_name
    source = None
    if root.is_file() and root.suffix.lower() == ".safetensors":
        source = root
        root = root.parent
    elif root.is_dir():
        for candidate in sorted(root.glob("*.safetensors")):
            source = candidate
            break
    if source is None:
        _fail_model_task(task_id, f"no .safetensors checkpoint found in {install_name}")
        return

    def on_progress(fraction: float, stage: str) -> None:
        with _MODEL_DOWNLOAD_LOCK:
            task = MODEL_DOWNLOAD_TASKS.get(task_id)
            if task and task.get("status") == "downloading":
                task["progress"] = max(0.0, min(0.99, fraction))
                task["status_text"] = f"Converting {install_name}: {stage or 'working'}"

    try:
        _check_cancelled(cancel_event)
        if not sdxl_convert.has_diffusers():
            _fail_model_task(task_id, sdxl_convert.missing_dependency_message())
            return
        sdxl_convert.convert(source, root, cancel_event=cancel_event, on_progress=on_progress, purge_source=True)
        if register_on_finish:
            import custom_models
            declaration = {
                "id": custom_id or custom_models._slugify_folder_name(root.name),
                "label": label or root.name,
                "family": "sdxl",
                "local_path": str(root),
                "source": "local",
            }
            reg_ok, reg_reason, _ = custom_models.verify_and_register(declaration)
            if not reg_ok and "already registered" not in (reg_reason or ""):
                _fail_model_task(task_id, reg_reason or "registration failed")
                return
        _finish_model_task(task_id, f"{install_name} converted and ready to use")
    except Exception as exc:
        _fail_model_task(task_id, exc)


@router.get("/api/sdxl/convert/available")
def sdxl_conversion_available():
    """Lets the UI disable the Convert action with a reason, instead of failing on click."""
    available = sdxl_convert.has_diffusers()
    return {
        "available": available,
        "reason": None if available else sdxl_convert.missing_dependency_message(),
    }


@router.post("/api/sdxl/convert")
def start_sdxl_conversion(req: ConversionRequest):
    install_name = hf_browse.safe_install_name(req.install_name)
    if install_name != req.install_name.strip():
        raise HTTPException(400, "invalid install name")
    if req.local_path:
        root = Path(req.local_path).expanduser().resolve()
        if root.is_file() and root.suffix.lower() == ".safetensors":
            root = root.parent
    else:
        root = generator.ASSET_DIR / "models" / install_name
    if not root.is_dir():
        raise HTTPException(404, f"{install_name} is not in the model store")
    if (root / "model_index.json").is_file():
        if req.register_on_finish:
            import custom_models
            declaration = {
                "id": req.custom_id or custom_models._slugify_folder_name(root.name),
                "label": req.label or root.name,
                "family": "sdxl",
                "local_path": str(root),
                "source": "local",
            }
            custom_models.verify_and_register(declaration)
        return {"status": "already_converted", "install_name": install_name}
    if not sdxl_convert.has_diffusers():
        raise HTTPException(503, sdxl_convert.missing_dependency_message())
    task_id = uuid.uuid4().hex[:12]
    cancel_event = threading.Event()
    task = {
        "id": task_id, "source": "convert", "model_id": f"convert-{install_name}",
        "repo_id": f"local/{install_name}", "model_name": install_name, "engine": "sdxl",
        "status": "downloading", "progress": 0.0, "downloaded_bytes": 0, "total_bytes": 0,
        "speed_mb_s": 0.0, "status_text": f"Preparing to convert {install_name}...",
        "started_at": time.time(), "finished_at": None, "error": None, "result": None,
        "install_dir": str(root), "worker_active": True, "cancel_event": cancel_event,
    }
    with _MODEL_DOWNLOAD_LOCK:
        MODEL_DOWNLOAD_TASKS[task_id] = task
    threading.Thread(
        target=_run_sdxl_conversion,
        args=(task_id, install_name, cancel_event),
        kwargs={
            "custom_id": req.custom_id,
            "label": req.label,
            "local_path": str(root),
            "register_on_finish": req.register_on_finish,
        },
        daemon=True,
    ).start()
    return {"task_id": task_id, "status": "downloading", "model_name": install_name, "source": "convert"}


@router.get("/api/civitai/models")
def search_civitai_models(
    query: str | None = None,
    types: str | None = None,
    sort: str = "Most Downloaded",
    limit: int = 40,
):
    source = sources.SOURCES["civitai"]()
    try:
        return source.search_raw(query=query, types=types, sort=sort, limit=limit)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))


@router.post("/api/civitai/models/download")
def download_civitai_model(req: CivitaiModelDownloadRequest):
    # Task 5: every call on this route is unlisted (no registry entry claims
    # a Civitai version), so the family declaration is always required; the
    # worker tail registers the download through verify_and_register.
    family = (req.family or "").strip()
    if not family:
        raise HTTPException(400, "family is required for Civitai versions outside the model registry (one of sdxl|flux2|flux3|krea|zit|qwen)")
    refusal = _unlisted_enum_refusal(family, req.variant)
    if refusal is not None:
        raise HTTPException(400, refusal)
    name = hf_browse.safe_install_name(req.name or f"civitai-{req.model_id}")
    # Same dots hole as the HF path (safe_install_name permits "..", which
    # resolves to ASSET_DIR itself): contain against models/, not the tree.
    install_dir = _resolve_inside_models(name)
    repo_id = f"civitai/{req.model_id}/{req.model_version_id}"
    declaration = {
        "id": None,
        "label": (req.name or "").strip() or None,
        "family": family,
        "variant": req.variant,
        "source": "civitai",
        "civitai_ids": [req.model_version_id],
        "civitai_model_id": req.model_id,
        "civitai_version_id": req.model_version_id,
    }
    with _MODEL_DOWNLOAD_LOCK:
        for task_id, task in MODEL_DOWNLOAD_TASKS.items():
            if task.get("repo_id") == repo_id and task.get("worker_active"):
                return {"task_id": task_id, "status": task.get("status", "downloading"),
                        "model_name": task.get("model_name"), "already_running": True}
    task_id = uuid.uuid4().hex[:12]
    cancel_event = threading.Event()
    task = {
        "id": task_id, "source": "civitai", "model_id": f"civitai-{req.model_version_id}",
        "repo_id": repo_id, "model_name": name,
        "engine": "mflux", "status": "downloading", "progress": 0.0, "downloaded_bytes": 0,
        "total_bytes": 0, "speed_mb_s": 0.0,
        "status_text": f"Preparing Civitai download of {name}...",
        "started_at": time.time(), "finished_at": None, "error": None, "result": None,
        "install_dir": str(install_dir), "worker_active": True, "cancel_event": cancel_event,
        "declaration": declaration,
    }
    with _MODEL_DOWNLOAD_LOCK:
        MODEL_DOWNLOAD_TASKS[task_id] = task
    threading.Thread(target=_run_civitai_model_download,
                     args=(task_id, req.model_version_id, name, req.token, cancel_event), daemon=True).start()
    return {"task_id": task_id, "status": "downloading", "model_name": name, "source": "civitai"}


@router.get("/api/civitai/models/version/{version_id}/prefill")
def civitai_version_prefill(version_id: int):
    """Suggest (never decide) the onboard family for a Civitai version.

    Returns {suggested_family, model_name, version_name, model_id}. The UI pre-selects
    the suggestion and the user can override it; declare+verify still
    decides. An unsupported architecture suggests None; a missing version
    is 404.
    """
    info = civitai_service.fetch_by_version_id(version_id)
    if not info:
        raise HTTPException(404, f"Civitai model version {version_id} not found")
    try:
        return civitai_service.prefill_for_version(info)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/api/hf/models")
def search_hf_models(
    search: str | None = None,
    author: str | None = None,
    architecture: str | None = None,
    quantization: str | None = None,
    kind: str | None = None,
    limit: int = 40,
    sort: str = "downloads",
    all_kinds: bool = False,
):
    """Search one Hugging Face org for downloadable models.

    Architecture and quantisation filters are applied by `hf_browse`, not here, because
    both only exist as substrings of the repo id -- there is no structured field for
    either anywhere in the HF API or in this app's model registry.

    The route owns nothing source-specific: it resolves "huggingface" from the source
    registry and forwards. Which module answers a Hugging Face query is not this file's
    business, and pinning it here is what made the previous split worthwhile.
    """
    source = sources.SOURCES["huggingface"]()
    try:
        return source.search_raw(
            search=search,
            author=author,
            architecture=architecture,
            quantization=quantization,
            kind=kind,
            limit=limit,
            sort=sort,
            models_only=not all_kinds,
        )
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))


class ModelRegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: str = Field(min_length=1, max_length=200)
    model_id: str = Field(min_length=1, max_length=120)


def _known_model_dirs() -> set[str]:
    """Directory names in the model store that belong to a registry model.

    Comparing directory names against generator.MODELS *keys* is wrong: the keys are model
    ids ("krea2-turbo", "juggernaut-xl-lightning" is an id too) while the directories are
    bundles ("krea2-turbo-q4") or model_dir basenames. Getting this wrong made the
    detection route report juggernaut-xl-lightning as an unknown leftover with "no engine
    can run this architecture", for a model the app can obviously run.
    """
    names: set[str] = set()
    for model_id, minfo in generator.MODELS.items():
        model_dir = minfo.get("model_dir")
        if model_dir:
            names.add(Path(str(model_dir)).name)
        repo = str(minfo.get("repo") or "")
        if repo.startswith("local:"):
            names.add(repo.split(":", 1)[1].strip("/"))
        try:
            if generator.model_download_repo(model_id, minfo) is None:
                label = str(minfo.get("label") or "")
                if label:
                    names.add(label.replace(" ", "").lower())
        except Exception:
            continue
    return names


def _resolve_inside_models(name: str) -> Path:
    """Resolve ASSET_DIR/models/<name> and prove it really is inside models/.

    Containment must be checked against models/, NOT against ASSET_DIR. safe_install_name
    permits dots, so ".." passes untouched; ASSET_DIR/models/.. resolves to ASSET_DIR,
    which is trivially "inside" ASSET_DIR. On the delete route that let a single
    `DELETE /api/hf/models/..` rmtree the entire asset directory -- every model, the
    LoRA store and hf_token.txt -- failing only on the final rmdir, i.e. a 500 returned
    after the damage. Requiring a non-empty relative path also rejects ".".

    Shared by the delete and register routes so the two cannot drift apart again.
    """
    models_root = (generator.ASSET_DIR / "models").resolve()
    target = generator.ASSET_DIR / "models" / name
    try:
        inside = target.resolve().relative_to(models_root)
    except ValueError:
        raise HTTPException(400, "path must stay inside the asset directory")
    if not inside.parts:
        raise HTTPException(400, "path must stay inside the asset directory")
    return target


def _dir_size(directory: Path) -> int:
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _model_dir_for_repo(repo_id: str) -> Path | None:
    """Locate an already-downloaded repo's directory, or None if it is not there."""
    target = generator.ASSET_DIR / "models" / hf_browse.safe_install_name(repo_id)
    if not target.is_dir():
        return None
    try:
        target.resolve().relative_to(generator.ASSET_DIR.resolve())
    except ValueError:
        return None
    return target


# The SDXL layout check lives in backend/sdxl_layout.py, not here. It is needed in
# three places that cannot import each other cleanly: this router (download worker
# and adopt route) and generator.py, on the generation path -- and routers/downloads.py
# already imports generator, so importing it back would be a cycle. Re-exported so
# existing callers and tests keep working.
from sdxl_layout import (  # noqa: E402,F401
    SDXL_REQUIRED,
    SDXL_WEIGHT_DIR,
    is_sdxl_engine,
    sdxl_layout_report,
)


def _has_weights(directory: Path) -> bool:
    return any(
        p.suffix in (".safetensors", ".npz", ".gguf") or p.name.endswith(".index.json")
        for p in directory.rglob("*")
        if p.is_file()
    )


@router.post("/api/hf/models/register")
def register_downloaded_model(req: ModelRegistrationRequest):
    """Point an existing engine at a downloaded repo, so the download becomes selectable.

    Deliberately reuses the app's existing `model_paths` override instead of extending
    generator.MODELS. Every generator build already does `model_path=local_arg` where
    local_arg comes from that override, so binding a repo is a settings write -- no
    generation code changes, and unbinding restores the registry default for free.
    """
    import app_settings

    model_id = req.model_id.strip()
    if model_id not in generator.MODELS:
        raise HTTPException(404, f"Unknown engine: {model_id}")

    # Two kinds of source reach here: a Hugging Face repo id, and a LOCAL directory name --
    # which is what a converted Civitai checkpoint is. Demanding an "org/repo" id made the
    # converted model impossible to bind at all.
    raw = (req.repo_id or "").strip()
    repo_id = hf_service._valid_repo_id(raw)
    if repo_id is not None:
        expected = hf_browse.usable_as(repo_id, "diffusion")[0]
        if expected != model_id:
            raise HTTPException(400, f"{repo_id} is a {expected or 'non-runnable'} model, not {model_id}")
        directory = _model_dir_for_repo(repo_id)
    else:
        name = hf_browse.safe_install_name(raw)
        if name != raw:
            raise HTTPException(400, "invalid local model name")
        # Same containment as the delete route: ".." would otherwise resolve to
        # ASSET_DIR itself and bind an override to the whole asset directory.
        candidate = _resolve_inside_models(name)
        directory = candidate if candidate.is_dir() else None

    if directory is None:
        raise HTTPException(404, f"{raw} is not in the model store")
    if not _has_weights(directory):
        raise HTTPException(400, f"{directory.name} contains no weights")

    # _has_weights searches recursively, so it is satisfied by a single-file
    # checkpoint -- which is how almost every SDXL checkpoint is published on
    # Hugging Face. An SDXL engine needs the diffusers folder layout, and finding
    # that out at generation time surfaced as a forty-line traceback in the UI
    # ending in "No .safetensors files in .../unet". Refuse here, and name the
    # missing part instead.
    if is_sdxl_engine(model_id):
        ok, reason = sdxl_layout_report(directory)
        if not ok:
            raise HTTPException(400, reason)
    elif generator.is_qwen_flux_tokenizer_model(model_id):
        generator._ensure_flux_tokenizer_available(directory)

    updated = app_settings.update_settings({"model_paths": {model_id: str(directory)}})
    paths = updated.get("model_paths") or {}
    return {
        "status": "registered",
        "model_id": model_id,
        "repo_id": repo_id or str(directory),
        "local_path": str(directory),
        "model_paths": paths,
    }


# Architecture key -> (frontend key, human label). Module level so the vocabulary can be
# tested against generator.detect_architecture; they drifted apart once already, which
# silently mislabelled every FLUX checkpoint as SDXL-derived.
ARCHITECTURE_LABELS = {
    "sdxl": ("sdxl", "SDXL"),
    "mflux": ("flux-2", "FLUX.2"),
    "flux2": ("flux-2", "FLUX.2"),
    "flux3": ("flux-3", "FLUX.3"),
    "krea2": ("krea", "Krea"),
    "z-image": ("z-image", "Z-Image"),
}


@router.delete("/api/hf/models/register/{model_id}")
def unregister_downloaded_model(model_id: str):
    """Drop the override so the engine falls back to its registry default."""
    import app_settings

    if generator.get_model_info(model_id) is None:
        raise HTTPException(404, f"Unknown engine: {model_id}")
    settings_now = app_settings.get_settings()
    paths = dict(settings_now.get("model_paths") or {})
    if model_id not in paths:
        return {"status": "not_registered", "model_id": model_id, "cleared": True}
    # An empty string is how app_settings deletes a model_paths entry, same as clearing a
    # model_defaults override. Passing the whole map minus the key does NOT delete it, which
    # is why unregistering used to report success while leaving the override in place and
    # silently hijacking that engine.
    updated = app_settings.update_settings({"model_paths": {model_id: ""}})
    remaining = updated.get("model_paths") or {}
    return {
        "status": "unregistered",
        "model_id": model_id,
        "model_paths": remaining,
        "cleared": model_id not in remaining,
    }


class DeleteDetectedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirm: bool


@router.delete("/api/hf/models/{install_name}")
def delete_detected_model(install_name: str, req: DeleteDetectedRequest):
    """Delete a model downloaded via HF Browser or Civitai from ASSET_DIR/models/.

    Removes the directory and clears any model_paths override pointing to it.

    `confirm: true` is required. shutil.rmtree on a model store is unrecoverable --
    there is no trash, no undo, and no second copy -- and until this gate existed the
    only confirmation was a window.confirm in the UI, which any other caller (a
    script, curl, a future panel) skipped entirely. The 409 body names the directory
    and its size so a caller can see the blast radius before committing.
    """
    import app_settings
    import shutil

    safe_name = hf_browse.safe_install_name(install_name)
    if safe_name != install_name:
        raise HTTPException(400, "invalid model name")

    # Containment rationale lives in _resolve_inside_models. It matters here because
    # this route then rmtree's whatever it resolves to.
    target_dir = _resolve_inside_models(safe_name)

    if not target_dir.is_dir():
        raise HTTPException(404, f"{safe_name} not found in model store")

    # Measured before the confirmation gate so the 409 can report it, and reused
    # afterwards so the success response reports what was really removed.
    size_mb = _dir_size(target_dir) / (1024 * 1024)
    if req.confirm is not True:
        raise HTTPException(
            409,
            f"deleting {safe_name} would permanently remove {target_dir.resolve()} "
            f"({size_mb:.1f} MB) and cannot be undone. "
            f"Re-send with confirm: true to delete it.",
        )

    # Clear any model_paths override pointing to this directory
    settings_now = app_settings.get_settings()
    paths = dict(settings_now.get("model_paths") or {})
    cleared = False
    for model_id, path in list(paths.items()):
        if path and Path(path).resolve() == target_dir.resolve():
            app_settings.update_settings({"model_paths": {model_id: ""}})
            cleared = True

    # Delete the directory
    try:
        shutil.rmtree(target_dir, ignore_errors=False)
    except OSError as e:
        raise HTTPException(500, f"failed to delete model directory: {e}")

    return {
        "status": "deleted",
        "install_name": safe_name,
        "cleared_overrides": cleared,
        "deleted_mb": round(size_mb, 1),
    }


class ConvertDetectedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    install_name: str = Field(min_length=1, max_length=120)


@router.post("/api/hf/models/convert")
def convert_detected_model(req: ConvertDetectedRequest):
    """Convert a detected model's .bin files to .safetensors (SDXL only)."""
    import sdxl_convert
    import shutil

    safe_name = hf_browse.safe_install_name(req.install_name)
    if safe_name != req.install_name:
        raise HTTPException(400, "invalid model name")

    target_dir = generator.ASSET_DIR / "models" / safe_name
    try:
        target_dir.resolve().relative_to(generator.ASSET_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "path must stay inside the asset directory")

    if not target_dir.is_dir():
        raise HTTPException(404, f"{safe_name} not found in model store")

    # Check if it needs conversion: must be a diffusers directory with .bin files
    has_bin = any(target_dir.rglob("*.bin"))
    if not has_bin:
        raise HTTPException(400, f"{safe_name} has no .bin files to convert")

    # Verify it's a diffusers layout (model_index.json + unet/)
    if not (target_dir / "model_index.json").is_file():
        raise HTTPException(400, f"{safe_name} is not a diffusers model directory")
    if not (target_dir / "unet").is_dir():
        raise HTTPException(400, f"{safe_name} has no unet/ directory")

    # Run conversion
    try:
        result = sdxl_convert.convert_bin_to_safetensors(target_dir)
    except Exception as e:
        raise HTTPException(500, f"Conversion failed: {e}")

    return {
        "status": "converted",
        "install_name": safe_name,
        "result": result,
    }


def _detected_model_rows(include_registered: bool = False) -> list[dict]:
    """Scan the model store and return one row per directory that holds weights.

    The body below is the original `list_detected_downloads` scan, moved here verbatim
    so `/api/models/inventory` can merge the same rows into one list. Two copies of
    this scan would drift, and the drift would surface as a model visible in one tab and
    missing from the other -- which is the bug the merged inventory exists to remove.
    """
    import app_settings

    root = generator.ASSET_DIR / "models"
    known = _known_model_dirs()
    paths = app_settings.get_settings().get("model_paths") or {}
    bound = {str(Path(v).name) for v in paths.values() if v}

    out = []
    if root.is_dir():
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            if directory.name in bound or not _has_weights(directory):
                continue
            if directory.name in known and not include_registered:
                continue
            # Inspect actual checkpoint to determine architecture (fixes Civitai models)
            arch_key, arch_label = hf_browse.guess_architecture(directory.name)
            quant = hf_browse.guess_quantization(directory.name)
            kind = hf_browse.guess_kind(directory.name)
            
            # Try to determine true architecture:
            # 1. Check model_index.json for diffusers models (most reliable)
            # 2. Fall back to inspecting safetensors for single-file checkpoints
            # One authority, shared with the tests: read what the checkpoint declares.
            # The old inline copy lowercased _class_name and substring-matched "krea" or
            # "z-image", so it invented architectures for pipelines we do not have and
            # could not tell "unknown" from "flux".
            inspected_arch = generator.detect_architecture(directory)
            
            # Fall back to safetensors inspection for single-file checkpoints
            if not inspected_arch:
                for ckpt in directory.rglob("*.safetensors"):
                    base, _ = _inspect_safetensors(ckpt)
                    if base:
                        inspected_arch = base
                        break
            
            if inspected_arch and inspected_arch in ARCHITECTURE_LABELS:
                arch_key, arch_label = ARCHITECTURE_LABELS[inspected_arch]
                kind = "diffusion"  # Checkpoints are diffusion models
            
            # Pass inspected architecture as hint for executable rules matching
            arch_hint = arch_label if inspected_arch else None
            usable, reason = hf_browse.usable_as(directory.name, kind, arch_hint=arch_hint)
            out.append(
                {
                    "name": directory.name,
                    "path": str(directory),
                    "architecture": arch_key,
                    "architecture_label": arch_label,
                    "quantization": quant,
                    "kind": kind,
                    "bytes": _dir_size(directory),
                    "usable_as": usable,
                    "usable_as_label": reason,
                    "registered": directory.name in known,
                }
            )
    return out


@router.get("/api/hf/models/detected")
def list_detected_downloads(include_registered: bool = False):
    """Directories in the model store that the registry does not know about.

    Without this, a repo downloaded from the browser is invisible: it is in neither
    MODELS nor /api/models, so the Models tab would show nothing and the user would have
    no idea the weights are on disk.

    Registered models are excluded by default. They are in the store, but the registry
    already owns them, so listing them here as "detected" is noise -- and it was actively
    misleading, reporting juggernaut-xl-lightning as an unknown leftover when it is the
    SDXL engine the app has always run. Pass include_registered=true to see everything.

    The scan itself is `_detected_model_rows`, shared with `/api/models/inventory`.
    """
    return {"items": _detected_model_rows(include_registered),
            "asset_dir": str(generator.ASSET_DIR / "models")}


@router.get("/api/models/downloads")
def get_model_downloads():
    # Also the heartbeat for the stall watchdog: the download worker can be blocked
    # inside hf_hub_download indefinitely, so this poller is the only place that
    # still gets to run and notice.
    #
    # A failure here used to be swallowed, which defeats the entire mechanism: this
    # poll is the only thing that can expire a wedged download, so an exception meant
    # the task hung forever with no error and no timeout. The endpoint still returns the
    # task list -- degrading beats failing here -- but the failure is recorded on the
    # task and logged, rather than vanishing.
    try:
        _enforce_download_stall_watchdog()
    except Exception as exc:
        print(f"[downloads] stall watchdog failed: {type(exc).__name__}: {exc}", flush=True)
        with _MODEL_DOWNLOAD_LOCK:
            for task_id, task in MODEL_DOWNLOAD_TASKS.items():
                if task.get("worker_active"):
                    task["watchdog_error"] = f"{type(exc).__name__}: {exc}"
                    break
    hidden_keys = {"cancel_event", "worker_active", "_speed_t", "_speed_b", "_last_progress_at", "watchdog_error", "declaration"}
    with _MODEL_DOWNLOAD_LOCK:
        now = time.time()
        for task_id in [
            task_id
            for task_id, task in MODEL_DOWNLOAD_TASKS.items()
            if task.get("status") in ("done", "error", "cancelled") and not task.get("worker_active") and task.get("finished_at") and now - task["finished_at"] > 600
        ]:
            MODEL_DOWNLOAD_TASKS.pop(task_id, None)
        result = [{key: value for key, value in task.items() if key not in hidden_keys} for task in MODEL_DOWNLOAD_TASKS.values()]
    result.sort(key=lambda item: item.get("started_at", 0), reverse=True)
    return result


@router.delete("/api/models/downloads/{task_id}")
def cancel_model_download(task_id: str):
    with _MODEL_DOWNLOAD_LOCK:
        task = MODEL_DOWNLOAD_TASKS.get(task_id)
        if not task:
            raise HTTPException(404, "Download task not found")
        if task.get("status") == "downloading":
            cancel_event = task.get("cancel_event")
            if cancel_event:
                cancel_event.set()
            task["status"] = "cancelled"
            task["status_text"] = "Download cancelled by user"
            task["finished_at"] = time.time()
            task.pop("cancel_event", None)
        return {"status": "ok", "task_id": task_id}
