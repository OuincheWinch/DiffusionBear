import hashlib
import os
import re
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
from .loras import _model_is_fully_cached
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


def _cleanup_registered_entry(entry: dict | None):
    if not entry:
        return
    name = entry.get("name")
    path = entry.get("path")
    _remove_lora_entries(names={name} if isinstance(name, str) else None, paths={path} if isinstance(path, str) else None)
    remaining_paths = {item.get("path") for item in _read_loras()}
    if not isinstance(path, str) or path in remaining_paths:
        return
    try:
        resolved = Path(path).expanduser().resolve()
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
            _cleanup_registered_entry(entry)
    except civitai_service.CivitaiAuthError as e:
        _fail_task(task_id, e, auth=True, auth_msg="Authentication required")
    except Exception as e:
        if not registered:
            _cleanup_registered_entry(entry)
        _fail_task(task_id, e)


def _run_hf_download(task_id: str, req: HFImportRequest, meta: dict, cancel_event: threading.Event):
    target_dir = SDXL_LORA_DIR if meta["base_model"] == "sdxl" else LORA_FILES_DIR
    entry = None
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
            _cleanup_registered_entry(entry)
    except hf_service.HFAuthError as e:
        if not registered:
            _cleanup_registered_entry(entry)
        _fail_task(task_id, e, auth=True, auth_msg="Hugging Face authentication required")
    except Exception as e:
        if not registered:
            _cleanup_registered_entry(entry)
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
        os.replace(tmp_path, final_path)
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
_DOWNLOAD_STALL_LIMIT_S = float(os.environ.get("DIFFUSIONBEAR_DOWNLOAD_STALL_SECONDS", "180"))


def _stalled_model_task_ids() -> list[str]:
    """Download tasks whose byte counter has not moved within the limit."""
    now = time.monotonic()
    out = []
    with _MODEL_DOWNLOAD_LOCK:
        for task_id, task in MODEL_DOWNLOAD_TASKS.items():
            if task.get("status") != "downloading" or not task.get("worker_active"):
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
            if now - last > _DOWNLOAD_STALL_LIMIT_S and (total == 0 or done < total):
                out.append(task_id)
    return out


def _enforce_download_stall_watchdog() -> None:
    for task_id in _stalled_model_task_ids():
        with _MODEL_DOWNLOAD_LOCK:
            task = MODEL_DOWNLOAD_TASKS.get(task_id, {})
            label = task.get("model_name") or task.get("model_id") or "model"
            got = (task.get("downloaded_bytes") or 0) >> 20
            total = (task.get("total_bytes") or 0) >> 20
        _fail_model_task(
            task_id,
            f"stalled: no data received for {_DOWNLOAD_STALL_LIMIT_S:.0f}s while "
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


def _run_model_download(
    task_id: str,
    model_id: str,
    minfo: dict,
    cancel_event: threading.Event,
    repo_override: str | None = None,
    install_dir: Path | None = None,
):
    target_dir = None
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
            # HF-browser downloads: a plain directory under ASSET_DIR, not the HF cache and
            # not a registry model_dir. Reject anything that would escape ASSET_DIR.
            target_dir = install_dir
            try:
                target_dir.resolve().relative_to(generator.ASSET_DIR.resolve())
            except ValueError:
                raise ValueError("install path must stay inside the asset directory")
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
        with _MODEL_DOWNLOAD_LOCK:
            task = MODEL_DOWNLOAD_TASKS.get(task_id)
            if task and task.get("status") == "downloading":
                task["total_bytes"] = total
                task["revision"] = revision
                task["status_text"] = f"Downloading {minfo.get('label', model_id)} weights..."
        from huggingface_hub import hf_hub_download

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
                local_dir=str(target_dir) if target_dir else None,
                tqdm_class=_ModelDownloadTqdm(task_id, cancel_event),
            )
        if not _finish_model_task(task_id, f"{minfo.get('label', model_id)} installed successfully!"):
            for path in _snapshot_files(cleanup_root) - before_files:
                try:
                    path.unlink()
                except OSError:
                    pass
            return
    except Exception as e:
        if repo:
            for path in _snapshot_files(cleanup_root) - before_files:
                try:
                    path.unlink()
                except OSError:
                    pass
            for path in _incomplete_paths(target_dir, repo) - before_incomplete:
                try:
                    path.unlink()
                except OSError:
                    pass
        _fail_model_task(task_id, e)


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
        if minfo is None:
            minfo = {"id": model_id, "label": name, "engine": "mflux"}
        install_dir = generator.ASSET_DIR / "models" / name
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
                if task.get("model_id") == model_id and task.get("worker_active"):
                    return {"task_id": task_id, "status": task.get("status", "downloading"), "model_name": task.get("model_name"), "already_running": True}
        if _model_is_fully_cached(model_id, minfo):
            return {"status": "already_installed", "model_id": model_id, "model_name": minfo.get("label", model_id)}
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
            "finished_at": None,
            "error": None,
            "result": None,
            "worker_active": True,
            "cancel_event": cancel_event,
        }
        with _MODEL_DOWNLOAD_LOCK:
            for existing_id, existing in MODEL_DOWNLOAD_TASKS.items():
                if existing.get("model_id") == model_id and existing.get("worker_active"):
                    return {"task_id": existing_id, "status": existing.get("status", "downloading"), "model_name": existing.get("model_name"), "already_running": True}
            MODEL_DOWNLOAD_TASKS[task_id] = task
        threading.Thread(target=_run_model_download, args=(task_id, model_id, minfo, cancel_event), daemon=True).start()
        return {"task_id": task_id, "status": "downloading", "model_name": task["model_name"], "source": "model"}



class CivitaiModelDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: int = Field(ge=1)
    model_version_id: int = Field(ge=1)
    name: str | None = Field(default=None, max_length=120)
    token: str | None = Field(default=None, max_length=8192)


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
                disposition.split("filename=", 1)[-1].strip('" ') if "filename=" in disposition else None,
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
            for chunk in response.iter_content(chunk_size=1 << 20):
                if cancel_event.is_set():
                    raise civitai_service.DownloadCancelled("Download cancelled by user")
                if not chunk:
                    continue
                with destination.open("ab") as fh:
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
            if not civitai_service.is_valid_safetensors(destination):
                destination.unlink(missing_ok=True)
                raise ValueError("downloaded file is not a valid safetensors checkpoint")
        _finish_model_task(task_id, f"{name} installed successfully!")
    except civitai_service.DownloadCancelled:
        _fail_model_task(task_id, "Download cancelled by user")
    except Exception as exc:
        _fail_model_task(task_id, exc)



class ConversionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    install_name: str = Field(min_length=1, max_length=120)


def _run_sdxl_conversion(task_id: str, install_name: str, cancel_event: threading.Event):
    """Convert a downloaded single-file checkpoint into a diffusers directory.

    Runs the real work in a subprocess (see sdxl_convert) so a hung or crashing
    conversion cannot take the backend down. On success the directory lands where the SDXL
    engine can find it, and only then is the task marked done -- a half-written diffusers
    folder would read as "installed" and then fail at generation time.
    """
    root = generator.ASSET_DIR / "models" / install_name
    source = None
    if root.is_dir():
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
        sdxl_convert.convert(source, root, cancel_event=cancel_event, on_progress=on_progress)
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
    root = generator.ASSET_DIR / "models" / install_name
    if not root.is_dir():
        raise HTTPException(404, f"{install_name} is not in the model store")
    if (root / "model_index.json").is_file():
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
    threading.Thread(target=_run_sdxl_conversion, args=(task_id, install_name, cancel_event), daemon=True).start()
    return {"task_id": task_id, "status": "downloading", "model_name": install_name, "source": "convert"}


@router.get("/api/civitai/models")
def search_civitai_models(
    query: str | None = None,
    types: str | None = None,
    sort: str = "Most Downloaded",
    limit: int = 40,
):
    try:
        return civitai_browse.search_models(query=query, types=types, sort=sort, limit=limit)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))


@router.post("/api/civitai/models/download")
def download_civitai_model(req: CivitaiModelDownloadRequest):
    name = hf_browse.safe_install_name(req.name or f"civitai-{req.model_id}")
    install_dir = generator.ASSET_DIR / "models" / name
    with _MODEL_DOWNLOAD_LOCK:
        for task_id, task in MODEL_DOWNLOAD_TASKS.items():
            if task.get("model_id") == f"civitai-{req.model_version_id}" and task.get("worker_active"):
                return {"task_id": task_id, "status": task.get("status", "downloading"),
                        "model_name": task.get("model_name"), "already_running": True}
    task_id = uuid.uuid4().hex[:12]
    cancel_event = threading.Event()
    task = {
        "id": task_id, "source": "civitai", "model_id": f"civitai-{req.model_version_id}",
        "repo_id": f"civitai/{req.model_id}/{req.model_version_id}", "model_name": name,
        "engine": "mflux", "status": "downloading", "progress": 0.0, "downloaded_bytes": 0,
        "total_bytes": 0, "speed_mb_s": 0.0,
        "status_text": f"Preparing Civitai download of {name}...",
        "started_at": time.time(), "finished_at": None, "error": None, "result": None,
        "install_dir": str(install_dir), "worker_active": True, "cancel_event": cancel_event,
    }
    with _MODEL_DOWNLOAD_LOCK:
        MODEL_DOWNLOAD_TASKS[task_id] = task
    threading.Thread(target=_run_civitai_model_download,
                     args=(task_id, req.model_version_id, name, req.token, cancel_event), daemon=True).start()
    return {"task_id": task_id, "status": "downloading", "model_name": name, "source": "civitai"}


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

    Architecture and quantisation filters are applied here rather than in the HF query,
    because both only exist as substrings of the repo id -- there is no structured field
    for either anywhere in the HF API or in this app's model registry.
    """
    try:
        return hf_browse.search_models(
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
        candidate = generator.ASSET_DIR / "models" / name
        try:
            candidate.resolve().relative_to(generator.ASSET_DIR.resolve())
        except ValueError:
            raise HTTPException(400, "path must stay inside the asset directory")
        directory = candidate if candidate.is_dir() else None

    if directory is None:
        raise HTTPException(404, f"{raw} is not in the model store")
    if not _has_weights(directory):
        raise HTTPException(400, f"{directory.name} contains no weights")

    updated = app_settings.update_settings({"model_paths": {model_id: str(directory)}})
    paths = updated.get("model_paths") or {}
    return {
        "status": "registered",
        "model_id": model_id,
        "repo_id": repo_id or str(directory),
        "local_path": str(directory),
        "model_paths": paths,
    }


@router.delete("/api/hf/models/register/{model_id}")
def unregister_downloaded_model(model_id: str):
    """Drop the override so the engine falls back to its registry default."""
    import app_settings

    if model_id not in generator.MODELS:
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
            arch_key, arch_label = hf_browse.guess_architecture(directory.name)
            quant = hf_browse.guess_quantization(directory.name)
            kind = hf_browse.guess_kind(directory.name)
            usable, reason = hf_browse.usable_as(directory.name, kind)
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
    return {"items": out, "asset_dir": str(root)}


@router.get("/api/models/downloads")
def get_model_downloads():
    # Also the heartbeat for the stall watchdog: the download worker can be blocked
    # inside hf_hub_download indefinitely, so this poller is the only place that
    # still gets to run and notice.
    try:
        _enforce_download_stall_watchdog()
    except Exception:
        pass
    hidden_keys = {"cancel_event", "worker_active", "_speed_t", "_speed_b", "_last_progress_at"}
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
