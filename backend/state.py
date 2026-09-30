import json
import os
import queue
import re
import struct
import tempfile
import time
import threading
from pathlib import Path
from typing import Literal
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

import generator
import civitai_service

# Directories & Files
def _ensure_private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


DATA_DIR = _ensure_private_dir(generator.DATA_DIR)
ASSET_DIR = _ensure_private_dir(generator.ASSET_DIR)
GENERATED_DIR = _ensure_private_dir(generator.GENERATED_DIR)
LORA_FILES_DIR = _ensure_private_dir(ASSET_DIR / "lora_files")
LORAS_FILE = DATA_DIR / "loras.json"
SDXL_LORA_DIR = _ensure_private_dir(ASSET_DIR / "SDXL")
UPLOADS_DIR = _ensure_private_dir(DATA_DIR / "uploads")

# Validation & Limits
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
SAFE_REF_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
SAFE_LORA_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*(?::[A-Za-z0-9][A-Za-z0-9._/-]*)?$")
MAX_LORA_UPLOAD_BYTES = 8 * (1 << 30)
MAX_REFERENCE_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_RUNTIME_JSON_BYTES = 64 * 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
SUPPORTED_REFERENCE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

_KNOWN_TRIGGERS = {
    "evi@decrepPunk": ["evi@decrepPunk"],
}


def _validate_image_id(image_id: str) -> str:
    if not isinstance(image_id, str) or not SAFE_ID_RE.fullmatch(image_id) or len(image_id) > 128:
        raise HTTPException(400, "invalid id")
    return image_id


def _atomic_write_text(path: Path, text: str):
    _ensure_private_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        fchmod = getattr(os, "fchmod", None)
        if fchmod is not None:
            try:
                fchmod(fd, 0o600)
            except OSError:
                pass
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def _sanitize_component(value: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", str(value or "").strip())
    return cleaned.strip("._")[:80]


def _sanitize_filename(value: str, fallback: str = "download.safetensors") -> str:
    raw = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = _sanitize_component(raw)
    if not cleaned or cleaned in (".", ".."):
        cleaned = _sanitize_component(fallback) or "download.safetensors"
    if not cleaned.lower().endswith(".safetensors"):
        cleaned = f"{cleaned[:150]}.safetensors"
    return cleaned[:180]


def _validate_lora_path(path: str) -> str | None:
    if not isinstance(path, str) or not path.strip() or len(path) > 4096 or "\x00" in path:
        return "invalid LoRA path"
    value = path.strip()
    try:
        p = Path(value).expanduser()
    except (OSError, RuntimeError):
        return "invalid LoRA path"
    if not p.is_absolute():
        if not SAFE_LORA_ID_RE.fullmatch(value) or any(part in (".", "..") for part in value.replace(":", "/").split("/")):
            return "LoRA must be an existing .safetensors file or a valid Hugging Face repo id"
        return None
    try:
        resolved = p.resolve()
        if not resolved.exists():
            return f"LoRA file not found: {value}"
        if not resolved.is_file():
            return "LoRA path is not a file"
        if not resolved.name.lower().endswith(".safetensors"):
            return "only .safetensors files are supported"
        if resolved.stat().st_size > MAX_LORA_UPLOAD_BYTES:
            return "LoRA file exceeds the 8 GB limit"
        if not civitai_service.is_valid_safetensors(resolved):
            return "LoRA file is not a valid .safetensors archive"
        return None
    except OSError:
        return "LoRA file is not readable"


def _validate_reference_images(reference_images: list[str]) -> list[str]:
    if len(reference_images) > 10:
        raise HTTPException(400, "a maximum of 10 reference images is allowed")
    roots = (GENERATED_DIR.resolve(), UPLOADS_DIR.resolve())
    normalized: list[str] = []
    seen: set[str] = set()
    for value in reference_images:
        if not isinstance(value, str) or not value.strip() or len(value) > 4096 or "\x00" in value:
            raise HTTPException(400, "invalid reference image path")
        raw = value.strip()
        try:
            candidate = Path(raw).expanduser()
            if candidate.is_absolute():
                resolved = candidate.resolve()
                if not any(resolved.is_relative_to(root) for root in roots):
                    raise HTTPException(400, "reference image is outside the gallery and uploads directories")
            else:
                if not SAFE_REF_NAME_RE.fullmatch(raw):
                    raise HTTPException(400, "invalid reference image name")
                names = [raw]
                if not Path(raw).suffix:
                    names.extend(f"{raw}{ext}" for ext in SUPPORTED_REFERENCE_EXTENSIONS)
                resolved = None
                for root in roots:
                    for name in names:
                        option = (root / name).resolve()
                        if option.is_file() and option.is_relative_to(root):
                            resolved = option
                            break
                    if resolved:
                        break
                if resolved is None:
                    raise HTTPException(400, f"reference image not found: {raw}")
        except HTTPException:
            raise
        except (OSError, RuntimeError) as e:
            raise HTTPException(400, "invalid reference image path") from e
        if not resolved.is_file() or resolved.suffix.lower() not in SUPPORTED_REFERENCE_EXTENSIONS:
            raise HTTPException(400, f"unsupported reference image: {raw}")
        value = str(resolved)
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


# --- Gallery Index ---
_gallery_lock = threading.Lock()
_IMAGE_MUTATION_LOCK = threading.RLock()
_IMAGE_TOMBSTONES: set[str] = set()
GALLERY_INDEX: dict[str, dict] = {}
_GALLERY_INDEX_BUILT_AT: float = 0.0


def _image_is_deleted(image_id: str) -> bool:
    return image_id in _IMAGE_TOMBSTONES


def _mark_image_deleted(image_id: str):
    _IMAGE_TOMBSTONES.add(image_id)


def _unmark_image_deleted(image_id: str):
    _IMAGE_TOMBSTONES.discard(image_id)


def _init_gallery_index() -> int:
    """Rebuild the gallery index from sidecars. Returns how many entries landed.

    The index is a cache, and it is only built at startup. If every read fails --
    which is exactly what happens when the data directory is on an external volume
    and macOS has not yet granted access, so every open() raises -- then the loop
    below skips every file and the app presents an EMPTY gallery indefinitely, with
    no error anywhere. That is not a cosmetic failure: the images are on disk and
    the API answers 200 with an empty list.

    So the result is now returned, and an empty index is treated as suspect rather
    than authoritative. See ensure_gallery_index().
    """
    valid_entries = []
    for jf in GENERATED_DIR.glob("*.json"):
        try:
            if not jf.resolve().is_relative_to(GENERATED_DIR.resolve()):
                continue
            if jf.stat().st_size > MAX_METADATA_BYTES:
                continue
            data = json.loads(jf.read_text(encoding="utf-8"))
            image_id = data.get("id") if isinstance(data, dict) else None
            if not isinstance(image_id, str) or not SAFE_ID_RE.fullmatch(image_id) or len(image_id) > 128:
                continue
            if jf.stem != image_id:
                continue
            valid_entries.append((image_id, data))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
    global _GALLERY_INDEX_BUILT_AT
    with _gallery_lock:
        GALLERY_INDEX.clear()
        GALLERY_INDEX.update(valid_entries)
        _GALLERY_INDEX_BUILT_AT = time.time()
    return len(GALLERY_INDEX)


def ensure_gallery_index(max_age: float = 0.0) -> int:
    """Return the index size, rebuilding if it is empty or older than max_age.

    An empty index next to a non-empty directory is the signature of a failed scan,
    not an empty gallery, so it is retried rather than believed. Retries are rate
    limited so a genuinely empty store cannot turn every request into a full scan.
    """
    now = time.time()
    with _gallery_lock:
        size = len(GALLERY_INDEX)
        age = now - _GALLERY_INDEX_BUILT_AT if _GALLERY_INDEX_BUILT_AT else None
    stale = size == 0 and any(GENERATED_DIR.glob("*.json"))
    if stale and (age is None or age > 5.0):
        return _init_gallery_index()
    if max_age and (age is None or age > max_age):
        return _init_gallery_index()
    return size



# --- Hard Limits ---
MAX_LORAS = 16
MAX_QUEUED_IMAGES = 512


# --- Pydantic Request Models ---
class LoRA(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    path: str = Field(min_length=1, max_length=4096)
    scale: float = Field(default=1.0, ge=0.0, le=10.0)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    prompt: str = Field(min_length=1, max_length=100_000)
    model: str = Field(default=generator.DEFAULT_MODEL_ID, min_length=1, max_length=200)
    # None means "use the model's own default_width/default_height". The old
    # hardcoded 1024x1024 default was a trap for qwen-image-2.1, whose engine
    # rejects anything above 589824 px, so a minimal API request could only ever
    # fail. Resolved at the request boundary so everything downstream sees ints.
    width: int | None = Field(default=None, ge=128, le=2048)
    height: int | None = Field(default=None, ge=128, le=2048)
    # None means "use the model's own default_steps", resolved in
    # generator.generate(). A hardcoded 4 here was applied to every model: an API
    # caller that omitted steps got 4 steps on juggernaut-xi (default 25),
    # realvis-xl-v5 (25), qwen-image-2.1 (25), z-image-turbo (8) and krea2 (8).
    # The frontend always sends steps explicitly, so only direct API callers were
    # affected - but they were getting badly under-sampled images.
    steps: int | None = Field(default=None, ge=1, le=50)
    guidance: float | None = Field(default=None, ge=0.0, le=10.0)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    quantization: Literal[4, 8] = 4
    loras: list[LoRA] = Field(default_factory=list, max_length=MAX_LORAS)
    batch: int = Field(default=1, ge=1, le=16)
    negative_prompt: str = Field(default="", max_length=100_000)
    sampler: str | None = Field(default=None, min_length=1, max_length=80)
    cache_interval: int = Field(default=1, ge=1, le=10)
    reference_images: list[str] = Field(default_factory=list, max_length=10)
    reference_strength: float | None = Field(default=None, gt=0.0, le=1.0)
    output_format: Literal["png", "jpeg", "jpg"] = "png"
    stealth: bool = False
    fast_vae: bool = True
    max_pixels: int | None = Field(default=None, ge=256 * 256, le=2048 * 2048)


class TokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(default="", max_length=4096)


# --- Jobs & Generation Queue ---
_JOBS_LOCK = threading.RLock()
_JOBS_TTL_SECONDS = 3600
_MAX_FINISHED_JOBS = 50

JOBS: dict[str, dict] = {}
_job_queue: queue.Queue = queue.Queue()


def _prune_jobs():
    """Caller must hold _JOBS_LOCK."""
    now = time.time()
    dead = [
        jid
        for jid, j in JOBS.items()
        if j["status"] in ("done", "error", "cancelled")
        and now - j.get("finished_at", now) > _JOBS_TTL_SECONDS
    ]
    for jid in dead:
        del JOBS[jid]
    finished = [jid for jid, j in JOBS.items() if j["status"] in ("done", "error", "cancelled")]
    for jid in finished[:-_MAX_FINISHED_JOBS]:
        JOBS.pop(jid, None)


# --- Queue Recovery & Crash Resilience ---
QUEUE_RECOVERY_FILE = DATA_DIR / "queue_recovery.json"
PENDING_QUEUE_FILE = DATA_DIR / "pending_queue.json"
_RECOVERY_LOCK = threading.RLock()
_MAX_PENDING_RECORDS = 512
_MAX_RECOVERABLE_RECORDS = 100
_RECOVERY_TOMBSTONES: set[str] = set()


def _load_json_list(path: Path) -> list[dict]:
    if path.exists():
        try:
            resolved = path.resolve()
            size = path.stat().st_size
        except OSError:
            return []
        if not resolved.is_relative_to(DATA_DIR.resolve()) or size > MAX_RUNTIME_JSON_BYTES:
            raise ValueError(f"{path.name} exceeds the runtime state size limit")
    try:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        print(f"[queue_recovery] warning loading {path.name}: {e}")
    return []


def _save_json_list(path: Path, data: list[dict]):
    payload = json.dumps(data, indent=2, allow_nan=False)
    if len(payload.encode("utf-8")) > MAX_RUNTIME_JSON_BYTES:
        raise ValueError(f"{path.name} exceeds the runtime state size limit")
    _atomic_write_text(path, payload)


def _validated_request(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    try:
        return GenerateRequest.model_validate(value).model_dump()
    except Exception:
        return None


def _validated_pending(items: list[dict]) -> list[dict]:
    valid = []
    seen = set()
    for item in items:
        job_id = item.get("id")
        request = _validated_request(item.get("request"))
        timestamp = item.get("enqueued_at")
        if not isinstance(job_id, str) or not SAFE_ID_RE.fullmatch(job_id) or job_id in seen or request is None:
            continue
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or not 0 < timestamp < time.time() + 86400:
            timestamp = time.time()
        seen.add(job_id)
        valid.append({"id": job_id, "request": request, "enqueued_at": float(timestamp)})
    return valid[-_MAX_PENDING_RECORDS:]


def _validated_recoverable(items: list[dict]) -> list[dict]:
    valid = []
    seen = set()
    for item in items:
        job_id = item.get("id")
        request = _validated_request(item.get("request"))
        timestamp = item.get("timestamp")
        if not isinstance(job_id, str) or not SAFE_ID_RE.fullmatch(job_id) or job_id in seen or request is None:
            continue
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or not 0 < timestamp < time.time() + 86400:
            timestamp = time.time()
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason or len(reason) > 64:
            reason = "cancelled"
        seen.add(job_id)
        valid.append({"id": job_id, "reason": reason, "timestamp": float(timestamp), "request": request})
    return valid[:_MAX_RECOVERABLE_RECORDS]


def _quarantine_recovery_file(path: Path) -> bool:
    if not path.exists():
        return True
    quarantine = path.with_name(f"{path.name}.invalid.{time.time_ns()}")
    try:
        os.replace(path, quarantine)
        return True
    except OSError as e:
        print(f"[queue_recovery] warning quarantining {path.name}: {e}")
        return False


def record_pending_jobs(records: list[tuple[str, dict]]):
    if not records:
        return
    with _RECOVERY_LOCK:
        raw_items = _load_json_list(PENDING_QUEUE_FILE)
        items = _validated_pending(raw_items)
        if len(items) != len(raw_items) and not _quarantine_recovery_file(PENDING_QUEUE_FILE):
            raise OSError("could not quarantine invalid pending queue data")
        for job_id, request_data in records:
            if not isinstance(job_id, str) or not SAFE_ID_RE.fullmatch(job_id) or len(job_id) > 128:
                raise ValueError("invalid pending job id")
            if job_id in _RECOVERY_TOMBSTONES:
                continue
            pending = {
                "id": job_id,
                "request": _validated_request(request_data),
                "enqueued_at": time.time(),
            }
            if pending["request"] is None:
                raise ValueError("invalid pending job request")
            items = [item for item in items if item["id"] != job_id]
            items.append(pending)
        _save_json_list(PENDING_QUEUE_FILE, _validated_pending(items))


def record_pending_job(job_id: str, request_data: dict):
    record_pending_jobs([(job_id, request_data)])


def remove_pending_jobs(job_ids: set[str]):
    if not job_ids:
        return
    with _RECOVERY_LOCK:
        raw_items = _load_json_list(PENDING_QUEUE_FILE)
        items = _validated_pending(raw_items)
        if len(items) != len(raw_items) and not _quarantine_recovery_file(PENDING_QUEUE_FILE):
            raise OSError("could not quarantine invalid pending queue data")
        new_items = [item for item in items if item["id"] not in job_ids]
        if len(new_items) != len(items):
            _save_json_list(PENDING_QUEUE_FILE, new_items)


def remove_pending_job(job_id: str):
    remove_pending_jobs({job_id})


def record_recoverable_jobs(records: list[tuple[str, dict, str]]):
    if not records:
        return
    with _RECOVERY_LOCK:
        raw_archive = _load_json_list(QUEUE_RECOVERY_FILE)
        archive = _validated_recoverable(raw_archive)
        if len(archive) != len(raw_archive) and not _quarantine_recovery_file(QUEUE_RECOVERY_FILE):
            raise OSError("could not quarantine invalid recovery data")
        now = time.time()
        for job_id, request_data, reason in records:
            if job_id in _RECOVERY_TOMBSTONES:
                continue
            request = _validated_request(request_data)
            if not isinstance(job_id, str) or not SAFE_ID_RE.fullmatch(job_id) or request is None:
                raise ValueError("invalid recoverable job")
            archive = [item for item in archive if item["id"] != job_id]
            archive.insert(0, {"id": job_id, "reason": str(reason)[:64], "timestamp": now, "request": request})
        _save_json_list(QUEUE_RECOVERY_FILE, _validated_recoverable(archive))
        remove_pending_jobs({job_id for job_id, _, _ in records})


def record_recoverable_job(job_id: str, request_data: dict, reason: str = "cancelled"):
    record_recoverable_jobs([(job_id, request_data, reason)])


def get_recoverable_jobs() -> list[dict]:
    with _RECOVERY_LOCK:
        return [
            item for item in _validated_recoverable(_load_json_list(QUEUE_RECOVERY_FILE))
            if item.get("id") not in _RECOVERY_TOMBSTONES
        ]


def delete_recoverable_jobs(job_ids: set[str] | None):
    with _RECOVERY_LOCK:
        raw_records = _load_json_list(QUEUE_RECOVERY_FILE)
        records = _validated_recoverable(raw_records)
        if len(records) != len(raw_records) and not _quarantine_recovery_file(QUEUE_RECOVERY_FILE):
            raise OSError("could not quarantine invalid recovery data")
        if job_ids is None:
            _RECOVERY_TOMBSTONES.update(str(record["id"]) for record in records)
            new_records = []
        else:
            _RECOVERY_TOMBSTONES.update(str(job_id) for job_id in job_ids)
            new_records = [record for record in records if record["id"] not in job_ids]
        if new_records != records or job_ids is None:
            _save_json_list(QUEUE_RECOVERY_FILE, new_records)


def delete_recoverable_job(job_id: str | None = None):
    delete_recoverable_jobs(None if job_id is None else {job_id})


def clear_recoverable_jobs():
    delete_recoverable_job(None)


def clear_recoverable_state():
    with _RECOVERY_LOCK:
        for item in _load_json_list(QUEUE_RECOVERY_FILE) + _load_json_list(PENDING_QUEUE_FILE):
            job_id = item.get("id")
            if isinstance(job_id, str) and SAFE_ID_RE.fullmatch(job_id):
                _RECOVERY_TOMBSTONES.add(job_id)
        _save_json_list(QUEUE_RECOVERY_FILE, [])
        _save_json_list(PENDING_QUEUE_FILE, [])


def _init_queue_recovery_on_startup():
    with _RECOVERY_LOCK:
        raw_pending = _load_json_list(PENDING_QUEUE_FILE)
        pending = _validated_pending(raw_pending)
        if len(pending) != len(raw_pending) and not _quarantine_recovery_file(PENDING_QUEUE_FILE):
            return
        if not pending:
            if raw_pending:
                _save_json_list(PENDING_QUEUE_FILE, [])
            return
        raw_records = _load_json_list(QUEUE_RECOVERY_FILE)
        records = _validated_recoverable(raw_records)
        if len(records) != len(raw_records) and not _quarantine_recovery_file(QUEUE_RECOVERY_FILE):
            return
        for item in pending:
            records = [record for record in records if record["id"] != item["id"]]
            records.insert(0, {
                "id": item["id"],
                "reason": "interrupted",
                "timestamp": item["enqueued_at"],
                "interrupted_at": time.time(),
                "request": item["request"],
            })
        _save_json_list(QUEUE_RECOVERY_FILE, _validated_recoverable(records))
        _save_json_list(PENDING_QUEUE_FILE, [])
        print(f"[queue_recovery] Restored {len(pending)} interrupted jobs to recovery archive", flush=True)


_init_queue_recovery_on_startup()


def _emit_generation_event(message: str):
    event_log = (
        os.environ.get("DIFFUSIONBEAR_EVENT_LOG", "").strip()
        or os.environ.get("MLX_DIFFUSION_EVENT_LOG", "").strip()
    )
    if not event_log:
        return
    try:
        with open(event_log, "a", encoding="utf-8") as stream:
            stream.write(f"{message}\n")
    except OSError:
        pass


def _generation_text(value: object, limit: int = 180) -> str:
    return " ".join(str(value or "").split())[:limit] or "-"


def _effective_request_steps(req) -> int:
    """Step count a job will actually run.

    GenerateRequest.steps is None when the caller omitted it, leaving the choice
    to the model registry. Resolved once per job so the progress readout and the
    engine agree; reading it inside the per-step callback crashed the job with
    "unsupported operand type(s) for -: 'NoneType' and 'int'".
    """
    if req.steps is not None:
        return int(req.steps)
    minfo = generator.get_model_info(req.model) or {}
    return int(minfo.get("default_steps") or 4)


def _worker():
    while True:
        try:
            job_id = _job_queue.get()
        except BaseException:
            break
        try:
            cancel_event = threading.Event()
            with _JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None or job.get("status") == "cancelled":
                    try:
                        remove_pending_job(job_id)
                        if job is not None:
                            req = job.get("request")
                            request_data = req.model_dump() if hasattr(req, "model_dump") else dict(req or {})
                            record_recoverable_job(job_id, request_data, reason="cancelled")
                    except Exception as e:
                        print(f"[queue_recovery] error finalizing cancelled job {job_id}: {e}", flush=True)
                    continue
                req = job["request"]
                ref_imgs = list(req.reference_images)
                variant = "edit" if (req.model in ("flux2-klein-4b", "flux2-klein-9b") and ref_imgs) else "standard"
                is_loaded = generator.is_pipeline_loaded(
                    model_id=req.model,
                    quantization=req.quantization,
                    loras=[lora.model_dump() for lora in req.loras],
                    variant=variant,
                )
                job["status"] = "generating"
                job["phase"] = "preparing" if is_loaded else "loading_model"
                job["phase_detail"] = "Preparing prompt conditioning & latents..." if is_loaded else "Loading model weights into Apple Silicon unified memory..."
                job["cancel_event"] = cancel_event

            generation_started = time.monotonic()
            generation_started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            lora_summary = ",".join(Path(lora.path).name for lora in req.loras if lora.path) or "none"
            _emit_generation_event(
                f"[generation] START date={generation_started_at} job={job_id} model={req.model} "
                f"size={req.width}x{req.height} steps={req.steps} seed={req.seed if req.seed is not None else 'auto'} "
                f"batch={req.batch} guidance={req.guidance if req.guidance is not None else 'auto'} "
                f"sampler={req.sampler or 'default'} quantization={req.quantization} "
                f"loras={_generation_text(lora_summary, 240)} prompt={_generation_text(req.prompt)}"
            )
            results = []
            status = "generating"
            try:
                req = job["request"]
                batch = max(1, req.batch)
                base_seed = req.seed if req.seed is not None else int(time.time())
                # req.steps is None when the caller omitted it, so the model owns
                # the default. Resolved once for the progress readout rather
                # than read inside the per-step callback.
                req_steps = _effective_request_steps(req)
                for i in range(batch):
                    img_start_time = time.time()
                    if i > 0:
                        with _JOBS_LOCK:
                            job["phase"] = "preparing"
                            job["phase_detail"] = f"Preparing image {i + 1}/{batch}..."

                    def on_step(t, _total=req_steps):
                        done = t + 1
                        elapsed = time.time() - img_start_time
                        eta = elapsed / done * (_total - done) if done else None
                        with _JOBS_LOCK:
                            job["phase"] = "generating"
                            job["phase_detail"] = f"Denoising step {done}/{_total}..."
                            job["progress"] = {
                                "step": done,
                                "steps": _total,
                                "elapsed": round(elapsed, 1),
                                "eta_seconds": round(eta, 1) if eta is not None else None,
                                "image_index": i,
                                "batch": batch,
                            }

                    def on_phase(phase, detail=""):
                        with _JOBS_LOCK:
                            job["phase"] = phase
                            if detail:
                                job["phase_detail"] = detail

                    with _JOBS_LOCK:
                        if job["status"] == "cancelled":
                            raise generator.GenerationCancelled()

                    ref_imgs = list(req.reference_images)

                    meta = generator.generate(
                        prompt=req.prompt,
                        model=req.model,
                        width=req.width,
                        height=req.height,
                        steps=req.steps,
                        guidance=req.guidance,
                        seed=base_seed + 1024 * i,
                        quantization=req.quantization,
                        loras=[lora.model_dump() for lora in req.loras],
                        progress_cb=on_step,
                        phase_cb=on_phase,
                        cancel_event=cancel_event,
                        negative_prompt=req.negative_prompt,
                        sampler=req.sampler,
                        cache_interval=req.cache_interval,
                        reference_images=ref_imgs,
                        image_strength=req.reference_strength,
                        output_format=req.output_format,
                        stealth=req.stealth,
                        fast_vae=req.fast_vae,
                        max_pixels=req.max_pixels,
                    )
                    results.append(meta)
                    with _gallery_lock:
                        GALLERY_INDEX[meta["id"]] = meta
                    with _JOBS_LOCK:
                        job["phase"] = "saving"
                        job["phase_detail"] = "Finalizing and saving image..."
                        job["last_saved"] = {"id": meta["id"], "index": i + 1, "batch": batch}
                        job["last_result"] = meta
                        job["partial_results"] = list(results)

                if len(results) == batch and results:
                    status = "done"
                else:
                    status = "error"
            except generator.GenerationCancelled:
                status = "cancelled"
            except Exception as e:
                import traceback
                traceback.print_exc()
                status = "error"
                with _JOBS_LOCK:
                    job["error"] = str(e)
                generator._drop_mflux_pipeline()
            except BaseException as be:
                import traceback
                traceback.print_exc()
                status = "error"
                with _JOBS_LOCK:
                    job["error"] = f"Interrupted: {type(be).__name__}"
                generator._drop_mflux_pipeline()
            finally:
                try:
                    remove_pending_job(job_id)
                except Exception as e:
                    print(f"[queue_recovery] error clearing pending job {job_id}: {e}", flush=True)
                recovery_data = None
                with _JOBS_LOCK:
                    job.pop("cancel_event", None)
                    if job.get("status") == "cancelled" or status == "cancelled":
                        job["status"] = "cancelled"
                        if results:
                            job["partial_results"] = results
                        req_data = job["request"].model_dump() if hasattr(job.get("request"), "model_dump") else dict(job.get("request") or {})
                        recovery_data = req_data
                    elif status == "done" and results:
                        job["status"] = "done"
                        job["phase"] = "done"
                        job["phase_detail"] = "Complete"
                        job["result"] = results[-1]
                        job["results"] = results
                        total_gen_time = round(sum(r.get("generation_time", 0) for r in results), 1)
                        if "progress" in job and isinstance(job["progress"], dict):
                            job["progress"]["elapsed"] = total_gen_time
                            job["progress"]["eta_seconds"] = 0
                    else:
                        job["status"] = "error"
                        if "error" not in job:
                            job["error"] = "Generation stopped or failed before producing an image"
                        if results:
                            job["partial_results"] = results
                    job["finished_at"] = time.time()
                    _prune_jobs()
                    final_status = str(job.get("status") or status)
                    error = str(job.get("error") or "").replace("\n", " ").strip()
                    error_suffix = f" error={error[:160]}" if error else ""
                    output_links = []
                    generation_seconds = 0.0
                    for result in results:
                        if not isinstance(result, dict):
                            continue
                        try:
                            generation_seconds += float(result.get("generation_time") or 0)
                        except (TypeError, ValueError):
                            pass
                        filename = result.get("file")
                        if filename:
                            try:
                                output_links.append((Path(generator.GENERATED_DIR) / str(filename)).resolve().as_uri())
                            except (OSError, ValueError):
                                pass
                    output_summary = ",".join(output_links) or "none"
                    _emit_generation_event(
                        f"[generation] END date={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
                        f"job={job_id} model={req.model} status={final_status} "
                        f"elapsed={time.monotonic() - generation_started:.1f}s generation_time={generation_seconds:.1f}s "
                        f"images={len(results)} output={output_summary}{error_suffix}"
                    )
                if recovery_data is not None:
                    try:
                        record_recoverable_job(job_id, recovery_data, reason="cancelled")
                    except Exception as e:
                        print(f"[queue_recovery] error archiving job {job_id}: {e}", flush=True)
        except Exception as outer_be:
            print(f"[worker] critical job loop error: {outer_be}", flush=True)
            with _JOBS_LOCK:
                if job_id in JOBS and JOBS[job_id].get("status") == "generating":
                    JOBS[job_id]["status"] = "error"
                    JOBS[job_id]["error"] = f"Fatal worker error: {outer_be}"
                    JOBS[job_id]["finished_at"] = time.time()
        finally:
            _job_queue.task_done()


# Start background worker daemon
threading.Thread(target=_worker, daemon=True).start()


# --- LoRA Registry State & Helpers ---
_loras_lock = threading.RLock()


def _valid_registry_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    name = entry.get("name")
    path = entry.get("path")
    valid = (
        isinstance(name, str)
        and 0 < len(name) <= 200
        and isinstance(path, str)
        and 0 < len(path) <= 4096
        and "\x00" not in path
    )
    if not valid:
        return False
    sha256 = entry.get("sha256")
    if sha256 is not None and (not isinstance(sha256, str) or not sha256 or len(sha256) > 128 or any(ord(char) < 32 for char in sha256)):
        return False
    return True


def _loras_signature():
    try:
        st = LORAS_FILE.stat()
    except OSError:
        return None
    try:
        resolved = str(LORAS_FILE.resolve())
    except OSError:
        resolved = ""
    return (st.st_mtime_ns, st.st_size, resolved)


# (signature, validated registry) for the last successful _read_loras(). Every
# caller holds _loras_lock, so no extra locking is needed here.
_loras_cache: dict = {}


def _read_loras() -> list[dict]:
    with _loras_lock:
        signature = _loras_signature()
        cached = _loras_cache.get("loras")
        if cached is not None and _loras_cache.get("signature") == signature:
            return [dict(entry) for entry in cached]
        try:
            if signature is None:
                return []
            if not LORAS_FILE.resolve().is_relative_to(DATA_DIR.resolve()) or signature[1] > MAX_RUNTIME_JSON_BYTES:
                raise ValueError("LoRA registry exceeds the runtime state size limit")
            data = json.loads(LORAS_FILE.read_text(encoding="utf-8"))
        except ValueError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError):
            return []
        if not isinstance(data, list):
            return []
        valid = [dict(entry) for entry in data[:512] if _valid_registry_entry(entry)]
        _loras_cache["signature"] = signature
        _loras_cache["loras"] = valid
        return [dict(entry) for entry in valid]


def _write_loras(loras: list[dict]):
    with _loras_lock:
        valid = [dict(entry) for entry in loras[:512] if _valid_registry_entry(entry)]
        _ensure_private_dir(LORAS_FILE.parent)
        _atomic_write_text(LORAS_FILE, json.dumps(valid, indent=2, allow_nan=False))
        _loras_cache.clear()


def _upsert_lora_entries(entries: list[dict]):
    with _loras_lock:
        loras = _read_loras()
        for entry in entries:
            if not _valid_registry_entry(entry):
                raise ValueError("invalid LoRA registry entry")
            loras = [
                item
                for item in loras
                if item.get("name") != entry.get("name") and item.get("path") != entry.get("path")
            ]
            loras.append(dict(entry))
        _write_loras(loras)


def _remove_lora_entries(names: set[str] | None = None, paths: set[str] | None = None):
    names = names or set()
    paths = paths or set()
    if not names and not paths:
        return []
    with _loras_lock:
        loras = _read_loras()
        removed = [entry for entry in loras if entry.get("name") in names or entry.get("path") in paths]
        remaining = [entry for entry in loras if entry not in removed]
        if removed:
            _write_loras(remaining)
        return removed


def _inspect_safetensors(path: Path) -> tuple[str | None, list[str]]:
    """Fast inspection of safetensors header to determine base_model and triggers."""
    if not civitai_service.is_valid_safetensors(path):
        return None, []
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                return None, []
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            if header_len <= 0 or header_len > min(size - 8, 100 * 1024 * 1024):
                return None, []
            header = json.loads(f.read(header_len).decode("utf-8"))
            if not isinstance(header, dict):
                return None, []
        meta = header.get("__metadata__", {})
        keys = [k for k in header.keys() if k != "__metadata__"]

        version = str(meta.get("ss_base_model_version", "")).lower()
        arch = str(meta.get("modelspec.architecture", "")).lower()
        title = str(meta.get("modelspec.title", "")).lower()
        name_lower = path.name.lower()

        if "sdxl" in version or "sdxl" in arch or "sdxl" in name_lower or any(k.startswith(("lora_unet_", "lora_te1_", "lora_te2_")) for k in keys[:20]):
            base_model = "sdxl"
        elif "krea" in name_lower or "krea" in title or "krea" in version:
            base_model = "krea2"
        elif (
            "z-image" in name_lower
            or "z_image" in name_lower
            or "zit" in name_lower
            or "zimage" in name_lower
            or "zimage" in version
            or "z-image" in version
            or "z_image" in version
            or "zimage" in arch
            or "z-image" in arch
        ):
            base_model = "z-image"
        elif any("double_blocks" in k for k in keys[:20]) or "flux" in version or "flux" in arch or "flux" in name_lower:
            base_model = "flux2"
        else:
            base_model = "sdxl" if "sdxl" in name_lower else "flux2"

        triggers = []
        if "ss_tag_frequency" in meta:
            try:
                tf = json.loads(meta["ss_tag_frequency"])
                for cat, tags in tf.items():
                    for tag, count in tags.items():
                        if count >= 10 and len(tag) > 2 and tag.lower() not in (
                            "simple background", "white background", "no humans", "1girl", "solo", "looking at viewer"
                        ):
                            if tag not in triggers:
                                triggers.append(tag)
            except Exception:
                pass
        if "modelspec.trigger_phrase" in meta:
            phrase = meta["modelspec.trigger_phrase"]
            if phrase and phrase not in triggers:
                triggers.append(phrase)

        return base_model, triggers[:3]
    except Exception:
        return None, []


def sync_lora_entry_with_civitai(entry: dict, force: bool = False) -> bool:
    """Enrich single LoRA entry with SHA256 and Civitai metadata. Returns True if changed."""
    changed = False
    p = Path(entry.get("path", ""))
    if not p.is_file():
        return False

    if not entry.get("sha256") or force:
        try:
            entry["sha256"] = civitai_service.compute_file_sha256(p)
            changed = True
        except Exception:
            return changed

    if (not entry.get("civitai_version_id") or force) and entry.get("sha256"):
        try:
            civitai_data = civitai_service.fetch_by_hash(entry["sha256"])
            if civitai_data:
                cm = civitai_service.extract_civitai_metadata(civitai_data)
                if cm.get("civitai_version_id"):
                    entry["civitai_version_id"] = cm["civitai_version_id"]
                    changed = True
                if cm.get("civitai_model_id"):
                    entry["civitai_model_id"] = cm["civitai_model_id"]
                    changed = True
                if cm.get("civitai_model_name"):
                    entry["civitai_model_name"] = cm["civitai_model_name"]
                    changed = True
                if cm.get("civitai_version_name"):
                    entry["civitai_version_name"] = cm["civitai_version_name"]
                    changed = True
                if cm.get("base_model"):
                    entry["base_model"] = cm["base_model"]
                    changed = True
                current_triggers = list(entry.get("triggers", []))
                for t in cm.get("triggers", []):
                    if t and t not in current_triggers:
                        current_triggers.append(t)
                        changed = True
                entry["triggers"] = current_triggers
        except Exception as e:
            print(f"[civitai] Failed querying Civitai for {entry.get('name')}: {e}", flush=True)

    return changed


def _discover_local_loras():
    """Register safetensors dropped into lora_files/ or SDXL/. Runs at startup and after uploads.

    Deliberately does NOT hold _loras_lock for the duration. The slow parts here
    are reading every safetensors header, streaming a full-file SHA-256 for any
    registry entry that lacks one, and a Civitai HTTP lookup (8s timeout) per
    entry needing metadata. That used to all happen inside the lock, so a single
    entry missing its sha256 stalled /api/loras and /api/gallery/loras - both of
    which need that same lock - for the length of a 4-12s hash plus the HTTP
    round trip, on a background thread, at every startup.

    _read_loras() and _upsert_lora_entries() each take the lock only for the
    duration of their own read or write, so the registry merge stays atomic.
    """
    loras = _read_loras()
    changed = False

    for lora in loras:
        path = Path(lora.get("path", ""))
        if path.is_absolute() and (not path.is_file() or not civitai_service.is_valid_safetensors(path)):
            print(f"[loras] Retaining unavailable LoRA entry: {lora.get('name')} ({path})", flush=True)

    for lora in loras:
        lora.setdefault("triggers", [])
        lora.setdefault("base_model", "sdxl" if "SDXL" in lora.get("path", "") or "sdxl" in lora.get("name", "").lower() else "flux2")
        if not lora["triggers"]:
            lora["triggers"] = next(
                (trigger for key, trigger in _KNOWN_TRIGGERS.items() if key in lora["name"]), [])
            if lora["triggers"]:
                changed = True
        if not lora.get("civitai_version_id") or not lora.get("sha256"):
            if sync_lora_entry_with_civitai(lora):
                changed = True

    known_paths = {lora["path"] for lora in loras}
    known_names = {lora["name"] for lora in loras}
    for folder in (LORA_FILES_DIR, SDXL_LORA_DIR):
        for f in sorted(folder.glob("*.safetensors")):
            if not f.resolve().is_relative_to(folder.resolve()):
                continue
            if "f42SDXL" in f.name or "Juggernaut" in f.name:
                continue
            try:
                if f.stat().st_size > 2 * (1 << 30):
                    continue
            except OSError:
                pass
            resolved = str(f.resolve())
            if resolved in known_paths:
                continue
            name = f.stem.split("__", 1)[-1]
            if name in known_names:
                continue
            base, triggers = _inspect_safetensors(f)
            if not base:
                continue
            entry = {
                "name": name,
                "path": resolved,
                "scale": 1.0,
                "triggers": triggers or next((t for k, t in _KNOWN_TRIGGERS.items() if k in name), []),
                "base_model": base,
            }
            old = next((lora for lora in loras if lora["name"] == entry["name"]), None)
            if old and old.get("base_model"):
                entry["base_model"] = old["base_model"]
            sync_lora_entry_with_civitai(entry)
            loras = [lora for lora in loras if lora["name"] != entry["name"]]
            loras.append(entry)
            known_paths.add(resolved)
            changed = True
    if changed:
        _upsert_lora_entries(loras)



# --- Async Downloads State ---
DOWNLOAD_TASKS: dict[str, dict] = {}
_DOWNLOAD_LOCK = threading.RLock()

# --- Async Model (weights) Download State ---
MODEL_DOWNLOAD_TASKS: dict[str, dict] = {}
_MODEL_DOWNLOAD_LOCK = threading.RLock()
