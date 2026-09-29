import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import generator
from state import (
    GenerateRequest,
    SAFE_ID_RE,
    _validate_lora_path,
    _validate_reference_images,
    _JOBS_LOCK,
    JOBS,
    _job_queue,
    _prune_jobs,
    MAX_LORAS,
    MAX_QUEUED_IMAGES,
    record_pending_job,
    record_pending_jobs,
    record_recoverable_jobs,
    remove_pending_jobs,
    get_recoverable_jobs,
    delete_recoverable_jobs,
    clear_recoverable_state,
)

router = APIRouter(tags=["jobs"])


def _prepare_request(value: GenerateRequest | dict) -> GenerateRequest:
    req = value if isinstance(value, GenerateRequest) else GenerateRequest.model_validate(value)
    if not req.prompt.strip():
        raise HTTPException(400, "prompt is required")
    info = next(
        (candidate for candidate in generator.MODELS.values() if req.model in (candidate.get("id"), candidate.get("repo"))),
        None,
    )
    if info is None:
        raise HTTPException(400, f"unknown model: {req.model}")
    req.model = info["id"]
    if req.width is None:
        req.width = int(info.get("default_width") or 512)
    if req.height is None:
        req.height = int(info.get("default_height") or 768)
    if not info.get("supports_fast_vae"):
        req.fast_vae = False
    if len(req.loras) > MAX_LORAS:
        raise HTTPException(400, f"Maximum of {MAX_LORAS} LoRAs allowed per generation")
    try:
        req.quantization = generator._effective_quantization(info["id"], req.quantization)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    lora_paths = set()
    for lora in req.loras:
        err = _validate_lora_path(lora.path)
        if err:
            raise HTTPException(400, err)
        resolved_key = str(Path(lora.path).expanduser().resolve()) if Path(lora.path).expanduser().is_absolute() else lora.path
        if resolved_key in lora_paths:
            raise HTTPException(400, "duplicate LoRA path")
        lora_paths.add(resolved_key)
    if req.loras and not info.get("supports_loras"):
        raise HTTPException(400, f"LoRAs are not supported on {info['label']}")
    if not info.get("supports_guidance") and req.guidance is not None and req.guidance > 1.0 and info.get("engine") != "sdxl":
        raise HTTPException(400, f"{info['label']} does not support guidance > 1.0")
    if req.negative_prompt and not info.get("supports_negative"):
        raise HTTPException(400, f"{info['label']} does not support negative prompts")
    if req.sampler:
        try:
            req.sampler = generator._validate_sampler(req.sampler, info)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    if req.reference_images:
        if not (info.get("supports_multi_reference") or info.get("supports_ref") or info.get("id") in {"z-image-turbo", "krea2-turbo", "qwen-image-2.1"}):
            raise HTTPException(400, f"reference images are not supported on {info['label']}")
        max_references = int(info.get("max_reference_images") or (10 if info.get("supports_multi_reference") else 1))
        if len(req.reference_images) > max_references:
            raise HTTPException(400, f"a maximum of {max_references} reference images is allowed")
        if not info.get("supports_multi_reference") and len(req.reference_images) > 1:
            raise HTTPException(400, "this model accepts one reference image")
        req.reference_images = _validate_reference_images(req.reference_images)
    elif req.reference_strength is not None:
        raise HTTPException(400, "reference_strength requires a reference image")
    if info.get("id") == "qwen-image-2.1":
        if req.width * req.height > 589824:
            raise HTTPException(400, "qwen-image-2.1 requests are limited to 589824 pixels")
        # Refused here rather than in the worker, so a 40-step request is a clean
        # 400 at submit time instead of a queued job that fails a second later.
        steps = req.steps if req.steps is not None else int(info.get("default_steps") or 25)
        if steps > generator._QWEN_MAX_STEPS:
            raise HTTPException(
                400,
                f"qwen-image-2.1 is limited to {generator._QWEN_MAX_STEPS} steps on this machine "
                f"(measured median 790.8s at 25 steps vs 910s at 40). Requested {steps}. "
                f"Change the per-model default in Parameters > Defaults if it is set there.",
            )
    return req


@router.post("/api/generate")
def generate(req: GenerateRequest):
    req = _prepare_request(req)
    job_id = uuid.uuid4().hex
    request_data = req.model_dump()
    with generator._model_maintenance_lock:
        with _JOBS_LOCK:
            active_images = sum(
                getattr(job.get("request"), "batch", 1)
                for job in JOBS.values()
                if job.get("status") in ("generating", "queued")
            )
            if active_images + req.batch > MAX_QUEUED_IMAGES:
                raise HTTPException(429, f"Queue limit reached ({active_images}/{MAX_QUEUED_IMAGES} images in queue)")
            JOBS[job_id] = {
                "id": job_id,
                "status": "queued",
                "request": req,
                "created_at": time.time(),
            }
            try:
                record_pending_job(job_id, request_data)
            except Exception as e:
                JOBS.pop(job_id, None)
                raise HTTPException(500, "failed to persist queued job") from e
            _prune_jobs()
    _job_queue.put(job_id)
    return {"job_id": job_id}


@router.get("/api/jobs")
def list_jobs(limit: int = 25):
    limit = min(max(1, limit), 50)
    with _JOBS_LOCK:
        active = [job for job in JOBS.values() if job.get("status") in ("generating", "queued")]
        active.sort(key=lambda job: (0 if job.get("status") == "generating" else 1, job.get("created_at", 0)))
        finished = [job for job in JOBS.values() if job.get("status") not in ("generating", "queued")]
        finished.sort(key=lambda job: job.get("finished_at", 0), reverse=True)
        output = []
        for job in (active + finished)[:limit]:
            progress = dict(job["progress"]) if isinstance(job.get("progress"), dict) else {}
            if job.get("last_saved"):
                saved = job["last_saved"]
                progress.update({"saved_id": saved["id"], "saved_index": saved["index"], "batch": saved["batch"]})
            req = job.get("request")
            batch = getattr(req, "batch", 1) if req else progress.get("batch", 1)
            if "batch" not in progress and batch > 1:
                progress["batch"] = batch
            output.append({
                "id": job["id"],
                "status": job["status"],
                "phase": job.get("phase"),
                "phase_detail": job.get("phase_detail"),
                "progress": progress or None,
                "batch": batch,
                "model": getattr(req, "model", "") if req else "",
                "prompt": getattr(req, "prompt", "")[:80] if req else "",
                "finished_at": job.get("finished_at"),
                "created_at": job.get("created_at"),
            })
        return output


@router.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if not isinstance(job_id, str) or not SAFE_ID_RE.fullmatch(job_id) or len(job_id) > 128:
        raise HTTPException(400, "invalid job id")
    with _JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        snapshot = dict(job)
        for key in ("progress", "last_result"):
            if isinstance(snapshot.get(key), dict):
                snapshot[key] = dict(snapshot[key])
        for key in ("results", "partial_results"):
            if isinstance(snapshot.get(key), list):
                snapshot[key] = list(snapshot[key])
    output = {key: value for key, value in snapshot.items() if key not in ("request", "cancel_event")}
    if hasattr(snapshot.get("request"), "model_dump"):
        output["request"] = snapshot["request"].model_dump()
    if snapshot.get("last_saved"):
        saved = dict(snapshot["last_saved"])
        output["last_saved"] = saved
        progress = dict(output.get("progress") or {})
        progress.update({"saved_id": saved["id"], "saved_index": saved["index"], "batch": saved["batch"]})
        output["progress"] = progress
    return output


@router.post("/api/jobs/cancel-all")
def cancel_all_jobs():
    cancelled = []
    running_event = None
    records = []
    with _JOBS_LOCK:
        for job in JOBS.values():
            if job.get("status") in ("done", "error", "cancelled"):
                continue
            was_running = job.get("status") == "generating"
            job["status"] = "cancelled"
            job["finished_at"] = time.time()
            cancelled.append(job["id"])
            req = job.get("request")
            request_data = req.model_dump() if hasattr(req, "model_dump") else dict(req or {})
            records.append((job["id"], request_data, "cancelled"))
            if was_running or job.get("cancel_event") is not None:
                running_event = job.get("cancel_event") or running_event
    if running_event is not None:
        running_event.set()
        generator.cancel_current()
    if records:
        try:
            record_recoverable_jobs(records)
        except Exception as e:
            raise HTTPException(500, "jobs were cancelled but recovery persistence failed") from e
    return {"cancelled": cancelled}


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not isinstance(job_id, str) or not SAFE_ID_RE.fullmatch(job_id) or len(job_id) > 128:
        raise HTTPException(400, "invalid job id")
    with _JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        if job["status"] in ("done", "error", "cancelled"):
            raise HTTPException(400, f"job already {job['status']}")
        was_running = job["status"] == "generating"
        job["status"] = "cancelled"
        job["finished_at"] = time.time()
        req = job.get("request")
        request_data = req.model_dump() if hasattr(req, "model_dump") else dict(req or {})
        event = job.get("cancel_event")
    if was_running:
        if event is not None:
            event.set()
        generator.cancel_current()
    try:
        record_recoverable_jobs([(job_id, request_data, "cancelled")])
    except Exception as e:
        raise HTTPException(500, "job was cancelled but recovery persistence failed") from e
    return {"status": "cancelled", "job_id": job_id}


@router.get("/api/queue/recovery")
def list_recoverable_queue():
    return {"items": get_recoverable_jobs()}


class RestoreQueueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_ids: list[str] = Field(default_factory=list, max_length=100)


@router.post("/api/queue/recovery/restore")
def restore_queue_jobs(req_body: RestoreQueueRequest):
    if any(not isinstance(job_id, str) or not SAFE_ID_RE.fullmatch(job_id) or len(job_id) > 128 for job_id in req_body.job_ids):
        raise HTTPException(400, "invalid recovery job id")
    items = get_recoverable_jobs()
    requested = set(req_body.job_ids)
    selected = [item for item in items if not requested or item["id"] in requested]
    if requested and len(selected) != len(requested):
        raise HTTPException(404, "one or more recovery jobs were not found")
    prepared = []
    for item in selected:
        try:
            prepared.append((item["id"], _prepare_request(item.get("request"))))
        except HTTPException as e:
            raise HTTPException(400, f"recovery job {item['id']} is invalid: {e.detail}") from e
        except Exception as e:
            raise HTTPException(400, f"recovery job {item['id']} is invalid") from e
    new_jobs = []
    source_ids = {item[0] for item in prepared}
    with _JOBS_LOCK:
        active_images = sum(
            getattr(job.get("request"), "batch", 1)
            for job in JOBS.values()
            if job.get("status") in ("generating", "queued")
        )
        requested_images = sum(req.batch for _, req in prepared)
        if active_images + requested_images > MAX_QUEUED_IMAGES:
            raise HTTPException(429, f"Queue limit reached ({active_images}/{MAX_QUEUED_IMAGES} images in queue)")
        for source_id, req in prepared:
            new_job_id = uuid.uuid4().hex
            new_jobs.append((new_job_id, req))
            JOBS[new_job_id] = {
                "id": new_job_id,
                "status": "queued",
                "request": req,
                "recovered_from": source_id,
                "created_at": time.time(),
            }
        try:
            record_pending_jobs([(new_job_id, req.model_dump()) for new_job_id, req in new_jobs])
            delete_recoverable_jobs(source_ids)
        except Exception as e:
            for new_job_id, _ in new_jobs:
                JOBS.pop(new_job_id, None)
            try:
                remove_pending_jobs({new_job_id for new_job_id, _ in new_jobs})
            except Exception:
                pass
            raise HTTPException(500, "failed to persist restored jobs") from e
        _prune_jobs()
    for new_job_id, _ in new_jobs:
        _job_queue.put(new_job_id)
    return {"restored": [job_id for job_id, _ in new_jobs], "count": len(new_jobs)}


@router.delete("/api/queue/recovery")
def clear_recovery(job_id: str | None = None, forget: bool = False):
    if job_id is not None and (not SAFE_ID_RE.fullmatch(job_id) or len(job_id) > 128):
        raise HTTPException(400, "invalid recovery job id")
    try:
        if job_id is None and forget:
            clear_recoverable_state()
        else:
            delete_recoverable_jobs({job_id} if job_id is not None else None)
    except Exception as e:
        raise HTTPException(500, "failed to update recovery archive") from e
    return {"status": "ok"}
