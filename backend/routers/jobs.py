import io
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import app_settings
import generator
import hardware_limits
from flux3_spatial import validate_bounding_box
from state import (
    GenerateRequest,
    DATA_DIR,
    _decode_mask_payload,
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


def _validate_mask(mask: str, width: int, height: int) -> str:
    """Reject a mask the worker could not apply, before any job exists."""
    from PIL import Image

    data = _decode_mask_payload(mask)
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            fmt = (image.format or "").upper()
            size = image.size
    except Exception as exc:
        raise HTTPException(400, "mask is not a readable image") from exc
    if fmt not in ("PNG", "WEBP"):
        raise HTTPException(400, "mask must be a PNG or WebP image")
    if size != (int(width), int(height)):
        raise HTTPException(
            400,
            f"mask is {size[0]}x{size[1]} but the output is "
            f"{int(width)}x{int(height)}; the mask must match the render size",
        )
    return mask


def _queue_cap() -> int:
    return min(MAX_QUEUED_IMAGES, hardware_limits.max_queued_images(DATA_DIR))


def _prepare_request(value: GenerateRequest | dict) -> GenerateRequest:
    req = value if isinstance(value, GenerateRequest) else GenerateRequest.model_validate(value)
    if not (req.prompt and req.prompt.strip()) and not req.spatial_boxes:
        raise HTTPException(400, "prompt is required")
    info = generator.get_model_info(req.model)
    if info is None:
        raise HTTPException(400, f"unknown model: {req.model}")
    req.model = info["id"]
    # Persisted preferences must drive the fallback: get_model_info returns the raw
    # registry entry, so a saved per-model width/height would be ignored for every
    # caller that omits the field. Only GET /api/models applied the overrides before.
    info = app_settings.apply_model_defaults(info["id"], info)
    max_prompt_bytes = int(info.get("max_prompt_bytes") or generator._MAX_PROMPT_BYTES)
    for label, text in (("prompt", req.prompt), ("negative_prompt", req.negative_prompt)):
        if len(text.encode("utf-8")) > max_prompt_bytes:
            raise HTTPException(
                400,
                f"{label} exceeds the {max_prompt_bytes}-byte budget for {info['label']}",
            )
    if req.width is None:
        req.width = int(info.get("default_width") or 512)
    if req.height is None:
        req.height = int(info.get("default_height") or 768)
    safe_cap = generator.safe_max_pixels_for_model(info)
    total_px = req.width * req.height
    if total_px > safe_cap:
        mem_gb = round(generator.device_memory_bytes() / (1 << 30), 1)
        safe_mp = round(safe_cap / 1e6, 2)
        req_mp = round(total_px / 1e6, 2)
        raise HTTPException(
            400,
            f"Requested resolution {req.width}×{req.height} ({req_mp} MP) exceeds the safe memory ceiling "
            f"of {safe_mp} MP for this machine's unified memory ({mem_gb} GB). "
            f"Reduce dimensions to avoid a Metal out-of-memory crash.",
        )
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
    if not info.get("supports_guidance") and req.guidance is not None and req.guidance > 1.0 and info.get("engine") != "sdxl" and info.get("id") != "flux3-klein":
        raise HTTPException(400, f"{info['label']} does not support guidance > 1.0")
    if req.negative_prompt and not info.get("supports_negative"):
        raise HTTPException(400, f"{info['label']} does not support negative prompts")
    if req.sampler:
        try:
            req.sampler = generator._validate_sampler(req.sampler, info, strict=True)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    if req.reference_images:
        if not (info.get("supports_multi_reference") or info.get("supports_ref") or info.get("id") in {"z-image-turbo", "krea2-turbo", "qwen-image-2.1"}):
            raise HTTPException(400, f"reference images are not supported on {info['label']}")
        max_references = generator.effective_max_reference_images(
            info,
            unlock_cap=bool(getattr(req, "unlock_reference_cap", False)),
        )
        if len(req.reference_images) > max_references:
            raise HTTPException(400, f"a maximum of {max_references} reference images is allowed")
        if not info.get("supports_multi_reference") and len(req.reference_images) > 1:
            raise HTTPException(400, "this model accepts one reference image")
        req.reference_images = _validate_reference_images(req.reference_images)
    elif req.reference_strength is not None:
        raise HTTPException(400, "reference_strength requires a reference image")
    if req.mask:
        req.mask = _validate_mask(req.mask, req.width, req.height)
    if req.spatial_boxes:
        # The schema has already rejected unknown keys, out-of-range coordinates
        # and inverted ranges. Convert back to dicts for the prompt composers.
        req.spatial_boxes = [
            box.model_dump(by_alias=True, exclude_none=True) for box in req.spatial_boxes
        ]
        if not info.get("supports_bounding_box"):
            raise HTTPException(400, f"bounding boxes are not supported on {info['label']}")
        if len(req.spatial_boxes) > 10:
            raise HTTPException(400, "a maximum of 10 spatial boxes is allowed")
        from spatial_boxes import compose_flux3_bbox_prompt, validate_and_clamp_bbox
        for b in req.spatial_boxes:
            if not isinstance(b, dict):
                raise HTTPException(400, "each spatial box must be an object")
            raw_box = b.get("bbox") or b.get("tgt_bbox") or b.get("src_bbox")
            if not raw_box:
                raise HTTPException(400, "each spatial box must specify a bounding box")
            if b.get("bbox"):
                b["bbox"] = validate_and_clamp_bbox(b["bbox"])
            if b.get("tgt_bbox"):
                b["tgt_bbox"] = validate_and_clamp_bbox(b["tgt_bbox"])
            if b.get("src_bbox"):
                b["src_bbox"] = validate_and_clamp_bbox(b["src_bbox"])
        if not req.bounding_boxes:
            req.bounding_boxes = [
                b.get("bbox") or b.get("tgt_bbox") or b.get("src_bbox")
                for b in req.spatial_boxes
            ]
        if info.get("id") == "flux3-klein" or info.get("family") == "flux3":
            req.prompt = compose_flux3_bbox_prompt(
                req.prompt, req.spatial_boxes, has_reference=bool(req.reference_images)
            )
        elif info.get("ecosystem") == "MOSS" or "moss-image" in str(info.get("id")):
            from spatial_boxes import compose_moss_bbox_prompt
            w = req.width or int(info.get("default_width") or 1024)
            h = req.height or int(info.get("default_height") or 1024)
            req.prompt = compose_moss_bbox_prompt(
                req.prompt, req.spatial_boxes, width=w, height=h
            )
    elif req.bounding_boxes:
        if not info.get("supports_bounding_box"):
            raise HTTPException(400, f"bounding boxes are not supported on {info['label']}")
        if len(req.bounding_boxes) > 10:
            raise HTTPException(400, "a maximum of 10 bounding boxes is allowed")
        for box in req.bounding_boxes:
            if not validate_bounding_box(box):
                raise HTTPException(
                    400,
                    f"invalid bounding box coordinates: {box}. Must be [ymin, xmin, ymax, xmax] on a 0-1000 grid with ymin <= ymax and xmin <= xmax",
                )
    policy = hardware_limits.step_policy(info, pixels=req.width * req.height)
    requested_steps = (
        req.steps if req.steps is not None else int(info.get("default_steps") or 1)
    )
    if requested_steps > policy.hard_max_steps:
        hint = ""
        if info.get("is_distilled") or str(info.get("id", "")).startswith("qwen-image"):
            hint = (
                " This model's schedule is fixed; pick a non-distilled model for a "
                "longer run."
            )
        raise HTTPException(
            400,
            detail={
                "code": "steps_above_hard_max",
                "steps": requested_steps,
                "hard_max_steps": policy.hard_max_steps,
                "message": (
                    f"{info['label']} is limited to {policy.hard_max_steps} steps on this "
                    f"machine ({hardware_limits.memory_profile()['total_gb']} GB).{hint}"
                ),
            },
        )
    if requested_steps > policy.safe_max_steps:
        if not req.allow_extended_steps:
            raise HTTPException(
                422,
                detail={
                    "code": "steps_requires_ack",
                    "steps": requested_steps,
                    "safe_max_steps": policy.safe_max_steps,
                    "hard_max_steps": policy.hard_max_steps,
                    "message": (
                        f"{requested_steps} steps exceeds the no-warning ceiling of "
                        f"{policy.safe_max_steps}. Resubmit with "
                        "allow_extended_steps=true and a fresh steps_ack."
                    ),
                },
            )
        if not hardware_limits.verify_steps_ack(
            req.steps_ack,
            info["id"],
            requested_steps,
            req.width,
            req.height,
            req.batch,
            policy.hard_max_steps,
        ):
            raise HTTPException(
                422,
                detail={
                    "code": "steps_ack_invalid",
                    "message": (
                        "steps_ack is missing, expired, or does not match this request."
                    ),
                },
            )
    req.steps = requested_steps

    batch_cap = hardware_limits.max_batch(
        info, requested_steps, req.width * req.height
    )
    if req.batch > batch_cap:
        raise HTTPException(
            422,
            detail={
                "code": "batch_above_budget",
                "batch": req.batch,
                "max_batch": batch_cap,
                "message": (
                    f"batch {req.batch} would exceed the 24h job budget on this "
                    f"machine (max {batch_cap}); reduce batch or steps."
                ),
            },
        )

    if info.get("id") == "qwen-image-2.1-turbo" and req.width * req.height > 768 * 768:
        min_steps = generator._qwen_turbo_min_steps(req.width, req.height)
        if requested_steps < min_steps:
            raise HTTPException(
                400,
                f"qwen-image-2.1-turbo needs at least {min_steps} steps above 768x768; "
                f"{requested_steps} steps leaves faded duplicate objects. "
                f"Requested {req.width}x{req.height}.",
            )
    return req


class StepPreflightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str = Field(min_length=1, max_length=200)
    steps: int = Field(ge=1, le=hardware_limits.ABSOLUTE_MAX_STEPS)
    width: int | None = Field(default=None, ge=128, le=8192)
    height: int | None = Field(default=None, ge=128, le=8192)
    batch: int = Field(default=1, ge=1, le=64)


@router.post("/api/generate/steps/preflight")
def preflight_steps(body: StepPreflightRequest):
    info = generator.get_model_info(body.model)
    if info is None:
        raise HTTPException(400, f"unknown model: {body.model}")
    info = app_settings.apply_model_defaults(info["id"], info)
    width = body.width or int(info.get("default_width") or 512)
    height = body.height or int(info.get("default_height") or 768)
    policy = hardware_limits.step_policy(info, pixels=width * height)
    if body.steps > policy.hard_max_steps:
        raise HTTPException(
            400,
            detail={
                "code": "steps_above_hard_max",
                "steps": body.steps,
                "hard_max_steps": policy.hard_max_steps,
                "model": info["id"],
                "message": (
                    f"{info['label']} is limited to {policy.hard_max_steps} steps "
                    "on this machine."
                ),
            },
        )
    if body.steps > policy.safe_max_steps:
        return {
            "allowed": True,
            "requires_ack": True,
            "safe_max_steps": policy.safe_max_steps,
            "hard_max_steps": policy.hard_max_steps,
            "estimated_seconds": policy.estimated_seconds,
            "ack_token": hardware_limits.issue_steps_ack(
                info["id"],
                body.steps,
                width,
                height,
                body.batch,
                policy.hard_max_steps,
            ),
            "warning": (
                f"{body.steps} steps is above the no-warning ceiling of "
                f"{policy.safe_max_steps}. Confirm you want to run it."
            ),
        }
    return {
        "allowed": True,
        "requires_ack": False,
        "safe_max_steps": policy.safe_max_steps,
        "hard_max_steps": policy.hard_max_steps,
        "estimated_seconds": policy.estimated_seconds,
    }


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
            queue_cap = _queue_cap()
            if active_images + req.batch > queue_cap:
                raise HTTPException(429, f"Queue limit reached ({active_images}/{queue_cap} images in queue)")
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
                # The worker has always stored the failure message (state.py sets
                # job["error"]); this projection just never exposed it. So the generation
                # stack could only ever render a bare "X Error" with no reason, while the
                # banner at the top of the form showed the real Metal message -- two
                # surfaces reporting the same failure in two different ways, which read as
                # two separate errors.
                "error": job.get("error"),
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
        queue_cap = _queue_cap()
        if active_images + requested_images > queue_cap:
            raise HTTPException(429, f"Queue limit reached ({active_images}/{queue_cap} images in queue)")
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
