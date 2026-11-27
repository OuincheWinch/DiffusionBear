"""Hardware-derived generation budgets.

Pure functions over MLX device info and model metadata. This module imports no
application module, so the generator, the routers and the settings layer can all
use it without a circular import.

Every ceiling here is derived at runtime. A 16 GB Mac clamps to protect itself;
a 64 GB or 128 GB Mac is allowed to push the limits. Nothing is a hardcoded
constant for one machine.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import shutil
import threading
import time
from dataclasses import asdict, dataclass

REFERENCE_MEMORY_GB = 16.0
REFERENCE_PIXELS = 512 * 768
SAFE_BASE_STEPS = 50
SAFE_MIN_STEPS = 25
SAFE_MAX_STEPS = 400
HARD_STEPS_MULTIPLIER = 8
HARD_MIN_STEPS = 400
HARD_MAX_STEPS = 3200
ABSOLUTE_MAX_STEPS = 10_000
MIN_PIXELS = 128 * 128
MAX_PIXELS_ABSOLUTE = 8192 * 8192
PIXEL_STATE_BYTES = 256
ACK_TTL_SECONDS = 300
MAX_TOTAL_JOB_SECONDS = 24 * 3600

_LOCK = threading.RLock()
_CACHE: dict = {}
_ACK_SECRET = secrets.token_bytes(32)


def _device_info() -> dict:
    try:
        import mlx.core as mx

        fn = getattr(mx, "device_info", None) or getattr(
            getattr(mx, "metal", None), "device_info", None
        )
        return dict(fn() or {}) if fn else {}
    except Exception:
        return {}


def memory_profile() -> dict:
    """Total, single-buffer and effective memory for this machine."""
    with _LOCK:
        cached = _CACHE.get("profile")
        if cached:
            return dict(cached)
    info = _device_info()
    total = int(info.get("memory_size") or 0)
    if total <= 0:
        try:
            total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError, AttributeError):
            total = 16 << 30
    max_buffer = int(info.get("max_buffer_length") or 0)
    # Metal caps a single allocation well below total memory. Do not let the
    # pixel ceiling assume memory a single buffer can never use.
    effective = total
    if max_buffer > 0:
        effective = min(total, int(max_buffer * 1.8))
    profile = {
        "total_bytes": total,
        "max_buffer_bytes": max_buffer,
        "effective_bytes": effective,
        "total_gb": round(total / (1 << 30), 2),
        "effective_gb": round(effective / (1 << 30), 2),
    }
    with _LOCK:
        _CACHE["profile"] = profile
    return dict(profile)


METAL_CACHE_LIMIT_ENV = "DIFFUSIONBEAR_METAL_CACHE_LIMIT_MB"
METAL_CACHE_MIN_BYTES = 512 << 20
METAL_CACHE_MAX_BYTES = 1 << 30
METAL_CACHE_MODE_ENV = "DIFFUSIONBEAR_METAL_CACHE_MODE"
# "balanced": capped cache, lower peaks, slower (SDXL 1024²: +27 % per image on
# an M4 Pro VM, peak footprint 10.3 -> 6.9 GB). "fast": MLX's own default
# during a job; the cache is still emptied at the end of every job.
METAL_CACHE_MODES = ("balanced", "fast")
DEFAULT_METAL_CACHE_MODE = "balanced"

# MLX's own cache limit, captured the first time this process overrides it.
_mlx_default_cache_limit: int | None = None


def metal_cache_mode() -> str:
    """Active cache mode: env override, then the ``metal_cache_mode`` setting.

    Read on every call so engine subprocesses (SDXL daemon) pick up a change
    from Settings at their next job without a restart.
    """
    raw = os.environ.get(METAL_CACHE_MODE_ENV, "").strip().lower()
    if raw in METAL_CACHE_MODES:
        return raw
    try:
        import app_settings

        value = app_settings.get_setting("metal_cache_mode")
    except Exception:
        value = None
    return value if value in METAL_CACHE_MODES else DEFAULT_METAL_CACHE_MODE


def metal_cache_limit_bytes() -> int | None:
    """Ceiling for MLX's freed-buffer cache, separate from live (active) memory.

    Unbounded, the cache kept several GB of freed Metal buffers after each
    stage, so the process footprint ran far above active memory. 1/16 of
    unified memory, clamped to [512 MB, 1 GB]: 1 GB on the 16 GB reference Mac,
    512 MB on 8 GB. ``DIFFUSIONBEAR_METAL_CACHE_LIMIT_MB`` overrides it (0
    disables the cache entirely). None in "fast" mode: leave MLX's default.
    """
    raw = os.environ.get(METAL_CACHE_LIMIT_ENV, "").strip()
    if raw:
        try:
            return max(0, int(float(raw) * (1 << 20)))
        except ValueError:
            pass
    if metal_cache_mode() == "fast":
        return None
    total = memory_profile()["total_bytes"]
    return max(METAL_CACHE_MIN_BYTES, min(METAL_CACHE_MAX_BYTES, total // 16))


def apply_metal_cache_limit() -> int | None:
    """Install the cache ceiling in this process; returns the bytes applied.

    In "fast" mode this restores MLX's default limit if an earlier
    "balanced" job lowered it, and returns None.
    """
    global _mlx_default_cache_limit
    try:
        import mlx.core as mx
    except Exception:
        return None
    setter = getattr(mx, "set_cache_limit", None) or getattr(
        getattr(mx, "metal", None), "set_cache_limit", None
    )
    if setter is None:
        return None
    with _LOCK:
        if _mlx_default_cache_limit is None:
            # MLX has no getter: the setter returns the previous value. Read it
            # on the first call, before vendored code (FLUX.3) lowers it, so
            # "fast" can always put the native limit back.
            try:
                native = setter(0)
                setter(native)
                if isinstance(native, int):
                    _mlx_default_cache_limit = native
            except Exception:
                pass
    limit = metal_cache_limit_bytes()
    target = limit if limit is not None else _mlx_default_cache_limit
    if target is None:
        return None
    try:
        setter(target)
    except Exception:
        return None
    return limit


def release_metal_cache() -> None:
    """Return freed Metal buffers to the OS (end of job, cancel, error).

    Also reinstalls the ceiling: vendored pipelines (FLUX.3 ``_generate``) set
    their own process-wide limit mid-job.
    """
    import gc

    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
    apply_metal_cache_limit()


def safe_max_pixels_for_bytes(total_bytes: int, max_buffer_bytes: int = 0) -> int:
    """Pixel ceiling for an explicit memory size.

    Used by ``generator.safe_max_pixels`` so its long-standing
    ``device_memory_bytes`` test hook still drives the calculation.
    """
    gb = max(0.5, int(total_bytes) / (1 << 30))
    scaled = int(4_194_304 * (gb / REFERENCE_MEMORY_GB))
    if max_buffer_bytes:
        scaled = min(scaled, int(max_buffer_bytes / PIXEL_STATE_BYTES))
    return max(MIN_PIXELS, min(MAX_PIXELS_ABSOLUTE, scaled))


def safe_max_pixels() -> int:
    """Total-pixel ceiling derived from the device.

    Byte-for-byte compatible with the historical ``generator.safe_max_pixels``
    (16 GB -> 4.19 MP, 32 GB -> 8.39 MP, 64 GB -> 16.78 MP, 128 GB -> 33.55 MP),
    plus a single-buffer clamp that is not binding on current Apple silicon.
    """
    profile = memory_profile()
    return safe_max_pixels_for_bytes(
        profile["total_bytes"], profile["max_buffer_bytes"]
    )


def safe_max_pixels_for_model_bytes(total_bytes: int, model_info: dict, max_buffer_bytes: int = 0) -> int:
    """Pixel ceiling after reserving the model's resident weights."""
    cap = safe_max_pixels_for_bytes(total_bytes, max_buffer_bytes)
    resident_gb = (model_info or {}).get("resident_gb")
    if not resident_gb:
        return cap
    total = max(1.0, int(total_bytes) / (1 << 30))
    usable = max(0.15 * total, total - float(resident_gb))
    return max(MIN_PIXELS, int(cap * usable / total))


def safe_max_pixels_for_model(model_info: dict) -> int:
    """Pixel ceiling after reserving the model's resident weights.

    ``resident_gb`` is optional registry metadata. When absent the total-memory
    ceiling is returned unchanged, so this is a no-op until a model is measured.
    """
    profile = memory_profile()
    return safe_max_pixels_for_model_bytes(
        profile["total_bytes"], model_info, profile["max_buffer_bytes"]
    )


def max_generation_seconds() -> int:
    """Wall-clock ceiling for one image, scaled by machine memory."""
    gb = memory_profile()["total_gb"]
    return int(max(1800, min(21600, gb * 300)))


def estimated_seconds(model_info: dict, steps: int, pixels: int | None = None) -> int | None:
    """Rough per-image wall clock, only when the model ships a per-step cost."""
    per_step = (model_info or {}).get("per_step_seconds")
    if not per_step:
        return None
    pixels = pixels or int(model_info.get("default_width") or 512) * int(
        model_info.get("default_height") or 768
    )
    scale = max(0.25, pixels / REFERENCE_PIXELS)
    return int(round(float(per_step) * int(steps) * scale))


def max_batch(model_info: dict, steps: int, pixels: int | None = None) -> int:
    """How many images of this cost fit in a 24h job before it is refused."""
    estimate = estimated_seconds(model_info, steps, pixels)
    if not estimate or estimate <= 0:
        return 64
    return max(1, min(64, MAX_TOTAL_JOB_SECONDS // estimate))


def max_queued_images(data_dir, bytes_per_image: int = 25 * 1024 * 1024) -> int:
    """Queue depth bounded by free disk, not by a constant."""
    try:
        free = shutil.disk_usage(str(data_dir)).free
    except OSError:
        return 128
    return max(16, min(2048, int(free // bytes_per_image)))


@dataclass(frozen=True)
class StepPolicy:
    safe_max_steps: int
    hard_max_steps: int
    model_max_steps: int | None
    requires_ack: bool
    estimated_seconds: int | None
    max_generation_seconds: int

    def to_dict(self) -> dict:
        return asdict(self)


def step_policy(model_info: dict, pixels: int | None = None) -> StepPolicy:
    """Safe and hard step ceilings for this model on this machine."""
    model_info = model_info or {}
    profile = memory_profile()
    ratio = max(0.5, min(8.0, profile["total_gb"] / REFERENCE_MEMORY_GB))
    safe = max(SAFE_MIN_STEPS, min(SAFE_MAX_STEPS, round(SAFE_BASE_STEPS * ratio)))
    hard = max(HARD_MIN_STEPS, min(HARD_MAX_STEPS, safe * HARD_STEPS_MULTIPLIER))

    model_cap = model_info.get("max_steps")
    if model_cap:
        model_cap = int(model_cap)
        safe, hard = min(safe, model_cap), min(hard, model_cap)
    if model_info.get("is_distilled"):
        cap = int(model_info.get("max_steps") or 12)
        model_cap = cap if model_cap is None else min(model_cap, cap)
        safe, hard = min(safe, cap), min(hard, cap)
    if model_info.get("id") in ("qwen-image-2.1", "qwen-image-2.1-turbo", "qwen-image-edit"):
        model_cap = 25 if model_cap is None else min(model_cap, 25)
        safe, hard = min(safe, 25), min(hard, 25)

    safe = max(1, safe)
    hard = max(safe, hard)
    return StepPolicy(
        safe_max_steps=safe,
        hard_max_steps=hard,
        model_max_steps=int(model_cap) if model_cap else None,
        requires_ack=hard > safe,
        estimated_seconds=estimated_seconds(model_info, safe, pixels),
        max_generation_seconds=max_generation_seconds(),
    )


def _ack_message(model_id, steps, width, height, batch, hard_max) -> bytes:
    return (
        f"{model_id}\x00{int(steps)}\x00{int(width)}\x00{int(height)}"
        f"\x00{int(batch)}\x00{int(hard_max)}"
    ).encode("utf-8")


def issue_steps_ack(model_id, steps, width, height, batch, hard_max) -> str:
    """Issue a short-lived token bound to this exact request shape."""
    issued = int(time.time())
    message = _ack_message(model_id, steps, width, height, batch, hard_max) + b"\x00" + str(issued).encode()
    signature = hmac.new(_ACK_SECRET, message, hashlib.sha256).hexdigest()[:32]
    return f"{issued}.{signature}"


def verify_steps_ack(token, model_id, steps, width, height, batch, hard_max) -> bool:
    """Verify a token from :func:`issue_steps_ack`. Expires with the process."""
    if not isinstance(token, str) or len(token) > 80 or "." not in token:
        return False
    issued_text, signature = token.split(".", 1)
    try:
        issued = int(issued_text)
    except ValueError:
        return False
    now = time.time()
    if issued > now + 60 or now - issued > ACK_TTL_SECONDS:
        return False
    expected = hmac.new(
        _ACK_SECRET,
        _ack_message(model_id, steps, width, height, batch, hard_max) + b"\x00" + issued_text.encode(),
        hashlib.sha256,
    ).hexdigest()[:32]
    return hmac.compare_digest(signature, expected)
