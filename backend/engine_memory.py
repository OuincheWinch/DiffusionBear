"""One memory policy for every DiffusionBear engine family.

Each table entry restates a decision that already exists next to the line that
made it, so the tradeoffs are visible in one place and cannot drift silently:

- **SDXL** keeps both CLIP text encoders resident in the daemon and tiles the
  VAE decode at or above 768x768 (`mlx_sdxl_engine._validate_request`).
- **Qwen** keeps its text encoder for a 16 GB-friendly render and frees the
  ~4.8 GB encoder at or above 768x768 (`qwen_engine._should_evict_text_encoder`);
  its VAE tiles whenever either side exceeds 512 px.
- **MOSS** loads its text encoder on a prompt-cache miss and always frees it
  before the transformer (`_run`'s `finally`); the diffusers VAE always has
  tiling enabled (`_get_pipeline`).
- **FIBO** also caches prompt embeddings and evicts the encoder inside the
  conditioning phase; its FLUX.2 decoder runs untiled.
- **mflux** keeps the pipeline's text encoder resident (Krea 2's q4 encoder is
  a persistent derived artifact) and assigns a 512 px tiling config after
  construction.

Memory is measured as process RSS. ``mx.get_peak_memory`` is not a memory
metric here: it has reported 24 GB on a 16 GB machine.

This module imports no MLX and no torch so the same policy can be read from the
backend and from any engine subprocess.
"""

from __future__ import annotations

import os
from typing import Callable

from engine_metrics import peak_rss_mb

# Text-encoder lifecycle after conditioning.
TE_RESIDENT = "resident"  # encoder stays loaded between requests
TE_EVICT = "evict"  # encoder is freed after conditioning at this size
TE_CACHED = "cached"  # embeddings are cached; encoder loads only on a miss

# Engine names as the dispatcher spells them; "fibo" is accepted as an alias.
_ALIASES = {"fibo": "fibo2"}

# Shared with the per-engine implementations; keep both dictionaries in sync
# with the cited functions, not with the UI model list.
TE_EVICT_MIN_PIXELS = 768 * 768
QWEN_VAE_MIN_SIDE = 512

_TE_POLICY: dict[str, Callable[[int, int], str]] = {
    "sdxl": lambda width, height: TE_RESIDENT,
    "qwen": lambda width, height: (
        TE_EVICT if width * height >= TE_EVICT_MIN_PIXELS else TE_RESIDENT
    ),
    "moss": lambda width, height: TE_CACHED,
    "fibo2": lambda width, height: TE_CACHED,
    "mflux": lambda width, height: TE_RESIDENT,
    # Ming helper is one-shot and stages the conditioner before the DiT; nothing survives the job.
    "ming": lambda width, height: TE_EVICT,
}

_VAE_POLICY: dict[str, Callable[[int, int], bool]] = {
    "sdxl": lambda width, height: width * height >= TE_EVICT_MIN_PIXELS,
    "qwen": lambda width, height: max(width, height) > QWEN_VAE_MIN_SIDE,
    "moss": lambda width, height: True,
    "fibo2": lambda width, height: False,
    "mflux": lambda width, height: True,
    "ming": lambda width, height: True,
}

# A machine at or above this much unified memory can hold a MOSS pipeline and
# an SDXL daemon in the wired budget, so the idle watchdogs, not a switch,
# decide when to release one. Matches generator.KEEP_ALIVE_MIN_UNIFIED_GB.
KEEP_ALIVE_MIN_UNIFIED_GB = 32


# MLX keeps freed Metal buffers in a cache for reuse. Unbounded, that cache
# pushed the physical footprint 2-5 GB above the live tensors at the end of the
# DiT and VAE stages (FLUX.2 staged: 3.5 GB active, 7.3 GB footprint), which is
# what swaps first on smaller machines. 1 GiB matches qwen_engine.
MLX_CACHE_LIMIT_DEFAULT_BYTES = 1 << 30
MLX_CACHE_LIMIT_ENV = "DIFFUSIONBEAR_MLX_CACHE_LIMIT_MB"


def mlx_cache_limit_bytes() -> int | None:
    """Metal buffer-cache cap in bytes, or None to leave MLX unbounded.

    DIFFUSIONBEAR_MLX_CACHE_LIMIT_MB overrides the default; a negative value
    disables the cap, 0 disables the cache entirely. Without it, the policy is
    ``hardware_limits.metal_cache_limit_bytes()`` (Settings > memory mode), so
    both caps agree: None in "fast" mode.
    """
    raw = os.environ.get(MLX_CACHE_LIMIT_ENV, "").strip()
    if not raw:
        try:
            import hardware_limits

            return hardware_limits.metal_cache_limit_bytes()
        except Exception:
            return MLX_CACHE_LIMIT_DEFAULT_BYTES
    try:
        mb = float(raw)
    except ValueError:
        return MLX_CACHE_LIMIT_DEFAULT_BYTES
    if mb < 0:
        return None
    return int(mb * (1 << 20))


def _normalize(engine: str) -> str:
    key = str(engine or "").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _TE_POLICY:
        raise ValueError(f"unknown engine: {engine!r}")
    return key


def _size(width: int, height: int) -> tuple[int, int]:
    try:
        width, height = int(width), int(height)
    except (TypeError, ValueError) as exc:
        raise ValueError("width and height must be integers") from exc
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return width, height


def te_residency(engine: str, width: int, height: int) -> str:
    """What happens to the text encoder after conditioning at this size."""
    key = _normalize(engine)
    width, height = _size(width, height)
    return _TE_POLICY[key](width, height)


def vae_tiling(engine: str, width: int, height: int) -> bool:
    """Whether this family tiles its VAE decode at this size."""
    key = _normalize(engine)
    width, height = _size(width, height)
    return _VAE_POLICY[key](width, height)


def policy_label(engine: str, width: int, height: int) -> str:
    """Compact, stable value for the one ``memory_policy`` sidecar field."""
    te = te_residency(engine, width, height)
    tiled = "tiled" if vae_tiling(engine, width, height) else "untiled"
    return f"te={te},vae={tiled}"


def one_heavy_engine_policy(active_engine: str) -> None:
    """Keep one heavy engine resident and release the other families.

    The eviction implementation stays in ``generator``; this entry point only
    validates the engine name and hands it to that implementation, which
    already skips ``active_engine``. Keeping the direction one-way avoids the
    import cycle: ``generator`` imports this module at import time, while this
    function imports ``generator`` lazily.
    """
    keep = _normalize(active_engine)
    import generator

    generator._evict_other_engines_impl(keep)
