"""Shared instrumentation for every DiffusionBear engine family.

This module is imported by the backend process and by each engine subprocess
(SDXL, Qwen, MOSS, FIBO), so it is deliberately dependency-free: the standard
library only, no MLX and no torch. It records per-phase wall timings, the
number of explicit synchronization boundaries, cache statistics, process
peak RSS and the macOS physical footprint, and it writes benchmark rows as
JSON sidecars.

RSS does not count Metal buffers in unified memory (a 1 GiB MLX array moves
``ru_maxrss`` by ~12 MiB), so memory pressure is read from the physical
footprint (``proc_pid_rusage``), the number the macOS memory pressure and
swap decisions use.

Wall clock is ``time.perf_counter``: monotonic, and unaffected by wall-clock
adjustments mid-generation. Peak RSS comes from ``getrusage`` -- note that
``ru_maxrss`` is bytes on macOS and kilobytes on Linux, so the conversion must
branch or the number is wrong by 1024x.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator


class EngineMetrics:
    """Per-generation timings, sync count and cache stats for one engine."""

    def __init__(
        self,
        engine: str = "",
        model: str = "",
        cache: dict | None = None,
    ) -> None:
        self.engine = engine
        self.model = model
        self.phases: dict[str, float] = {}
        self.sync_count = 0
        self.cache: dict = dict(cache) if cache else {}

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time a named phase; repeated phases accumulate."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.phases[name] = self.phases.get(name, 0.0) + elapsed

    def add_sync(self, count: int = 1) -> None:
        """Record a host-device synchronization boundary."""
        self.sync_count += count

    def snapshot(self) -> dict:
        """A JSON-ready view; mutating it does not affect the live counters."""
        return {
            "engine": self.engine,
            "model": self.model,
            "phases": dict(self.phases),
            "sync_count": self.sync_count,
            "cache": dict(self.cache),
            "peak_rss_mb": peak_rss_mb(),
            "footprint_mb": footprint_mb(),
            "peak_footprint_mb": peak_footprint_mb(),
        }


def peak_rss_mb() -> float:
    """Process peak resident set size in MiB."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024 * 1024)
    return raw / 1024


# rusage_info_v4 as uint64 words: 16-byte uuid (2 words), then
# ri_phys_footprint at word 9 and ri_lifetime_max_phys_footprint at word 30.
_RUSAGE_INFO_V4 = 4
_RUSAGE_WORDS = 64
_PHYS_FOOTPRINT = 9
_LIFETIME_MAX_PHYS_FOOTPRINT = 30
_libproc = None


def _rusage_v4() -> tuple | None:
    global _libproc
    if sys.platform != "darwin":
        return None
    try:
        if _libproc is None:
            _libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        buf = (ctypes.c_uint64 * _RUSAGE_WORDS)()
        if _libproc.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(buf)) != 0:
            return None
        return tuple(buf)
    except Exception:
        return None


def footprint_mb() -> float | None:
    """Current physical footprint in MiB (includes Metal buffers); None off macOS."""
    info = _rusage_v4()
    return None if info is None else info[_PHYS_FOOTPRINT] / (1024 * 1024)


def peak_footprint_mb() -> float | None:
    """Lifetime peak physical footprint in MiB; None off macOS."""
    info = _rusage_v4()
    return None if info is None else info[_LIFETIME_MAX_PHYS_FOOTPRINT] / (1024 * 1024)


def write_metrics(path: Path, metrics: dict) -> None:
    """Write one benchmark row atomically, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
