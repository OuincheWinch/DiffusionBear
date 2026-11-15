"""Shared instrumentation for every DiffusionBear engine family.

This module is imported by the backend process and by each engine subprocess
(SDXL, Qwen, MOSS, FIBO), so it is deliberately dependency-free: the standard
library only, no MLX and no torch. It records per-phase wall timings, the
number of explicit synchronization boundaries, cache statistics, and process
peak RSS, and it writes benchmark rows as JSON sidecars.

Wall clock is ``time.perf_counter``: monotonic, and unaffected by wall-clock
adjustments mid-generation. Peak RSS comes from ``getrusage`` -- note that
``ru_maxrss`` is bytes on macOS and kilobytes on Linux, so the conversion must
branch or the number is wrong by 1024x.
"""

from __future__ import annotations

import contextlib
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
        }


def peak_rss_mb() -> float:
    """Process peak resident set size in MiB."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024 * 1024)
    return raw / 1024


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
