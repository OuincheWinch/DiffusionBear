#!/usr/bin/env python3
"""SDXL (Juggernaut XL Lightning) first baseline — exercises the app's real
subprocess engine via generator.generate() so the standard data/generated
artifacts ({id}.png, {id}.json, {id}_thumb.png) are produced, then records the
run to test/manifest.json alongside a test/ PNG copy.

Usage:
  ./venv/bin/python backend/scripts/profile_sdxl.py
"""
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
PROJECT_ROOT = BACKEND_DIR.parent
TEST_DIR = PROJECT_ROOT / "test"
TEST_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = TEST_DIR / "manifest.json"

from image_meta import atomic_write_json
import generator
import mlx.core as mx

PROMPT = (
    "a detailed imperial stormtrooper standing in a scenic swiss alpine landscape, "
    "green valley, snow-capped mountains, wooden chalet, cinematic lighting"
)
W, H, STEPS = 1024, 1024, 4
SEED = 1337


def _sysload() -> float:
    try:
        return float(os.popen("uptime").read().split()[-3].strip(","))
    except Exception:
        return 0.0


def _dependency_versions() -> dict:
    names = ("mlx", "mflux", "numpy", "pillow")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _engine_dependency_versions() -> dict:
    python = PROJECT_ROOT / "venv-sdxl" / "bin" / "python"
    if not python.exists():
        return {"available": False}
    query = (
        "import importlib.metadata as m,json; "
        "n=('mlx','mlx-diffuser','numpy','pillow'); "
        "print(json.dumps({x:m.version(x) for x in n}))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", query],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {"available": True, **json.loads(result.stdout)}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _memory_snapshot() -> dict:
    snapshot = {}
    for key, name in (
        ("active_bytes", "get_active_memory"),
        ("peak_bytes", "get_peak_memory"),
        ("cache_bytes", "get_cache_memory"),
    ):
        getter = getattr(mx, name, None) or getattr(mx.metal, name, None)
        if getter:
            snapshot[key] = int(getter())
    return snapshot


def _reset_peak_memory() -> None:
    reset = getattr(mx, "reset_peak_memory", None) or getattr(mx.metal, "reset_peak_memory", None)
    if reset:
        reset()


def _phase_timings(events: list[tuple[str, float]], start: float, end: float) -> dict:
    if not events:
        return {"unattributed": round(max(0.0, end - start), 3)}
    totals = {"unattributed_before_first_phase": round(max(0.0, events[0][1] - start), 3)}
    for index, (phase, timestamp) in enumerate(events):
        next_timestamp = events[index + 1][1] if index + 1 < len(events) else end
        duration = max(0.0, next_timestamp - timestamp)
        totals[phase] = round(totals.get(phase, 0.0) + duration, 3)
    return totals


if __name__ == "__main__":
    print(f"[sys] {os.popen('uptime').read().strip()}", flush=True)
    daemon = getattr(generator, "_sdxl_daemon", None)
    resident = daemon is not None and daemon.poll() is None
    current_model = str(getattr(generator, "_current_sdxl_model", "") or "")
    pipeline_cache = "warm" if resident and "juggernaut-xl-lightning" in current_model else "cold"
    phase_events = []

    def phase_cb(phase, msg=None):
        phase_events.append((phase, time.perf_counter()))

    _reset_peak_memory()
    memory_before = _memory_snapshot()
    t0 = time.perf_counter()
    meta = generator.generate(
        prompt=PROMPT,
        width=W,
        height=H,
        steps=STEPS,
        guidance=1.0,
        seed=SEED,
        quantization=4,
        model="juggernaut-xl-lightning",
        sampler="euler_trailing",
        progress_cb=lambda *a, **k: None,
        phase_cb=phase_cb,
        fast_vae=True,
    )
    wall = time.perf_counter() - t0
    memory_after = _memory_snapshot()
    phases = _phase_timings(phase_events, t0, t0 + wall)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    img_name = f"sdxl_lightning_{W}x{H}_q4_s{SEED}_{stamp}.png"
    img_path = TEST_DIR / img_name
    gen_dir = BACKEND_DIR / "data" / "generated"
    src = gen_dir / meta["file"]
    if src.exists():
        shutil.copy2(src, img_path)
    data_artifacts = {
        "png": f"{meta['id']}.{meta.get('format', 'png')}",
        "json": f"{meta['id']}.json",
        "thumb": f"{meta['id']}_thumb.png",
    }
    present = {k: (gen_dir / v).exists() for k, v in data_artifacts.items()}

    runs = []
    if MANIFEST.exists():
        try:
            runs = json.loads(MANIFEST.read_text())
            if not isinstance(runs, list):
                runs = []
        except Exception:
            runs = []
    runs.append({
        "image": img_name,
        "model": "juggernaut-xl-lightning",
        "variant": "e2e-sdxl-daemon-app-path",
        "width": W,
        "height": H,
        "steps": STEPS,
        "seed": SEED,
        "guidance": 1.0,
        "sampler": meta.get("sampler", "euler_trailing"),
        "quantization_bits": 4,
        "fast_vae": True,
        "cache_interval": meta.get("cache_interval"),
        "pipeline_cache": pipeline_cache,
        "wired_limit_gb": generator._wired_limit_bytes() / (1 << 30),
        "prompt": PROMPT,
        "dependencies": _dependency_versions(),
        "engine_dependencies": _engine_dependency_versions(),
        "memory": {
            "scope": "api-parent-only; SDXL subprocess memory unavailable",
            "before": memory_before,
            "after": memory_after,
        },
        "system_load": _sysload(),
        "wall_clock": round(wall, 3),
        "wall_clock_scope": f"generator.generate() with {pipeline_cache} SDXL daemon, including load and save",
        "generation_time": meta.get("generation_time"),
        "generation_time_scope": "SDXL engine timing reported by generator, excluding parent-side orchestration",
        "phase_sequence": [phase for phase, _ in phase_events],
        "phases": phases,
        "data_generated": {"artifacts": data_artifacts, "present": present},
        "created_at_utc": stamp,
        "pipeline": "generator.generate() -> sdxl subprocess daemon incl. save",
    })
    atomic_write_json(MANIFEST, runs, ensure_ascii=False)

    print(f"\n== SDXL Lightning | {W}x{H} | {STEPS} steps | seed {SEED} ==", flush=True)
    print(f"  wall {wall:6.2f}s | generation_time={meta.get('generation_time')}s | cache {pipeline_cache}", flush=True)
    print(f"  data/generated: {data_artifacts} -> {present}", flush=True)
    print(f"  image saved : {img_path}", flush=True)
    print(f"  manifest run(s) -> {MANIFEST}", flush=True)
