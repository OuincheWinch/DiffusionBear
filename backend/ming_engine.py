"""Ming-Image subprocess bridge.

The Ming runtime is the parity-gated Swift/MLX port in backend/ming_helper.
This Python shim owns helper discovery/building, JSONL progress forwarding and
error normalization; generator.py stays engine-agnostic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

BACKEND_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = BACKEND_DIR / "ming_helper"


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _candidate_paths() -> list[Path]:
    paths = []
    override = os.environ.get("DIFFUSIONBEAR_MING_HELPER")
    if override:
        paths.append(Path(override).expanduser())
    paths.extend([
        BACKEND_DIR / "ming-helper",
        BACKEND_DIR / "MingHelper",
        PACKAGE_DIR / ".build" / "release" / "MingHelper",
        PACKAGE_DIR / ".build" / "arm64-apple-macosx" / "release" / "MingHelper",
    ])
    return paths


def find_helper() -> Path | None:
    for path in _candidate_paths():
        if path.is_file() and os.access(path, os.X_OK) and (path.parent / "mlx.metallib").is_file():
            return path.resolve()
    return None


def build_helper() -> Path:
    existing = find_helper()
    if existing is not None:
        return existing
    if not PACKAGE_DIR.is_dir():
        raise RuntimeError(f"Ming helper source is missing at {PACKAGE_DIR}")
    if subprocess.run(["/usr/bin/env", "swift", "--version"], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError(
            "Ming requires a one-time Swift helper build, but no Swift toolchain is installed. "
            "Install Xcode Command Line Tools or use a packaged DiffusionBear build that includes MingHelper."
        )
    emit({"phase": "building_helper", "detail": "Building the Ming Swift/MLX helper (first run only)..."})
    started = time.time()
    proc = subprocess.run(
        ["/usr/bin/env", "swift", "build", "-c", "release", "--product", "MingHelper"],
        cwd=PACKAGE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=2400,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2400:]
        raise RuntimeError(f"Ming helper build failed: {tail}")
    metal = subprocess.run(
        ["/bin/bash", str(PACKAGE_DIR / "build_metallib.sh")],
        cwd=PACKAGE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1200,
    )
    if metal.returncode != 0:
        tail = (metal.stderr or metal.stdout or "")[-2400:]
        raise RuntimeError(f"MLX Metal shader build failed: {tail}")
    helper = find_helper()
    if helper is None:
        raise RuntimeError("Ming helper build finished but the executable was not found")
    emit({"phase": "building_helper", "detail": f"Ming helper ready in {time.time() - started:.1f}s."})
    return helper


def _bool(value) -> bool:
    return bool(value) and str(value).lower() not in ("0", "false", "no")


def run_request(req: dict) -> int:
    root = str(req.get("model_path") or "").strip()
    prompt = str(req.get("prompt") or "").strip()
    dest = str(req.get("dest") or "").strip()
    if not root or not prompt or not dest:
        raise ValueError("model_path, prompt and dest are required")
    if not (Path(root) / "mllm" / "config.json").is_file():
        raise ValueError(f"not a Ming snapshot: {root}")
    helper = build_helper()
    args = [
        str(helper),
        "--root", root,
        "--prompt", prompt,
        "--output", dest,
        "--width", str(int(req.get("width", 1024))),
        "--height", str(int(req.get("height", 1024))),
        "--steps", str(int(req.get("steps", 12))),
        "--guidance", str(float(req.get("guidance", 1.0))),
        "--seed", str(int(req.get("seed", 42))),
        "--resolution", str(int(req.get("resolution", 1024))),
        "--cache-gb", str(int(req.get("cache_gb", 2))),
        "--dtype", str(req.get("dtype") or "bf16"),
    ]
    if _bool(req.get("transparent")):
        args.append("--transparent")

    stderr_file = tempfile.TemporaryFile(mode="w+")
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    final = None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        emit(payload)
        if payload.get("status") == "success":
            final = payload
    code = proc.wait()
    stderr_file.seek(0)
    stderr = stderr_file.read()[-4000:]
    stderr_lines = [stderr] if stderr else []
    if code != 0:
        detail = (stderr_lines[-1] if stderr_lines else "").strip() or f"exit code {code}"
        emit({"status": "error", "error": f"Ming helper failed: {detail}"})
        return 1
    if final is None:
        emit({"status": "error", "error": "Ming helper exited without a success payload"})
        return 1
    return 0


def main() -> int:
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
        return run_request(req)
    except Exception as exc:
        emit({"status": "error", "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
