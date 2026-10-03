"""Convert a single-file Civitai SDXL checkpoint into a diffusers directory.

Why this exists
---------------
Civitai serves one `.safetensors`. The SDXL engine builds its pipeline with
`StableDiffusionXLPipeline.from_diffusers(...)` (sdxl_engine.py:258), which needs a
diffusers *directory*: `model_index.json`, `unet/`, `text_encoder/`, `text_encoder_2/`,
`vae/`, `tokenizer/`, `tokenizer_2/`, `scheduler/`. A downloaded Civitai checkpoint is
therefore invisible to the engine until it is converted, which is exactly the gap the
Models tab reported as "single-file checkpoint, needs converting".

Why it runs in a subprocess
---------------------------
The conversion needs `torch` plus `diffusers`, and the inference process must never carry
that. `mlx_diffuser`'s own `converters/` package does the opposite direction (diffusers ->
MLX-native) and cannot help. So this runs out-of-process with a timeout and reports
structured progress back over stdout; a failed or hung conversion cannot take the backend
with it, and cancelling is a process kill rather than a hope.

THE DEPENDENCY IS NOT INSTALLED
--------------------------------
Neither venv has `diffusers` (checked 2026-10-02: venv-sdxl has torch + transformers +
omegaconf + accelerate but no diffusers; the main venv has torch + transformers but no
diffusers, omegaconf or accelerate). So today this exits with a clear, actionable message
rather than pretending to work. That behaviour is tested.

To enable it, install into the MAIN venv (Python 3.10, already carries torch; the SDXL
runtime in venv-sdxl stays untouched, which matters because AGENTS.md records that the
runtime must not depend on torch):

    ./venv/bin/pip install "diffusers" omegaconf accelerate

Then it runs under the main interpreter, which is the default here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# Where the conversion runs: the SAME interpreter that is running the backend.
#
# It used to be a hardcoded ../venv/bin/python relative to this file. That silently picked
# the DEV checkout's own venv, which is a different environment from the one the app
# actually runs on -- diffusers was installed in the external venv and has_diffusers()
# still reported False. sys.executable is correct in every context: a source checkout, the
# standalone app's bundled venv, or a launchd unit.
#
# That interpreter also already imports torch on the mflux inference path, so nothing new
# is being added to the running process beyond diffusers itself. venv-sdxl, which is
# deliberately torch-free at runtime, is never used here.
DEFAULT_PYTHON = Path(sys.executable)

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
STAGE_WEIGHT = 0.75  # share of progress that is download-vs-convert, roughly


class ConversionError(RuntimeError):
    pass


def has_diffusers(python: Path | None = None) -> bool:
    """Cheap probe so the UI can disable the action instead of failing on click."""
    exe = python or Path(os.environ.get("MLX_SDXL_CONVERT_PYTHON") or DEFAULT_PYTHON)
    if not exe.exists():
        return False
    try:
        result = subprocess.run(
            [str(exe), "-c", "import diffusers, omegaconf, accelerate"],
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def missing_dependency_message() -> str:
    return (
        "Converting a checkpoint needs `diffusers`, which is not installed. "
        "Run: ./venv/bin/pip install diffusers omegaconf accelerate"
    )


_WORKER = r'''
import json, os, sys, time
from pathlib import Path

src, dest, base, cancel_path = sys.argv[1:5]

def emit(kind, **fields):
    print(json.dumps({"kind": kind, **fields}), flush=True)

def cancelled():
    return os.path.exists(cancel_path)

try:
    import torch
    from diffusers import StableDiffusionXLPipeline
except Exception as exc:
    emit("fatal", error="conversion dependencies are missing: %s" % exc)
    sys.exit(3)

emit("stage", stage="loading", progress=0.05)
try:
    pipeline = StableDiffusionXLPipeline.from_single_file(
        str(src),
        torch_dtype=torch.float16,
        local_files_only=False,
        safety_checker=None,
    )
except Exception as exc:
    emit("fatal", error="could not load the checkpoint: %s" % exc)
    sys.exit(4)

if cancelled():
    emit("fatal", error="cancelled")
    sys.exit(5)

emit("stage", stage="converting", progress=0.80)
target = Path(dest)
target.mkdir(parents=True, exist_ok=True)
try:
    pipeline.save_pretrained(str(target))
except Exception as exc:
    emit("fatal", error="could not write the diffusers directory: %s" % exc)
    sys.exit(6)

# Normalise CLIP key naming. diffusers 0.40 serialises the two SDXL text encoders
# inconsistently: text_encoder_2 (CLIPTextModelWithProjection) writes 516 of 517 keys
# with a "text_model." prefix, while text_encoder (plain CLIPTextModel) writes none of its
# 196. mlx_diffuser's converter expects the prefixed form, so the engine rejects the
# directory with "missing 196 keys ... extra 196 keys" -- a pure naming mismatch, with the
# weights themselves fine.
#
# Normalising toward the convention the OTHER file already uses, rather than a hardcoded
# prefix, means the fix follows whichever convention the installed diffusers actually
# writes instead of assuming one.
def _normalise_clip_keys(target):
    from safetensors.torch import load_file, save_file
    import torch
    te = target / "text_encoder" / "model.safetensors"
    te2 = target / "text_encoder_2" / "model.safetensors"
    if not te.is_file() or not te2.is_file():
        return
    with safe_open_keys(te) as k1, safe_open_keys(te2) as k2:
        prefixed_elsewhere = any(x.startswith("text_model.") for x in k2)
        needs = bool(k1) and not any(x.startswith("text_model.") for x in k1)
    if not (prefixed_elsewhere and needs):
        return
    tensors = load_file(str(te))
    renamed = {(k if k.startswith("text_model.") else "text_model." + k): v for k, v in tensors.items()}
    save_file(renamed, str(te), metadata={"format": "pt"})
    emit("stage", stage="normalised clip keys", progress=0.99)

def safe_open_keys(path):
    from safetensors import safe_open
    class _Ctx:
        def __enter__(self): self._h = safe_open(str(path), "pt"); return list(self._h.keys())
        def __exit__(self, *a): return False
    return _Ctx()

_normalise_clip_keys(target)

# Tokenizers: save_pretrained writes ONLY the fast tokenizer (tokenizer.json +
# tokenizer_config.json), but mlx_diffuser loads the SLOW CLIPTokenizer, which reads
# vocab.json and merges.txt. With those absent, transformers resolves vocab_file/merges_file
# to None and dies with the deeply unhelpful
#   TypeError: expected str, bytes or os.PathLike object, not NoneType
# Both files are recoverable from tokenizer.json's "model" section: "vocab" is the token->id
# map, "merges" holds ["left","right"] pairs that have to be joined with a space.
def _restore_slow_tokenizer_files(target):
    import json as _json
    for sub in ("tokenizer", "tokenizer_2"):
        folder = target / sub
        src = folder / "tokenizer.json"
        if not src.is_file() or (folder / "vocab.json").is_file():
            continue
        model = _json.loads(src.read_text("utf-8")).get("model") or {}
        vocab, merges = model.get("vocab"), model.get("merges")
        if not vocab or not merges:
            continue
        (folder / "vocab.json").write_text(_json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        lines = [" ".join(m) if isinstance(m, (list, tuple)) else str(m) for m in merges]
        (folder / "merges.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        emit("stage", stage="restored tokenizer files", progress=0.995)

_restore_slow_tokenizer_files(target)

emit("stage", stage="done", progress=1.0)
'''


def convert(
    source: Path,
    dest: Path,
    python: Path | None = None,
    cancel_event: threading.Event | None = None,
    on_progress=None,
    timeout: int = 60 * 60,
) -> dict:
    """Convert `source` into a diffusers directory at `dest`.

    Runs out-of-process. `on_progress(fraction, text)` is called from the reader thread.
    """
    exe = python or Path(os.environ.get("MLX_SDXL_CONVERT_PYTHON") or DEFAULT_PYTHON)
    if not source.is_file():
        raise ConversionError(f"checkpoint not found: {source.name}")
    if not exe.exists():
        raise ConversionError(f"interpreter not found: {exe}")

    with tempfile_dir() as workdir:
        script = Path(workdir) / "convert_sdxl.py"
        script.write_text(_WORKER, encoding="utf-8")
        cancel_path = Path(workdir) / "cancel"
        proc = subprocess.Popen(
            [str(exe), str(script), str(source), str(dest), BASE_MODEL, str(cancel_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        result: dict = {"ok": False, "error": None, "dest": str(dest)}
        started = time.monotonic()
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("{"):
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("kind") == "fatal":
                    result["error"] = payload.get("error") or "conversion failed"
                    break
                if payload.get("kind") == "stage":
                    if on_progress:
                        on_progress(float(payload.get("progress") or 0.0),
                                    str(payload.get("stage") or ""))
            code = proc.wait(timeout=max(30, timeout - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            proc.kill()
            raise ConversionError("conversion timed out")
        finally:
            if proc.poll() is None:
                proc.kill()

        if cancel_event is not None and cancel_event.is_set():
            shutil.rmtree(dest, ignore_errors=True)
            raise ConversionError("cancelled")
        if result["error"]:
            shutil.rmtree(dest, ignore_errors=True)
            raise ConversionError(result["error"])
        if code != 0:
            shutil.rmtree(dest, ignore_errors=True)
            raise ConversionError(f"conversion exited with status {code}")
        if not (dest / "model_index.json").is_file():
            shutil.rmtree(dest, ignore_errors=True)
            raise ConversionError("conversion produced no model_index.json")
        result["ok"] = True
        return result


class tempfile_dir:
    """Tiny scoped temp dir; avoids importing tempfile for one use."""

    def __enter__(self):
        import tempfile

        self._dir = tempfile.TemporaryDirectory(prefix="sdxl-convert-")
        return self._dir.name

    def __exit__(self, *exc):
        self._dir.cleanup()
        return False