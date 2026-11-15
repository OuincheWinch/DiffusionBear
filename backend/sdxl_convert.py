"""Convert a single-file Civitai SDXL checkpoint into a diffusers directory.

Why this exists
---------------
Civitai serves one `.safetensors`. The SDXL engine loads a diffusers *directory*:
`model_index.json`, `unet/`, `text_encoder/`, `text_encoder_2/`, `vae/`,
`tokenizer/`, `tokenizer_2/`, `scheduler/`. A downloaded Civitai checkpoint is
therefore invisible to the engine until it is converted, which is exactly the gap the
Models tab reported as "single-file checkpoint, needs converting".

Why it runs in a subprocess
---------------------------
The conversion needs `torch` plus `diffusers`, and the inference process must never import
them. So this runs out-of-process with a timeout and reports structured progress back over
stdout; a failed or hung conversion cannot take the backend with it, and cancelling is a
process kill rather than a hope.

THE DEPENDENCY
--------------
Conversion of raw single-file SDXL `.safetensors` checkpoints uses external `torch`,
`diffusers`, `omegaconf`, and `accelerate` when available; the packaged v0.5.0+ runtime
drops PyTorch completely (all bundled engines run on pure MLX/safetensors). When those
optional conversion packages are missing, this exits cleanly with a clear, actionable
message rather than pretending to work. That behaviour is tested.
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
# The inference process itself never imports diffusers: only this subprocess does.
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
# 196. mlx_sdxl_engine expects the prefixed form, so the engine rejects the
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
# tokenizer_config.json), but mlx_sdxl_engine loads the SLOW CLIPTokenizer, which reads
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
    purge_source: bool = False,
) -> dict:
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
        if purge_source and source.is_file():
            try:
                source.unlink()
            except OSError:
                pass
        return result


def _load_torch_bin_as_mx(bin_path: Path):
    """Read a zipped PyTorch .bin state dict into {key: mx.array} without torch."""
    import io
    import pickle
    import zipfile
    from collections import OrderedDict

    import mlx.core as mx
    import numpy as np

    _STORAGE_DTYPE = {
        "FloatStorage": (np.float32, 4, False),
        "HalfStorage": (np.float16, 2, False),
        "BFloat16Storage": (np.uint16, 2, True),
        "DoubleStorage": (np.float64, 8, False),
        "IntStorage": (np.int32, 4, False),
        "LongStorage": (np.int64, 8, False),
        "ByteStorage": (np.uint8, 1, False),
        "BoolStorage": (np.bool_, 1, False),
    }

    def _make_storage_cls(name: str):
        return _STORAGE_DTYPE.get(name, (np.float32, 4, False))

    def _rebuild_tensor(storage, offset, size, stride, *_extra):
        buf, np_dtype, is_bf16 = storage
        arr = np.frombuffer(buf, dtype=np_dtype)
        numel = 1
        for dim in size:
            numel *= int(dim)
        if numel == 0:
            out_np = np.empty(size, dtype=np_dtype)
        else:
            elem_size = np.dtype(np_dtype).itemsize
            byte_offset = int(offset) * elem_size
            byte_strides = tuple(int(s) * elem_size for s in stride)
            base_arr = np.frombuffer(buf, dtype=np_dtype, offset=byte_offset)
            out_np = np.lib.stride_tricks.as_strided(
                base_arr, shape=tuple(int(d) for d in size), strides=byte_strides, writeable=False
            ).copy()
        mx_arr = mx.array(out_np)
        if is_bf16:
            mx_arr = mx.view(mx_arr.astype(mx.uint16), mx.bfloat16)
        return mx_arr

    with zipfile.ZipFile(bin_path, "r") as zf:
        names = zf.namelist()
        pkl_name = next((n for n in names if n.endswith("data.pkl")), None)
        if pkl_name is None:
            raise ConversionError(f"Unsupported .bin archive (missing data.pkl): {bin_path.name}")
        prefix = pkl_name[: -len("data.pkl")]
        storage_cache: dict[str, tuple[bytes, object, bool]] = {}

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch._utils" and name in ("_rebuild_tensor", "_rebuild_tensor_v2"):
                    return _rebuild_tensor
                if module == "torch" and name in _STORAGE_DTYPE:
                    return _make_storage_cls(name)
                if module == "collections" and name == "OrderedDict":
                    return OrderedDict
                if module == "torch._utils" and name == "_rebuild_parameter":
                    return lambda data, requires_grad, hooks: data
                raise pickle.UnpicklingError(f"Forbidden global in checkpoint: {module}.{name}")

            def persistent_load(self, saved_id):
                if isinstance(saved_id, tuple) and len(saved_id) >= 3 and saved_id[0] == "storage":
                    storage_type, key = saved_id[1], str(saved_id[2])
                    if key not in storage_cache:
                        np_dtype, _, is_bf16 = storage_type
                        raw = zf.read(f"{prefix}data/{key}")
                        storage_cache[key] = (raw, np_dtype, is_bf16)
                    return storage_cache[key]
                raise pickle.UnpicklingError(f"Unsupported persistent id: {saved_id!r}")

        pkl_bytes = zf.read(pkl_name)
        state = _Unpickler(io.BytesIO(pkl_bytes)).load()

    if not isinstance(state, dict):
        raise ConversionError(f"Unexpected checkpoint payload in {bin_path.name}")
    return {str(k): v for k, v in state.items() if isinstance(v, mx.array)}


def convert_bin_to_safetensors(
    source_dir: Path,
    dest_dir: Path | None = None,
    cancel_event: threading.Event | None = None,
    on_progress=None,
) -> dict:
    """Convert a diffusers directory with .bin weights to .safetensors format (pure MLX).

    Some HF repos (e.g., RunDiffusion/Juggernaut-XI-v11) ship diffusers-format
    models with PyTorch .bin files in unet/text_encoder/vae. This converts those
    weights directly to .safetensors via pure Python + MLX without torch or diffusers.

    If dest_dir is None, converts in-place (source_dir is modified).
    """
    import mlx.core as mx
    from safetensors import safe_open

    source_dir = Path(source_dir)
    if dest_dir is None:
        dest_dir = source_dir
    else:
        dest_dir = Path(dest_dir)
        if dest_dir.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)
        shutil.copytree(source_dir, dest_dir)

    if not (source_dir / "model_index.json").is_file():
        raise ConversionError(f"not a diffusers directory: {source_dir}")

    if on_progress:
        on_progress(0.1, "scanning .bin components")

    components = ("unet", "vae", "text_encoder", "text_encoder_2")
    bin_jobs: list[tuple[Path, Path]] = []
    for comp in components:
        comp_dir = dest_dir / comp
        if not comp_dir.is_dir():
            continue
        if any(comp_dir.glob("*.safetensors")):
            continue
        bin_files = sorted(comp_dir.glob("*.bin"))
        if not bin_files:
            continue
        out_name = "model.safetensors" if comp.startswith("text_encoder") else "diffusion_pytorch_model.safetensors"
        bin_jobs.append((bin_files[0], comp_dir / out_name))

    total = max(1, len(bin_jobs))
    for idx, (bin_file, sf_file) in enumerate(bin_jobs):
        if cancel_event is not None and cancel_event.is_set():
            raise ConversionError("cancelled")
        if on_progress:
            on_progress(0.1 + 0.7 * (idx / total), f"converting {bin_file.parent.name}/{bin_file.name}")
        tensors = _load_torch_bin_as_mx(bin_file)
        part = sf_file.with_name(f".{sf_file.name}.part")
        mx.save_safetensors(str(part), tensors, metadata={"format": "pt"})
        os.replace(part, sf_file)
        del tensors

    def _safe_open_keys(path: Path) -> list[str]:
        with safe_open(str(path), "numpy") as h:
            return list(h.keys())

    # Normalize CLIP keys if text_encoder_2 uses "text_model." and text_encoder does not
    te = dest_dir / "text_encoder" / "model.safetensors"
    te2 = dest_dir / "text_encoder_2" / "model.safetensors"
    if te.is_file() and te2.is_file():
        k1 = _safe_open_keys(te)
        k2 = _safe_open_keys(te2)
        if any(x.startswith("text_model.") for x in k2) and k1 and not any(x.startswith("text_model.") for x in k1):
            tensors = mx.load(str(te))
            renamed = {(k if k.startswith("text_model.") else "text_model." + k): v for k, v in tensors.items()}
            mx.save_safetensors(str(te), renamed, metadata={"format": "pt"})

    # Restore slow tokenizer files
    for sub in ("tokenizer", "tokenizer_2"):
        folder = dest_dir / sub
        src = folder / "tokenizer.json"
        if not src.is_file() or (folder / "vocab.json").is_file():
            continue
        model = json.loads(src.read_text("utf-8")).get("model") or {}
        vocab, merges = model.get("vocab"), model.get("merges")
        if not vocab or not merges:
            continue
        (folder / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        lines = [" ".join(m) if isinstance(m, (list, tuple)) else str(m) for m in merges]
        (folder / "merges.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if on_progress:
        on_progress(1.0, "done")

    return {"ok": True, "dest": str(dest_dir)}


class tempfile_dir:
    """Tiny scoped temp dir; avoids importing tempfile for one use."""

    def __enter__(self):
        import tempfile

        self._dir = tempfile.TemporaryDirectory(prefix="sdxl-convert-")
        return self._dir.name

    def __exit__(self, *exc):
        self._dir.cleanup()
        return False