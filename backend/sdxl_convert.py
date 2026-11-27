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

    preexisting = set(dest.iterdir()) if dest.is_dir() else None

    def discard_output() -> None:
        # Only remove what the conversion wrote: dest may be the user's own folder.
        if preexisting is None:
            shutil.rmtree(dest, ignore_errors=True)
            return
        for entry in dest.iterdir() if dest.is_dir() else ():
            if entry in preexisting:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                try:
                    entry.unlink()
                except OSError:
                    pass

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
            discard_output()
            raise ConversionError("cancelled")
        if result["error"]:
            discard_output()
            raise ConversionError(result["error"])
        if code != 0:
            discard_output()
            raise ConversionError(f"conversion exited with status {code}")
        if not (dest / "model_index.json").is_file():
            discard_output()
            raise ConversionError("conversion produced no model_index.json")
        result["ok"] = True
        if purge_source and source.is_file():
            try:
                source.unlink()
            except OSError:
                pass
        return result


_ST_DTYPES = {
    "FloatStorage": ("F32", "<f4"),
    "HalfStorage": ("F16", "<f2"),
    "BFloat16Storage": ("BF16", "<u2"),
    "DoubleStorage": ("F64", "<f8"),
    "IntStorage": ("I32", "<i4"),
    "LongStorage": ("I64", "<i8"),
    "ByteStorage": ("U8", "u1"),
    "BoolStorage": ("BOOL", "?"),
}


def stream_bin_to_safetensors(bin_path: Path, out_path: Path, cancel_event=None) -> int:
    """Rewrite a zipped PyTorch .bin state dict as .safetensors, one tensor at a time.

    torch stores tensor storages uncompressed, so they are memory-mapped in place
    and written straight to disk: peak RAM is one tensor, not 2-3x the checkpoint
    (Juggernaut XI's 10 GB fp32 UNet used to reach ~29 GB and get killed).
    Returns the number of tensors written.
    """
    import io
    import mmap
    import pickle
    import struct
    import zipfile
    import zlib
    from collections import OrderedDict

    import numpy as np

    bin_path, out_path = Path(bin_path), Path(out_path)
    # No context manager on the mmap: numpy views keep it alive and it is
    # unmapped once the last one is dropped (an explicit close() raises
    # BufferError while a view is still referenced, e.g. by a traceback).
    with open(bin_path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    with zipfile.ZipFile(bin_path) as zf:
        names = zf.namelist()
        pkl_name = next((n for n in names if n.endswith("data.pkl")), None)
        if pkl_name is None:
            raise ConversionError(f"Unsupported .bin archive (missing data.pkl): {bin_path.name}")
        prefix = pkl_name[: -len("data.pkl")]
        storages: dict[str, tuple] = {}

        def _storage(kind: str, key: str):
            if key not in storages:
                info = zf.getinfo(f"{prefix}data/{key}")
                if info.compress_type == zipfile.ZIP_STORED:
                    fn_len, extra_len = struct.unpack("<HH", mm[info.header_offset + 26:info.header_offset + 30])
                    start = info.header_offset + 30 + fn_len + extra_len
                    buf = np.frombuffer(mm, dtype=np.uint8, count=info.file_size, offset=start)
                    crc = 0
                    for pos in range(0, info.file_size, 64 << 20):
                        crc = zlib.crc32(buf[pos:pos + (64 << 20)], crc)
                    if crc != info.CRC:
                        raise ConversionError(f"corrupted tensor data in {bin_path.name} ({info.filename}): CRC mismatch")
                else:
                    buf = np.frombuffer(zf.read(info.filename), dtype=np.uint8)
                storages[key] = (kind, buf)
            return storages[key]

        def _rebuild_tensor(storage, offset, size, stride, *_extra):
            return (storage, int(offset), tuple(int(d) for d in size), tuple(int(s) for s in stride))

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch._utils" and name in ("_rebuild_tensor", "_rebuild_tensor_v2"):
                    return _rebuild_tensor
                if module == "torch" and name in _ST_DTYPES:
                    return name
                if module == "collections" and name == "OrderedDict":
                    return OrderedDict
                if module == "torch._utils" and name == "_rebuild_parameter":
                    return lambda data, requires_grad, hooks: data
                raise pickle.UnpicklingError(f"Forbidden global in checkpoint: {module}.{name}")

            def persistent_load(self, saved_id):
                if isinstance(saved_id, tuple) and len(saved_id) >= 3 and saved_id[0] == "storage":
                    return _storage(saved_id[1], str(saved_id[2]))
                raise pickle.UnpicklingError(f"Unsupported persistent id: {saved_id!r}")

        state = _Unpickler(io.BytesIO(zf.read(pkl_name))).load()
        if not isinstance(state, dict):
            raise ConversionError(f"Unexpected checkpoint payload in {bin_path.name}")
        tensors = [(str(k), v) for k, v in state.items() if isinstance(v, tuple) and len(v) == 4]

        header: dict = {"__metadata__": {"format": "pt"}}
        pos = 0
        for key, ((kind, _), _, size, _) in tensors:
            st_dtype, np_dtype = _ST_DTYPES[kind]
            nbytes = int(np.prod(size, dtype=np.int64)) * np.dtype(np_dtype).itemsize
            header[key] = {"dtype": st_dtype, "shape": list(size), "data_offsets": [pos, pos + nbytes]}
            pos += nbytes
        raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        raw_header += b" " * (-len(raw_header) % 8)

        with open(out_path, "wb") as out:
            out.write(struct.pack("<Q", len(raw_header)))
            out.write(raw_header)
            for i, (key, ((kind, buf), offset, size, stride)) in enumerate(tensors):
                if cancel_event is not None and i % 64 == 0 and cancel_event.is_set():
                    raise ConversionError("cancelled")
                np_dtype = np.dtype(_ST_DTYPES[kind][1])
                numel = int(np.prod(size, dtype=np.int64))
                if numel == 0:
                    continue
                typed = buf.view(np_dtype)
                arr = np.lib.stride_tricks.as_strided(
                    typed[offset:],
                    shape=size,
                    strides=tuple(s * np_dtype.itemsize for s in stride),
                    writeable=False,
                )
                out.write(memoryview(np.ascontiguousarray(arr)).cast("B"))
                del arr, typed
        del state, tensors
        storages.clear()
    return len(header) - 1


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
        part = sf_file.with_name(f".{sf_file.name}.part")
        # The .safetensors copy is the same size as the .bin; check before
        # writing instead of filling the disk with a partial file.
        need = bin_file.stat().st_size + (512 << 20)
        free = shutil.disk_usage(sf_file.parent).free
        if free < need:
            raise ConversionError(
                f"not enough disk space to convert {bin_file.parent.name}: "
                f"need {need >> 30} GB free, have {free >> 30} GB"
            )
        try:
            stream_bin_to_safetensors(bin_file, part, cancel_event=cancel_event)
            os.replace(part, sf_file)
        finally:
            part.unlink(missing_ok=True)
        if dest_dir == source_dir:
            # In place, the .bin is now redundant: drop it so the next
            # component has room and the install does not keep both copies.
            bin_file.unlink(missing_ok=True)

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
            import mlx.core as mx

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