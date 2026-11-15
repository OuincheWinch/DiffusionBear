#!/usr/bin/env python3
"""
FIBO 2.0 Edit Turbo Engine for DiffusionBear.

Runs under venv-fibo (Python >= 3.10, MLX >= 0.32, mlx-vlm). Invoked by the main
backend as a daemon subprocess (--serve) or per-job with a JSON request on stdin/argv[1].
Outputs newline-delimited JSON messages (phase, progress, success/error) on stdout.

Memory Safety:
  - Phase 1: 4-bit Qwen3-VL-4B text encoder (~2.8 GB VRAM) -> extracted and evicted.
  - Phase 2: 4-bit FIBO 2.0 DiT Transformer (~4.2 GB VRAM) -> denoised and evicted.
  - Phase 3: FLUX.2 AutoencoderKL VAE (~189 MB VRAM) -> decoded to RGB PNG.
  - Peak resident memory stays strictly < 6.8 GB on 16 GB Apple Silicon Macs.
"""

import argparse
import gc
import hashlib
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

# Add backend directory to path
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import mlx.core as mx
import mlx.nn as nn
from PIL import Image, ImageFilter

from engine_metrics import EngineMetrics
from prompt_embedding_cache import PromptEmbeddingCache
from fibo2_mlx import (
    Fibo2Transformer,
    VaeDecoder,
    VaeEncoder,
    load_encoder,
    TextEncoder,
    sigmas_for,
    seed_noise,
    generation_prompt,
    grey_out,
)

_ipc_stdout_lock = threading.Lock()
_LAST_METRICS = None


def emit(payload: dict):
    """Write an atomic JSON line to stdout for the parent process."""
    line = json.dumps(payload) + "\n"
    with _ipc_stdout_lock:
        try:
            sys.stdout.write(line)
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            sys.exit(0)


class Q4TextEncoder(TextEncoder):
    """4-bit quantized Qwen3-VL-4B-Instruct text encoder."""

    def __init__(self, folder, dtype=mx.bfloat16):
        import mlx_vlm
        self.model, self.processor = mlx_vlm.load(folder)
        self.tokenizer = self.processor.tokenizer
        self.images = self.processor.image_processor
        with open(os.path.join(folder, "config.json")) as f:
            cfg = json.load(f)
        self.merge = int(cfg.get("vision_config", {}).get("spatial_merge_size", 2))
        self.image_token_id = int(getattr(self.model.config, "image_token_index", 151655))
        vocab = self.tokenizer.get_vocab()
        self.stop_ids = {vocab[t] for t in ("<|im_end|>", "<|endoftext|>") if t in vocab}
        self.dtype = dtype
        self._fibo_cache_model_path = str(folder)


# Private references so test monkeypatching of the module-level names cannot
# redirect the wrapper installation at a mock object.
_FIBO_TEXT_ENCODER_CLASS = TextEncoder
_FIBO_Q4_TEXT_ENCODER_CLASS = Q4TextEncoder


# ---------------------------------------------------------------------------
# Prompt-embedding disk cache
#
# The FIBO text encoder is a 4-bit Qwen3-VL-4B that is loaded, run and evicted
# before the transformer loads. Its output depends only on the prompt, the
# reference images, the tokenizer and any text-encoder adapter, so it can be
# persisted across engine processes. Only the embedding array is written --
# never VAE latents and never the prompt text, which is a hash preimage only.
# Every failure path degrades to a real encode and never raises into
# generation.
# ---------------------------------------------------------------------------

_FIBO_CACHE_ENGINE = "fibo2"
# Bumped by hand, not derived. ANY change to fibo2_mlx's PROMPT_TEMPLATE,
# TEXT_ENCODER_OUT_LAYERS, get_edit_prompt, generation_prompt,
# mark_text_images or VISION_MARKER changes the embedding produced for the
# same prompt, so this salt must be bumped to invalidate old cache entries.
_FIBO_TE_SCHEMA = "fibo2-te-v1"
_PROMPT_STORE = None
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.json",
)


def _prompt_store():
    global _PROMPT_STORE
    if _PROMPT_STORE is None:
        _PROMPT_STORE = PromptEmbeddingCache()
    return _PROMPT_STORE


def _json_scalar(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    return None


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_fingerprint(encoder):
    """Stable identity of the tokenizer state that produces the token ids.

    The tokenizer's own config files are hashed when its ``name_or_path`` is a
    directory; attributes cover in-memory tokenizers and fake encoders. The
    result is cached on the encoder for its lifetime because files do not
    change under a running engine.
    """
    explicit = getattr(encoder, "_fibo_tokenizer_fingerprint", None)
    if explicit is not None:
        return explicit

    tokenizer = getattr(encoder, "tokenizer", None)
    fingerprint = {
        "class": type(tokenizer).__name__ if tokenizer is not None else None,
        "files": [],
    }
    for attr in (
        "name_or_path",
        "vocab_size",
        "model_max_length",
        "padding_side",
        "truncation_side",
        "clean_up_tokenization_spaces",
    ):
        value = _json_scalar(getattr(tokenizer, attr, None))
        if value is not None:
            fingerprint[attr] = value

    root = getattr(tokenizer, "name_or_path", None)
    if isinstance(root, Path):
        root = str(root)
    if isinstance(root, str) and root and os.path.isdir(root):
        for name in _TOKENIZER_FILES:
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            try:
                fingerprint["files"].append([name, _file_sha256(path)])
            except OSError:
                fingerprint["files"].append([name, None])

    try:
        encoder._fibo_tokenizer_fingerprint = fingerprint
    except Exception:
        pass
    return fingerprint


def _image_digest(image):
    """Content identity of one reference image: size plus RGB pixel hash.

    Pixel content -- not the file path -- is what the encoder sees, and the
    mask path rewrites the pixels before encoding. Any failure returns None so
    the caller skips caching rather than risking a wrong hit.
    """
    try:
        rgb = image.convert("RGB")
        return [
            int(rgb.size[0]),
            int(rgb.size[1]),
            hashlib.sha256(rgb.tobytes()).hexdigest(),
        ]
    except Exception:
        return None


def _reference_signature(images):
    if not images:
        return []
    if not isinstance(images, (list, tuple)):
        images = [images]
    signature = []
    for image in list(images)[:4]:
        digest = _image_digest(image)
        if digest is None:
            return None
        signature.append(digest)
    return signature


def _text_image_signature(images):
    if not images:
        return []
    if not isinstance(images, (list, tuple)):
        images = [images]
    signature = []
    for image in images:
        digest = _image_digest(image)
        if digest is None:
            return None
        signature.append(digest)
    return signature


def _model_path(encoder):
    for attr in ("_fibo_cache_model_path", "model_path", "folder", "model_dir"):
        value = getattr(encoder, attr, None)
        if value:
            return str(value)
    tokenizer = getattr(encoder, "tokenizer", None)
    value = _json_scalar(getattr(tokenizer, "name_or_path", None))
    return value or "unknown"


def _model_fingerprint(encoder):
    """Model path plus size/mtime of the text-encoder files.

    Hashing multi-GB shards on every generation would cost more than the cache
    saves, so the text-encoder directory is pinned by file name, size and
    mtime_ns: an in-place re-download or re-quantisation changes at least one
    of them and invalidates every entry. Tokenizer file contents are hashed
    separately; see ``_tokenizer_fingerprint``.
    """
    model_path = _model_path(encoder)
    files = []
    if model_path != "unknown":
        root = Path(model_path)
        te_dir = root / "text_encoder"
        if not te_dir.is_dir():
            te_dir = root
        try:
            candidates = [te_dir / "config.json"]
            candidates.extend(sorted(te_dir.glob("*.safetensors")))
            for path in candidates:
                try:
                    stat = path.stat()
                except OSError:
                    continue
                files.append(
                    [
                        f"text_encoder/{path.name}",
                        int(stat.st_size),
                        int(stat.st_mtime_ns),
                    ]
                )
        except OSError:
            files = []
    return {"path": model_path, "te_files": files}


# The only derived artifact this engine persists is the text-encoder embedding.
# Transformer, VAE and adapter weights are read in place and never copied into
# a cache.
DERIVED_ARTIFACT_KINDS = ("text-embedding",)


def model_fingerprint(encoder):
    """Public deterministic fingerprint: model path plus TE file size/mtime."""
    return _model_fingerprint(encoder)


def _file_signature(path):
    try:
        stat = Path(path).stat()
        return [str(path), int(stat.st_size), int(stat.st_mtime_ns)]
    except Exception:
        return [str(path), None, None]


def _adapter_signature(entries):
    """Path, size, mtime and scale for every adapter given to the engine."""
    signature = []
    for entry in entries or []:
        try:
            if isinstance(entry, dict):
                path = entry.get("path") or entry.get("name") or ""
                scale = entry.get("scale", 1.0)
            else:
                path, scale = entry, 1.0
            if not path:
                continue
            signature.append([_file_signature(path), round(float(scale), 4)])
        except Exception:
            continue
    return signature


def _cache_key(encoder, prompt, images=None, text_images=None):
    """SHA-256 over every input that can change the text-encoder output."""
    references = _reference_signature(images)
    if references is None:
        return None
    text_references = _text_image_signature(text_images)
    if text_references is None:
        return None
    try:
        return PromptEmbeddingCache.key(
            _FIBO_CACHE_ENGINE,
            _model_fingerprint(encoder),
            prompt,
            negative_prompt="",
            clip_skip=0,
            extra={
                "reference": references,
                "text_images": text_references,
                "tokenizer": _tokenizer_fingerprint(encoder),
                "adapters": _adapter_signature(
                    getattr(encoder, "_fibo_cache_adapters", [])
                ),
                "te_schema": _FIBO_TE_SCHEMA,
            },
        )
    except Exception:
        return None


def _array_to_payload(array):
    """Serialise one MLX array losslessly, including bfloat16.

    numpy has no bfloat16 buffer format, so the raw bytes are stored as uint8
    and a small JSON tensor carries dtype and shape. Evaluating first is
    mandatory: converting an un-evaluated MLX array to numpy can abort the
    whole process.
    """
    import numpy as np

    if not isinstance(array, mx.array):
        return None
    mx.eval(array)
    raw = mx.contiguous(array).view(mx.uint8)
    mx.eval(raw)
    meta = {
        "dtype": str(array.dtype).rsplit(".", 1)[-1],
        "shape": [int(n) for n in array.shape],
    }
    return {
        "data": np.ascontiguousarray(np.asarray(raw)),
        "meta": np.frombuffer(
            json.dumps(meta, separators=(",", ":")).encode("utf-8"),
            dtype=np.uint8,
        ),
    }


def _payload_to_array(data):
    """Rebuild the cached MLX array; None for anything unreadable."""
    import numpy as np

    try:
        meta = json.loads(
            np.asarray(data["meta"], dtype=np.uint8).tobytes().decode("utf-8")
        )
        dtype = getattr(mx, str(meta["dtype"]).rsplit(".", 1)[-1])
        shape = tuple(int(n) for n in meta["shape"])
        raw = np.asarray(data["data"], dtype=np.uint8)
        return mx.array(raw).view(dtype).reshape(shape)
    except Exception:
        return None


def _call_original(encoder, original, prompt, images, text_images):
    if getattr(original, "__self__", None) is not None:
        return original(prompt, images=images, text_images=text_images)
    return original(encoder, prompt, images=images, text_images=text_images)


def _encode_with_cache(encoder, original, prompt, images=None, text_images=None):
    try:
        store = _prompt_store()
        disabled = bool(getattr(store, "disabled", False))
    except Exception:
        return _call_original(encoder, original, prompt, images, text_images)
    if disabled:
        return _call_original(encoder, original, prompt, images, text_images)

    key = _cache_key(encoder, prompt, images, text_images)
    if key is not None:
        try:
            data = store.get(key)
        except Exception:
            data = None
        if data:
            cached = _payload_to_array(data)
            if cached is not None:
                return cached

    result = _call_original(encoder, original, prompt, images, text_images)
    if key is not None and isinstance(result, mx.array):
        try:
            payload = _array_to_payload(result)
            if payload is not None:
                store.put(key, payload)
        except Exception:
            pass
    return result


def _install_prompt_cache_wrapper(cls, original):
    if cls.__dict__.get("_fibo_prompt_cache_wrapper", False):
        return

    def encode(self, prompt, images=None, text_images=None):
        source = getattr(self, "_fibo_original_encode", None)
        if source is None:
            source = original
        return _encode_with_cache(self, source, prompt, images, text_images)

    encode.__name__ = "encode"
    encode.__qualname__ = f"{cls.__name__}.encode"
    encode._fibo_cache_wrapper = True
    cls._fibo_original_encode = original
    cls.encode = encode
    cls._fibo_prompt_cache_wrapper = True


def _install_prompt_cache_wrappers():
    """Patch both text-encoder classes once, from the engine entrypoint.

    Deliberately not run at import time: importing fibo_engine (as the test
    suite does) must not globally modify vendor classes. ``run_job`` calls this
    before constructing an encoder; repeated calls are no-ops.
    """
    text_cls = _FIBO_TEXT_ENCODER_CLASS
    q4_cls = _FIBO_Q4_TEXT_ENCODER_CLASS
    q4_original = q4_cls.__dict__.get("_fibo_original_encode")
    if q4_original is None:
        q4_original = getattr(text_cls, "_fibo_original_encode", q4_cls.encode)
    if not text_cls.__dict__.get("_fibo_prompt_cache_wrapper", False):
        _install_prompt_cache_wrapper(text_cls, text_cls.encode)
    if not q4_cls.__dict__.get("_fibo_prompt_cache_wrapper", False):
        _install_prompt_cache_wrapper(q4_cls, q4_original)


def load_q4_transformer(folder, dtype=mx.bfloat16):
    """Load the 4-bit quantized FIBO 2.0 Transformer."""
    with open(os.path.join(folder, "config.json")) as f:
        cfg = json.load(f)

    model = Fibo2Transformer(cfg)

    def should_quantize(path, mod):
        if not isinstance(mod, nn.Linear):
            return False
        if any(k in path for k in ["embed", "norm", "final_layer.adaLN_modulation.1"]):
            return False
        return mod.weight.shape[1] % 64 == 0

    nn.quantize(model, group_size=64, bits=4, class_predicate=should_quantize)

    import glob
    weights = {}
    for p in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
        weights.update(mx.load(p))

    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def _host_schedule(sigmas):
    """Build the per-step timesteps and sigma differences once, on the host.

    ``sigmas_for`` already returns plain Python floats. Converting them to
    arrays and computing the differences before the loop keeps every sigma
    read and every allocation out of the denoising hot path.
    """
    timesteps = [mx.array([float(s)], dtype=mx.float32) for s in sigmas[:-1]]
    deltas = [float(s_next) - float(s) for s, s_next in zip(sigmas[:-1], sigmas[1:])]
    return timesteps, deltas


def _denoise_loop(tr, latents, cond, context_latents, timesteps, deltas, steps,
                  emit_fn=None):
    """One transformer call and exactly one ``mx.eval`` per step.

    The initial latents may be lazy; the first step folds them into the graph
    and evaluates them with that step. No cache clear or phase boundary belongs
    inside the loop; eviction happens after the loop returns.
    """
    progress = emit if emit_fn is None else emit_fn
    for i, (timestep, delta) in enumerate(zip(timesteps, deltas), 1):
        velocity = tr(latents, timestep, cond, context=context_latents)
        latents = latents + delta * velocity
        mx.eval(latents)
        progress({"progress": {"step": i, "total": steps}})
    return latents


def resolve_fibo2_model_dir(req: dict) -> Path:
    """Find the valid FIBO 2.0 4-bit directory."""
    if req.get("model_path"):
        p = Path(req["model_path"])
        if (p / "transformer").is_dir():
            return p

    if req.get("model_dir"):
        p = Path(req["model_dir"])
        if (p / "transformer").is_dir():
            return p

    # Check DiffusionBear standard models directory
    app_asset_dir = os.environ.get("MLX_DIFFUSION_ASSET_DIR") or os.environ.get("DIFFUSIONBEAR_ASSET_DIR")
    if app_asset_dir:
        cand1 = Path(app_asset_dir) / "models" / "fibo2-turbo"
        if (cand1 / "transformer").is_dir():
            return cand1
        cand2 = Path(app_asset_dir) / "models" / "FIBO2-Edit-Turbo-MLX-4bit"
        if (cand2 / "transformer").is_dir():
            return cand2

    # Check local repo workspace development directory
    repo_root = Path(__file__).resolve().parent.parent
    local_dev = repo_root / "tools" / "fibo2" / "output"
    if (local_dev / "transformer").is_dir():
        return local_dev

    raise FileNotFoundError(
        "FIBO 2.0 Edit Turbo 4-bit weights not found. Please download "
        "'Ouinche/FIBO2-Edit-Turbo-MLX-4bit' via the Models tab."
    )


def run_job(req: dict) -> int:
    global _LAST_METRICS
    t_start = time.perf_counter()
    _install_prompt_cache_wrappers()
    try:
        prompt = req.get("prompt", "")
        width = int(req.get("width", 1024)) // 16 * 16
        height = int(req.get("height", 1024)) // 16 * 16
        steps = int(req.get("steps", 4))
        seed = int(req.get("seed", 4242))
        dest = Path(req["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)

        ref_paths = req.get("ref_paths") or req.get("reference_images") or []
        if isinstance(ref_paths, str):
            ref_paths = [ref_paths]
        ref_paths = [str(p) for p in ref_paths if p and os.path.exists(str(p))][:4]

        ref_images = []
        for p in ref_paths:
            try:
                ref_images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                emit({"phase": "warning", "detail": f"Failed to load reference {p}: {e}"})

        original_ref = ref_images[0].copy() if ref_images else None
        mask_path = req.get("mask_path") or req.get("mask")
        mask_im = None
        if mask_path and os.path.exists(str(mask_path)):
            try:
                mask_im = Image.open(str(mask_path)).convert("L")
            except Exception as e:
                emit({"phase": "warning", "detail": f"Failed to load mask {mask_path}: {e}"})

        if mask_im is not None and ref_images:
            emit({"phase": "preparing", "detail": "Applying 100% hard mid-grey inpainting mask to reference image..."})
            ref_images[0] = grey_out(ref_images[0], mask_im)

        model_dir = resolve_fibo2_model_dir(req)
        dtype = mx.bfloat16

        metrics = EngineMetrics(engine="fibo2", model=model_dir.name)
        _LAST_METRICS = metrics

        # -------------------------------------------------------------
        # STAGE 1: Text Encoding (Qwen3-VL-4B Q4)
        # -------------------------------------------------------------
        emit({"phase": "preparing", "detail": "Loading Qwen3-VL-4B text encoder into memory..."})
        with metrics.phase("load"):
            t0 = time.perf_counter()
            te_dir = str(model_dir / "text_encoder")
            te = Q4TextEncoder(te_dir, dtype=dtype)
            te._fibo_cache_model_path = str(model_dir)
            te._fibo_cache_adapters = req.get("loras") or req.get("adapters") or []
            load_te_time = time.perf_counter() - t0

        context_latents = None
        with metrics.phase("conditioning"):
            if ref_images:
                emit({"phase": "preparing", "detail": f"Encoding prompt with {len(ref_images)} reference image(s)..."})
                cond = te.encode(prompt, images=ref_images)
                mx.eval(cond)

                # Evict text encoder before loading VAE encoder
                del te
                gc.collect()
                mx.clear_cache()

                emit({"phase": "preparing", "detail": f"Encoding {len(ref_images)} reference image latents with VAE..."})
                vae_enc = load_encoder(model_dir, dtype=dtype)
                context_latents = [vae_enc.encode(im, width, height) for im in ref_images]
                for cl in context_latents:
                    mx.eval(cl)
                del vae_enc
                gc.collect()
                mx.clear_cache()
            else:
                emit({"phase": "preparing", "detail": "Encoding structured prompt conditioning..."})
                gen_text = generation_prompt(prompt)
                cond = te.encode(gen_text)
                mx.eval(cond)
                del te
                gc.collect()
                mx.clear_cache()

        # -------------------------------------------------------------
        # STAGE 2: Denoising Loop (FIBO 2.0 DiT Transformer Q4)
        # -------------------------------------------------------------
        emit({"phase": "loading_model", "detail": "Loading FIBO 2.0 DiT Transformer (4.2 GB 4-bit)..."})
        with metrics.phase("load"):
            t0 = time.perf_counter()
            tr_dir = str(model_dir / "transformer")
            tr = load_q4_transformer(tr_dir, dtype=dtype)
            load_tr_time = time.perf_counter() - t0

        emit({"phase": "denoising", "detail": f"Denoising {steps} steps at {width}×{height}..."})
        t0 = time.perf_counter()
        with metrics.phase("denoise"):
            sigmas = sigmas_for(steps, width, height)
            timesteps, deltas = _host_schedule(sigmas)
            latents = seed_noise((1, tr.in_channels, height // 8, width // 8), seed).astype(mx.float32)
            latents = _denoise_loop(
                tr, latents, cond, context_latents, timesteps, deltas, steps)

        denoise_time = round(time.perf_counter() - t0, 2)

        # Evict transformer immediately
        del tr, cond, context_latents
        gc.collect()
        mx.clear_cache()

        # -------------------------------------------------------------
        # STAGE 3: Latent Decoding (FLUX.2 VAE)
        # -------------------------------------------------------------
        emit({"phase": "decoding_vae", "detail": "Decoding latents with FLUX.2 VAE..."})
        t0 = time.perf_counter()
        with metrics.phase("decode"):
            vae_dir = str(model_dir / "vae")
            vae = VaeDecoder(vae_dir, dtype=mx.float16)
            img = vae.decode(latents)
        decode_time = round(time.perf_counter() - t0, 2)

        del vae, latents
        gc.collect()
        mx.clear_cache()

        with metrics.phase("save"):
            if req.get("composite", True) and mask_im is not None and original_ref is not None:
                bin_m = mask_im.resize(img.size, Image.Resampling.BILINEAR).convert("L").point(lambda v: 255 if v >= 128 else 0)
                orig_resized = original_ref.resize(img.size, Image.Resampling.LANCZOS)
                feathered = bin_m.filter(ImageFilter.GaussianBlur(radius=1.5))
                img = Image.composite(img, orig_resized, feathered)

            # Save output image
            img.save(dest)
        total_time = round(time.perf_counter() - t_start, 2)

        emit({
            "status": "success",
            "generation_time": total_time,
            "denoise_time": denoise_time,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "dest_size": [width, height],
            "model_path": str(model_dir),
            "reference_images": [Path(p).name for p in ref_paths],
        })
        return 0

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        emit({"error": f"{type(e).__name__}: {e}"})
        return 1


def _parent_watchdog(parent_pid: int):
    """Exit immediately if parent process dies."""
    while True:
        try:
            os.kill(parent_pid, 0)
        except OSError:
            sys.exit(0)
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description="DiffusionBear FIBO 2.0 Subprocess Engine")
    parser.add_argument("--serve", action="store_true", help="Run as long-lived daemon reading JSON on stdin")
    parser.add_argument("--parent-pid", type=int, help="Parent process ID for heartbeat watchdog")
    parser.add_argument("payload", nargs="?", help="JSON payload for single-job execution")
    args = parser.parse_args()

    if args.parent_pid:
        t = threading.Thread(target=_parent_watchdog, args=(args.parent_pid,), daemon=True)
        t.start()

    if args.serve:
        emit({"status": "ready"})
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as err:
                emit({"error": f"Invalid JSON request: {err}"})
                continue
            run_job(req)
        return 0

    if args.payload:
        try:
            req = json.loads(args.payload)
        except json.JSONDecodeError as err:
            emit({"error": f"Invalid JSON payload: {err}"})
            return 1
        return run_job(req)

    # Read single job from stdin
    raw = sys.stdin.read().strip()
    if raw:
        req = json.loads(raw)
        return run_job(req)

    emit({"error": "No payload provided"})
    return 1


if __name__ == "__main__":
    sys.exit(main())
