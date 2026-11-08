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
    t_start = time.perf_counter()
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

        # -------------------------------------------------------------
        # STAGE 1: Text Encoding (Qwen3-VL-4B Q4)
        # -------------------------------------------------------------
        emit({"phase": "preparing", "detail": "Loading Qwen3-VL-4B text encoder into memory..."})
        t0 = time.perf_counter()
        te_dir = str(model_dir / "text_encoder")
        te = Q4TextEncoder(te_dir, dtype=dtype)
        load_te_time = time.perf_counter() - t0

        context_latents = None
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
        t0 = time.perf_counter()
        tr_dir = str(model_dir / "transformer")
        tr = load_q4_transformer(tr_dir, dtype=dtype)
        load_tr_time = time.perf_counter() - t0

        emit({"phase": "denoising", "detail": f"Denoising {steps} steps at {width}×{height}..."})
        t0 = time.perf_counter()
        sigmas = sigmas_for(steps, width, height)
        latents = seed_noise((1, tr.in_channels, height // 8, width // 8), seed).astype(mx.float32)

        for i, (s, s_next) in enumerate(zip(sigmas[:-1], sigmas[1:]), 1):
            t = mx.array([s], dtype=mx.float32)
            v = tr(latents, t, cond, context=context_latents)
            latents = latents + (s_next - s) * v
            mx.eval(latents)
            emit({"progress": {"step": i, "total": steps}})

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
        vae_dir = str(model_dir / "vae")
        vae = VaeDecoder(vae_dir, dtype=mx.float16)
        img = vae.decode(latents)
        decode_time = round(time.perf_counter() - t0, 2)

        del vae, latents
        gc.collect()
        mx.clear_cache()

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
