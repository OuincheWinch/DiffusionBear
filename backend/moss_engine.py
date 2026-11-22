"""Pure-MLX MOSS-Image engine.

Three physical stages, one component tree resident at a time:
    TE -> embeddings -> evict TE
    DiT -> latents -> evict DiT
    VAE -> pixels -> evict VAE

No torch, no diffusers, no MPS. The Q4 checkpoint is the mflux-saved
community pack; the text encoder is the local MLX q8-g64 Qwen 3.5 sibling.
"""
from __future__ import annotations

import gc
import json
import os
import signal
import sys
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any

import mlx.core as mx

import hardware_limits

hardware_limits.apply_metal_cache_limit()
from PIL import Image

from moss_mlx import MOSSImageLatentCreator, MOSSImageScheduler
from moss_mlx.loader import (
    encode_moss_prompt,
    load_moss_text_encoder,
    load_moss_transformer,
    release_text_encoder,
)
from moss_mlx.vae import load_moss_vae, release_vae
from moss_mlx.lora import apply_moss_lora, apply_dmd_time_modules

_stdout_lock = threading.Lock()
try:
    import app_settings as _app_settings
    _DEFAULT_ASSET_DIR = Path(_app_settings.ASSET_DIR) / "models"
except Exception:
    _DEFAULT_ASSET_DIR = Path(__file__).resolve().parent / "data" / "models"

_ASSET_DIR = Path(
    os.environ.get("MOSS_MLX_ASSET_DIR") or _DEFAULT_ASSET_DIR
).expanduser()

_MOSS_PROMPT_CACHE: OrderedDict[tuple, tuple[mx.array, mx.array | None]] = OrderedDict()
_MOSS_PROMPT_CACHE_MAX = 8
_MOSS_CACHED_TRANSFORMER: tuple[tuple, Any] | None = None


@mx.compile
def _compiled_cfg(uncond: mx.array, pred: mx.array, guidance: float) -> mx.array:
    return uncond + guidance * (pred - uncond)


@mx.compile
def _compiled_dmd_step(z: mx.array, q: mx.array, x0: mx.array) -> mx.array:
    return z + mx.einsum("oc,bchw->bohw", q, x0)


def emit(payload: dict) -> None:
    line = json.dumps(payload, separators=(",", ":")) + "\n"
    with _stdout_lock:
        sys.stdout.write(line)
        sys.stdout.flush()


def _clear():
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def _resolve_q4(model_path: str) -> Path:
    requested = Path(model_path).expanduser()
    lower = str(model_path).lower()
    fallback = _ASSET_DIR / ("kromoss" if "kromoss" in lower else "moss-image-sft")
    candidates = [requested, fallback]
    for candidate in candidates:
        if (
            (candidate / "mflux_model.json").is_file()
            or (
                (candidate / "transformer").is_dir()
                and any((candidate / "transformer").glob("*.safetensors"))
            )
        ) and (candidate / "vae").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"MOSS pure-MLX requires the MLX/Diffusers model pack ({fallback.name}); "
        "no transformer/VAE layout found at " + ", ".join(str(c) for c in candidates)
    )


def _validate(req: dict) -> dict:
    prompt = str(req.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    dest = req.get("dest")
    if not dest:
        raise ValueError("dest is required")
    model_path = req.get("model_path")
    if not model_path:
        raise ValueError("model_path is required")
    width = int(req.get("width", 1024))
    height = int(req.get("height", 1024))
    if width % 16 or height % 16:
        raise ValueError("width and height must be multiples of 16")
    steps = int(req.get("steps", 50))
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return {
        "prompt": prompt,
        "dest": dest,
        "model_path": model_path,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": float(req.get("guidance", 3.0)),
        "seed": req.get("seed"),
        "negative_prompt": str(req.get("negative_prompt") or "").strip(),
        "adapter_dir": str(req.get("adapter_dir") or "").strip(),
        "loras": list(req.get("loras") or []),
    }


def _run(req: dict, keep_alive: bool = False) -> int:
    global _MOSS_CACHED_TRANSFORMER
    hardware_limits.apply_metal_cache_limit()
    started = time.time()
    try:
        req = _validate(req)
        prompt = req["prompt"]
        negative_prompt = req["negative_prompt"]
        width, height = req["width"], req["height"]
        steps, guidance = req["steps"], req["guidance"]
        seed = req["seed"] if req["seed"] is not None else int(time.time())
        root = _resolve_q4(req["model_path"])
        adapter_dir = req["adapter_dir"]
        loras = req["loras"]
        distilled = bool(adapter_dir)
        do_cfg = guidance != 1.0 and not distilled
        dest = Path(req["dest"])
        load_time = 0.0
        use_cache = bool(keep_alive or req.get("use_cache"))

        prompt_key = (str(root), prompt, negative_prompt, bool(do_cfg))
        cached_prompt = _MOSS_PROMPT_CACHE.get(prompt_key) if use_cache else None
        if cached_prompt is not None:
            _MOSS_PROMPT_CACHE.move_to_end(prompt_key)
            positive, negative = cached_prompt
            emit({"phase": "encoding_prompt", "detail": "Stage 1/3: reusing cached prompt embeddings."})
        else:
            if _MOSS_CACHED_TRANSFORMER is not None:
                # Evict resident DiT before loading the 9.3 GB Q8 text encoder.
                _MOSS_CACHED_TRANSFORMER = None
                _clear()
            emit({"phase": "loading_text_encoder", "detail": "Stage 1/3: loading MLX Qwen 3.5 text encoder..."})
            t0 = time.perf_counter()
            text_encoder, tokenizer = load_moss_text_encoder(root)
            load_time += time.perf_counter() - t0
            emit({"phase": "encoding_prompt", "detail": "Stage 1/3: encoding prompt embeddings..."})
            positive = encode_moss_prompt(text_encoder, tokenizer, prompt)
            negative = encode_moss_prompt(text_encoder, tokenizer, negative_prompt) if do_cfg else None
            mx.eval(positive)
            if negative is not None:
                mx.eval(negative)
            release_text_encoder(text_encoder)
            text_encoder = None
            del tokenizer
            _clear()
            emit({"phase": "evicting_text_encoder", "detail": "Stage 1/3: text encoder evicted."})
            if use_cache:
                _MOSS_PROMPT_CACHE[prompt_key] = (positive, negative)
                _MOSS_PROMPT_CACHE.move_to_end(prompt_key)
                while len(_MOSS_PROMPT_CACHE) > _MOSS_PROMPT_CACHE_MAX:
                    _MOSS_PROMPT_CACHE.popitem(last=False)

        def _lora_cache_sig(entry: dict) -> tuple[str, float, int, int]:
            path = Path(str(entry.get("path", "")))
            try:
                st = path.stat()
                return (
                    str(path),
                    float(entry.get("scale", 1.0)),
                    int(st.st_size),
                    int(st.st_mtime_ns),
                )
            except OSError:
                return (str(path), float(entry.get("scale", 1.0)), -1, -1)

        dit_key = (
            str(root),
            str(adapter_dir),
            tuple(_lora_cache_sig(e) for e in loras),
        )
        transformer = None
        if keep_alive and _MOSS_CACHED_TRANSFORMER is not None and _MOSS_CACHED_TRANSFORMER[0] == dit_key:
            transformer = _MOSS_CACHED_TRANSFORMER[1]
            emit({"phase": "loading_transformer", "detail": "Stage 2/3: reusing warm MLX MOSS DiT (Q4)."})
        else:
            if _MOSS_CACHED_TRANSFORMER is not None:
                _MOSS_CACHED_TRANSFORMER = None
                _clear()
            emit({"phase": "loading_transformer", "detail": "Stage 2/3: loading MLX MOSS DiT (Q4)..."})
            t0 = time.perf_counter()
            transformer = load_moss_transformer(root, quantize=4)
            load_time += time.perf_counter() - t0
            if distilled:
                apply_dmd_time_modules(transformer, adapter_dir)
                lora_file = Path(adapter_dir) / "lora.safetensors"
                if not lora_file.is_file():
                    lora_file = Path(adapter_dir) / "lora_latest.safetensors"
                applied = apply_moss_lora(transformer, lora_file, 1.0)
                if applied <= 0:
                    raise RuntimeError(f"DMD adapter applied no LoRA layers: {lora_file}")
                adapter_lora_names = {lora_file.name, "lora.safetensors", "lora_latest.safetensors"}
                for entry in loras:
                    path = Path(str(entry.get("path", "")))
                    if path.name in adapter_lora_names and path.parent == Path(adapter_dir):
                        continue
                    if not path.is_file():
                        raise FileNotFoundError(f"MOSS LoRA file not found: {path}")
                    apply_moss_lora(transformer, path, float(entry.get("scale", 1.0)))
            elif loras:
                for entry in loras:
                    apply_moss_lora(transformer, str(entry["path"]), float(entry.get("scale", 1.0)))
            if keep_alive:
                _MOSS_CACHED_TRANSFORMER = (dit_key, transformer)
        scheduler = MOSSImageScheduler()
        if distilled:
            ds = mx.load(str(Path(adapter_dir) / "dataspace.safetensors"))
            q = ds["latent_mix.Q"].astype(mx.float32)
            codes = [float(v) for v in ds["step_codes"].reshape(-1).tolist()]
            steps = min(steps, len(codes))
            latents = MOSSImageLatentCreator.create_latents(1, height, width, int(seed)).astype(mx.float32)
            z = latents
            x0 = mx.zeros_like(z)
            emit({"phase": "denoising", "detail": f"Stage 2/3: DMD denoising {steps} steps..."})
            for index, code in enumerate(codes[:steps]):
                x = _compiled_dmd_step(z, q, x0)
                x0 = z - transformer(x, code, positive)
                mx.eval(x0)
                emit({"progress": {"step": index + 1, "steps": steps}})
            latents = x0
            del z, x0
            sampler_label = "moss_dataspace_dmd"
        else:
            timesteps = scheduler.set_timesteps(steps)
            latents = MOSSImageLatentCreator.create_latents(1, height, width, int(seed))
            emit({"phase": "denoising", "detail": f"Stage 2/3: denoising {len(timesteps)} steps..."})
            for index, timestep in enumerate(timesteps):
                pred = transformer(latents, timestep, positive)
                if do_cfg:
                    uncond = transformer(latents, timestep, negative, skip_layers=(22,))
                    pred = _compiled_cfg(uncond, pred, guidance)
                latents = scheduler.step(pred, index, latents)
                mx.eval(latents)
                emit({"progress": {"step": index + 1, "steps": len(timesteps)}})
            sampler_label = "moss_flow_match"
        del positive, negative, scheduler
        if not keep_alive:
            del transformer
            transformer = None
            _clear()
            emit({"phase": "evicting_transformer", "detail": "Stage 2/3: DiT evicted."})
        else:
            transformer = None
            _clear()

        emit({"phase": "loading_vae", "detail": "Stage 3/3: loading MLX FLUX.2 VAE..."})
        t0 = time.perf_counter()
        vae = load_moss_vae(root)
        load_time += time.perf_counter() - t0
        decoded = vae.decode_packed_latents(latents)
        mx.eval(decoded)
        image = _to_pil(decoded)
        del vae, decoded, latents
        vae = None
        _clear()
        emit({"phase": "evicting_vae", "detail": "Stage 3/3: VAE evicted."})

        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(f".{dest.name}.part{os.getpid()}")
        image.save(part, format="PNG")
        os.replace(part, dest)
        size = dest.stat().st_size
        emit({
            "status": "success",
            "generation_time": round(time.time() - started, 2),
            "load_time": round(load_time, 2),
            "dest_size": size,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "sampler": sampler_label,
            "distilled": distilled,
            "engine": "moss_mlx",
            "load": "te+dit+vae",
            "memory_stages": ["text_encoder", "transformer", "vae"],
        })
        return 0
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        msg = str(exc)
        emit({"error": f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__})
        return 1
    finally:
        # Stage ownership is local to this request. Always release the MLX
        # allocator cache even when keep_alive=True; the warm daemon must not
        # retain the failed request's Metal scratch buffers.
        _clear()


def _to_pil(decoded: mx.array) -> Image.Image:
    import numpy as np

    decoded = decoded.astype(mx.float32)
    mx.eval(decoded)
    arr = np.asarray(decoded, dtype=np.float32)
    if arr.ndim == 5:
        arr = arr[:, :, 0]
    if arr.shape[1] == 3:
        arr = arr.transpose(0, 2, 3, 1)
    arr = arr[0]
    arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def _serve() -> int:
    try:
        from engine_watchdog import arm_orphan_exit, parent_pid_from_argv

        parent = parent_pid_from_argv(sys.argv[1:])
        if parent is not None:
            arm_orphan_exit(parent)
    except Exception:
        pass
    emit({"status": "ready", "engine": "moss", "backend": "mlx"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"error": f"invalid json: {exc}"})
            continue
        if req.get("command") == "ping":
            emit({"status": "pong"})
            continue
        if req.get("command") == "shutdown":
            break
        _run(req, keep_alive=True)
    return 0


def main() -> int:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: sys.exit(0))
        except (ValueError, OSError):
            pass
    if "--serve" in sys.argv:
        return _serve()
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        req = json.loads(arg) if arg.startswith("{") else json.loads(Path(arg).read_text("utf-8"))
        return _run(req)
    sys.stderr.write("Usage: moss_engine.py [--serve] [<request_json>]\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
