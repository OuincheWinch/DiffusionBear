#!/usr/bin/env python3
import argparse
import importlib.metadata
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import mlx.core as mx
import mlx.nn as nn

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

PROJECT_ROOT = BACKEND_DIR.parent
TEST_DIR = PROJECT_ROOT / "test"
TEST_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = TEST_DIR / "manifest.json"

from image_meta import atomic_write_json
import generator  # reuse the app's exact pipeline construction & exotic wraps

CANONICAL = (
    "a detailed imperial stormtrooper standing in a scenic swiss alpine landscape, "
    "green valley, snow-capped mountains, wooden chalet, cinematic lighting"
)

MODELS = {
    "flux2-klein-4b": {
        "prompt": CANONICAL,
        "default_steps": 4,
        "default_res": "768x768",
        "guidance": 1.0,
    },
    "z-image-turbo": {
        "prompt": (
            "a dramatic portrait of a cyberpunk samurai, cinematic rim lighting, "
            "neon-lit city bokeh background, sharp details, 8k"
        ),
        "default_steps": 8,
        "default_res": "768x768",
        "guidance": 1.0,
    },
    "krea2-turbo": {
        "prompt": (
            "a cozy cafe on a rainy Paris street at night, moody cinematic lighting, "
            "wet pavement reflections, highly detailed"
        ),
        "default_steps": 8,
        "default_res": "1024x1024",
        "guidance": 1.0,
    },
}


def _dependency_versions() -> dict:
    names = ("mlx", "mflux", "mlx-lm", "mlx-taef", "mlx-teacache", "numpy", "pillow")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


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


def _eval_value(value) -> None:
    if isinstance(value, mx.array):
        mx.eval(value)
    elif isinstance(value, dict):
        for item in value.values():
            _eval_value(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _eval_value(item)


def _phase_timings(events: list[tuple[str, float]], start: float, end: float) -> dict:
    if not events:
        return {"unattributed": round(max(0.0, end - start), 3)}
    totals = {"unattributed_before_first_phase": round(max(0.0, events[0][1] - start), 3)}
    for index, (phase, timestamp) in enumerate(events):
        next_timestamp = events[index + 1][1] if index + 1 < len(events) else end
        duration = max(0.0, next_timestamp - timestamp)
        totals[phase] = round(totals.get(phase, 0.0) + duration, 3)
    return totals


def _record(run: dict) -> None:
    """Append one benchmark run to test/manifest.json (read-modify-write)."""
    runs = []
    if MANIFEST.exists():
        try:
            runs = json.loads(MANIFEST.read_text())
            if not isinstance(runs, list):
                runs = []
        except Exception:
            runs = []
    runs.append(run)
    atomic_write_json(MANIFEST, runs, ensure_ascii=False)
    print(f"[manifest] {len(runs)} run(s) -> {MANIFEST}", flush=True)


def _sysload() -> float:
    try:
        return float(os.popen("uptime").read().split()[-3].strip(","))
    except Exception:
        return 0.0


def _load():
    print(f"[sys] {os.popen('uptime').read().strip()}", flush=True)
    if hasattr(mx, "device_info"):
        info = mx.device_info()
        print(f"[sys] device: {info.get('device_name')} | mem {info.get('memory_size', 0) // (1 << 30)} GiB"
              f" | wired cap {generator._wired_limit_bytes() // (1 << 30)} GiB", flush=True)
    load1 = _sysload()
    if load1 > 5:
        print(f"[warn] system load {load1:.1f} > 5 — generation times inflated per lesson", flush=True)


def _set_wired_like_app(override_gb: float | None = None):
    limit = generator._wired_limit_bytes()
    if override_gb is not None:
        limit = int(override_gb * (1 << 30))
    if limit <= 0:
        return None
    set_limit = getattr(mx, "set_wired_limit", None) or mx.metal.set_wired_limit
    return set_limit(limit)


def _taef2_wrap(pipe):
    """Identical fast-VAE hook the app installs (generator.py _generate_flux, fast_vae=True)."""
    taef2 = generator._get_taef_decoder("taef2")
    orig = pipe.vae.decode_packed_latents

    def _decode(packed_latents, tiling_config=None, *args, **kwargs):
        if packed_latents.ndim == 5:
            packed_latents = packed_latents[:, :, 0, :, :]
        unpacked = pipe.vae._unpatchify_latents(packed_latents)
        nhwc = mx.transpose(unpacked, (0, 2, 3, 1))
        decoded_nhwc = taef2.decode(nhwc)
        return mx.transpose(decoded_nhwc, (0, 3, 1, 2)) * 2.0 - 1.0

    pipe.vae.decode_packed_latents = _decode
    return orig


def _save_png(decoded_nchw, path):
    """Mirror mflux ImageUtil.to_image pixel math for a bit-identical reference file."""
    x = mx.clip((decoded_nchw / 2 + 0.5), 0, 1)
    x = x[0].transpose(1, 2, 0).astype(mx.float32)
    mx.eval(x)
    arr = np.array(x)
    img = (arr * 255).round().astype("uint8")
    Image.fromarray(img).save(str(path), format="PNG")
    return path


def _bf16_transformer(pipe):
    """Rebuild transformer QuantizedLinears as plain bf16 Linears (timing-only)."""
    replacements = []
    for name, m in pipe.transformer.named_modules():
        if isinstance(m, nn.QuantizedLinear):
            w = mx.dequantize(m.weight, m.scales, biases=m.biases, group_size=m.group_size, bits=m.bits)
            out_d, in_d = w.shape
            lin = nn.Linear(in_d, out_d)
            lin.weight = w
            lin.bias = getattr(m, "bias", None)
            replacements.append((name, lin))
    for name, newm in replacements:
        parts = name.split(".")
        node = pipe.transformer
        for p in parts[:-1]:
            if p.isdigit():
                node = node[int(p)]
            else:
                node = getattr(node, p)
        setattr(node, parts[-1], newm)
    mx.eval(pipe.transformer)
    print(f"[variant] rebuilt {len(replacements)} transformer linears as bf16 (timing-only proxy)", flush=True)


def _instrument_transformer(pipe):
    """Wrap transformer leaf ops; mx.eval output after each call to force GPU sync
    so wall-time == per-op GPU time (serialized — for ranking only).

    mlx nn layers dispatch `call` via the CLASS '__call__' (special-method lookup
    is type-based, so instance monkeypatching never fires). We therefore patch the
    leaf classes once and gate on instance-id so only this transformer's ops are
    timed; originals are restored when the returned restore() is invoked.
    """
    acc = {}
    id2name = {}
    leaf_types = (nn.Linear, nn.QuantizedLinear, nn.Embedding, nn.RMSNorm, nn.LayerNorm, nn.Conv2d)
    for name, m in pipe.transformer.named_modules():
        if isinstance(m, leaf_types):
            id2name[id(m)] = name

    saved = {}
    for cls in set(type(m) for _, m in pipe.transformer.named_modules() if isinstance(m, leaf_types)):
        orig = cls.__call__

        def _wrap(self, *a, o=orig, **k):
            rid = id(self)
            if rid in id2name:
                t0 = time.perf_counter()
                r = o(self, *a, **k)
                outs = r if isinstance(r, (tuple, list)) else (r,)
                for t in outs:
                    if isinstance(t, mx.array):
                        mx.eval(t)
                acc[id2name[rid]] = acc.get(id2name[rid], 0.0) + (time.perf_counter() - t0)
                return r
            return o(self, *a, **k)

        cls.__call__ = _wrap
        saved[cls] = orig

    def _restore():
        for cls, orig in saved.items():
            cls.__call__ = orig

    print(f"[variant] instrumented {len(id2name)} leaf ops (class-patched, serialized, ranking only)", flush=True)
    return acc, _restore


def _taef_wrap(pipe, model_id):
    """Mirror the app's fast-VAE hook for ZIT/Krea (generator.py generate())."""
    if model_id == "z-image-turbo" and hasattr(pipe, "_decode_latents"):
        taef1 = generator._get_taef_decoder("taef1")
        orig = pipe._decode_latents
        from mflux.models.z_image.latent_creator import ZImageLatentCreator

        def _decode(*, latents, config, prompt, seed, **kwargs):
            unpacked = ZImageLatentCreator.unpack_latents(latents, config.height, config.width)
            nhwc = mx.transpose(unpacked, (0, 2, 3, 1))
            decoded_nhwc = taef1.decode(nhwc)
            return mx.transpose(decoded_nhwc, (0, 3, 1, 2)) * 2.0 - 1.0

        pipe._decode_latents = _decode
        return orig
    if model_id == "krea2-turbo" and hasattr(pipe, "_decode_latents"):
        krea = generator._get_taef_decoder("krea2")
        orig = pipe._decode_latents

        def _decode(*, latents, prompt, seed, **kwargs):
            nhwc = mx.transpose(latents, (0, 2, 3, 1))
            decoded_nhwc = krea.decode(nhwc)
            return mx.transpose(decoded_nhwc, (0, 3, 1, 2)) * 2.0 - 1.0

        pipe._decode_latents = _decode
        return orig
    return None


def profile_generic(model_id, width, height, steps, seed, no_cache, wired_gb):
    """Standalone detailed timing for Z-Image / Krea2 via their own generate_image loop
    (app parity: fast-VAE wrap, wired limit, and — unless --no-cache — the rope memo
    + text-encoder LRU that generator now installs). Saves PNG + sidecar JSON to test/.
    """
    _reset_peak_memory()
    memory_before = _memory_snapshot()
    expected_key = generator._make_pipeline_key(model_id, 4, [], variant="standard")
    pipeline_cache = "warm" if getattr(generator, "_current_pipeline_key", None) == expected_key else "cold"
    t0 = time.perf_counter()
    pipe = generator._get_pipeline(model_id, 4, [], variant="standard")
    load_t = time.perf_counter() - t0
    actual_wired = int(wired_gb * (1 << 30)) if wired_gb is not None else generator._wired_limit_bytes()
    actual_wired = max(actual_wired, 0)

    taef_orig = _taef_wrap(pipe, model_id)
    try:
        # wrap the CURRENT _encode_prompts (which may be the memoized LRU) for timing
        encode_times = []
        orig_enc = pipe._encode_prompts

        def enc(prompt, negative_prompt=None, guidance=1.0, **kw):
            t = time.perf_counter()
            r = orig_enc(prompt=prompt, negative_prompt=negative_prompt, guidance=guidance, **kw)
            _eval_value(r)
            encode_times.append(time.perf_counter() - t)
            return r

        pipe._encode_prompts = enc

        decode_times = []
        orig_dec = pipe._decode_latents

        def dec(*, latents, prompt, seed, **kw):
            t = time.perf_counter()
            r = orig_dec(latents=latents, prompt=prompt, seed=seed, **kw)
            _eval_value(r)
            decode_times.append(time.perf_counter() - t)
            return r

        pipe._decode_latents = dec

        t = time.perf_counter()
        out = pipe.generate_image(
            seed=seed,
            prompt=MODELS[model_id]["prompt"],
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=MODELS[model_id]["guidance"],
        )
        wall = time.perf_counter() - t
    finally:
        pipe._encode_prompts = orig_enc
        if taef_orig is not None:
            pipe._decode_latents = taef_orig

    # bit-identical save of the returned image + companion sidecar
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = model_id.replace("-", "_")
    img_name = f"{slug}_{width}x{height}_q4_s{seed}_{stamp}.png"
    img_path = TEST_DIR / img_name
    out.image.save(str(img_path), format="PNG")

    memory_after = _memory_snapshot()
    encode_total = sum(encode_times)
    decode_total = sum(decode_times)
    generate_remainder = wall - encode_total - decode_total
    run = {
        "image": img_name,
        "model": model_id,
        "variant": "q4-standalone",
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "guidance": MODELS[model_id]["guidance"],
        "sampler": "model-native",
        "quantization_bits": 4,
        "transformer_format": "q4",
        "wired_limit_gb": actual_wired / (1 << 30),
        "rope_memo": "off" if no_cache else "on",
        "prompt": MODELS[model_id]["prompt"],
        "text_encoder_cache": "off" if no_cache else "on",
        "pipeline_cache": pipeline_cache,
        "dependencies": _dependency_versions(),
        "memory": {"scope": "profile-process", "before": memory_before, "after": memory_after},
        "system_load": _sysload(),
        "generation_time": round(wall, 3),
        "timing_scope": "generate_image wall time, excluding model load",
        "phases": {
            "model_load": round(load_t, 3),
            "prompt_encode": round(encode_total, 3),
            "prompt_encode_calls": len(encode_times),
            "generate_remainder": round(generate_remainder, 3),
            "vae_decode_taef": round(decode_total, 3),
        },
        "created_at_utc": stamp,
        "pipeline": "profile_generic standalone generate_image loop",
    }
    _record(run)

    print(f"\n== {model_id} | {width}x{height} | {steps} steps | seed {seed} | wired {actual_wired / (1 << 30)} GiB | rope_memo {run['rope_memo']} ==", flush=True)
    print(f"  model load/build     : {load_t:6.2f}s", flush=True)
    print(f"  prompt encode        : {encode_times[0]:6.2f}s" if encode_times else "", flush=True)
    print(f"  generate_image wall  : {wall:6.2f}s  (encode {encode_total:5.2f} | remainder {generate_remainder:5.2f} | decode {decode_total:5.2f})", flush=True)
    print(f"  image saved          : {img_path}", flush=True)
    return run


def profile(width, height, steps, seed, variant="q4", wired_gb=None, no_cache=False):
    if no_cache:
        os.environ["MLX_DISABLE_ROPE_CACHE"] = "1"
    _reset_peak_memory()
    memory_before = _memory_snapshot()
    expected_key = generator._make_pipeline_key("flux2-klein-4b", 4, [], variant="standard")
    pipeline_cache = "warm" if getattr(generator, "_current_pipeline_key", None) == expected_key else "cold"
    t0 = time.perf_counter()
    pipe = generator._get_pipeline("flux2-klein-4b", 4, [], variant="standard")
    load_t = time.perf_counter() - t0
    actual_wired = int(wired_gb * (1 << 30)) if wired_gb is not None else generator._wired_limit_bytes()

    if variant == "bf16-transformer":
        _bf16_transformer(pipe)
    acc = None
    restore_instrument = None
    if variant == "internals":
        acc, restore_instrument = _instrument_transformer(pipe)

    from mflux.models.common.config.config import Config

    cfg = Config(
        model_config=pipe.model_config,
        num_inference_steps=steps,
        height=height,
        width=width,
        guidance=1.0,
        image_path=None,
        image_strength=None,
        scheduler="flow_match_euler_discrete",
    )

    t = time.perf_counter()
    prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids = pipe._encode_prompt_pair(
        prompt=CANONICAL, negative_prompt=" ", guidance=1.0
    )
    mx.eval(prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids)
    encode_t = time.perf_counter() - t

    t = time.perf_counter()
    latents, latent_ids, latent_height, latent_width = pipe._prepare_generation_latents(seed=seed, config=cfg)
    mx.eval(latents, latent_ids)
    lat_t = time.perf_counter() - t

    predict = pipe._predict(pipe.transformer)
    iter_t, build_t, gpu_t = [], [], []
    for i in range(steps):
        a = time.perf_counter()
        noise = predict(
            latents=latents,
            latent_ids=latent_ids,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            guidance=1.0,
            timestep=cfg.scheduler.timesteps[i],
        )
        b = time.perf_counter()
        latents = cfg.scheduler.step(noise=noise, timestep=i, latents=latents, sigmas=cfg.scheduler.sigmas)
        c = time.perf_counter()
        mx.eval(latents)
        d = time.perf_counter()
        iter_t.append(d - a)
        build_t.append(b - a)
        gpu_t.append(d - c)

    if restore_instrument is not None:
        restore_instrument()

    t = time.perf_counter()
    packed = latents.reshape(latents.shape[0], latent_height, latent_width, latents.shape[-1]).transpose(0, 3, 1, 2)
    decoded = pipe.vae.decode_packed_latents(packed)
    mx.eval(decoded)
    decode_t = time.perf_counter() - t
    memory_after = _memory_snapshot()

    # store image + manifest entry, always
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    img_name = f"klein_{width}x{height}_{variant}_s{seed}_{stamp}.png"
    img_path = TEST_DIR / img_name
    _save_png(decoded, img_path)

    denoise_total = sum(iter_t)
    gen_total = encode_t + lat_t + denoise_total + decode_t
    top_ops = None
    if acc is not None:
        ranked = sorted(acc.items(), key=lambda kv: -kv[1])
        print("\n-- transformer per-op time (serialized; ranking only, all steps summed) --", flush=True)
        for i, (op, dt) in enumerate(ranked[:15], 1):
            print(f"  {i:2d}. {op:55s} {dt:7.2f}s", flush=True)
        top_ops = {name: round(dt, 3) for name, dt in ranked[:15]}
        print(f"  ... {len(ranked) - 15} more instrumented ops", flush=True)
    run = {
        "image": img_name,
        "model": "flux2-klein-4b",
        "variant": variant,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "guidance": 1.0,
        "sampler": "flow_match_euler_discrete",
        "quantization_bits": 4,
        "transformer_format": "q4" if variant == "q4" else ("bf16(proxy)" if variant == "bf16-transformer" else "q4(instrumented)"),
        "top_ops": top_ops,
        "wired_limit_gb": actual_wired / (1 << 30),
        "rope_memo": "off" if no_cache else "on",
        "prompt": CANONICAL,
        "text_encoder_cache": "off" if no_cache else "on",
        "pipeline_cache": pipeline_cache,
        "dependencies": _dependency_versions(),
        "memory": {"scope": "profile-process", "before": memory_before, "after": memory_after},
        "system_load": _sysload(),
        "generation_time": round(gen_total, 3),
        "timing_scope": "encode + latent prep + denoise + decode, excluding model load and image save",
        "phases": {
            "model_load": round(load_t, 3),
            "prompt_encode_cold": round(encode_t, 3),
            "latent_prep": round(lat_t, 3),
            "step_times": [round(x, 3) for x in iter_t],
            "host_build": [round(x, 3) for x in build_t],
            "denoise_total": round(denoise_total, 3),
            "vae_decode_taef2": round(decode_t, 3),
        },
        "created_at_utc": stamp,
        "pipeline": "profile_klein.py standalone loop",
    }
    _record(run)

    print(f"\n== FLUX.2-klein 4B | {width}x{height} | {steps} steps | seed {seed} | variant {variant} | wired {actual_wired / (1 << 30)} GiB ==", flush=True)
    print(f"  model load/build     : {load_t:6.2f}s", flush=True)
    print(f"  prompt encode (cold) : {encode_t:6.2f}s", flush=True)
    print(f"  latent prep          : {lat_t:6.2f}s", flush=True)
    for i, (it, bd, ev) in enumerate(zip(iter_t, build_t, gpu_t)):
        print(f"  step {i + 1}              : {it:6.2f}s  (host build {bd:5.2f}s | eval {ev:5.2f}s)", flush=True)
    print(f"  vae decode (TAEF2)   : {decode_t:6.2f}s", flush=True)
    print(f"  denoise total        : {denoise_total:6.2f}s  ({denoise_total / steps:5.2f}s/step avg)", flush=True)
    print(f"  generation time      : {gen_total:6.2f}s  (excl. model load)", flush=True)
    print(f"  image saved          : {img_path}", flush=True)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flux2-klein-4b", choices=list(MODELS))
    ap.add_argument("--resolution", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--variant", default="q4", choices=["q4", "bf16-transformer", "internals"])
    ap.add_argument("--no-cache", action="store_true",
                    help="disable lossless rope memo + text-encoder LRU (honest baseline)")
    ap.add_argument("--e2e", action="store_true", help="also time full generator.generate() wall-clock (writes data/generated artifacts)")
    ap.add_argument("--wired-gb", type=float, default=None, help="override wired memory limit (bypasses app 45%% cap)")
    args = ap.parse_args()

    _load()
    res = args.resolution or MODELS[args.model]["default_res"]
    steps = args.steps or MODELS[args.model]["default_steps"]
    w, h = (int(v) for v in res.lower().split("x"))
    if args.no_cache:
        os.environ["MLX_DISABLE_ROPE_CACHE"] = "1"
    if args.wired_gb is not None and args.wired_gb < 0:
        raise SystemExit("--wired-gb must be >= 0")
    prev = _set_wired_like_app(override_gb=args.wired_gb)
    try:
        if args.model == "flux2-klein-4b":
            profile(w, h, steps, args.seed, variant=args.variant, wired_gb=args.wired_gb, no_cache=args.no_cache)
        else:
            profile_generic(args.model, w, h, steps, args.seed, args.no_cache, args.wired_gb)

        if args.e2e:
            generator._drop_mflux_pipeline()
            _reset_peak_memory()
            memory_before = _memory_snapshot()
            phase_events = []

            def phase_cb(phase, msg=None):
                phase_events.append((phase, time.perf_counter()))

            t0 = time.perf_counter()
            meta = generator.generate(
                prompt=MODELS[args.model]["prompt"],
                width=w,
                height=h,
                steps=steps,
                guidance=MODELS[args.model]["guidance"],
                seed=args.seed,
                quantization=4,
                model=args.model,
                progress_cb=lambda *a, **k: None,
                phase_cb=phase_cb,
                fast_vae=True,
            )
            wall = time.perf_counter() - t0
            memory_after = _memory_snapshot()
            phases = _phase_timings(phase_events, t0, t0 + wall)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            slug = args.model.replace("-", "_")
            img_name = f"{slug}_{w}x{h}_e2e_q4_s{args.seed}_{stamp}.png"
            img_path = TEST_DIR / img_name
            gen_dir = BACKEND_DIR / "data" / "generated"
            src = gen_dir / meta["file"]
            if src.exists():
                import shutil

                shutil.copy2(src, img_path)
            data_artifacts = {
                "png": f"{meta['id']}.{meta.get('format', 'png')}",
                "json": f"{meta['id']}.json",
                "thumb": f"{meta['id']}_thumb.png",
            }
            present = {
                k: (gen_dir / v).exists()
                for k, v in data_artifacts.items()
            }
            print(f"\n== E2E generator.generate() wall {wall:6.2f}s | generation_time={meta['generation_time']}s ==", flush=True)
            print(f"   data/generated: {data_artifacts} -> {present}", flush=True)
            _record({
                "image": img_name,
                "model": args.model,
                "variant": "e2e-q4-app-path",
                "width": w,
                "height": h,
                "steps": steps,
                "seed": args.seed,
                "guidance": MODELS[args.model]["guidance"],
                "sampler": meta.get("sampler"),
                "quantization_bits": 4,
                "transformer_format": "q4",
                "fast_vae": True,
                "wired_limit_gb": generator._wired_limit_bytes() / (1 << 30),
                "rope_memo": "off" if args.no_cache else "on",
                "text_encoder_cache": "off" if args.no_cache else "on",
                "pipeline_cache": "cold",
                "prompt_cache": "cold",
                "dependencies": _dependency_versions(),
                "memory": {"scope": "profile-process", "before": memory_before, "after": memory_after},
                "prompt": MODELS[args.model]["prompt"],
                "system_load": _sysload(),
                "wall_clock": round(wall, 3),
                "generation_time": meta.get("generation_time"),
                "wall_clock_scope": "cold generator.generate path including model load and save",
                "phase_sequence": [phase for phase, _ in phase_events],
                "phases_cb": phases,
                "data_generated": {"artifacts": data_artifacts, "present": present},
                "created_at_utc": stamp,
                "pipeline": "generator.generate() full app path incl. save",
            })
    finally:
        if prev is not None:
            set_limit = getattr(mx, "set_wired_limit", None) or mx.metal.set_wired_limit
            set_limit(prev)
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()