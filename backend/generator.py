import atexit
import collections
import gc
import json
import math
import os
import queue
import re
import select
import signal
import subprocess
import sys
import traceback
import threading
import time
import uuid
from pathlib import Path

from collections import OrderedDict
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo

from image_meta import (
    atomic_write_json,
    save_image_with_metadata,
    extract_image_metadata,
)
from upscale import upscale_image, thumbnail_path

import app_settings

# Anti-SIGABRT Safety Patch for mflux ImageUtil:
# By default, mflux calls np.array(images) on an un-evaluated MLX array.
# If an allocation/Metal failure occurs during eval_impl inside Python's C buffer protocol
# (getbufferproc), the unhandled C++ exception causes std::terminate() -> SIGABRT (Abort trap 6).
# Forcing mx.eval(images) in Python scope before calling np.array(images) ensures any Metal error
# is raised as a standard, catchable Python RuntimeError.
try:
    import mlx.core as mx
    import numpy as np
    import mflux.utils.image_util as _mflux_img_util

    _orig_mflux_to_numpy = _mflux_img_util.ImageUtil._to_numpy

    @staticmethod
    def _safe_mflux_to_numpy(images: mx.array) -> np.ndarray:
        if len(images.shape) == 5:
            images = mx.squeeze(images, axis=2)
        images = mx.transpose(images, (0, 2, 3, 1))
        images = mx.array.astype(images, mx.float32)
        mx.eval(images)
        return np.array(images)

    _mflux_img_util.ImageUtil._to_numpy = _safe_mflux_to_numpy
except Exception as _patch_err:
    print(f"[generator] WARN failed to patch ImageUtil._to_numpy: {_patch_err}", flush=True)

# Second layer, added after the 2026-10-02 abort.
#
# The patch above only protects ImageUtil._to_numpy, and the backend still died with
# SIGABRT. The crash report is unambiguous about the mechanism:
#
#   mlx::core::eval_impl -> mlx::core::array::eval -> _PyManagedBuffer_FromObject
#     -> numpy PyArray_FromAny -> numpy array_array
#
# i.e. numpy pulled the buffer protocol off an MLX array. pybind11's exception
# translator only wraps Python->C calls, so a Metal error raised inside a C callback
# that numpy invokes escapes it entirely and hits std::terminate() -> abort(). Asking
# for 40GB directly demonstrates both halves of this, in a subprocess:
#
#   mx.eval(x) in Python scope -> RuntimeError (catchable)
#   np.array(x) un-evaluated    -> SIGABRT (process gone)
#
# so the fix is to make sure eval happens in Python scope first. Measured independently,
# and the distinction is the whole point: it is not a heuristic.
#
# This wrapper applies that rule to EVERY np.array() call in the process rather than one
# known call site, because the 2026-10-02 crash proves the offending call site is not
# ImageUtil._to_numpy and we have not identified it yet. It also prints the caller's
# stack when it fires, so the first real occurrence names the culprit instead of
# leaving us guessing again.
try:
    import mlx.core as _mx_for_np_guard

    _orig_np_array = np.array

    def _np_array_eval_guard(obj, *args, **kwargs):
        if type(obj).__module__.split(".")[0] == "mlx":
            # Force evaluation in Python scope while we can still catch the error.
            _mx_for_np_guard.eval(obj)
            print(
                "[generator] mlx->numpy conversion outside ImageUtil._to_numpy:\n"
                + "".join(traceback.format_stack()[-8:]),
                flush=True,
            )
        return _orig_np_array(obj, *args, **kwargs)

    np.array = _np_array_eval_guard
except Exception as _guard_err:
    print(f"[generator] WARN np.array eval guard not installed: {_guard_err}", flush=True)

try:
    from mflux.models.krea2 import Krea2
    _orig_krea2_decode = Krea2._decode_latents

    def _safe_krea2_decode(self, *args, **kwargs):
        import mlx.core as _mx
        _mx.clear_cache()
        decoded = _orig_krea2_decode(self, *args, **kwargs)
        _mx.eval(decoded)
        _mx.clear_cache()
        return decoded

    Krea2._decode_latents = _safe_krea2_decode
except Exception:
    pass

DATA_DIR = app_settings.DATA_DIR
ASSET_DIR = app_settings.ASSET_DIR
ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = DATA_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_orphan_artifacts(max_age_seconds: float = 3600.0) -> int:
    cutoff = time.time() - max(0.0, float(max_age_seconds))
    removed = 0
    try:
        candidates = list(GENERATED_DIR.glob("*.raw.png"))
        candidates.extend(
            path for path in GENERATED_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg")
            and re.fullmatch(r"[a-f0-9]{32}", path.stem)
            and not (GENERATED_DIR / f"{path.stem}.json").exists()
        )
        for path in candidates:
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                path.unlink()
                removed += 1
            except OSError:
                pass
    except OSError:
        pass
    return removed


DEFAULT_MODEL = "mlx-community/flux2-klein-4b-4bit"
DEFAULT_MODEL_ID = "flux2-klein-4b"

# Qwen-Image 2.1 sampler name -> mflux scheduler id. "linear" is mflux's native
# default and renders saturated mono-color subjects correctly; the euler
# empirical-mu scheduler collapses them to magenta/green.
_QWEN_SCHEDULERS = {
    "linear": "linear",
    "euler": "flow_match_euler_discrete",
}
_MAX_PROMPT_BYTES = 128 * (1 << 10)

MODELS = {
    "flux2-klein-4b": {
        "id": "flux2-klein-4b",
        "label": "FLUX.2-klein 4B",
        "repo": DEFAULT_MODEL,
        "civitai_model_id": 2387000,
        "civitai_version_id": 2684089,
        "civitai_model_name": "FLUX.2-klein 4B",
        "civitai_version_name": "4-bit",
        "sha256": "D9055F16B1",
        "ecosystem": "FLUX.2",
        "is_distilled": True,
        "default_steps": 4,
        "default_width": 512,
        "default_height": 768,
        "default_guidance": 1.0,
        "supports_guidance": True,
        "supports_negative": False,
        "supports_loras": True,
        "supports_multi_reference": True,
        "supports_ref": True,
        "max_reference_images": 10,
        "supports_fast_vae": True,
        "lora_format": "FLUX.2",
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft 512×768 (~56s)", "width": 512, "height": 768, "steps": 4},
            {"id": "fast", "label": "⚡ Fast 768×768 (~90s)", "width": 768, "height": 768, "steps": 4},
            {"id": "quality", "label": "✦ Quality 1024×1024 (~2min)", "width": 1024, "height": 1024, "steps": 4},
        ],
    },
    "flux2-klein-9b": {
        "id": "flux2-klein-9b",
        "label": "FLUX.2-klein 9B",
        "repo": "mlx-community/flux2-klein-9b-4bit",
        "civitai_model_id": 2387000,
        "civitai_version_id": 2684090,
        "civitai_model_name": "FLUX.2-klein 9B",
        "civitai_version_name": "4-bit",
        "ecosystem": "FLUX.2",
        "default_steps": 4,
        "default_width": 512,
        "default_height": 768,
        "is_distilled": True,
        "default_guidance": 1.0,
        "supports_guidance": True,
        "supports_negative": False,
        "supports_loras": True,
        "supports_multi_reference": True,
        "supports_ref": True,
        "max_reference_images": 10,
        "supports_fast_vae": True,
        "lora_format": "FLUX.2",
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft 512×768 (~110s)", "width": 512, "height": 768, "steps": 4},
            {"id": "fast", "label": "⚡ Fast 768×768", "width": 768, "height": 768, "steps": 4},
            {"id": "quality", "label": "✦ Quality 1024×1024", "width": 1024, "height": 1024, "steps": 4},
        ],
    },
    "juggernaut-xl-lightning": {
        "id": "juggernaut-xl-lightning",
        "label": "Juggernaut XL Lightning (MLX)",
        "repo": "RunDiffusion/Juggernaut-XL-Lightning",
        "model_dir": ASSET_DIR / "models" / "juggernaut-xl-lightning",
        "engine": "sdxl",
        "civitai_version_id": 357609,
        "civitai_model_id": 133005,
        "civitai_model_name": "Juggernaut XL",
        "civitai_version_name": "V9+RDPhoto2-Lightning_4S",
        "sha256": "357609",
        "ecosystem": "SDXL",
        "default_steps": 4,
        "default_width": 512,
        "default_height": 768,
        "default_guidance": 1.0,
        "supports_guidance": True,
        "supports_negative": True,
        "supports_loras": True,
        "supports_ref": False,
        "max_reference_images": 0,
        "supports_fast_vae": True,
        "samplers": ["euler_trailing", "dpmpp_2m_karras", "euler_a_substep", "euler_a", "euler", "ddim"],
        "lora_format": "SDXL",
        "is_distilled": True,
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft 512×768 (~10s)", "width": 512, "height": 768, "steps": 4, "sampler": "euler_trailing", "guidance": 1.0, "cache_interval": 2},
            {"id": "photo", "label": "✦ Carré 1024×1024 (~70s)", "width": 1024, "height": 1024, "steps": 4, "sampler": "euler_trailing", "guidance": 1.0, "cache_interval": 1},
            {"id": "portrait", "label": "✦ Portrait 832×1216 (~60s)", "width": 832, "height": 1216, "steps": 4, "sampler": "euler_trailing", "guidance": 1.0, "cache_interval": 1},
            {"id": "cinematic", "label": "✦ Cinématique 1216×832 (~45s)", "width": 1216, "height": 832, "steps": 4, "sampler": "euler_trailing", "guidance": 1.0, "cache_interval": 1},
            {"id": "fast", "label": "⚡ Fast 4-step DeepCache 1024×1024", "width": 1024, "height": 1024, "steps": 4, "sampler": "euler_trailing", "guidance": 1.0, "cache_interval": 2},
        ],
    },
    "realvis-xl-v5-lightning": {
        "id": "realvis-xl-v5-lightning",
        "label": "RealVisXL V5.0 Lightning (MLX)",
        "repo": "SG161222/RealVisXL_V5.0_Lightning",
        "model_dir": ASSET_DIR / "models" / "realvis-xl-v5-lightning",
        "engine": "sdxl",
        "civitai_version_id": 361593,
        "civitai_model_id": 139562,
        "civitai_model_name": "RealVisXL V5.0",
        "civitai_version_name": "V5.0 Lightning",
        "sha256": "B620C6B8D3",
        "ecosystem": "SDXL",
        "default_steps": 6,
        "default_width": 512,
        "default_height": 768,
        "default_guidance": 1.0,
        "supports_guidance": True,
        "supports_negative": True,
        "supports_loras": True,
        "supports_ref": False,
        "max_reference_images": 0,
        "supports_fast_vae": True,
        "samplers": ["euler_trailing", "dpmpp_2m_karras", "euler_a_substep", "euler_a", "euler", "ddim"],
        "lora_format": "SDXL",
        "is_distilled": True,
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft 512×768 (~16s)", "width": 512, "height": 768, "steps": 6, "sampler": "euler_trailing", "guidance": 1.0, "cache_interval": 1},
            {"id": "photo", "label": "✦ Carré 1024×1024 (~56s)", "width": 1024, "height": 1024, "steps": 6, "sampler": "euler_trailing", "guidance": 1.0, "cache_interval": 1},
            {"id": "portrait", "label": "✦ Portrait 832×1216 (~1.8min)", "width": 832, "height": 1216, "steps": 6, "sampler": "euler_trailing", "guidance": 1.0, "cache_interval": 1},
        ],
    },
    "realvis-xl-v5": {
        "id": "realvis-xl-v5",
        "label": "RealVisXL V5.0 Standard / Hyper-SD",
        "repo": "SG161222/RealVisXL_V5.0",
        "model_dir": ASSET_DIR / "models" / "realvis-xl-v5",
        "engine": "sdxl",
        "civitai_version_id": 361592,
        "civitai_model_id": 139562,
        "civitai_model_name": "RealVisXL V5.0",
        "civitai_version_name": "V5.0",
        "sha256": "D7E84FE269",
        "ecosystem": "SDXL",
        "default_steps": 25,
        "default_width": 512,
        "default_height": 768,
        "default_guidance": 4.5,
        "supports_guidance": True,
        "supports_negative": True,
        "supports_loras": True,
        "supports_ref": False,
        "max_reference_images": 0,
        "supports_fast_vae": True,
        "samplers": ["dpmpp_2m_karras", "euler_a", "euler", "euler_trailing", "ddim"],
        "lora_format": "SDXL",
        "is_distilled": False,
        "presets": [
            {"id": "draft", "label": "⚡ Hyper-SD 8-step Draft 512×768 (~45s)", "width": 512, "height": 768, "steps": 8, "sampler": "euler_trailing", "guidance": 2.0, "cache_interval": 1},
            {"id": "photo", "label": "✦ Carré 1024×1024 (25 steps)", "width": 1024, "height": 1024, "steps": 25, "sampler": "dpmpp_2m_karras", "guidance": 4.5, "cache_interval": 1},
            {"id": "portrait", "label": "✦ Portrait 832×1216 (25 steps)", "width": 832, "height": 1216, "steps": 25, "sampler": "dpmpp_2m_karras", "guidance": 4.5, "cache_interval": 1},
        ],
    },
    "juggernaut-xi": {
        "id": "juggernaut-xi",
        "label": "Juggernaut XI v11 (MLX)",
        "repo": "RunDiffusion/Juggernaut-XI-v11",
        "model_dir": ASSET_DIR / "models" / "juggernaut-xi",
        "engine": "sdxl",
        "civitai_version_id": 782002,
        "civitai_model_id": 133005,
        "civitai_model_name": "Juggernaut XL",
        "civitai_version_name": "XI",
        "sha256": "782002",
        "ecosystem": "SDXL",
        "default_steps": 25,
        "default_width": 512,
        "default_height": 768,
        "default_guidance": 4.0,
        "supports_guidance": True,
        "supports_negative": True,
        "supports_loras": True,
        "supports_ref": False,
        "max_reference_images": 0,
        "supports_fast_vae": True,
        "samplers": ["dpmpp_2m_karras", "euler_a_substep", "euler_a", "euler", "euler_trailing", "ddim"],
        "lora_format": "SDXL",
        "is_distilled": False,
        "presets": [
            {"id": "draft", "label": "⚡ Hyper-SD 8-step Draft 512×768 (~49s)", "width": 512, "height": 768, "steps": 8, "sampler": "euler_trailing", "guidance": 2.0, "cache_interval": 1},
            {"id": "photo", "label": "✦ Carré 1024×1024 (25 steps)", "width": 1024, "height": 1024, "steps": 25, "sampler": "dpmpp_2m_karras", "guidance": 4.0, "cache_interval": 1},
            {"id": "portrait", "label": "✦ Portrait 832×1216 (25 steps)", "width": 832, "height": 1216, "steps": 25, "sampler": "dpmpp_2m_karras", "guidance": 4.0, "cache_interval": 1},
        ],
    },
    "z-image-turbo": {
        "id": "z-image-turbo",
        "label": "Z-Image Turbo 6B",
        "repo": "filipstrand/Z-Image-Turbo-mflux-4bit",
        "civitai_model_id": 2168935,
        "civitai_version_id": 2442439,
        "civitai_model_name": "Z Image Turbo",
        "civitai_version_name": "Turbo",
        "sha256": "2407613050",
        "ecosystem": "ZImageTurbo",
        "is_distilled": True,
                # 8 -> 6 steps, measured 2026-09-29 on this M1 16GB. 6/8 steps is a
        # 1.30-1.32x saving. Six steps was ALREADY shipped as the "draft" preset
        # at 512x768, so this is not a new quality level there; at 1024x1024 and
        # 1280x720 the side-by-side sheets are near-identical. At 512x768 six
        # steps is mildly smoother on fine texture (PSNR 26.9/27.1 dB, i.e. a
        # different image rather than a degraded one). Revert is this one number.
"default_steps": 6,
        "default_width": 512,
        "default_height": 768,
        "default_guidance": None,
        "supports_guidance": False,
        "supports_negative": False,
        "supports_loras": True,
        "supports_ref": True,
        "max_reference_images": 1,
        "supports_fast_vae": True,
        "lora_format": "Z-Image",
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft 512×768 (~1.5min)", "width": 512, "height": 768, "steps": 6},
            {"id": "turbo", "label": "✦ Turbo 1024×1024 (~4.9min)", "width": 1024, "height": 1024, "steps": 6},
            {"id": "wide", "label": "✦ Wide HD 1280×720 (~4.9min)", "width": 1280, "height": 720, "steps": 6},
        ],
    },
    "krea2-turbo": {
        "id": "krea2-turbo",
        "label": "Krea 2 Turbo 13B",
        "repo": "local:krea2-turbo-q4",
        "civitai_model_id": 2726029,
        "civitai_version_id": 3064584,
        "civitai_model_name": "Krea 2 Turbo Official Comfy-Org Checkpoints (Krea2)",
        "civitai_version_name": "krea2_turbo_bf16",
        "sha256": "78BBF8F416",
        "ecosystem": "Krea 2",
        "is_distilled": True,
        "default_steps": 8,
        "default_width": 512,
        "default_height": 768,
        "default_guidance": 1.0,
        "supports_guidance": False,
        "supports_negative": False,
        "supports_loras": True,
        "supports_ref": True,
        "max_reference_images": 1,
        "supports_fast_vae": True,
        "lora_format": "Krea 2",
        # Size caps removed 2026-09-22 (user decision). 16GB M1 note: the 3D
        # causal VAE decode has high activation memory; very large sizes may OOM.
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft 4-step 512×768 (~3min)", "width": 512, "height": 768, "steps": 4},
            {"id": "turbo", "label": "✦ 8-step Quality 512×512 (~3.5min)", "width": 512, "height": 512, "steps": 8},
            {"id": "portrait", "label": "✦ Portrait 512×768 8-step (~5.5min)", "width": 512, "height": 768, "steps": 8},
            {"id": "fast", "label": "⚡ Fast 4-step 512×512 (~2.5min)", "width": 512, "height": 512, "steps": 4},
        ],
    },
    "qwen-image-2.1": {
        "id": "qwen-image-2.1",
        "label": "Qwen-Image 2.1 (MLX q4, experimental)",
        "repo": "mlx-community/Qwen-Image-2.1-MLX-4bit",
        # mflux-community/mflux fork (Qwen-Image-2.1 port, PR #736) patched to load the
        # pre-quantized mlx-community repo with a QUANTIZED Qwen3-VL text encoder.
        # Upstream hard-codes the TE at bf16 (~17.5GB resident) which can't fit 16GB.
        "ecosystem": "Qwen-Image 2.1",
        "default_steps": 25,
        "default_width": 512,
        "default_height": 768,
        "default_guidance": 1.0,
        "supports_guidance": True,
        "supports_negative": True,
        "supports_loras": False,
        "supports_ref": True,
        "max_reference_images": 1,
        "supports_fast_vae": False,
        # 2026-09-23: default sampler flipped to "linear". The forced
        # flow_match_euler_discrete (empirical-mu) sampler catastrophically
        # collapses saturated mono-color subjects to magenta/green (same seed &
        # steps as the linear run, verified on a yellow-banana control: linear =
        # clean, euler = 71% magenta) while linear renders them correctly; euler
        # is still offered for scene/landscape looks where it was preferred.
        "samplers": ["linear", "euler"],
        "default_sampler": "linear",
        # Size caps removed 2026-09-22 (user decision). 16GB M1 note: the q4
        # pipeline is ~10.5GB resident; 1024² can OOM in the bf16 VAE decode.
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft 512×768 25 steps (~12min)", "width": 512, "height": 768, "steps": 25},
            # No time claims on these two: the measured medians only exist for
            # 512x768, and a test refuses a claim without one. Bench pending a
            # quiet machine - a 25-step run is ~13min, so it is not something to
            # guess at on a loaded one.
            {"id": "wide", "label": "✦ Wide 768×768 25 steps (~12min)", "width": 768, "height": 768, "steps": 25},
            {"id": "landscape", "label": "✦ Landscape 768×512 25 steps (~9min)", "width": 768, "height": 512, "steps": 25},
        ],
    },
}


def get_model_info(model_id: str) -> dict | None:
    if not model_id:
        return None
    if model_id in MODELS:
        return MODELS[model_id]
    for m in MODELS.values():
        if m.get("repo") == model_id or m.get("id") == model_id:
            return m
    return None

# Fraction of unified memory a wired budget may claim, shared by the generic and
# krea2 paths so the two cannot drift apart again. 0.68 of 16GB is ~10.9GB, which
# leaves room for macOS itself; the effective budget is still the lower of this
# fraction, the configured GB value, and Apple's recommended working set.
_WIRED_MEMORY_FRACTION = 0.68

# Wired memory hint used during generation to keep Metal from swapping.
#
# THE DEFAULT IS NOW DERIVED FROM THE MACHINE, NOT A CONSTANT.
#
# This used to be a hardcoded 9 GB (generic) and 0 (krea2). Those were numbers tuned
# by hand on ONE 16GB M1, stored in settings.json, and carried to every other Mac the
# app ran on. A 9 GB ceiling is far too tight on a 64GB machine and, worse, was actively
# harmful here: krea2 persists a 9 GB pin that predates the 2026-09-29 revert, and the
# Krea 2 text-encoder quantisation pass -- which transiently holds a 7.5GB bf16 copy
# AND its ~1.9GB q4 result -- no longer fits under it, and dies with a Metal
# CommandBuffer OOM during the loading_model phase.
#
# `None` means "auto": derive the budget from the device. Explicit GB values are still
# honoured as deliberate overrides, and 0 still means unbounded.
def _parse_wired_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "" or raw.strip().lower() == "auto":
        return None
    try:
        return float(raw)
    except ValueError:
        print(f"[generator] invalid {name}={raw!r}, deriving from device instead", flush=True)
        return None


# Generic budget: FLUX.2-klein, Z-Image, SDXL and qwen-image-2.1. qwen is in this
# group because its q4 pipeline is ~10.5GB resident and needs the larger allowance; it
# is NOT here because of any measured speedup.
_WIRED_LIMIT_GB = _parse_wired_env("MLX_WIRED_LIMIT_GB")

# krea2 (13B q4). It used to carry a 9 GB pin, justified by an AGENTS.md note claiming
# ~18% faster for identical output. Measured on 2026-09-29 (test/wired_budget/, 6 runs
# per arm, interleaved): the arms are statistically indistinguishable (exact permutation
# p=1.00, 150-239s bounded vs 139-198s unbounded) while all 12 renders are
# pixel-identical. So the pin bought nothing and was reverted -- but the REVERT ONLY
# CHANGED THE ENV DEFAULT, leaving a stale 9 GB in settings.json, which is why the cap
# was still in force long after the note said it was gone.
_KREA_WIRED_LIMIT_GB = _parse_wired_env("MLX_KREA_WIRED_LIMIT_GB")


def _env_wired_gb() -> float | None:
    return _WIRED_LIMIT_GB


def _env_krea_gb() -> float:
    return _KREA_WIRED_LIMIT_GB / (1 << 30) if _KREA_WIRED_LIMIT_GB > 0 else 0


def _wired_limit_gb() -> float | None:
    """Configured generic wired limit in GB. None means auto (derive from the device)."""
    v = app_settings.get_setting("memory_wired_limit_gb")
    return float(v) if v is not None else _env_wired_gb()


def _krea_wired_limit_gb() -> float | None:
    """Configured krea2 wired limit in GB. None means auto (derive from the device)."""
    v = app_settings.get_setting("memory_krea_wired_limit_gb")
    return float(v) if v is not None else _env_wired_gb()


_device_budget_cache: dict = {}


def _derived_wired_budget_bytes() -> int:
    """Wired budget derived from this Mac, in bytes.

    The lower of a fixed fraction of unified memory and Apple's own recommended working
    set for the device. On a 16GB M1 that is min(16GB x 0.68, 11.84GB) = 10.88GB; the
    same code yields ~43GB on a 64GB Mac, which is the point -- the old 9 GB was a
    hand-tuned constant for one machine and meaningless on any other.

    Cached: the device does not change under us, and this is read on the status path.
    """
    if "bytes" in _device_budget_cache:
        return _device_budget_cache["bytes"]
    value = 0
    try:
        import mlx.core as mx

        dev_info = getattr(mx, "device_info", None) or getattr(mx, "metal", "device_info", None)
        if dev_info:
            d = dev_info() or {}
            cap = d.get("max_recommended_working_set_size") or d.get(
                "recommended_max_working_set_size"
            )
            mem = d.get("memory_size") or 0
            if mem > 0:
                value = int(mem * _WIRED_MEMORY_FRACTION)
                if cap:
                    value = min(value, int(cap))
            elif cap:
                value = int(cap * _WIRED_MEMORY_FRACTION)
    except Exception:
        value = 0
    if value <= 0:
        # No device info: fall back to the historical constant rather than to unbounded.
        value = 9 * (1 << 30)
    _device_budget_cache["bytes"] = value
    return value


def _derived_wired_budget_gb() -> float:
    return round(_derived_wired_budget_bytes() / (1 << 30), 2)


def _resolve_wired_budget(setting_gb: float | None, fallback_gb: int) -> int:
    """Effective wired budget in bytes.

    `setting_gb` is None for auto (derive from the device), 0 for an explicit
    unbounded request, and a positive number for a deliberate pin.
    """
    if setting_gb is None:
        return _derived_wired_budget_bytes()
    if setting_gb <= 0:
        return 0
    return _wired_budget_bytes(int(setting_gb * (1 << 30)), fallback_gb)


def _wired_budget_bytes(limit_gb: float, fallback_gb: int) -> int:
    """Effective wired budget: the lowest of the configured GB, the shared memory
    fraction, and Apple's recommended working set."""
    limit = int(limit_gb)
    if limit <= 0:
        return 0
    try:
        import mlx.core as mx

        dev_info = getattr(mx, "device_info", None) or getattr(mx.metal, "device_info", None)
        if dev_info:
            d = dev_info()
            cap = d.get("max_recommended_working_set_size") or d.get("recommended_max_working_set_size") or 0
            mem = d.get("memory_size", 0)
            if mem > 0:
                budget = int(mem * _WIRED_MEMORY_FRACTION)
                if cap > 0:
                    budget = min(budget, cap)
                return min(limit, budget)
            if cap > 0:
                return min(limit, int(cap * _WIRED_MEMORY_FRACTION))
    except Exception:
        pass
    return min(limit, fallback_gb * (1 << 30))


def _wired_limit_bytes() -> int:
    return _resolve_wired_budget(_wired_limit_gb(), 9)


def _krea_wired_limit_bytes() -> int:
    """Wired budget for krea2 (13B q4) only. Defaults to 0 = unbounded, which is
    what the 2026-09-29 A/B measured: no speed difference (p=1.00) and a
    pixel-identical result, so there is nothing to buy with a pin. qwen does NOT
    use this helper -- it uses the generic 9 GB budget, because its pipeline is
    ~10.5GB resident and a too-low cap starves the load.
    """
    return _resolve_wired_budget(_krea_wired_limit_gb(), 9)

_lock = threading.Lock()
_model_maintenance_lock = threading.RLock()
_cancel_event = threading.Event()
_pipeline = None
_current_pipeline_key = None
_current_pipeline_model = None
_current_sdxl_model = None
_prompt_cache: OrderedDict[str, tuple] = OrderedDict()
_PROMPT_CACHE_MAX_SIZE = 32
_PROMPT_CACHE_MAX_BYTES = 256 * (1 << 20)

_taef_models: dict = {}

# Quantized Krea 2 text encoder, reused across pipeline rebuilds. See
# _krea_quantized_text_encoder. Held outside the pipeline cache on purpose: the
# pipeline is dropped and rebuilt on every LoRA/step change, the encoder is not.
_krea_te_cache: dict = {}
_krea_te_lock = threading.Lock()


def _value_bytes(value) -> int:
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    if isinstance(value, dict):
        return sum(_value_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_value_bytes(v) for v in value)
    return 0


def _trim_prompt_cache():
    total = sum(_value_bytes(v) for v in _prompt_cache.values())
    while _prompt_cache and (
        len(_prompt_cache) > _PROMPT_CACHE_MAX_SIZE or total > _PROMPT_CACHE_MAX_BYTES
    ):
        _, value = _prompt_cache.popitem(last=False)
        total -= _value_bytes(value)


def _cache_prompt(key, value):
    _prompt_cache[key] = value
    _prompt_cache.move_to_end(key)
    _trim_prompt_cache()


_PROCESS_READER_CHUNK = 65536
_PROCESS_READER_MAX_LINE = 4 * (1 << 20)
_ENGINE_TIMEOUT_DEFAULT_S = 1800.0
_qwen_process = None
_qwen_reader_messages = None
_qwen_reader_done = None
_qwen_reader_thread = None
_qwen_stderr_thread = None
_qwen_stderr_done = None
_qwen_watchdog = None
_qwen_idle_since = None


def _engine_timeout(name: str) -> float:
    try:
        value = float(os.environ.get(name, _ENGINE_TIMEOUT_DEFAULT_S))
    except (TypeError, ValueError):
        return _ENGINE_TIMEOUT_DEFAULT_S
    return max(1.0, value)


def _write_process_request(proc, payload: bytes, cancel_events, timeout_s: float):
    if proc is None or proc.stdin is None:
        raise RuntimeError("engine stdin is unavailable")
    try:
        fd = proc.stdin.fileno()
        os.set_blocking(fd, False)
    except (AttributeError, OSError, ValueError) as e:
        raise RuntimeError("engine stdin is unavailable") from e
    deadline = time.monotonic() + timeout_s
    offset = 0
    while offset < len(payload):
        if any(event is not None and event.is_set() for event in cancel_events):
            _terminate_process(proc)
            raise GenerationCancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process(proc, force=True)
            raise TimeoutError("engine request write timed out")
        try:
            _, writable, _ = select.select([], [fd], [], min(0.1, remaining))
        except (OSError, ValueError) as e:
            raise RuntimeError("engine stdin is unavailable") from e
        if not writable:
            continue
        try:
            written = os.write(fd, payload[offset:])
        except BlockingIOError:
            continue
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError("engine closed its input") from e
        if written <= 0:
            raise RuntimeError("engine accepted no request bytes")
        offset += written


def _read_process_stream(stream, callback):
    fd = None
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        fd = None
    pending = bytearray()
    if fd is not None:
        while True:
            chunk = os.read(fd, _PROCESS_READER_CHUNK)
            if not chunk:
                break
            pending.extend(chunk)
            while True:
                index = pending.find(b"\n")
                if index < 0:
                    break
                line = bytes(pending[:index])
                del pending[: index + 1]
                callback(line.decode("utf-8", errors="replace"))
            if len(pending) > _PROCESS_READER_MAX_LINE:
                raise ValueError("subprocess output line exceeded size limit")
    else:
        while True:
            chunk = stream.read(_PROCESS_READER_CHUNK)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", errors="replace")
            pending.extend(chunk)
            while True:
                index = pending.find(b"\n")
                if index < 0:
                    break
                line = bytes(pending[:index])
                del pending[: index + 1]
                callback(line.decode("utf-8", errors="replace"))
            if len(pending) > _PROCESS_READER_MAX_LINE:
                raise ValueError("subprocess output line exceeded size limit")
    if pending:
        callback(bytes(pending).decode("utf-8", errors="replace"))


def _start_json_reader(proc):
    messages = queue.Queue()
    done = threading.Event()

    def _run():
        try:
            _read_process_stream(proc.stdout, messages.put)
        except Exception as exc:
            messages.put(exc)
        finally:
            messages.put(None)
            done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return messages, done, thread


def _start_stderr_reader(proc, tail):
    done = threading.Event()

    def _run():
        try:
            _read_process_stream(proc.stderr, lambda line: tail.append(f"{line}\n"))
        except Exception as exc:
            tail.append(f"[reader] {exc}\n")
        finally:
            done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return done, thread


def _drain_stderr(proc):
    _read_process_stream(proc.stderr, lambda line: _sdxl_stderr_tail.append(line))


def _close_process_streams(proc):
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(proc, stream_name, None)
        if stream is None:
            continue
        try:
            stream.close()
        except Exception:
            pass


def _terminate_process(proc, force=False):
    if proc is None:
        return
    try:
        alive = proc.poll() is None
    except Exception:
        alive = True
    if alive:
        sig = signal.SIGKILL if force else signal.SIGTERM
        delivered = False
        if os.name == "posix" and getattr(proc, "_mlx_process_group", False):
            try:
                os.killpg(proc.pid, sig)
                delivered = True
            except (OSError, ProcessLookupError):
                pass
        if not delivered:
            try:
                if force:
                    proc.kill()
                else:
                    proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix" and getattr(proc, "_mlx_process_group", False):
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=5.0)
            except Exception:
                pass
        except Exception:
            try:
                proc.wait()
            except Exception:
                pass
    else:
        try:
            proc.wait(timeout=0.1)
        except Exception:
            pass
    _close_process_streams(proc)


def _join_reader(thread, timeout=1.0):
    if thread is not None:
        thread.join(timeout=timeout)


def _read_engine_message(proc, messages, done, cancel_events, timeout_s, label, tail=None):
    if proc is None or messages is None or done is None:
        raise RuntimeError(f"{label} reader is not initialized")
    deadline = time.monotonic() + timeout_s
    while True:
        if any(event is not None and event.is_set() for event in cancel_events):
            _terminate_process(proc)
            raise GenerationCancelled()
        try:
            message = messages.get(timeout=0.1)
        except queue.Empty:
            if done.is_set() and messages.empty():
                return None
            if proc.poll() is not None and done.is_set():
                return None
            if time.monotonic() >= deadline:
                _terminate_process(proc, force=True)
                detail = "".join(tail)[-2000:] if tail is not None else ""
                suffix = f": {detail}" if detail else ""
                raise TimeoutError(f"{label} timed out after {timeout_s:.0f}s{suffix}")
            continue
        if message is None:
            return None
        if isinstance(message, BaseException):
            raise RuntimeError(f"{label} reader failed: {message}") from message
        return message


def _kill_qwen_process():
    global _qwen_process, _qwen_reader_messages, _qwen_reader_done, _qwen_reader_thread, _qwen_stderr_thread, _qwen_stderr_done
    _cancel_qwen_watchdog()
    proc = _qwen_process
    if proc is not None:
        _terminate_process(proc)
        _remove_qwen_pid(proc.pid)
    _join_reader(_qwen_stderr_thread)
    _join_reader(_qwen_reader_thread)
    if _qwen_process is proc:
        _qwen_process = None
    _qwen_reader_messages = None
    _qwen_reader_done = None
    _qwen_reader_thread = None
    _qwen_stderr_thread = None
    _qwen_stderr_done = None
    if proc is not None:
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass


def _get_taef_decoder(variant: str):
    """Lazy loader and in-memory cache for fast TAEF mini-VAEs."""
    global _taef_models
    if variant not in _taef_models:
        if variant == "taef2":
            from mlx_taef import TAEF2
            _taef_models["taef2"] = TAEF2.from_pretrained(include_encoder=False)
        elif variant == "taef1":
            from mlx_taef import TAEF1
            _taef_models["taef1"] = TAEF1.from_pretrained(include_encoder=False)
        elif variant == "krea2":
            from mlx_taef import Krea2
            _taef_models["krea2"] = Krea2.from_pretrained(include_encoder=False)
        else:
            raise ValueError(f"unknown TAEF variant: {variant}")
    return _taef_models[variant]


class GenerationCancelled(Exception):
    pass


def cancel_current():
    _cancel_event.set()
    for proc in (_qwen_process, globals().get("_sdxl_daemon")):
        try:
            if proc is not None and proc.poll() is None:
                _terminate_process(proc)
        except Exception:
            pass


def model_download_repo(model_id: str, minfo: dict) -> str | None:
    """Resolve the Hugging Face repo whose weights are fetched when (re)installing a model.

    Returns None for models with no remote source (e.g. ``local:`` krea2 checkpoints).

    This must be the repo the pipeline actually loads. It used to special-case
    flux2-klein-4b onto black-forest-labs/FLUX.2-klein-4B on the belief that
    mflux falls back to its own default when model_path is absent. It does not:
    the 4B branch passes model_path=local_arg, and when that resolves the
    configured repo the pipeline reads mlx-community/flux2-klein-4b-4bit. So a
    fresh install downloaded 22.1 GB of upstream BFL weights -- a duplicate
    transformer, a 7.4 GB single-file checkpoint and a pile of demo JPEGs --
    none of which the app ever loads, while the 4.3 GB it does need arrived
    separately on first use.
    """
    if minfo.get("engine") == "sdxl":
        # diffusers-format HF repos; snapshot_download(local_dir=<model_dir>) installs them.
        return minfo.get("repo")
    if model_id == "krea2-turbo":
        return None  # local checkpoint bundle, no remote source
    return minfo.get("repo")


def _model_path_has_incomplete(path: Path) -> bool:
    ok, value = _probe_fs(
        lambda: any(
            candidate.name.endswith((".incomplete", ".part"))
            for candidate in path.rglob("*")
            if candidate.is_file() and ".cache" not in candidate.relative_to(path).parts
        )
    )
    if ok:
        return value
    # Could not tell. Claiming "incomplete" on an unreadable directory is the
    # dangerous direction: it makes the model look un-downloaded and invites a
    # re-fetch of the whole checkpoint.
    return False


def _model_path_has_weights(path: Path) -> bool:
    try:
        indexes = list(path.rglob("*.safetensors.index.json"))
        if indexes:
            for index_path in indexes:
                data = json.loads(index_path.read_text("utf-8"))
                weight_map = data.get("weight_map") if isinstance(data, dict) else None
                if not isinstance(weight_map, dict) or not weight_map:
                    return False
                if any(not (index_path.parent / str(filename)).is_file() for filename in weight_map.values()):
                    return False
            return True
        return any(path.rglob("*.safetensors"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


_MODEL_CACHED_TTL_S = 5.0
_model_cached_cache: tuple[float, str, bool] | None = None


def is_model_cached(model_id: str) -> bool:
    # Short TTL so a directory walk that fails outright cannot flip the
    # answer for every caller that follows within a few seconds.
    global _model_cached_cache
    cached = _model_cached_cache
    now = time.monotonic()
    if cached is not None and cached[1] == model_id and now - cached[0] < _MODEL_CACHED_TTL_S:
        return cached[2]
    result = _is_model_cached_uncached(model_id)
    _model_cached_cache = (now, model_id, result)
    return result


def _is_model_cached_uncached(model_id: str) -> bool:
    minfo = get_model_info(model_id)
    if not minfo:
        return False
    local = resolve_local_model_path(model_id)
    if local is not None:
        if minfo.get("engine") == "sdxl":
            return (local / "model_index.json").is_file() and not _model_path_has_incomplete(local) and _model_path_has_weights(local)
        return not _model_path_has_incomplete(local) and _model_path_has_weights(local)
    if minfo.get("engine") == "sdxl":
        model_dir = minfo.get("model_dir")
        if not model_dir:
            return False
        path = Path(model_dir)
        return (path / "model_index.json").is_file() and not _model_path_has_incomplete(path) and _model_path_has_weights(path)
    if model_id == "krea2-turbo":
        path = ASSET_DIR / "models" / "krea2-turbo-q4"
        return path.is_dir() and not _model_path_has_incomplete(path) and _model_path_has_weights(path)
    repo = model_download_repo(model_id, minfo) or ""
    if "/" not in repo:
        return False
    cache_dir = _hf_repo_cache_dir(repo)
    snapshots = cache_dir / "snapshots"
    if not snapshots.is_dir():
        return False
    try:
        for snapshot in snapshots.iterdir():
            if snapshot.is_dir() and not _model_path_has_incomplete(snapshot) and _model_path_has_weights(snapshot) and any(snapshot.rglob("*.json")):
                return True
    except OSError:
        return False
    return False


def _newest_hf_snapshot(repo_path: Path) -> Path | None:
    """Newest snapshot directory inside an HF cache repo folder, if any."""
    snaps = repo_path / "snapshots"
    if not snaps.is_dir():
        return None
    try:
        candidates = sorted(
            (p for p in snaps.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    return candidates[0] if candidates else None


def _resolve_hf_cache_snapshot(path: Path) -> Path:
    """If ``path`` is an HF cache repo root (``models--org--name``), return its
    newest snapshot directory containing weights; otherwise return it unchanged."""
    snaps = path / "snapshots"
    if not snaps.is_dir():
        return path
    try:
        candidates = sorted(
            (p for p in snaps.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return path
    for candidate in candidates:
        try:
            if any(candidate.rglob("*.safetensors")) or any(candidate.rglob("*.index.json")):
                return candidate
        except OSError:
            continue
    return candidates[0] if candidates else path


def resolve_local_model_path(model_id: str, path: Path | str | None = None) -> Path | None:
    """Normalized local weights directory registered for a model, or None.

    HF-cache repo roots are resolved to their snapshot directory so the returned
    path can be handed straight to mflux's ``model_path`` / the SDXL ``model_dir``.
    """
    if path is None:
        raw = app_settings.model_local_path(model_id)
        if not raw:
            return None
        path = Path(raw).expanduser()
    p = Path(path).expanduser()
    if not p.exists():
        return None
    return _resolve_hf_cache_snapshot(p)


def validate_local_model_dir(model_id: str, path: Path | str) -> tuple[bool, str | None]:
    """Sanity-check that ``path`` plausibly holds this model's weights."""
    p = Path(path).expanduser()
    if not p.exists():
        return False, "path does not exist"
    if not p.is_dir():
        return False, "path is not a directory"
    resolved = _resolve_hf_cache_snapshot(p)
    minfo = get_model_info(model_id)
    if not minfo:
        return False, "unknown model"
    if minfo.get("engine") == "sdxl":
        if (resolved / "model_index.json").exists() or any(resolved.rglob("*.safetensors")):
            return True, None
        return False, "no diffusers model_index.json or .safetensors found in directory"
    try:
        has_weights = any(resolved.rglob("*.safetensors")) or any(
            resolved.rglob("*.safetensors.index.json")
        )
    except OSError as e:
        return False, f"cannot read directory: {e}"
    if not has_weights:
        return False, "no .safetensors weights found in directory"
    return True, None


def list_local_model_sources(model_id: str | None = None) -> list[dict]:
    """Already-on-disk weight locations for the "install from local" UI.

    Lists every repo in the Hugging Face hub cache with a resolved snapshot path,
    annotating whether it looks like the requested model so the UI can rank it."""
    minfo = get_model_info(model_id) if model_id else None
    expected: set[str] = set()
    if minfo:
        for candidate in (
            model_download_repo(minfo.get("id"), minfo),
            minfo.get("repo"),
            minfo.get("id"),
        ):
            if candidate and "/" in str(candidate) and not str(candidate).startswith("local:"):
                expected.add(str(candidate))
    sources: list[dict] = []
    try:
        from huggingface_hub import scan_cache_dir

        for r in scan_cache_dir().repos:
            snapshot = _newest_hf_snapshot(Path(r.repo_path))
            sources.append(
                {
                    "type": "hf_cache",
                    "repo_id": r.repo_id,
                    "path": str(snapshot or r.repo_path),
                    "size_bytes": int(getattr(r, "size_on_disk", 0) or 0),
                    "matches_model": r.repo_id in expected,
                }
            )
    except Exception as e:
        print(f"[generator] local source scan failed: {e}", flush=True)
    sources.sort(key=lambda s: (not s["matches_model"], -s["size_bytes"]))
    return sources


def is_pipeline_loaded(model_id: str, quantization: int = 4, loras: list | None = None, variant: str = "standard") -> bool:
    """Check if the requested model and weights are already warm & resident in unified memory."""
    minfo = get_model_info(model_id) or {}
    if minfo.get("engine") == "sdxl":
        model_dir = str(
            resolve_local_model_path(model_id)
            or minfo.get("model_dir")
            or (ASSET_DIR / "models" / "juggernaut-xl-lightning")
        )
        return (
            _qwen_process is None
            and _sdxl_daemon is not None
            and _sdxl_daemon.poll() is None
            and _current_sdxl_model == model_dir
        )
    key = _make_pipeline_key(model_id, quantization, loras or [], variant)
    return (
        _qwen_process is None
        and _sdxl_daemon is None
        and _pipeline is not None
        and _current_pipeline_key == key
    )


def _dir_size(path) -> int:
    total = 0
    try:
        for p in Path(path).rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _hf_repo_cache_dir(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"


def model_disk_usage(model_id: str, minfo: dict | None = None) -> int:
    """Installed-on-disk size in bytes for a model (SDXL local dir, krea2 bundle,
    or the HF hub cache snapshot for mflux repos)."""
    try:
        minfo = minfo or get_model_info(model_id)
    except Exception:
        return 0
    if minfo:
        local = resolve_local_model_path(model_id)
        if local is not None:
            return _dir_size(local)
        engine = minfo.get("engine")
        if engine == "sdxl":
            d = minfo.get("model_dir")
            if not d:
                return 0
            return _dir_size(d)
        if model_id == "krea2-turbo":
            return _dir_size(ASSET_DIR / "models" / "krea2-turbo-q4")
        repo = model_download_repo(model_id, minfo) or ""
        if "/" in repo:
            return _dir_size(_hf_repo_cache_dir(repo))
    return 0


def uninstall_model(model_id: str) -> tuple[bool, str | None]:
    with _model_maintenance_lock:
        return _uninstall_model(model_id)


def _uninstall_model(model_id: str) -> tuple[bool, str | None]:
    """Remove a model's weights from disk. Refuses (False, reason) while the model
    is queued/generating or a weights download is in-flight for it."""
    minfo = get_model_info(model_id)
    if not minfo:
        return False, "unknown model"
    try:
        import state

        with state._JOBS_LOCK:
            for j in state.JOBS.values():
                if j.get("status") not in ("generating", "queued"):
                    continue
                req = j.get("request")
                if req is None:
                    continue
                rid = getattr(req, "model", "") if not isinstance(req, dict) else req.get("model", "")
                if rid == model_id or rid == minfo.get("repo") or (get_model_info(rid) or {}).get("id") == model_id:
                    return False, f"{model_id} has an active generation job"
        with state._MODEL_DOWNLOAD_LOCK:
            for t in state.MODEL_DOWNLOAD_TASKS.values():
                if t.get("model_id") == model_id and t.get("status") == "downloading":
                    return False, f"{model_id} is currently downloading"
    except Exception as e:
        print(f"[generator] uninstall guard error: {e}", flush=True)

    configured_local = app_settings.configured_model_local_path(model_id)
    local_raw = app_settings.model_local_path(model_id)
    if configured_local and not local_raw:
        app_settings.update_settings({"model_paths": {model_id: ""}})
        return True, "local path unlinked; files left on disk"
    unlinked = False
    try:
        removed_any = False
        if local_raw:
            # Pointer install: unlink only; never delete the user's own files.
            app_settings.update_settings({"model_paths": {model_id: ""}})
            removed_any = True
            unlinked = True
        elif minfo.get("engine") == "sdxl":
            d = minfo.get("model_dir")
            if d and Path(d).exists():
                if ASSET_DIR != DATA_DIR and Path(d).resolve().is_relative_to(ASSET_DIR.resolve()):
                    return False, "model is in the shared asset store; remove it from the source application"
                import shutil

                shutil.rmtree(Path(d), ignore_errors=True)
                removed_any = not Path(d).exists()
                if not removed_any:
                    return False, "failed to remove model directory"
        elif model_id == "krea2-turbo":
            import shutil

            d = ASSET_DIR / "models" / "krea2-turbo-q4"
            if d.exists():
                if ASSET_DIR != DATA_DIR:
                    return False, "model is in the shared asset store; remove it from the source application"
                shutil.rmtree(d, ignore_errors=True)
                removed_any = not d.exists()
                if not removed_any:
                    return False, "failed to remove local model bundle"
        else:
            repo = model_download_repo(model_id, minfo) or ""
            if "/" not in repo:
                return False, "model has no removable weights"
            removed_any = _delete_hf_repo_cache(repo)
    except Exception as e:
        return False, f"uninstall error: {e}"

    with _lock:
        if _current_pipeline_model == model_id:
            _drop_mflux_pipeline()
        if minfo.get("engine") == "sdxl" and _sdxl_daemon is not None and _sdxl_daemon.poll() is None:
            _kill_sdxl_daemon()
        if minfo.get("engine") == "qwen" and _qwen_process is not None and _qwen_process.poll() is None:
            _kill_qwen_process()

    if unlinked:
        return True, "local path unlinked; files left on disk"
    return True, None


def _delete_hf_repo_cache(repo_id: str) -> bool:
    """Remove one repo from ~/.cache/huggingface/hub (snapshots + blobs)."""
    removed_any = False
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        for r in info.repos:
            if r.repo_id == repo_id:
                r.delete()
                removed_any = True
                break
    except Exception as e:
        print(f"[generator] hf cache delete failed: {e}", flush=True)
    # belt & braces: drop a half-removed / stale directory (e.g. interrupted downloads)
    stale = _hf_repo_cache_dir(repo_id)
    try:
        if stale.exists():
            import shutil

            shutil.rmtree(stale, ignore_errors=True)
    except OSError:
        pass
    return removed_any or not stale.exists()


_STORAGE_STATS_TTL_S = 60.0
_storage_stats_cache: tuple[float, dict] | None = None


def _storage_stats() -> dict:
    """Disk counters for the engine panel, cached for a minute.

    Walking GENERATED_DIR costs ~32ms with ~4400 entries, and the frontend polls
    /api/engine/status every 5s from a component that is mounted on every tab, so
    the scan was burning ~0.6s of CPU per minute away from the generation thread.
    The numbers only move when an image is written or deleted, which the panel
    already reflects within one poll anyway.
    """
    global _storage_stats_cache
    now = time.monotonic()
    cached = _storage_stats_cache
    if cached is not None and now - cached[0] < _STORAGE_STATS_TTL_S:
        return dict(cached[1])
    stats: dict = {}
    try:
        images = [f for f in GENERATED_DIR.iterdir() if f.is_file() and f.suffix.lower() in (".png", ".jpeg", ".jpg", ".webp", ".heic")]
        stats["generated_dir"] = str(GENERATED_DIR)
        stats["image_count"] = len(images)
        stats["image_bytes"] = sum(f.stat().st_size for f in images)
        stats["settings_file"] = str(app_settings.SETTINGS_FILE)
        stats["settings_exists"] = app_settings.SETTINGS_FILE.exists()
    except Exception as e:
        stats["error"] = str(e)
    _storage_stats_cache = (now, stats)
    return dict(stats)


def get_engine_status() -> dict:
    """Live snapshot of the Apple Silicon / MLX runtime, resident pipelines and
    disk state — drives the ⚙️ Parameters > Engine & GPU section."""
    status: dict = {
        "meta": {},
        "metal": {},
        "wired": {},
        "mflux": {},
        "sdxl": {},
        "taef": [],
        "storage": {},
    }
    try:
        import platform

        status["meta"] = {
            "engine": "MLX",
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
    except Exception:
        pass

    try:
        import mlx.core as mx

        dev_info = getattr(mx, "device_info", None) or getattr(mx.metal, "device_info", None)
        if dev_info:
            d = dev_info()
            status["metal"]["memory_size"] = d.get("memory_size")
            status["metal"]["recommended_max_working_set_size"] = (
                d.get("max_recommended_working_set_size") or d.get("recommended_max_working_set_size")
            )
            status["metal"]["name"] = d.get("name") or d.get("model")
    except Exception:
        pass

    try:
        import mlx.core as mx

        peak = getattr(mx, "get_peak_memory", None) or getattr(mx.metal, "get_peak_memory", None)
        if peak:
            status["metal"]["peak_memory"] = peak()
        cache = getattr(mx, "get_cache_memory", None) or getattr(mx.metal, "get_cache_memory", None)
        if cache:
            status["metal"]["cache_memory"] = cache()
    except Exception:
        pass

    # auto/derived are reported separately from the configured number so the UI can say
    # "Auto (10.88 GB)" instead of showing a derived figure as though the user had typed
    # it -- which would then be persisted on the next save and freeze a machine-specific
    # constant into the settings again, the exact bug this change exists to remove.
    status["wired"] = {
        "generic_limit_gb": _wired_limit_gb(),
        "generic_auto": _wired_limit_gb() is None,
        "generic_derived_gb": _derived_wired_budget_gb(),
        "generic_budget_bytes": _wired_limit_bytes(),
        "krea_limit_gb": _krea_wired_limit_gb(),
        "krea_auto": _krea_wired_limit_gb() is None,
        "krea_derived_gb": _derived_wired_budget_gb(),
        "krea_budget_bytes": _krea_wired_limit_bytes(),
    }

    now = time.time()
    mflux_idle = _mflux_idle_kill_s()
    status["mflux"] = {
        "resident": _pipeline is not None and _sdxl_daemon is None and _qwen_process is None,
        "model": _current_pipeline_model,
        "idle_kill_s": mflux_idle,
        "watchdog_armed": _mflux_watchdog is not None and _mflux_watchdog.is_alive(),
        "idle_since": _mflux_idle_since,
        "seconds_until_release": (
            max(0, mflux_idle - (now - _mflux_idle_since)) if _mflux_idle_since else None
        ),
        "prompt_cache_size": len(_prompt_cache),
    }
    sdxl_idle = _sdxl_idle_kill_s()
    status["sdxl"] = {
        "resident": _pipeline is None and _qwen_process is None and _sdxl_daemon is not None and _sdxl_daemon.poll() is None,
        "pid": _sdxl_daemon.pid if _sdxl_daemon is not None else None,
        "model": Path(_current_sdxl_model).name if _current_sdxl_model else None,
        "idle_kill_s": sdxl_idle,
        "watchdog_armed": _sdxl_watchdog is not None and _sdxl_watchdog.is_alive(),
        "idle_since": _sdxl_idle_since,
        "seconds_until_release": (
            max(0, sdxl_idle - (now - _sdxl_idle_since)) if _sdxl_idle_since else None
        ),
        "stderr_tail": "".join(_sdxl_stderr_tail)[-1200:],
    }
    qwen_idle = _qwen_idle_kill_s()
    status["qwen"] = {
        "resident": _pipeline is None and _sdxl_daemon is None and _qwen_process is not None and _qwen_process.poll() is None,
        "pid": _qwen_process.pid if _qwen_process is not None else None,
        "idle_kill_s": qwen_idle,
        "watchdog_armed": _qwen_watchdog is not None and _qwen_watchdog.is_alive(),
        "idle_since": _qwen_idle_since,
        "seconds_until_release": (
            max(0, qwen_idle - (now - _qwen_idle_since)) if _qwen_idle_since else None
        ),
    }
    status["taef"] = sorted(_taef_models.keys())

    status["storage"].update(_storage_stats())
    return status


class _ProgressCallback:
    def __init__(self, cb, cancel_event=None, phase_cb=None):
        self._cb = cb
        self._cancel_event = cancel_event
        self._phase_cb = phase_cb

    def call_before_loop(self, **kwargs):
        if self._phase_cb:
            self._phase_cb("compiling", "Compiling Metal graph & encoding prompt...")

    def call_in_loop(self, t, **kwargs):
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise GenerationCancelled()
        if self._phase_cb:
            self._phase_cb("generating", f"Denoising step {t + 1}...")
        self._cb(t)

    def call_after_loop(self, **kwargs):
        if self._phase_cb:
            self._phase_cb("saving", "Decoding VAE latents & saving image...")


def _normalize_lora_path(path: str) -> str:
    raw = str(path)
    candidate = Path(raw).expanduser()
    if candidate.exists() or raw.startswith(("/", "~", "./", "../")):
        return str(candidate.resolve())
    return raw


def _normalize_loras(loras: list[dict] | None) -> list[dict]:
    normalized = []
    for lora in loras or []:
        if not isinstance(lora, dict) or not lora.get("path"):
            continue
        item = dict(lora)
        item["path"] = _normalize_lora_path(item["path"])
        normalized.append(item)
    return normalized


_STAT_RETRY_ATTEMPTS = 3
_STAT_RETRY_DELAY_S = 0.05


def _probe_fs(fn):
    """Run a filesystem probe, retrying transient OSError. Returns (ok, value).

    The model store lives on an external volume, which was
    observed returning a transient ENOENT from stat() in the middle of a
    benchmark. Every consumer used to read a single failed probe as a fact
    about the filesystem, and each of those readings was expensive:

      _lora_signature  -> a different pipeline key -> full 20-40s reload of an
                          otherwise warm pipeline
      _model_path_has_incomplete -> "the download is incomplete" -> the Engine
                          panel offers to re-download 13GB of weights
      _model_is_fully_cached -> installed=False in /api/models

    A retry costs at most 100ms and only on an actual failure.
    """
    last = None
    for attempt in range(_STAT_RETRY_ATTEMPTS):
        try:
            return True, fn()
        except (OSError, ValueError) as e:
            last = e
            if attempt + 1 < _STAT_RETRY_ATTEMPTS:
                time.sleep(_STAT_RETRY_DELAY_S)
    return False, last


_lora_sig_cache: dict = {}


def _lora_signature(path: str) -> tuple:
    normalized = _normalize_lora_path(path)
    ok, stat = _probe_fs(lambda: Path(normalized).stat())
    if ok:
        signature = (normalized, stat.st_size, stat.st_mtime_ns)
        _lora_sig_cache[normalized] = signature
        return signature
    # Never let a failed probe invalidate a warm pipeline: fall back to the last
    # signature actually observed for this path. Only a path we have never seen
    # reports None, which is what a genuinely missing file should look like.
    previous = _lora_sig_cache.get(normalized)
    if previous is not None:
        return previous
    return (normalized, None, None)


def _make_pipeline_key(model_id: str, quantization: int, loras: list[dict], variant: str = "standard") -> tuple:
    return (
        model_id,
        quantization,
        tuple(sorted(_lora_signature(l["path"]) for l in loras)),
        variant,
    )


def _update_pipeline_lora_scales(pipe, scales: list[float]):
    if not scales or not hasattr(pipe, "transformer"):
        return
    try:
        from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
        from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear

        for mod in pipe.transformer.modules():
            if isinstance(mod, LoRALinear):
                mod.scale = scales[0]
            elif isinstance(mod, FusedLoRALinear):
                for idx, l in enumerate(mod.loras):
                    if idx < len(scales):
                        l.scale = scales[idx]
        if hasattr(pipe, "lora_scales"):
            pipe.lora_scales = scales
    except Exception as e:
        print(f"[generator] note on dynamic lora scaling: {e}", flush=True)


def _apply_prompt_cache(pipe):
    """Cache text-encoder output keyed by prompt with true LRU eviction.
    Encoding runs a Qwen3 forward pass which is a significant fraction of a short generation."""
    import mlx.core as mx

    orig_encode = pipe._encode_prompt_pair

    def cached_encode(*, prompt, negative_prompt, guidance):
        if prompt in _prompt_cache:
            _prompt_cache.move_to_end(prompt)
            prompt_embeds, text_ids = _prompt_cache[prompt]
        else:
            prompt_embeds, text_ids = orig_encode(
                prompt=prompt,
                negative_prompt=None,
                guidance=1.0,  # encode only the positive prompt here
            )[0:2]
            mx.eval(prompt_embeds, text_ids)
            _cache_prompt(prompt, (prompt_embeds, text_ids))
        negative_embeds, negative_ids = None, None
        if guidance is not None and guidance > 1.0 and negative_prompt is not None:
            neg_key = f"\x00neg:{negative_prompt}"
            if neg_key in _prompt_cache:
                _prompt_cache.move_to_end(neg_key)
                negative_embeds, negative_ids = _prompt_cache[neg_key]
            else:
                negative_embeds, negative_ids = orig_encode(
                    prompt=negative_prompt,
                    negative_prompt=None,
                    guidance=1.0,
                )[0:2]
                mx.eval(negative_embeds, negative_ids)
                _cache_prompt(neg_key, (negative_embeds, negative_ids))
        return prompt_embeds, text_ids, negative_embeds, negative_ids

    pipe._encode_prompt_pair = cached_encode


def _apply_prompt_cache_txt(pipe):
    """Shared LRU for Z-Image / Krea2 text encoders.

    Both mflux engines expose ``_encode_prompts(prompt, negative_prompt,
    guidance) -> (embeds, neg_embeds)`` and re-encode on every generate_image
    call even though the pipe (with its TextEncoder) is cached in memory.
    prompt/negative/guidance fully determine the embeddings, so memoizing the
    raw arrays (never mutated) is bit-identical. Uses _PROMPT_CACHE_MAX_SIZE;
    tuple keys cannot collide with the string keys used by the FLUX wrapper.
    """
    orig_encode = pipe._encode_prompts

    def cached_encode(*, prompt, negative_prompt=None, guidance=1.0):
        import mlx.core as mx

        key = (prompt, negative_prompt or "", float(guidance))
        if key in _prompt_cache:
            _prompt_cache.move_to_end(key)
            embeds, neg = _prompt_cache[key]
            return embeds, neg
        embeds, neg = orig_encode(prompt=prompt, negative_prompt=negative_prompt, guidance=guidance)
        mx.eval(embeds)
        if neg is not None:
            mx.eval(neg)
        _cache_prompt(key, (embeds, neg))
        return embeds, neg

    pipe._encode_prompts = cached_encode


def _install_flux_text_encoder_tail_trim(pipe):
    if os.environ.get("MLX_DISABLE_FLUX_TE_TRIM") == "1":
        return
    text_encoder = getattr(pipe, "text_encoder", None)
    layers = getattr(text_encoder, "layers", None)
    if text_encoder is None or layers is None or len(layers) != 36:
        return
    if getattr(text_encoder, "num_hidden_layers", None) != 36:
        return
    text_encoder.layers = layers[:28]
    text_encoder.num_hidden_layers = 28
    text_encoder._mlxdiffusion_te_tail_trimmed = True


def _install_krea_sampler_scalar_cache():
    if os.environ.get("MLX_DISABLE_KREA_SIGMA_CACHE") == "1":
        return
    try:
        import mlx.core as mx
        from mflux.models.krea2.model.krea2_sampler import ErSdeStepper

        if getattr(ErSdeStepper, "_mlxdiffusion_sigma_cache", False):
            return

        def step(self, i, x, v, denoised):
            sigmas = self.sigmas
            zero_flags = getattr(self, "_sigma_zero_flags", None)
            if zero_flags is None:
                mx.eval(sigmas)
                zero_flags = [float(value.item()) == 0.0 for value in sigmas]
                self._sigma_zero_flags = zero_flags
            if zero_flags[i + 1]:
                self._old_denoised = denoised
                return denoised

            ls, lt = sigmas[i], sigmas[i + 1]
            r = self._noise_scaler(lt) / self._noise_scaler(ls)

            x = r * x + (1 - r) * denoised

            stage = min(self.max_stage, i + 1)
            if stage >= 2 and self._old_denoised is not None:
                dt = lt - ls
                step_size = -dt / self.NUM_POINTS
                lam_pos = lt + self._point_indice * step_size
                scaled = self._noise_scaler(lam_pos)

                s = mx.sum(1.0 / scaled) * step_size
                denoised_d = (denoised - self._old_denoised) / (ls - sigmas[i - 1])
                x = x + (dt + s * self._noise_scaler(lt)) * denoised_d

                if stage >= 3 and self._old_denoised_d is not None:
                    s_u = mx.sum((lam_pos - ls) / scaled) * step_size
                    denoised_u = (denoised_d - self._old_denoised_d) / ((ls - sigmas[i - 2]) / 2)
                    x = x + ((dt**2) / 2 + s_u * self._noise_scaler(lt)) * denoised_u
                self._old_denoised_d = denoised_d

            if self.s_noise > 0:
                self._key, sub = mx.random.split(self._key)
                noise = mx.random.normal(x.shape, key=sub)
                var = lt**2 - (ls**2) * (r**2)
                std = mx.sqrt(mx.maximum(var, 0.0))
                x = x + noise * self.s_noise * std

            self._old_denoised = denoised
            return x

        ErSdeStepper.step = step
        ErSdeStepper._mlxdiffusion_sigma_cache = True
    except Exception:
        return


def _install_zit_dynamic_padding(pipe):
    if os.environ.get("MLX_DISABLE_ZIT_DYNAMIC_PADDING") == "1":
        return
    tokenizers = getattr(pipe, "tokenizers", None)
    tokenizer = tokenizers.get("z_image") if hasattr(tokenizers, "get") else None
    if tokenizer is not None and getattr(tokenizer, "padding", None) == "max_length":
        tokenizer.padding = "longest"


def _install_rope_cache(model_id: str):
    """Memoize each engine's position-embedding cos/sin across denoise steps.

    cos/sin depend only on position ids, which are constant for one generation
    (FLUX.2 img_ids/txt_ids; Z-Image x_pos_ids/cap_pos_ids from patchify;
    Krea2 pos from image/text lengths) but every mflux engine recomputes them on
    each forward. ids are never mutated after creation, so caching per
    (shape, dtype, flat values) is bit-identical while skipping the per-step
    trig work. Class-level patch — mlx dispatches __call__ via the class, so
    instance monkeypatching never fires. Idempotent per rope class.
    """
    rope_classes = {
        "flux2-klein-4b": ("mflux.models.flux2.model.flux2_transformer.pos_embed", "Flux2PosEmbed"),
        "flux2-klein-9b": ("mflux.models.flux2.model.flux2_transformer.pos_embed", "Flux2PosEmbed"),
        "z-image-turbo": ("mflux.models.z_image.model.z_image_transformer.rope_embedder", "RopeEmbedder"),
        "krea2-turbo": ("mflux.models.krea2.model.krea2_transformer.rope_embedder", "Krea2RopeEmbedder"),
    }
    if os.environ.get("MLX_DISABLE_ROPE_CACHE") == "1":
        return
    spec = rope_classes.get(model_id)
    if spec is None:
        return
    import importlib
    try:
        mod = importlib.import_module(spec[0])
        cls = getattr(mod, spec[1])
    except Exception as e:
        print(f"[generator] rope cache unavailable for {model_id}: {e}", flush=True)
        return
    if spec in _ROPE_CACHE_INSTALLED:
        return
    orig = cls.__call__
    cache = OrderedDict()

    def _call(self, ids):
        key = (ids.shape, str(ids.dtype), tuple(ids.reshape(-1).tolist()))
        hit = cache.get(key)
        if hit is None:
            hit = orig(self, ids)
            cache[key] = hit
            total = sum(_value_bytes(v) for v in cache.values())
            while cache and (len(cache) > 8 or total > 8 * (1 << 20)):
                _, value = cache.popitem(last=False) if isinstance(cache, OrderedDict) else (None, None)
                if value is None:
                    cache.clear()
                    break
                total -= _value_bytes(value)
        return hit

    cls.__call__ = _call
    _ROPE_CACHE_INSTALLED.add(spec)
    print(f"[generator] installed rope memo cache for {model_id} ({spec[1]}, lossless; ids constant across steps)", flush=True)


_ROPE_CACHE_INSTALLED = set()


def _get_pipeline(model_id: str, quantization: int, loras: list[dict], variant: str = "standard", phase_cb=None):
    global _pipeline, _current_pipeline_key, _current_pipeline_model
    loras = sorted(_normalize_loras(loras), key=lambda item: item["path"])
    key = _make_pipeline_key(model_id, quantization, loras, variant)
    if _pipeline is not None and _current_pipeline_key == key:
        _kill_qwen_process()
        _kill_sdxl_daemon()
        return _pipeline
    info = get_model_info(model_id)
    if not info:
        raise ValueError(f"unknown model: {model_id}")
    if phase_cb is not None:
        if is_model_cached(model_id):
            phase_cb("loading_model", f"Loading {info['label']} weights into Apple Silicon unified memory...")
        else:
            phase_cb("downloading", f"Downloading {info['label']} weights from repository...")
    _kill_qwen_process()
    _kill_sdxl_daemon()
    _drop_mflux_pipeline()
    scales = [l.get("scale", 1.0) for l in loras]
    local_path = resolve_local_model_path(model_id)
    local_arg = str(local_path) if local_path is not None else None
    if model_id == "flux2-klein-4b":
        if variant == "edit":
            from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

            _pipeline = Flux2KleinEdit(
                model_path=local_arg,
                quantize=quantization,
                lora_paths=[l["path"] for l in loras] or None,
                lora_scales=scales or None,
                bake_lora=False,
            )
        else:
            from mflux.models.flux2 import Flux2Klein

            _pipeline = Flux2Klein(
                model_path=local_arg,
                quantize=quantization,
                lora_paths=[l["path"] for l in loras] or None,
                lora_scales=scales or None,
                bake_lora=False,
            )
    elif model_id == "flux2-klein-9b":
        from mflux.models.common.config.model_config import ModelConfig

        if variant == "edit":
            from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

            _pipeline = Flux2KleinEdit(
                model_config=ModelConfig.flux2_klein_9b(),
                model_path=local_arg or info["repo"],
                quantize=quantization,
                lora_paths=[l["path"] for l in loras] or None,
                lora_scales=scales or None,
                bake_lora=False,
            )
        else:
            from mflux.models.flux2 import Flux2Klein

            _pipeline = Flux2Klein(
                model_config=ModelConfig.flux2_klein_9b(),
                model_path=local_arg or info["repo"],
                quantize=quantization,
                lora_paths=[l["path"] for l in loras] or None,
                lora_scales=scales or None,
                bake_lora=False,
            )
    elif model_id == "z-image-turbo":
        from mflux.models.z_image import ZImageTurbo

        _pipeline = ZImageTurbo(
            quantize=quantization,
            model_path=local_arg or info["repo"],
            lora_paths=[l["path"] for l in loras] or None,
            lora_scales=scales or None,
            bake_lora=False,  # runtime fp16 adapters — baking erodes deltas in q4 weights
        )
        _install_zit_dynamic_padding(_pipeline)
    elif model_id == "krea2-turbo":
        from mflux.models.krea2 import Krea2
        import mlx.core as mx
        import mlx.nn as nn

        _install_krea_sampler_scalar_cache()
        local = ASSET_DIR / "models" / "krea2-turbo-q4"
        krea_source = local_arg or str(local)

        # Load the quantised encoder from disk instead of letting mflux materialise
        # 7.5GB of bf16 first. The flag has to be set BEFORE Krea2() is constructed,
        # because by the time it returns the bf16 weights are already resident.
        disk_te = None
        if os.environ.get("MLX_DISABLE_KREA_TE_CACHE") != "1":
            _install_krea_te_bf16_skip()
            disk_te = _krea_te_from_disk(krea_source, bits=4, group_size=64)
            _krea_te_skip_bf16["on"] = disk_te is not None
        try:
            _pipeline = Krea2(
                quantize=4,
                model_path=krea_source,
                lora_paths=[l["path"] for l in loras] or None,
                lora_scales=scales or None,
                bake_lora=False,
            )
        finally:
            _krea_te_skip_bf16["on"] = False

        if disk_te is not None:
            # The encoder module mflux built is now weightless (we skipped its load),
            # so drop it before adopting the on-disk copy.
            stale = getattr(_pipeline, "text_encoder", None)
            if stale is not None:
                _pipeline.text_encoder = None
                del stale
                gc.collect()
                mx.clear_cache()
            _pipeline.text_encoder = disk_te
        # Tiled VAE decode.
        #
        # mflux 0.20.0 made --vae-tiling actually work on Krea 2, where the flag was
        # previously accepted and ignored; the changelog puts the peak decode at ~7GB for
        # a 1024px output, cut to ~3GB tiled. That matters here because the decode is the
        # second place this pipeline can run out of room, right after the weights load.
        #
        # Krea2 is a correct target: FLUX.2 Klein, Klein edit, Lens and Ideogram 4 all set
        # supports_implicit_tiling=False, because their decoder takes normalisation
        # statistics per tile and shifts colour between tiles. Krea2 opts in.
        #
        # Krea2.__init__ does not accept tiling_config, but self.tiling_config is read at
        # decode time (krea2.py:186) and at encode time (:167) and defaults to None, so
        # assigning it after construction is enough -- no patch to mflux is needed.
        _apply_krea_vae_tiling(_pipeline)

# Explicitly quantize the Qwen3-VL text encoder (7.5GB bf16 -> ~1.9GB q4).
        # mflux skips it by default (skip_quantization=True), leaving 7.5GB of raw bf16 weight
        # which causes resident memory to exceed 15GB and trigger Metal CommandBuffer OOM on 1
        # The pass costs 65-69s, so it is cached in-process AND on disk: the first build
        # pays it once and writes the result out, and every later process reads the q4
        # weights directly instead of loading bf16 at all.
        if disk_te is None and hasattr(_pipeline, "text_encoder") and _pipeline.text_encoder is not None:
            _krea_quantized_text_encoder(_pipeline, krea_source, bits=4, group_size=64)
    elif model_id == "qwen-image-2.1":
        from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

        # mlx-community repo ships EVERYTHING pre-quantized (TE q4 + transformer q4 +
        # VAE bf16). The fork's qwen21_initializer rebuilds packed layers as
        # QuantizedLinear/QuantizedEmbedding at load (patch) and, with the TE
        # skip_quantization/precision overrides, keeps them quantized - no bf16 TE.
        _pipeline = QwenImage21(
            quantize=quantization,
            model_path=local_arg or info["repo"],
        )
    else:
        raise ValueError(f"unknown model: {model_id}")
    _current_pipeline_key = key
    _current_pipeline_model = model_id
    _prompt_cache.clear()
    if model_id in ("flux2-klein-4b", "flux2-klein-9b"):
        _install_flux_text_encoder_tail_trim(_pipeline)
    if model_id in ("flux2-klein-4b", "flux2-klein-9b"):
        if os.environ.get("MLX_DISABLE_ROPE_CACHE") != "1":
            _install_rope_cache(model_id)
            _apply_prompt_cache(_pipeline)
    elif model_id in ("z-image-turbo", "krea2-turbo"):
        _install_rope_cache(model_id)
        if os.environ.get("MLX_DISABLE_ROPE_CACHE") != "1":
            _apply_prompt_cache_txt(_pipeline)
    return _pipeline


def _enrich_loras_with_registry(loras: list[dict] | None) -> list[dict]:
    """Enrich LoRA list with civitai_version_id, civitai_model_name, and triggers from loras.json."""
    if not loras:
        return []
    loras_file = DATA_DIR / "loras.json"
    registry = []
    if loras_file.exists():
        try:
            registry = json.loads(loras_file.read_text())
        except Exception:
            registry = []

    enriched = []
    for lora in loras:
        if not isinstance(lora, dict):
            continue
        entry = dict(lora)
        path_str = entry.get("path", "")
        name_str = entry.get("name", "")
        match = next(
            (r for r in registry if (path_str and r.get("path") == path_str) or (name_str and r.get("name") == name_str)),
            None
        )
        if match:
            entry.setdefault("name", match.get("name"))
            if match.get("sha256"):
                entry["sha256"] = match["sha256"]
            if match.get("civitai_version_id"):
                entry["modelVersionId"] = match["civitai_version_id"]
                entry["civitai_version_id"] = match["civitai_version_id"]
            if match.get("civitai_model_id"):
                entry["modelId"] = match["civitai_model_id"]
                entry["civitai_model_id"] = match["civitai_model_id"]
            if match.get("civitai_model_name"):
                entry["civitai_model_name"] = match["civitai_model_name"]
            if match.get("civitai_version_name"):
                entry["civitai_version_name"] = match["civitai_version_name"]
            if match.get("triggers") and not entry.get("triggers"):
                entry["triggers"] = match["triggers"]
        enriched.append(entry)
    return enriched

def _validate_dimensions(width, height, minfo):
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError) as e:
        raise ValueError("width and height must be integers") from e
    if width < 128 or height < 128 or width > 2048 or height > 2048:
        raise ValueError("width and height must be between 128 and 2048")
    alignment = 8 if minfo.get("engine") == "sdxl" else 16
    if width % alignment or height % alignment:
        raise ValueError(f"{minfo['label']} dimensions must be multiples of {alignment}")
    return width, height


def _validate_sampler(sampler, minfo):
    allowed = minfo.get("samplers") or []
    if not allowed:
        value = str(sampler or "").lower()
        model_id = minfo.get("id")
        accepted = {
            "flux2-klein-4b": {"euler", "flow_match_euler_discrete"},
            "flux2-klein-9b": {"euler", "flow_match_euler_discrete"},
            "z-image-turbo": {"euler", "linear", "flow_match_euler_discrete"},
            "krea2-turbo": {"euler", "er_sde", "linear"},
        }.get(model_id, set())
        if value and value not in accepted:
            raise ValueError(f"{minfo['label']} does not support custom samplers")
        return "Euler"
    value = sampler or (minfo.get("default_sampler") or allowed[0])
    if value not in allowed:
        # A sampler picked on one model and carried over to another whose list differs is
        # a stale UI value, not a hostile one: the picker only ever offers samplers from
        # `allowed`, so by the time it arrives here it is known to be a value this app
        # itself produced. Refusing the whole generation over it cost the user their
        # prompt for a reason they could neither see nor fix, so fall back to the model's
        # default instead. Unknown values from anywhere else are still rejected upstream.
        fallback = minfo.get("default_sampler") or allowed[0]
        print(
            f"[generator] sampler {value!r} not offered by {minfo.get('id')}; "
            f"falling back to {fallback!r}",
            flush=True,
        )
        return fallback
    return value


def _effective_guidance(minfo, guidance, negative_prompt=""):
    if not minfo.get("supports_guidance"):
        return 1.0
    value = minfo.get("default_guidance", 1.0) if guidance is None else guidance
    try:
        value = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError("guidance must be numeric") from e
    if not math.isfinite(value) or value < 0:
        raise ValueError("guidance must be a finite non-negative number")
    if minfo.get("id") == "qwen-image-2.1" and negative_prompt and value <= 1.0:
        return 3.0
    return value


def _effective_steps(minfo, steps, overrides=None):
    """Clamp the step count for guidance-distilled models.

    A distilled checkpoint (SDXL-Lightning, FLUX.2-klein, Z-Image Turbo) is
    trained for a handful of steps. The request schema and the settings UI both
    allow up to 50, and app_settings permits a persisted model_defaults.steps of
    50, so a stray override silently turned a 4-step run into a 12x slower one
    with no quality gain - the extra steps land on a distribution the model was
    never distilled for. Non-distilled models are left alone.
    """
    value = int(minfo.get("default_steps", 1) if steps is None else steps)
    if not minfo.get("is_distilled"):
        return value
    cap = _DISTILLED_MAX_STEPS
    override = (overrides or {}).get("steps")
    if override is not None:
        try:
            cap = min(cap, max(1, int(override)))
        except (TypeError, ValueError):
            pass
    if value > cap:
        print(
            f"[generator] {minfo.get('label')}: {value} steps clamped to {cap} "
            "(guidance-distilled model)",
            flush=True,
        )
    return min(value, cap)


def _effective_quantization(model_id, quantization):
    try:
        value = int(quantization)
    except (TypeError, ValueError) as e:
        raise ValueError("quantization must be an integer") from e
    if model_id in ("krea2-turbo", "qwen-image-2.1"):
        if value != 4:
            raise ValueError(f"{model_id} supports only 4-bit quantization")
        return 4
    if value not in (4, 8):
        raise ValueError("quantization must be 4 or 8")
    return value


def _validate_reference_capability(minfo, ref_paths, image_strength):
    if ref_paths:
        if not minfo.get("supports_ref"):
            raise ValueError(f"reference images are not supported on {minfo['label']}")
        maximum = int(minfo.get("max_reference_images") or 1)
        if len(ref_paths) > maximum:
            raise ValueError(
                f"{minfo['label']} accepts at most {maximum} reference image(s)"
            )
    if image_strength is not None:
        try:
            strength = float(image_strength)
        except (TypeError, ValueError) as e:
            raise ValueError("image_strength must be numeric") from e
        if not math.isfinite(strength) or not 0.0 < strength <= 1.0:
            raise ValueError("image_strength must be in the interval (0, 1]")


def _resolve_reference_path(value: str) -> str:
    raw = str(value or "").strip()
    if not raw or "\x00" in raw:
        raise ValueError("invalid reference image path")
    candidate = Path(raw).expanduser()
    roots = (GENERATED_DIR.resolve(), (DATA_DIR / "uploads").resolve())
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not any(resolved.is_relative_to(root) for root in roots):
            raise ValueError("reference image is outside the gallery and uploads directories")
        return str(resolved)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", raw):
        raise ValueError("invalid reference image name")
    names = [raw]
    if not Path(raw).suffix:
        names.extend(f"{raw}{ext}" for ext in (".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"))
    for root in roots:
        for name in names:
            option = (root / name).resolve()
            if option.is_file() and option.is_relative_to(root):
                return str(option)
    raise ValueError(f"reference image not found: {raw}")


def _qwen_local_model(minfo):
    local = resolve_local_model_path("qwen-image-2.1")
    return str(local.resolve()) if local is not None else str(minfo.get("repo") or "mlx-community/Qwen-Image-2.1-MLX-4bit")


def _flow_scheduler_id(model_id):
    return {
        "flux2-klein-4b": "flow_match_euler_discrete",
        "flux2-klein-9b": "flow_match_euler_discrete",
        "z-image-turbo": "linear",
        "krea2-turbo": "er_sde",
    }.get(model_id)


def generate(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int | None = None,
    guidance: float | None = 1.0,
    seed: int | None = None,
    quantization: int = 4,
    loras: list[dict] | None = None,
    progress_cb=None,
    phase_cb=None,
    model: str = DEFAULT_MODEL_ID,
    cancel_event: threading.Event | None = None,
    negative_prompt: str = "",
    sampler: str | None = None,
    cache_interval: int = 1,
    reference_images: list[str] | None = None,
    image_strength: float | None = None,
    output_format: str = "png",
    stealth: bool = False,
    fast_vae: bool = True,
    max_pixels: int | None = None,  # per-request overridable hard pixel cap (defaults to model max_pixels)
) -> dict:
    """Blocking generation. Caller must hold no other heavy work."""
    if loras is not None:
        if not isinstance(loras, list) or len(loras) > 16:
            raise ValueError("LoRA list must contain at most 16 entries")
        if any(not isinstance(lora, dict) or not lora.get("path") for lora in loras):
            raise ValueError("each LoRA must contain a path")
        for lora in loras:
            try:
                scale = float(lora.get("scale", 1.0))
            except (TypeError, ValueError) as e:
                raise ValueError("LoRA scales must be numeric") from e
            if not math.isfinite(scale) or not 0.0 <= scale <= 10.0:
                raise ValueError("LoRA scales must be between 0 and 10")
    loras = _enrich_loras_with_registry(loras)
    minfo = get_model_info(model)
    if not minfo:
        raise ValueError(f"unknown model: {model}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt is required")
    if str(output_format).lower() not in ("png", "jpeg", "jpg"):
        raise ValueError("output_format must be png or jpeg")
    if len(prompt.encode("utf-8")) > _MAX_PROMPT_BYTES:
        raise ValueError("prompt exceeds the 128 KiB limit")
    try:
        steps = int(minfo.get("default_steps", 4) if steps is None else steps)
    except (TypeError, ValueError) as e:
        raise ValueError("steps must be an integer") from e
    if steps < 1 or steps > 50:
        raise ValueError("steps must be between 1 and 50")
    steps = _effective_steps(minfo, steps, {})
    width, height = _validate_dimensions(width, height, minfo)
    quantization = _effective_quantization(minfo["id"], quantization)
    if loras and not minfo.get("supports_loras"):
        raise ValueError(
            f"LoRAs are not supported on {minfo['label']} (needs {minfo.get('lora_format', 'compatible')}-format LoRAs)"
        )
    if not minfo.get("supports_negative"):
        negative_prompt = ""
    if len(str(negative_prompt).encode("utf-8")) > _MAX_PROMPT_BYTES:
        raise ValueError("negative prompt exceeds the 128 KiB limit")
    chosen_sampler = _validate_sampler(sampler, minfo)
    effective_guidance = _effective_guidance(minfo, guidance, negative_prompt)
    if minfo["id"] == "qwen-image-2.1" and steps > _QWEN_MAX_STEPS:
        raise ValueError(
            f"{minfo['label']}: {steps} steps exceeds the {_QWEN_MAX_STEPS}-step ceiling for this "
            f"machine. Measured median at 25 steps is 790.8s vs 910s at 40 steps, so the extra "
            f"steps are not worth the time. Use ≤ {_QWEN_MAX_STEPS} steps, or change the "
            f"per-model default in Parameters > Defaults."
        )
    if minfo["id"] == "qwen-image-2.1" and width * height > 589824:
        raise ValueError(
            f"{minfo['label']}: resolutions above 768×768 (589k px) OOM the bf16 VAE "
            f"decode on 16GB Apple Silicon and can crash the app. Keep ≤ 512×768 / 768×512 / 768×768."
        )

    ref_paths: list[str] = []
    if reference_images:
        for p in reference_images:
            if p and isinstance(p, str) and p.strip():
                ref_paths.append(p.strip())

    if ref_paths:
        _validate_reference_capability(minfo, ref_paths, image_strength)
        resolved_ref_paths = [_resolve_reference_path(p) for p in ref_paths]

        # If any reference image is raw HEIC/HEIF, normalize to PNG to prevent codec issues in downstream MLX
        normalized_ref_paths: list[str] = []
        for p in resolved_ref_paths:
            p_obj = Path(p)
            if p_obj.suffix.lower() in (".heic", ".heif"):
                uploads_dir = DATA_DIR / "uploads"
                uploads_dir.mkdir(parents=True, exist_ok=True)
                cached_png = uploads_dir / f"converted_{p_obj.stem}.png"
                if not cached_png.exists() or cached_png.stat().st_mtime < p_obj.stat().st_mtime:
                    try:
                        from PIL import ImageOps
                        with Image.open(p_obj) as img:
                            img.load()
                            trans = ImageOps.exif_transpose(img) or img
                            if trans.mode not in ("RGB", "RGBA"):
                                trans = trans.convert("RGB")
                            trans.save(cached_png, format="PNG")
                    except Exception:
                        try:
                            subprocess.run(
                                ["/usr/bin/sips", "-s", "format", "png", str(p_obj), "--out", str(cached_png)],
                                capture_output=True,
                                timeout=15,
                            )
                        except Exception:
                            pass
                if cached_png.exists():
                    normalized_ref_paths.append(str(cached_png.resolve()))
                else:
                    normalized_ref_paths.append(p)
            else:
                normalized_ref_paths.append(p)
        ref_paths = normalized_ref_paths
    else:
        _validate_reference_capability(minfo, [], image_strength)

    if image_strength is not None and not ref_paths:
        raise ValueError("image_strength requires at least one reference image")
    if ref_paths and image_strength is None:
        image_strength = 0.6

    max_side = minfo.get("max_side")
    cap_pixels = minfo.get("max_pixels")
    if max_pixels is not None:
        try:
            requested_cap = int(max_pixels)
        except (TypeError, ValueError) as e:
            raise ValueError("max_pixels must be an integer") from e
        if requested_cap < 256 * 256:
            raise ValueError("max_pixels must be at least 256×256")
        if cap_pixels is None or requested_cap < cap_pixels:
            cap_pixels = requested_cap
    if max_side:
        width = min(width, int(max_side))
        height = min(height, int(max_side))
    if cap_pixels and width * height > cap_pixels:
        scale = (cap_pixels / (width * height)) ** 0.5
        width = max(256, int(width * scale) // 16 * 16)
        height = max(256, int(height * scale) // 16 * 16)
    if cap_pixels and width * height > cap_pixels:
        raise ValueError("dimensions cannot be reduced below the requested pixel cap")

    if minfo.get("engine") == "sdxl":
        return _generate_sdxl(
            prompt=prompt, width=width, height=height, steps=steps,
            guidance=effective_guidance, seed=seed, loras=loras, progress_cb=progress_cb,
            phase_cb=phase_cb, model=model, cancel_event=cancel_event,
            negative_prompt=negative_prompt, sampler=chosen_sampler,
            cache_interval=cache_interval, quantization=quantization,
            output_format=output_format, stealth=stealth, fast_vae=fast_vae,
        )
    if minfo["id"] == "qwen-image-2.1":
        return _generate_qwen_subprocess(
            prompt=prompt, width=width, height=height, steps=steps,
            guidance=effective_guidance, seed=seed, progress_cb=progress_cb,
            phase_cb=phase_cb, model=model, cancel_event=cancel_event,
            negative_prompt=negative_prompt, sampler=chosen_sampler,
            ref_paths=ref_paths, image_strength=image_strength,
            quantization=quantization, output_format=output_format,
            stealth=stealth, fast_vae=fast_vae,
        )
    with _lock:
        _cancel_event.clear()
        _cancel_mflux_watchdog()
        seed = seed if seed is not None else int(time.time())

        variant = "standard"
        scheduler_id = _flow_scheduler_id(minfo["id"])
        is_flux2 = minfo["id"] in ("flux2-klein-4b", "flux2-klein-9b")
        if is_flux2 and ref_paths:
            variant = "edit"

        actual_loras = sorted(_normalize_loras(list(loras or [])[:16]), key=lambda item: item["path"])
        if minfo["id"] == "krea2-turbo" and steps <= 4:
            krea_distill_path = ASSET_DIR / "lora_files" / "krea2_turbo_4step_rank_64_lora_latest.safetensors"
            if not krea_distill_path.exists():
                krea_distill_path = ASSET_DIR / "lora_files" / "krea2_turbo_4step_rank_64_lora.safetensors"
            if krea_distill_path.exists():
                has_krea_distill = any(
                    "krea2_turbo_4step" in str(l.get("path", "") if isinstance(l, dict) else l)
                    for l in actual_loras
                )
                if not has_krea_distill:
                    if len(actual_loras) >= 16:
                        raise ValueError("cannot auto-add the Krea distillation LoRA when 16 LoRAs are already active")
                    actual_loras.append({
                        "path": str(krea_distill_path),
                        "scale": 1.0,
                        "name": "krea2_turbo_4step_rank_64_lora_latest",
                    })
            actual_loras = _enrich_loras_with_registry(actual_loras)
            actual_loras.sort(key=lambda item: item["path"])

        is_already_loaded = (
            _pipeline is not None
            and _current_pipeline_key == _make_pipeline_key(model, quantization, actual_loras, variant)
        )
        if is_already_loaded and phase_cb is not None:
            phase_cb("preparing", "Preparing prompt conditioning & latents...")

        t_load_start = time.time()
        pipe = _get_pipeline(model, quantization, actual_loras, variant=variant, phase_cb=phase_cb)
        load_time = round(time.time() - t_load_start, 2)
        lora_scales = [
            float(item.get("scale", 1.0)) if isinstance(item, dict) else 1.0
            for item in actual_loras
        ]
        _update_pipeline_lora_scales(pipe, lora_scales)
        if progress_cb is not None:
            pipe.callbacks.register(
                _ProgressCallback(progress_cb, cancel_event=cancel_event or _cancel_event, phase_cb=phase_cb)
            )
        import mlx.core as mx

        prev_wired = None
        restore_callbacks = []
        try:
            if model == "krea2-turbo":
                # krea2 only. Unbounded by default: its 9 GB pin showed no speed
                # benefit and a pixel-identical result (see the A/B note above).
                limit = _krea_wired_limit_bytes()
            else:
                # FLUX.2-klein, Z-Image, SDXL and qwen-image-2.1. qwen is on the
                # generic budget rather than krea2's because its q4 pipeline is
                # ~10.5GB resident and a too-low cap starves the load.
                limit = _wired_limit_bytes()
            if limit > 0:
                try:
                    set_limit = getattr(mx, "set_wired_limit", None) or mx.metal.set_wired_limit
                    prev_wired = set_limit(limit)
                except Exception:
                    prev_wired = None

            # Acceleration Hooks: Fast VAE
            if is_flux2:
                if fast_vae and hasattr(pipe, "vae") and hasattr(pipe.vae, "decode_packed_latents"):
                    taef2_dec = _get_taef_decoder("taef2")
                    orig_decode_packed = pipe.vae.decode_packed_latents

                    def _taef2_decode_packed(packed_latents, tiling_config=None, *args, **kwargs):
                        # Ollin's TAEF2 expects raw unpatchified latents without VAE BatchNorm
                        # denormalization (identity BN). unpack_packed_latents multiplies by bn_std
                        # (~1.77) which oversaturates and crushes shadows.
                        if packed_latents.ndim == 5:
                            packed_latents = packed_latents[:, :, 0, :, :]
                        unpacked = pipe.vae._unpatchify_latents(packed_latents)
                        nhwc = mx.transpose(unpacked, (0, 2, 3, 1))
                        decoded_nhwc = taef2_dec.decode(nhwc)
                        return mx.transpose(decoded_nhwc, (0, 3, 1, 2)) * 2.0 - 1.0

                    pipe.vae.decode_packed_latents = _taef2_decode_packed
                    restore_callbacks.append(lambda: setattr(pipe.vae, "decode_packed_latents", orig_decode_packed))
                elif not fast_vae and hasattr(pipe, "vae") and hasattr(pipe.vae, "decoder"):
                    if not getattr(pipe.vae.decoder, "_compiled", False):
                        try:
                            pipe.vae.decoder = mx.compile(pipe.vae.decoder)
                            pipe.vae.decoder._compiled = True
                        except Exception:
                            pass

            elif model == "krea2-turbo":
                if fast_vae and hasattr(pipe, "_decode_latents"):
                    krea_dec = _get_taef_decoder("krea2")
                    orig_krea_decode = pipe._decode_latents

                    def _taew_decode_krea(*, latents, prompt, seed, **kwargs):
                        nhwc = mx.transpose(latents, (0, 2, 3, 1))
                        decoded_nhwc = krea_dec.decode(nhwc)
                        return mx.transpose(decoded_nhwc, (0, 3, 1, 2)) * 2.0 - 1.0

                    pipe._decode_latents = _taew_decode_krea
                    restore_callbacks.append(lambda: setattr(pipe, "_decode_latents", orig_krea_decode))

            elif model == "z-image-turbo":
                if fast_vae and hasattr(pipe, "_decode_latents"):
                    taef1_dec = _get_taef_decoder("taef1")
                    orig_zit_decode = pipe._decode_latents

                    def _taef1_decode_zit(*, latents, config, prompt, seed, **kwargs):
                        from mflux.models.z_image.latent_creator import ZImageLatentCreator
                        unpacked = ZImageLatentCreator.unpack_latents(latents, config.height, config.width)
                        nhwc = mx.transpose(unpacked, (0, 2, 3, 1))
                        decoded_nhwc = taef1_dec.decode(nhwc)
                        return mx.transpose(decoded_nhwc, (0, 3, 1, 2)) * 2.0 - 1.0

                    pipe._decode_latents = _taef1_decode_zit
                    restore_callbacks.append(lambda: setattr(pipe, "_decode_latents", orig_zit_decode))

            t_infer_start = time.time()
            sampler_label = chosen_sampler
            if variant == "edit":
                out = pipe.generate_image(
                    seed=seed,
                    prompt=prompt,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=effective_guidance,
                    image_paths=[Path(p) for p in ref_paths],
                    scheduler=scheduler_id,
                )
            else:
                gen_kwargs = {
                    "seed": seed,
                    "prompt": prompt,
                    "num_inference_steps": steps,
                    "height": height,
                    "width": width,
                    "guidance": effective_guidance,
                    "image_path": ref_paths[0] if ref_paths else None,
                    "image_strength": image_strength,
                }
                if minfo.get("supports_negative"):
                    gen_kwargs["negative_prompt"] = negative_prompt or None
                if scheduler_id is not None:
                    gen_kwargs["scheduler"] = scheduler_id
                sampler_label = chosen_sampler
                out = pipe.generate_image(**gen_kwargs)
            infer_time = round(time.time() - t_infer_start, 2)
        finally:
            for _cb in reversed(restore_callbacks):
                try:
                    _cb()
                except Exception:
                    pass
            if progress_cb is not None:
                pipe.callbacks.in_loop = [
                    c for c in pipe.callbacks.in_loop
                    if not isinstance(c, _ProgressCallback)
                ]
                pipe.callbacks.before_loop = [
                    c for c in getattr(pipe.callbacks, "before_loop", [])
                    if not isinstance(c, _ProgressCallback)
                ]
                pipe.callbacks.after_loop = [
                    c for c in getattr(pipe.callbacks, "after_loop", [])
                    if not isinstance(c, _ProgressCallback)
                ]
            if prev_wired is not None:
                try:
                    set_limit = getattr(mx, "set_wired_limit", None) or mx.metal.set_wired_limit
                    set_limit(prev_wired)
                except Exception:
                    pass
            _arm_mflux_watchdog()
        if phase_cb is not None:
            phase_cb("saving", "Decoding VAE latents & saving image...")
        elapsed = infer_time
        if load_time > 0.5:
            print(f"[{model}] Model load/quantize: {load_time}s | Pure inference: {elapsed}s", flush=True)
        image_id = uuid.uuid4().hex
        fmt = "jpeg" if str(output_format).lower() in ("jpeg", "jpg") else "png"
        image_path = GENERATED_DIR / f"{image_id}.{fmt}"
        meta = {
            "id": image_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "sampler": sampler_label,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": effective_guidance,
            "seed": seed,
            "quantization": quantization,
            "loras": actual_loras,
            "generation_time": elapsed,
            "load_time": load_time,
            "created_at": time.time(),
            "software": "DiffusionBear",
            "generator": "DiffusionBear",
            "artist": app_settings.metadata_artist(),
            "tags": [],
            "file": image_path.name,
            "model": minfo["repo"],
            "reference_images": [Path(p).name for p in ref_paths],
            "stealth": stealth,
            "format": fmt,
            "fast_vae": bool(fast_vae) and bool(minfo.get("supports_fast_vae")),
        }
        if scheduler_id is not None:
            meta["scheduler"] = scheduler_id
        # Strength is meaningful only for strength-based single-ref conditioning
        # (Z-Image / Krea2). FLUX.2 in-context edit ignores strength entirely.
        if ref_paths and variant == "standard":
            meta["reference_strength"] = round(float(image_strength if image_strength is not None else 0.6), 3)
        if minfo.get("civitai_version_id"):
            meta["modelVersionId"] = minfo["civitai_version_id"]
        if minfo.get("civitai_model_id"):
            meta["modelId"] = minfo["civitai_model_id"]
        if minfo.get("civitai_model_name"):
            meta["model_label"] = minfo["civitai_model_name"]
        if minfo.get("civitai_version_name"):
            meta["modelVersionName"] = minfo["civitai_version_name"]

        save_image_with_metadata(
            image=out.image,
            dest_path=image_path,
            meta=meta,
            output_format=fmt,
            stealth=stealth,
        )
        atomic_write_json(GENERATED_DIR / f"{image_id}.json", meta)
        gc.collect()
        return meta


_sdxl_daemon = None
_sdxl_watchdog = None
_sdxl_idle_since = None
_sdxl_reader_messages = None
_sdxl_reader_done = None
_sdxl_reader_thread = None
_sdxl_stderr_thread = None
_sdxl_stderr_done = None
# Ceiling applied to guidance-distilled models by _effective_steps. The slowest
# distilled model in the registry runs 6 steps, so this is generous.
_DISTILLED_MAX_STEPS = 12

# Hard ceiling for qwen-image-2.1, by owner decision (2026-09-28): 40 steps is
# never worth running on a 16GB M1. Its own measured medians are 790.8s at
# 512x768 / 25 steps and 910s at 512x768 / 40 steps, so the extra 15 steps buy
# almost nothing for ~15% more time. Refused loudly rather than silently
# clamped, so a saved model_defaults of 40 cannot quietly keep costing that.
_QWEN_MAX_STEPS = 25
SDXL_IDLE_KILL_S = 300


def _arm_sdxl_watchdog():
    global _sdxl_watchdog, _sdxl_idle_since
    idle_s = _sdxl_idle_kill_s()
    proc = _sdxl_daemon
    if idle_s <= 0 or proc is None or proc.poll() is not None:
        _cancel_sdxl_watchdog()
        return
    if _sdxl_watchdog is not None:
        _sdxl_watchdog.cancel()
    _sdxl_idle_since = time.time()
    timer_ref = [None]

    def _kill():
        global _sdxl_daemon, _sdxl_watchdog, _current_sdxl_model, _sdxl_idle_since
        with _lock:
            if _sdxl_watchdog is not timer_ref[0] or _sdxl_daemon is not proc:
                return
            if proc.poll() is None:
                _terminate_process(proc)
                _remove_sdxl_pid(proc.pid)
                print("[sdxl] watchdog: idle daemon terminated", flush=True)
            _remove_sdxl_pid(proc.pid)
            if _sdxl_daemon is proc:
                _sdxl_daemon = None
                _current_sdxl_model = None
            if _sdxl_watchdog is timer_ref[0]:
                _sdxl_watchdog = None
                _sdxl_idle_since = None

    timer = threading.Timer(idle_s, _kill)
    timer.daemon = True
    timer_ref[0] = timer
    _sdxl_watchdog = timer
    timer.start()


def _cancel_sdxl_watchdog():
    global _sdxl_watchdog, _sdxl_idle_since
    if _sdxl_watchdog is not None:
        _sdxl_watchdog.cancel()
        _sdxl_watchdog = None
    _sdxl_idle_since = None


_mflux_watchdog = None
_mflux_idle_since = None
MFLUX_IDLE_KILL_S = 300

# The Qwen daemon holds ~10.5GB resident, so it follows the same 5-minute idle
# release policy as the SDXL daemon. 0 = keep it warm indefinitely.
QWEN_IDLE_KILL_S = 300


def _mflux_idle_kill_s() -> int:
    """Effective mflux idle-kill seconds: persisted setting > built-in 300. 0 = never release."""
    v = app_settings.get_setting("idle_kill_s_mflux")
    if v is None:
        return MFLUX_IDLE_KILL_S
    return max(0, int(v))


def _qwen_idle_kill_s() -> int:
    """Effective Qwen idle-kill seconds: persisted setting > built-in 300. 0 = never release."""
    v = app_settings.get_setting("idle_kill_s_qwen")
    if v is None:
        return QWEN_IDLE_KILL_S
    return max(0, int(v))


def _qwen_process_is_healthy() -> bool:
    """A live daemon is reusable only if it is still running and its readers are up."""
    proc = _qwen_process
    if proc is None or proc.stdin is None:
        return False
    try:
        if proc.poll() is not None:
            return False
    except Exception:
        return False
    return _qwen_reader_messages is not None and not _qwen_reader_done.is_set()


def _arm_qwen_watchdog():
    global _qwen_watchdog, _qwen_idle_since
    idle_s = _qwen_idle_kill_s()
    proc = _qwen_process
    if idle_s <= 0 or proc is None or proc.poll() is not None:
        _cancel_qwen_watchdog()
        return
    if _qwen_watchdog is not None:
        _qwen_watchdog.cancel()
    _qwen_idle_since = time.time()
    timer_ref = [None]

    def _kill():
        global _qwen_watchdog, _qwen_idle_since
        with _lock:
            if _qwen_watchdog is not timer_ref[0] or _qwen_process is not proc:
                return
            _kill_qwen_process()
            print("[qwen] watchdog: idle daemon terminated", flush=True)
            if _qwen_watchdog is timer_ref[0]:
                _qwen_watchdog = None
                _qwen_idle_since = None

    timer = threading.Timer(idle_s, _kill)
    timer.daemon = True
    timer_ref[0] = timer
    _qwen_watchdog = timer
    timer.start()


def _cancel_qwen_watchdog():
    global _qwen_watchdog, _qwen_idle_since
    if _qwen_watchdog is not None:
        _qwen_watchdog.cancel()
        _qwen_watchdog = None
    _qwen_idle_since = None


def _sdxl_idle_kill_s() -> int:
    """Effective SDXL idle-kill seconds: persisted setting > built-in 300. 0 = never release."""
    v = app_settings.get_setting("idle_kill_s_sdxl")
    if v is None:
        return SDXL_IDLE_KILL_S
    return max(0, int(v))


def _arm_mflux_watchdog():
    global _mflux_watchdog, _mflux_idle_since
    idle_s = _mflux_idle_kill_s()
    pipe_ref = _pipeline
    if idle_s <= 0 or pipe_ref is None:
        _cancel_mflux_watchdog()
        return
    if _mflux_watchdog is not None:
        _mflux_watchdog.cancel()
    _mflux_idle_since = time.time()
    timer_ref = [None]

    def _release():
        global _mflux_watchdog, _mflux_idle_since
        with _lock:
            if _mflux_watchdog is not timer_ref[0] or _pipeline is not pipe_ref:
                return
            _drop_mflux_pipeline()
            print("[mflux] watchdog: idle pipeline released", flush=True)
            if _mflux_watchdog is timer_ref[0]:
                _mflux_watchdog = None
                _mflux_idle_since = None

    timer = threading.Timer(idle_s, _release)
    timer.daemon = True
    timer_ref[0] = timer
    _mflux_watchdog = timer
    timer.start()


def _cancel_mflux_watchdog():
    global _mflux_watchdog, _mflux_idle_since
    if _mflux_watchdog is not None:
        _mflux_watchdog.cancel()
        _mflux_watchdog = None
    _mflux_idle_since = None


def rearm_engine_watchdogs():
    if not _lock.acquire(blocking=False):
        return
    try:
        _cancel_mflux_watchdog()
        _cancel_sdxl_watchdog()
        _cancel_qwen_watchdog()
        if _pipeline is not None and _mflux_idle_kill_s() > 0:
            _arm_mflux_watchdog()
        if _sdxl_daemon is not None and _sdxl_daemon.poll() is None and _sdxl_idle_kill_s() > 0:
            _arm_sdxl_watchdog()
        if _qwen_process is not None and _qwen_process.poll() is None and _qwen_idle_kill_s() > 0:
            _arm_qwen_watchdog()
    finally:
        _lock.release()


def _clear_krea_te_cache():
    """Release the cached quantized Krea 2 text encoder (~1.9GB q4).

    Kept alive across mflux pipeline rebuilds (see _krea_quantized_text_encoder)
    but dropped whenever we hand the machine to another engine, so the pinned
    memory never competes with SDXL's ~11GB daemon or Qwen's ~10.5GB pipeline.
    """
    global _krea_te_cache
    with _krea_te_lock:
        if not _krea_te_cache:
            return
        _krea_te_cache = {}
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def _krea_vae_tile_size() -> int:
    """VAE decode tile size in px; 0 disables tiled decoding."""
    v = app_settings.get_setting("krea_vae_tile_size")
    if v is None:
        v = os.environ.get("MLX_KREA_VAE_TILE_SIZE", "256")
    try:
        return max(0, int(v))
    except (TypeError, ValueError):
        return 256


def _apply_krea_vae_tiling(pipe) -> None:
    """Give the Krea 2 pipeline a TilingConfig so the final VAE decode tiles.

    mflux only tiles when tiles_per_dim > 1, and the tile size is a cap on each tile's
    working set. 256 is the value the changelog suggests for cutting peak decode memory
    further; the default 512 is the library's own. A value of 0 restores untiled decoding.
    """
    size = _krea_vae_tile_size()
    if size <= 0:
        return
    try:
        from mflux.models.common.vae.tiling_config import TilingConfig

        pipe.tiling_config = TilingConfig(
            vae_decode_tile_size=size,
            vae_decode_tiles_per_dim=8,
            vae_decode_overlap=8,
        )
    except Exception as exc:  # pragma: no cover - depends on the mflux build
        print(f"[generator] krea2 VAE tiling unavailable: {exc}", flush=True)


_krea_te_skip_bf16: dict = {"on": False}
_krea_te_skip_installed = False


def _krea_te_disk_path(model_path, bits, group_size):
    """Where the quantised Krea 2 text encoder is persisted between processes.

    The in-memory cache below dies with the process, so every app restart used to pay
    the full bf16 load again. Hashing the model path keeps this tied to the checkpoint
    it was built from, so swapping the model re-derives instead of loading stale weights.
    """
    import hashlib

    digest = hashlib.sha256(str(model_path).encode()).hexdigest()[:10]
    name = Path(str(model_path)).name or "krea2"
    return ASSET_DIR / "models" / f"{name}-te-q{bits}-g{group_size}-{digest}.safetensors"


def _install_krea_te_bf16_skip() -> None:
    """Stop mflux loading the 7.5GB bf16 encoder when we already have a q4 copy on disk.

    mflux builds Krea2TextEncoder as a plain bf16 module and WeightApplier._set_weights
    then loads the bf16 tensors into it (krea2's weight definition sets
    skip_quantization=True on that component). Those two steps are the spike: the encoder
    is resident before we ever get a chance to quantise it.

    _set_weights simply iterates `models` and calls model.update(...), so dropping the
    entry from the dict is enough to skip the load entirely. The patch is deliberately
    narrow twice over: it only fires while the flag is set, and only for a models dict
    that actually contains a Krea2TextEncoder, so no other engine can be affected.
    """
    global _krea_te_skip_installed
    if _krea_te_skip_installed:
        return
    from mflux.models.common.weights.loading.weight_applier import WeightApplier

    original = WeightApplier._set_weights

    def _patched(weights, models, components=None):
        if _krea_te_skip_bf16["on"] and any(
            type(m).__name__ == "Krea2TextEncoder" for m in models.values()
        ):
            models = {k: v for k, v in models.items() if k != "text_encoder"}
        return original(weights, models, components)

    WeightApplier._set_weights = staticmethod(_patched)
    _krea_te_skip_installed = True


def _krea_te_from_disk(model_path, bits, group_size):
    """Rebuild the quantised encoder straight from disk, or None if there is no cache."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder

    path = _krea_te_disk_path(model_path, bits, group_size)
    if not path.exists():
        return None
    try:
        module = Krea2TextEncoder()
        # Build the quantised *structure* first, then drop the saved q4 tensors in --
        # the same order mflux uses when a checkpoint declares a quantization level.
        nn.quantize(module, bits=bits, group_size=group_size)
        module.load_weights(str(path))
        mx.eval(module)
        return module
    except Exception as exc:  # a corrupt or truncated cache must never be fatal
        print(f"[generator] WARN Krea 2 q4 text-encoder cache unusable ({exc}); falling back", flush=True)
        # Self-heal: a truncated or mismatched cache would otherwise be re-read (and
        # re-rejected) on every single start, forever.
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def _krea_te_save_disk(module, model_path, bits, group_size) -> None:
    """Persist the quantised encoder so later processes never load bf16 at all."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    path = _krea_te_disk_path(model_path, bits, group_size)
    # mx.save_safetensors APPENDS .safetensors to any path that does not already end in
    # it, so a plain ".partial" temp name silently lands on ".partial.safetensors" and
    # the rename below then fails with ENOENT. The temp must keep the real suffix.
    tmp = path.with_name(path.stem + ".partial.safetensors")
    try:
        mx.save_safetensors(str(tmp), dict(tree_flatten(module.parameters())))
        os.replace(tmp, path)  # atomic: a crash mid-write can never poison the cache
    except Exception as exc:
        print(f"[generator] WARN could not persist Krea 2 q4 text encoder ({exc})", flush=True)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _krea_quantized_text_encoder(pipe, model_path, bits, group_size):
    """Return a 4-bit Krea 2 Qwen3-VL text encoder, reusing the last one if we can.

    mflux ships this encoder at bf16 (7.5GB) and leaves it unquantized, so the
    app quantizes it after construction. That nn.quantize() pass measured
    65.7s and 68.9s on two consecutive pipeline loads (test/manifest.json), and
    it was being repeated on EVERY krea2 rebuild - any LoRA change, any 4-step
    vs 8-step switch, any engine switch back to krea2.

    The encoder is a pure function of (model_path, bits, group_size): Krea2's LoRA
    path only ever touches the transformer, never the text encoder. So the
    quantized module is cacheable and safe to hand to a freshly built pipeline.
    """
    import mlx.core as mx
    import mlx.nn as nn

    key = (str(model_path), bits, group_size)
    disabled = os.environ.get("MLX_DISABLE_KREA_TE_CACHE") == "1"
    with _krea_te_lock:
        cached = None if disabled else (_krea_te_cache.get("module") if _krea_te_cache.get("key") == key else None)
        fresh = getattr(pipe, "text_encoder", None)
        if cached is not None:
            # Drop the bf16 encoder we just paid to load before adopting the cache.
            if fresh is not None and fresh is not cached:
                pipe.text_encoder = None
                del fresh
                gc.collect()
                mx.clear_cache()
            pipe.text_encoder = cached
            return cached

        if fresh is None:
            return None
        nn.quantize(fresh, bits=bits, group_size=group_size)
        mx.eval(fresh)
        mx.clear_cache()
        gc.collect()
        if not disabled:
            _krea_te_cache["key"] = key
            _krea_te_cache["module"] = fresh
            _krea_te_save_disk(fresh, model_path, bits, group_size)
        return fresh


def _drop_mflux_pipeline():
    global _pipeline, _current_pipeline_key, _current_pipeline_model, _prompt_cache
    _cancel_mflux_watchdog()
    _pipeline = None
    _current_pipeline_key = None
    _current_pipeline_model = None
    _prompt_cache.clear()
    _taef_models.clear()
    try:
        from taesd_mlx import clear_taesd_cache
        clear_taesd_cache()
    except Exception:
        pass
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def _kill_sdxl_daemon():
    global _sdxl_daemon, _current_sdxl_model
    global _sdxl_reader_messages, _sdxl_reader_done, _sdxl_reader_thread
    global _sdxl_stderr_thread, _sdxl_stderr_done
    proc = _sdxl_daemon
    had_sdxl = proc is not None or _current_sdxl_model is not None
    _cancel_sdxl_watchdog()
    if proc is not None:
        _terminate_process(proc)
        _remove_sdxl_pid(proc.pid)
    _join_reader(_sdxl_stderr_thread)
    _join_reader(_sdxl_reader_thread)
    if _sdxl_daemon is proc:
        _sdxl_daemon = None
    _current_sdxl_model = None
    _sdxl_reader_messages = None
    _sdxl_reader_done = None
    _sdxl_reader_thread = None
    _sdxl_stderr_thread = None
    _sdxl_stderr_done = None
    if had_sdxl:
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        gc.collect()


_QWEN_PID_FILE = DATA_DIR / ".tmp" / "qwen_daemon.pid"


def _read_qwen_pid() -> int | None:
    try:
        value = int(_QWEN_PID_FILE.read_text("utf-8").strip())
        return value if value > 1 else None
    except (OSError, UnicodeError, ValueError):
        return None


def _write_qwen_pid(pid: int):
    _QWEN_PID_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = _QWEN_PID_FILE.with_name(f".{_QWEN_PID_FILE.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(str(pid), encoding="utf-8")
        os.replace(tmp, _QWEN_PID_FILE)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass


def _remove_qwen_pid(pid: int | None = None):
    current = _read_qwen_pid()
    if current is not None and (pid is None or current == pid):
        try:
            _QWEN_PID_FILE.unlink()
        except OSError:
            pass


def _cleanup_stale_qwen_daemon() -> bool:
    """Reap a Qwen daemon left behind by a crashed parent.

    The daemon is resident for ~10.5GB, so unlike the old one-shot subprocess it
    cannot simply be allowed to outlive its parent. Mirrors
    _cleanup_stale_sdxl_daemon: only ever signals a pid whose command line still
    looks like our own `--serve` engine.
    """
    pid = _read_qwen_pid()
    if pid is None or pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        _remove_qwen_pid(pid)
        return True
    except PermissionError:
        return False
    try:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=2)
        command = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    if not command:
        return False
    if "qwen_engine.py" not in command or "--serve" not in command:
        _remove_qwen_pid(pid)
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _remove_qwen_pid(pid)
        return True
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            _remove_qwen_pid(pid)
            return True
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _remove_qwen_pid(pid)
    return True


_SDXL_PID_FILE = DATA_DIR / ".tmp" / "sdxl_daemon.pid"


def _read_sdxl_pid() -> int | None:
    try:
        value = int(_SDXL_PID_FILE.read_text("utf-8").strip())
        return value if value > 1 else None
    except (OSError, UnicodeError, ValueError):
        return None


def _write_sdxl_pid(pid: int):
    _SDXL_PID_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = _SDXL_PID_FILE.with_name(f".{_SDXL_PID_FILE.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(str(pid), encoding="utf-8")
        os.replace(tmp, _SDXL_PID_FILE)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass


def _remove_sdxl_pid(pid: int | None = None):
    current = _read_sdxl_pid()
    if current is not None and (pid is None or current == pid):
        try:
            _SDXL_PID_FILE.unlink()
        except OSError:
            pass


def _cleanup_stale_sdxl_daemon() -> bool:
    pid = _read_sdxl_pid()
    if pid is None:
        return True
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        _remove_sdxl_pid(pid)
        return True
    except PermissionError:
        return False
    try:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=2)
        command = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    if not command:
        return False
    if "sdxl_engine.py" not in command or "--serve" not in command:
        _remove_sdxl_pid(pid)
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _remove_sdxl_pid(pid)
        return True
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            _remove_sdxl_pid(pid)
            return True
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _remove_sdxl_pid(pid)
    return True


_sdxl_stderr_tail = collections.deque(maxlen=100)


def _drain_stderr(proc):
    try:
        _read_process_stream(proc.stderr, lambda line: _sdxl_stderr_tail.append(f"{line}\n"))
    except Exception:
        pass


def _get_sdxl_daemon():
    global _sdxl_daemon, _sdxl_reader_messages, _sdxl_reader_done, _sdxl_reader_thread
    global _sdxl_stderr_thread, _sdxl_stderr_done
    if _sdxl_daemon is not None and _sdxl_daemon.poll() is None:
        return _sdxl_daemon
    if _sdxl_daemon is not None:
        _kill_sdxl_daemon()
    if not _cleanup_stale_sdxl_daemon():
        raise RuntimeError("could not verify the previous SDXL daemon")
    script = Path(__file__).parent / "sdxl_engine.py"
    engine = Path(__file__).parent.parent / "venv-sdxl" / "bin" / "python"
    _sdxl_stderr_tail.clear()
    proc = subprocess.Popen(
        [str(engine), str(script), "--serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
        start_new_session=True,
        cwd=str(Path(__file__).resolve().parent),
    )
    proc._mlx_process_group = True
    _write_sdxl_pid(proc.pid)
    _sdxl_daemon = proc
    _sdxl_reader_messages, _sdxl_reader_done, _sdxl_reader_thread = _start_json_reader(proc)
    _sdxl_stderr_done, _sdxl_stderr_thread = _start_stderr_reader(proc, _sdxl_stderr_tail)
    return proc


def _generate_sdxl(prompt, width, height, steps, guidance, seed, loras,
                   progress_cb, model, cancel_event, negative_prompt, sampler,
                   cache_interval=1, quantization=4, output_format="png", stealth=False,
                   fast_vae=True, phase_cb=None):
    global _current_sdxl_model
    raw_image_path = None
    with _lock:
        _kill_qwen_process()
        _cancel_sdxl_watchdog()
        try:
            _drop_mflux_pipeline()
            _clear_krea_te_cache()
            minfo = get_model_info(model) or {}
            sampler = sampler or (minfo.get("samplers")[0] if minfo.get("samplers") else "euler_trailing")
            seed = seed if seed is not None else int(time.time())
            try:
                cache_interval = int(cache_interval)
            except (TypeError, ValueError) as e:
                raise ValueError("cache_interval must be an integer") from e
            if not 1 <= cache_interval <= 10:
                raise ValueError("cache_interval must be between 1 and 10")
            image_id = uuid.uuid4().hex
            fmt = "jpeg" if str(output_format).lower() in ("jpeg", "jpg") else "png"
            raw_image_path = GENERATED_DIR / f"{image_id}.raw.png"
            final_image_path = GENERATED_DIR / f"{image_id}.{fmt}"
            actual_loras = _normalize_loras(list(loras or []))
            lightning_path = str(ASSET_DIR / "SDXL" / "sdxl_lightning_4step_lora.safetensors")
            if not minfo.get("is_distilled"):
                if (steps <= 4 or sampler in ("euler_trailing", "trailing")) and os.path.exists(lightning_path):
                    has_lightning = any(
                        Path(str(l.get("path", ""))).name == "sdxl_lightning_4step_lora.safetensors"
                        for l in actual_loras
                    )
                    if not has_lightning:
                        if len(actual_loras) >= 16:
                            raise ValueError("cannot auto-add the SDXL Lightning LoRA when 16 LoRAs are already active")
                        actual_loras.append({"path": lightning_path, "scale": 1.0})
            actual_loras = _enrich_loras_with_registry(actual_loras)
            sdxl_wired_limit = _wired_limit_bytes()
            if sdxl_wired_limit > 0:
                sdxl_wired_limit = min(sdxl_wired_limit, int(6.5 * (1 << 30)))
            tile_vae = width * height >= 768 * 768
            model_dir = str(
                resolve_local_model_path(model)
                or minfo.get("model_dir")
                or (ASSET_DIR / "models" / "juggernaut-xl-lightning")
            )
            req = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "height": height,
                "width": width,
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
                "sampler": sampler,
                "image_id": image_id,
                "dest": str(raw_image_path),
                "loras": actual_loras,
                "cache_interval": cache_interval,
                "tile_vae": tile_vae,
                "model_dir": model_dir,
                "fast_vae": bool(fast_vae),
                "quantize_unet": 4,
                "wired_limit_bytes": sdxl_wired_limit,
            }
            is_daemon_alive = (
                _sdxl_daemon is not None
                and _sdxl_daemon.poll() is None
                and _current_sdxl_model == model_dir
            )
            if phase_cb is not None:
                if is_daemon_alive:
                    phase_cb("preparing", "Preparing prompt conditioning & latents...")
                else:
                    phase_cb("loading_model", "Loading SDXL UNet & text encoders into unified memory...")
            _cancel_event.clear()
            proc = _get_sdxl_daemon()
            _current_sdxl_model = model_dir
            _write_process_request(
                proc,
                (json.dumps(req, separators=(",", ":")) + "\n").encode("utf-8"),
                (cancel_event, _cancel_event),
                _engine_timeout("SDXL_ENGINE_TIMEOUT_S"),
            )
            result = None
            while True:
                line = _read_engine_message(
                    proc,
                    _sdxl_reader_messages,
                    _sdxl_reader_done,
                    (cancel_event, _cancel_event),
                    _engine_timeout("SDXL_ENGINE_TIMEOUT_S"),
                    "SDXL engine",
                    _sdxl_stderr_tail,
                )
                if line is None:
                    err = "".join(_sdxl_stderr_tail)[-2000:] or "EOF on stdout"
                    raise RuntimeError(f"SDXL engine died: {err}")
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if "progress" in payload:
                    step = payload["progress"].get("step")
                    if progress_cb is not None and isinstance(step, int):
                        progress_cb(max(0, step - 1))
                elif "phase" in payload:
                    if phase_cb is not None:
                        phase_cb(str(payload["phase"]), str(payload.get("detail", "")))
                else:
                    result = payload
                    break
            if result is None:
                raise RuntimeError("SDXL engine returned no result")
            if result.get("error"):
                # The daemon returns a bare error STRING and prints the traceback to
                # stderr. The stderr reader has been capturing it all along, but only the
                # EOF path used the tail -- so a normal error surfaced as
                # "TypeError: expected str, bytes or os.PathLike object, not NoneType"
                # with nothing to act on. The real cause (a CLIP tokenizer missing
                # vocab.json) only ever appeared because the tail was read by hand out of
                # the engine's own process. Attach it here, and log it either way.
                tail = "".join(_sdxl_stderr_tail).strip()
                message = str(result["error"])
                if tail:
                    print(f"[generator] SDXL engine traceback:\n{tail[-2000:]}", flush=True)
                    message = f"{message}\n--- engine traceback ---\n{tail[-2000:]}"
                _kill_sdxl_daemon()
                raise RuntimeError(message)
            if phase_cb is not None:
                phase_cb("saving", "Finalizing image and metadata...")
            actual_w = int(result.get("width", width))
            actual_h = int(result.get("height", height))
            actual_sampler = result.get("sampler") or sampler
            actual_guidance = float(result.get("guidance", guidance if guidance is not None else 1.0))
            actual_quantization = result.get("quantization", quantization or 4)
            actual_loras = result.get("loras") or actual_loras
            raw_bytes = raw_image_path.read_bytes()
            expected_bytes = actual_w * actual_h * 3
            if len(raw_bytes) != expected_bytes:
                raise RuntimeError(
                    f"SDXL raw image has {len(raw_bytes)} bytes, expected {expected_bytes}"
                )
            img_rgb = Image.frombytes("RGB", (actual_w, actual_h), raw_bytes)
            meta = {
                "id": image_id,
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "sampler": actual_sampler,
                "scheduler": result.get("scheduler"),
                "width": actual_w,
                "height": actual_h,
                "steps": int(result.get("steps", steps)),
                "guidance": actual_guidance,
                "seed": seed,
                "quantization": actual_quantization,
                "loras": actual_loras,
                "generation_time": result["generation_time"],
                "load_time": result.get("load_time", 0.0),
                "created_at": time.time(),
                "software": "DiffusionBear",
                "generator": "DiffusionBear",
                "artist": app_settings.metadata_artist(),
                "tags": [],
                "file": final_image_path.name,
                "model": minfo["repo"],
                "engine": "sdxl",
                "stealth": stealth,
                "format": fmt,
                "fast_vae": bool(result.get("fast_vae", fast_vae)),
                "cache_interval": int(result.get("cache_interval", cache_interval)),
                "cache_final_step": bool(result.get("cache_final_step", cache_interval > 1)),
                "tile_vae": bool(result.get("tile_vae", tile_vae)),
            }
            if minfo.get("civitai_version_id"):
                meta["modelVersionId"] = minfo["civitai_version_id"]
            if minfo.get("civitai_model_id"):
                meta["modelId"] = minfo["civitai_model_id"]
            if minfo.get("civitai_model_name"):
                meta["model_label"] = minfo["civitai_model_name"]
            if minfo.get("civitai_version_name"):
                meta["modelVersionName"] = minfo["civitai_version_name"]
            save_image_with_metadata(
                image=img_rgb,
                dest_path=final_image_path,
                meta=meta,
                output_format=fmt,
                stealth=stealth,
            )
            atomic_write_json(GENERATED_DIR / f"{image_id}.json", meta)
            gc.collect()
            return meta
        except BaseException:
            _kill_sdxl_daemon()
            raise
        finally:
            if raw_image_path is not None:
                try:
                    raw_image_path.unlink()
                except OSError:
                    pass
            _arm_sdxl_watchdog()


_qwen_stderr_tail = collections.deque(maxlen=100)


def _drain_qwen_stderr(proc):
    try:
        _read_process_stream(proc.stderr, lambda line: _qwen_stderr_tail.append(f"{line}\n"))
    except Exception:
        pass


def _generate_qwen_subprocess(prompt, width, height, steps, guidance, seed,
                              progress_cb=None, phase_cb=None, model="qwen-image-2.1",
                              cancel_event=None, negative_prompt="", sampler=None,
                              ref_paths=None, image_strength=None, quantization=4,
                              output_format="png", stealth=False, fast_vae=True):
    global _qwen_process, _qwen_reader_messages, _qwen_reader_done, _qwen_reader_thread
    global _qwen_stderr_thread, _qwen_stderr_done
    raw_image_path = None
    with _lock:
        _kill_sdxl_daemon()
        _cancel_mflux_watchdog()
        try:
            _drop_mflux_pipeline()
            _clear_krea_te_cache()
            minfo = get_model_info(model) or {}
            seed = seed if seed is not None else int(time.time())
            try:
                quantization = int(quantization)
            except (TypeError, ValueError) as e:
                raise ValueError("Qwen quantization must be an integer") from e
            sampler_label = str(sampler or minfo.get("default_sampler") or "linear").lower()
            if sampler_label not in _QWEN_SCHEDULERS:
                raise ValueError(f"unsupported Qwen sampler: {sampler_label}")
            image_id = uuid.uuid4().hex
            fmt = "jpeg" if str(output_format).lower() in ("jpeg", "jpg") else "png"
            raw_image_path = GENERATED_DIR / f"{image_id}.raw.png"
            final_image_path = GENERATED_DIR / f"{image_id}.{fmt}"
            model_path = _qwen_local_model(minfo)
            req = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "width": width,
                "height": height,
                "steps": steps,
                "guidance": guidance if guidance is not None else (minfo.get("default_guidance") or 1.0),
                "seed": seed,
                "sampler": sampler_label,
                "dest": str(raw_image_path),
                "ref_paths": [str(Path(p).resolve()) for p in (ref_paths or [])],
                "image_strength": image_strength,
                "fast_vae": bool(fast_vae),
                "phase_cb": phase_cb is not None,
                "model_path": model_path,
                "quantization": quantization,
                "wired_limit_bytes": _krea_wired_limit_bytes(),
            }
            if phase_cb is not None:
                phase_cb("preparing", "Preparing prompt & Qwen-Image 2.1 subprocess...")
            _cancel_event.clear()
            engine = Path(__file__).parent / "qwen_engine.py"
            _qwen_stderr_tail.clear()
            repo_python = Path(__file__).resolve().parent.parent / "venv" / "bin" / "python"
            python_exe = str(repo_python) if repo_python.exists() else sys.executable
            proc = _qwen_process
            if proc is None or proc.poll() is not None:
                if not _cleanup_stale_qwen_daemon():
                    raise RuntimeError("could not verify the previous Qwen daemon")
                proc = subprocess.Popen(
                    [python_exe, str(engine), "--serve"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                    bufsize=0,
                    start_new_session=True,
                    cwd=str(Path(__file__).resolve().parent),
                )
                proc._mlx_process_group = True
                _qwen_process = proc
                _write_qwen_pid(proc.pid)
                _qwen_reader_messages, _qwen_reader_done, _qwen_reader_thread = _start_json_reader(proc)
                _qwen_stderr_done, _qwen_stderr_thread = _start_stderr_reader(proc, _qwen_stderr_tail)
            _write_process_request(
                proc,
                (json.dumps(req, separators=(",", ":")) + "\n").encode("utf-8"),
                (cancel_event, _cancel_event),
                _engine_timeout("QWEN_ENGINE_TIMEOUT_S"),
            )
            result = None
            while True:
                line = _read_engine_message(
                    proc,
                    _qwen_reader_messages,
                    _qwen_reader_done,
                    (cancel_event, _cancel_event),
                    _engine_timeout("QWEN_ENGINE_TIMEOUT_S"),
                    "Qwen engine",
                    _qwen_stderr_tail,
                )
                if line is None:
                    err = "".join(_qwen_stderr_tail)[-2000:] or "EOF on stdout"
                    raise RuntimeError(f"Qwen engine died: {err}")
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if "progress" in payload:
                    step = payload["progress"].get("step")
                    if progress_cb is not None and isinstance(step, int):
                        progress_cb(max(0, step - 1))
                elif "phase" in payload:
                    if phase_cb is not None:
                        phase_cb(str(payload["phase"]), str(payload.get("detail", "")))
                elif "diag" in payload:
                    print(f"[qwen-engine diag] {json.dumps(payload)}", flush=True)
                else:
                    result = payload
                    break
            if result is None:
                err = "".join(_qwen_stderr_tail)[-2000:] or "no result received"
                raise RuntimeError(f"Qwen engine returned no result: {err}")
            if result.get("error"):
                raise RuntimeError(str(result["error"]))
            if phase_cb is not None:
                phase_cb("saving", "Finalizing image and metadata...")
            with Image.open(raw_image_path) as source:
                image = source.copy()
            actual_w = int(result.get("width", image.width))
            actual_h = int(result.get("height", image.height))
            if image.size != (actual_w, actual_h):
                raise RuntimeError(
                    f"Qwen returned {image.width}x{image.height}, expected {actual_w}x{actual_h}"
                )
            actual_sampler = result.get("sampler") or sampler_label
            actual_guidance = float(result.get("guidance", guidance if guidance is not None else 1.0))
            actual_quantization = result.get("quantization", quantization)
            meta = {
                "id": image_id,
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "sampler": actual_sampler,
                "scheduler": result.get("scheduler"),
                "width": actual_w,
                "height": actual_h,
                "steps": int(result.get("steps", steps)),
                "guidance": actual_guidance,
                "seed": seed,
                "quantization": actual_quantization,
                "loras": [],
                "generation_time": result["generation_time"],
                "load_time": result.get("load_time", 0.0),
                "created_at": time.time(),
                "software": "DiffusionBear",
                "generator": "DiffusionBear",
                "artist": app_settings.metadata_artist(),
                "tags": [],
                "file": final_image_path.name,
                "model": minfo["repo"],
                "engine": "qwen",
                "reference_images": [Path(p).name for p in (ref_paths or [])],
                "stealth": stealth,
                "format": fmt,
                "fast_vae": False,
                "tiled_vae": bool(result.get("tiled_vae")),
            }
            if ref_paths and image_strength is not None:
                meta["reference_strength"] = round(float(image_strength), 3)
            if minfo.get("civitai_version_id"):
                meta["modelVersionId"] = minfo["civitai_version_id"]
            if minfo.get("civitai_model_id"):
                meta["modelId"] = minfo["civitai_model_id"]
            if minfo.get("civitai_model_name"):
                meta["model_label"] = minfo["civitai_model_name"]
            if minfo.get("civitai_version_name"):
                meta["modelVersionName"] = minfo["civitai_version_name"]
            save_image_with_metadata(
                image=image,
                dest_path=final_image_path,
                meta=meta,
                output_format=fmt,
                stealth=stealth,
            )
            atomic_write_json(GENERATED_DIR / f"{image_id}.json", meta)
            gc.collect()
            return meta
        except Exception:
            # A failed request leaves the daemon in an unknown state (it may be
            # mid-graph or holding a bad tiling config), so drop it and let the
            # next generation pay a cold load rather than inherit the damage.
            _kill_qwen_process()
            raise
        finally:
            if not _qwen_process_is_healthy():
                _kill_qwen_process()
            else:
                _arm_qwen_watchdog()
            if raw_image_path is not None:
                try:
                    raw_image_path.unlink()
                except OSError:
                    pass
            _arm_mflux_watchdog()


def _shutdown_engines():
    try:
        _kill_qwen_process()
    except Exception:
        pass
    try:
        _kill_sdxl_daemon()
    except Exception:
        pass
    try:
        _drop_mflux_pipeline()
        _clear_krea_te_cache()
    except Exception:
        pass


atexit.register(_shutdown_engines)

