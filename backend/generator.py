import atexit
import collections
import copy
import gc
import json
import math
import os
import queue
import re
import select
import signal
import sqlite3
import subprocess
import sys
import tempfile
import traceback
import threading
import time
import uuid
from pathlib import Path

from collections import deque, OrderedDict
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo

from image_meta import (
    atomic_write_json,
    save_image_with_metadata,
    extract_image_metadata,
)
from samplers_store import (
    custom_ids_for_engine,
    ensure_mflux_registered,
    is_custom as sampler_is_custom,
)
from upscale import upscale_image, thumbnail_path
from sdxl_layout import sdxl_layout_report

import app_settings

# The Qwen21 text_encoder layout patch lives in qwen_engine.py ONLY.
#
# It used to be duplicated here as well, and that was a bug factory: generate() returns
# for both Qwen models before ever reaching _get_pipeline, so the copy in this file was
# unreachable, and it silently drifted out of date (it lacked the index.json-naming-a-
# single-file case, and returned None for every other layout -- the same defect that
# produced "AttributeError: 'NoneType' object has no attribute 'download_url'").
# Two copies of a load-bearing monkey-patch with no test asserting they agree is exactly
# how that happened. Qwen generation runs in a separate interpreter, so this process
# never needs it.

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

# TeaCache integration (mlx-teacache) - optional step-skipping for FLUX/Krea/Z-Image
try:
    from mlx_teacache.api import apply_teacache
    _TEACACHE_AVAILABLE = True
except Exception as _teacache_err:
    apply_teacache = None
    _TEACACHE_AVAILABLE = False
    print(f"[generator] TeaCache not available: {_teacache_err}", flush=True)

DATA_DIR = app_settings.DATA_DIR
ASSET_DIR = app_settings.ASSET_DIR
ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = DATA_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Tokenizer Resolution & Zero-Duplication Qwen/FLUX.3 Support
# ----------------------------------------------------------------------
# FLUX.3 Klein and FLUX.2 Klein use Qwen2TokenizerFast (vocab size 151,936),
# exactly identical to Qwen text encoders. In many MLX/HF model distributions
# (including vanch007/FLUX.3-Image-Klein-mlx-4bit and standard community repos),
# tokenizer files live in 'text_encoder/', 'processor/', or repository root '.',
# but mflux hardcodes hf_subdir='tokenizer' with fallback_subdirs=None.
#
# We patch Flux2KleinWeightDefinition.get_tokenizers (and QwenWeightDefinition.get_tokenizers)
# to supply fallback_subdirs=["text_encoder", "processor", "."].
# Additionally, TokenizerLoader._resolve_fallback_path is extended so that if a local
# FLUX.3 model lacks tokenizer files altogether, it resolves the tokenizer from any
# existing local Qwen model on disk (e.g. Qwen-Image-2.1-mflux-q4/processor) with
# zero file copying / zero disk duplication.

def is_tokenizer_sibling(filename: str) -> bool:
    """Check if a file in an HF repo is a tokenizer configuration or vocabulary file."""
    fn = filename.lower()
    base = Path(filename).name.lower()
    if base in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.json",
        "chat_template.jinja",
        "added_tokens.json",
        "special_tokens_map.json",
    ):
        return True
    if fn.startswith("tokenizer/"):
        return True
    if fn.startswith("processor/") and any(k in base for k in ("tokenizer", "vocab", "merges", "chat_template")):
        return True
    if fn.startswith("text_encoder/") and any(k in base for k in ("tokenizer", "vocab", "merges", "chat_template")):
        return True
    return False


def is_qwen_flux_tokenizer_model(model_id: str, minfo: dict | None = None, repo: str | None = None) -> bool:
    """Check if model uses the shared Qwen2Tokenizer (FLUX.2, FLUX.3, Qwen-Image, etc.)."""
    minfo = minfo or {}
    ecosystem = str(minfo.get("ecosystem", "")).lower()
    family = str(minfo.get("family", "")).lower()
    mid = str(model_id or "").lower()
    r = str(repo or "").lower()
    return (
        "flux2" in mid or "flux3" in mid or "qwen" in mid
        or "flux" in ecosystem or "qwen" in ecosystem
        or family in ("flux2", "flux3", "qwen")
        or "flux" in r or "qwen" in r
    )


def _is_valid_qwen_tokenizer_dir(path: Path | None) -> bool:
    if not path or not path.is_dir():
        return False
    cfg_file = path / "tokenizer_config.json"
    if not cfg_file.is_file():
        return False
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        tok_cls = str(cfg.get("tokenizer_class", "")).lower()
        return "qwen" in tok_cls
    except Exception:
        return False


def _find_local_qwen_tokenizer_path(exclude_dir: Path | None = None) -> Path | None:
    """Find a usable Qwen tokenizer directory across installed models with zero duplication."""
    models_dir = ASSET_DIR / "models"
    resolved_exclude = None
    if exclude_dir:
        try:
            resolved_exclude = Path(exclude_dir).resolve()
        except Exception:
            pass

    def _is_excluded(p: Path) -> bool:
        if resolved_exclude is None:
            return False
        try:
            res_p = p.resolve()
            return res_p == resolved_exclude or res_p.is_relative_to(resolved_exclude)
        except Exception:
            return False

    if models_dir.is_dir():
        priority_dirs = [
            models_dir / "tokenizers" / "qwen2",
            models_dir / "Qwen-Image-2.1-mflux-q4" / "processor",
            models_dir / "Qwen-Image-2.1-mflux-q4" / "tokenizer",
            models_dir / "krea2-turbo-q4" / "tokenizer",
            models_dir / "flux3-klein" / "text_encoder",
            models_dir / "flux2-klein-4b" / "text_encoder",
            models_dir / "flux2-klein-9b" / "text_encoder",
        ]
        for p in priority_dirs:
            if not _is_excluded(p) and _is_valid_qwen_tokenizer_dir(p):
                return p

        try:
            for cfg in models_dir.glob("**/tokenizer_config.json"):
                parent = cfg.parent
                if not _is_excluded(parent) and _is_valid_qwen_tokenizer_dir(parent):
                    return parent
        except Exception:
            pass

    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_hub.is_dir():
        try:
            for cfg in hf_hub.glob("**/tokenizer_config.json"):
                parent = cfg.parent
                if not _is_excluded(parent) and _is_valid_qwen_tokenizer_dir(parent):
                    return parent
        except Exception:
            pass

    return None


def download_and_register_qwen_tokenizer(target_dir: Path | str | None = None) -> Path:
    """Download and register the universal Qwen2 tokenizer from Hugging Face if none is installed.

    Saves into ASSET_DIR / "models" / "tokenizers" / "qwen2", and links to target_dir if provided.
    """
    reg_dir = ASSET_DIR / "models" / "tokenizers" / "qwen2"
    reg_dir.mkdir(parents=True, exist_ok=True, mode=0o755)

    cfg_file = reg_dir / "tokenizer_config.json"
    if not cfg_file.is_file():
        from huggingface_hub import hf_hub_download
        import hf_service
        token = hf_service.get_hf_token()
        for f in ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"):
            hf_hub_download(
                repo_id="Qwen/Qwen2-7B",
                filename=f,
                local_dir=str(reg_dir),
                token=token,
            )

    if target_dir:
        try:
            td = Path(target_dir).resolve()
            if td.is_dir():
                target_tok = td / "tokenizer"
                if not (target_tok.exists() or target_tok.is_symlink()):
                    try:
                        target_tok.symlink_to(reg_dir)
                    except OSError:
                        pass
                # For Qwen-specific layouts expecting processor/
                target_proc = td / "processor"
                if not (target_proc.exists() or target_proc.is_symlink()):
                    try:
                        target_proc.symlink_to(reg_dir)
                    except OSError:
                        pass
        except Exception:
            pass

    return reg_dir


def _ensure_flux_tokenizer_available(local_path: str | Path | None) -> None:
    """Ensure local FLUX model path has a valid tokenizer directory without copying files.

    If the weights directory contains tokenizer files in 'text_encoder/', 'processor/',
    or '.', a relative symlink 'tokenizer' is created. If no local tokenizer files exist,
    links to a shared local Qwen tokenizer if available, or automatically downloads & registers it.
    """
    if not local_path:
        return
    try:
        lp = Path(local_path).resolve()
        if not lp.is_dir():
            return

        target_tok = lp / "tokenizer"
        if target_tok.exists() or target_tok.is_symlink():
            return

        # 1. Check inside local model folder
        for candidate in ("text_encoder", "processor"):
            if (lp / candidate / "tokenizer_config.json").exists():
                try:
                    target_tok.symlink_to(candidate)
                    return
                except OSError:
                    pass

        if (lp / "tokenizer_config.json").exists():
            try:
                target_tok.symlink_to(".")
                return
            except OSError:
                pass

        # 2. Cross-model fallback: link to shared Qwen tokenizer
        shared = _find_local_qwen_tokenizer_path(exclude_dir=lp)
        if shared and (shared / "tokenizer_config.json").exists():
            try:
                target_tok.symlink_to(shared)
                return
            except OSError:
                pass

        # 3. If no tokenizer is installed anywhere on disk, download & register it
        try:
            download_and_register_qwen_tokenizer(target_dir=lp)
        except Exception as _dl_err:
            print(f"[generator] WARN failed to auto-download Qwen tokenizer: {_dl_err}", flush=True)
    except Exception as e:
        print(f"[generator] WARN failed to ensure tokenizer for {local_path}: {e}", flush=True)


_DEFAULT_CHATML_TEMPLATE = (
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- endif -%}"
)


def _ensure_tokenizer_chat_template(pipeline) -> None:
    """Ensure pipeline tokenizer has a valid chat_template for ChatML models."""
    try:
        tokenizers = []
        if hasattr(pipeline, "tokenizer") and pipeline.tokenizer is not None:
            tokenizers.append(pipeline.tokenizer)
        if hasattr(pipeline, "tokenizers") and isinstance(pipeline.tokenizers, dict):
            tokenizers.extend(pipeline.tokenizers.values())
        for tok in tokenizers:
            hf = getattr(tok, "tokenizer", tok)
            if hasattr(hf, "chat_template") and not getattr(hf, "chat_template", None):
                hf.chat_template = _DEFAULT_CHATML_TEMPLATE
    except Exception:
        pass


try:
    from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

    # 1. Ensure Flux2KleinWeightDefinition.get_tokenizers supplies fallback_subdirs
    _orig_flux2_get_tokenizers = Flux2KleinWeightDefinition.get_tokenizers

    def _patched_flux2_get_tokenizers():
        defs = _orig_flux2_get_tokenizers()
        for d in defs:
            if d.fallback_subdirs is None:
                d.fallback_subdirs = ["text_encoder", "processor", "."]
            else:
                for candidate in ("text_encoder", "processor", "."):
                    if candidate not in d.fallback_subdirs:
                        d.fallback_subdirs.append(candidate)
        return defs

    Flux2KleinWeightDefinition.get_tokenizers = _patched_flux2_get_tokenizers

    # 2. Also patch QwenWeightDefinition if available
    try:
        from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
        _orig_qwen_get_tokenizers = QwenWeightDefinition.get_tokenizers

        def _patched_qwen_get_tokenizers():
            defs = _orig_qwen_get_tokenizers()
            for d in defs:
                if d.fallback_subdirs is None:
                    d.fallback_subdirs = ["processor", "text_encoder", "."]
                else:
                    for candidate in ("processor", "text_encoder", "."):
                        if candidate not in d.fallback_subdirs:
                            d.fallback_subdirs.append(candidate)
            return defs

        QwenWeightDefinition.get_tokenizers = _patched_qwen_get_tokenizers
    except Exception:
        pass

    # 3. Patch TokenizerLoader._resolve_fallback_path to support text_encoder / processor
    # and fallback to any installed local Qwen tokenizer if missing
    _orig_resolve_fallback_path = TokenizerLoader._resolve_fallback_path

    @staticmethod
    def _safe_resolve_fallback_path(
        root_path: Path,
        fallback_subdirs: list[str] | None,
        tokenizer_class: str,
    ) -> Path | None:
        subdirs = list(fallback_subdirs or [])
        for d in ("text_encoder", "processor", "."):
            if d not in subdirs:
                subdirs.append(d)

        res = _orig_resolve_fallback_path(root_path, subdirs, tokenizer_class)
        if res is not None:
            return res

        # Cross-model zero-duplication fallback for Qwen2Tokenizer / Qwen2TokenizerFast
        if tokenizer_class in ("Qwen2Tokenizer", "Qwen2TokenizerFast"):
            qwen_path = _find_local_qwen_tokenizer_path()
            if qwen_path and TokenizerLoader._is_tokenizer_loadable(qwen_path, tokenizer_class):
                return qwen_path

        return None

    TokenizerLoader._resolve_fallback_path = _safe_resolve_fallback_path

    # 4. Patch TokenizerLoader._load_raw_tokenizer to supply fallback chat_template
    _orig_load_raw_tokenizer = TokenizerLoader._load_raw_tokenizer

    @staticmethod
    def _safe_load_raw_tokenizer(
        tokenizer_path: Path,
        tokenizer_class: str,
        chat_template: str | None = None,
    ):
        raw_tok = _orig_load_raw_tokenizer(tokenizer_path, tokenizer_class, chat_template)
        if hasattr(raw_tok, "chat_template") and not getattr(raw_tok, "chat_template", None):
            raw_tok.chat_template = _DEFAULT_CHATML_TEMPLATE
        return raw_tok

    TokenizerLoader._load_raw_tokenizer = _safe_load_raw_tokenizer

    # 5. Patch Flux3PromptEncoder to guarantee ChatML template fallback, minimum padding, and non-empty text_ids
    try:
        from mflux.models.flux3.model.flux3_text_encoder.prompt_encoder import Flux3PromptEncoder
        _orig_flux3_tokenize = Flux3PromptEncoder.tokenize

        @staticmethod
        def _safe_flux3_tokenize(prompt: str, tokenizer) -> tuple:
            import math
            import mlx.core as mx
            hf = getattr(tokenizer, "tokenizer", tokenizer)
            if hasattr(hf, "chat_template") and not getattr(hf, "chat_template", None):
                hf.chat_template = _DEFAULT_CHATML_TEMPLATE
            try:
                inp_ids, att_mask = _orig_flux3_tokenize(prompt, tokenizer)
                if inp_ids.ndim == 2 and inp_ids.shape[1] >= getattr(Flux3PromptEncoder, "PAD_MULTIPLE", 80):
                    return inp_ids, att_mask
            except Exception:
                pass
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            cap = getattr(Flux3PromptEncoder, "MAX_LENGTH", 4096)
            pad_mult = getattr(Flux3PromptEncoder, "PAD_MULTIPLE", 80)
            try:
                real = len(hf(text, truncation=True, max_length=cap)["input_ids"])
            except Exception:
                real = pad_mult
            length = max(pad_mult, min(math.ceil(real / pad_mult) * pad_mult, cap))
            tokens = hf(text, return_tensors="np", padding="max_length", truncation=True, max_length=length)
            input_ids = np.asarray(tokens["input_ids"])
            if input_ids.ndim == 0:
                input_ids = input_ids.reshape(1, 1)
            elif input_ids.ndim == 1:
                input_ids = input_ids.reshape(1, -1)
            raw_mask = tokens.get("attention_mask") if isinstance(tokens, dict) else getattr(tokens, "attention_mask", None)
            if raw_mask is not None:
                att_mask = np.asarray(raw_mask)
                if att_mask.ndim == 0:
                    att_mask = att_mask.reshape(1, 1)
                elif att_mask.ndim == 1:
                    att_mask = att_mask.reshape(1, -1)
            else:
                att_mask = np.ones_like(input_ids)
            if input_ids.shape[1] < length:
                raw_pad = getattr(hf, "pad_token_id", 0)
                pad_id = int(raw_pad) if isinstance(raw_pad, (int, np.integer)) else 0
                pad_width = length - input_ids.shape[1]
                input_ids = np.pad(input_ids, ((0, 0), (0, pad_width)), mode="constant", constant_values=pad_id)
                att_mask = np.pad(att_mask, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
            return mx.array(input_ids), mx.array(att_mask)

        Flux3PromptEncoder.tokenize = _safe_flux3_tokenize

        _orig_prepare_text_ids = Flux3PromptEncoder.prepare_text_ids

        @staticmethod
        def _safe_prepare_text_ids(length: int):
            return _orig_prepare_text_ids(max(1, length))

        Flux3PromptEncoder.prepare_text_ids = _safe_prepare_text_ids

        # 6. Patch Flux3PosEmbed.__call__ against empty ids and broadcasting mismatch
        from mflux.models.flux3.model.flux3_transformer.flux3_pos_embed import Flux3PosEmbed
        _orig_flux3_pos_embed_call = Flux3PosEmbed.__call__

        def _safe_flux3_pos_embed_call(self, ids):
            import mlx.core as mx
            if ids.size == 0 or (hasattr(ids, "shape") and len(ids.shape) > 0 and ids.shape[0] == 0):
                dim_total = sum(d // 2 for d in getattr(self, "axes_dim", (32, 32, 32, 32)))
                return mx.zeros((0, dim_total), dtype=mx.float32), mx.zeros((0, dim_total), dtype=mx.float32)
            return _orig_flux3_pos_embed_call(self, ids)

        Flux3PosEmbed.__call__ = _safe_flux3_pos_embed_call
    except Exception as _f3_tok_err:
        print(f"[generator] WARN failed to patch Flux3 text/pos modules: {_f3_tok_err}", flush=True)
except Exception as _tok_patch_err:
    print(f"[generator] WARN failed to patch TokenizerLoader: {_tok_patch_err}", flush=True)



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
        "engine": "mflux",
        "label": "FLUX.2-klein 4B",
        "repo": "mlx-community/flux2-klein-4b-4bit",
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
        "samplers": ["flow_match_euler_discrete", "euler"],
        "default_sampler": "flow_match_euler_discrete",
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft 512×768 (~56s)", "width": 512, "height": 768, "steps": 4},
            {"id": "fast", "label": "⚡ Fast 768×768 (~90s)", "width": 768, "height": 768, "steps": 4},
            {"id": "quality", "label": "✦ Quality 1024×1024 (~2min)", "width": 1024, "height": 1024, "steps": 4},
        ],
    },
    "flux2-klein-9b": {
        "id": "flux2-klein-9b",
        "engine": "mflux",
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
        "samplers": ["flow_match_euler_discrete", "euler"],
        "default_sampler": "flow_match_euler_discrete",
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft 512×768 (~110s)", "width": 512, "height": 768, "steps": 4},
            {"id": "fast", "label": "⚡ Fast 768×768", "width": 768, "height": 768, "steps": 4},
            {"id": "quality", "label": "✦ Quality 1024×1024", "width": 1024, "height": 1024, "steps": 4},
        ],
    },
    "flux3-klein": {
        "id": "flux3-klein",
        "engine": "mflux",
        "label": "FLUX.3-klein 5.6B",
        "repo": "zimageapp/FLUX-3-Klein-MLX-Q4",
        "ecosystem": "FLUX.3",
        # Export manifests name the architecture ("flux-3-image-klein"), not
        # this registry id, and the package ships no root model_index.json --
        # so directory-name heuristics cannot identify an install. These
        # declared names let _auto_bind_unbound_model recover an install that
        # lost its model_paths registration.
        "identity": ["flux-3-image-klein"],
        "is_distilled": True,
        "default_steps": 16,
        "max_steps": 24,
        "default_width": 768,
        "default_height": 512,
        "default_guidance": 1.0,
        "supports_guidance": False,
        "supports_negative": False,
        "supports_loras": False,
        "supports_multi_reference": True,
        "supports_ref": True,
        "max_reference_images": 10,
        "supports_fast_vae": False,
        "supports_bounding_box": True,
        "lora_format": "FLUX.3",
        "samplers": ["flow_match_euler_discrete", "euler"],
        "default_sampler": "flow_match_euler_discrete",
        "presets": [
            {"id": "fast", "label": "⚡ Fast 512×512", "width": 512, "height": 512, "steps": 16},
            {"id": "balanced", "label": "✦ Balanced 768×512", "width": 768, "height": 512, "steps": 16},
            {"id": "quality", "label": "✦ Quality 1024×1024", "width": 1024, "height": 1024, "steps": 16},
        ],
    },
    "moss-image-sft": {
        "id": "moss-image-sft",
        "engine": "moss",
        "label": "MOSS-Image SFT 5B",
        "repo": "OpenMOSS-Team/MOSS-Image-SFT",
        # The install task strips the upstream 16.79 GB bf16 text_encoder/ from
        # this repo: the engine's primary encoder is its MLX q8-g64 conversion,
        # downloaded from te_mlx_repo into te_mlx_sibling (a directory NEXT TO
        # the model dir, which _mlx_te_dir auto-detects). The bf16 torch
        # reference encoder therefore no longer ships with fresh installs and
        # MOSS_TE_MLX=off has nothing local to fall back to -- that escape
        # hatch needs the full upstream snapshot instead. File total after the
        # strip: 10.4 GB upstream + 10.0 GB MLX encoder.
        "te_mlx_repo": "Ouinche/moss-image-qwen3.5-te-mlx-q8g64",
        "te_mlx_sibling": "qwen3_5-te-q8g64",
        "ecosystem": "MOSS",
        "is_distilled": False,
        "default_steps": 50,
        "default_width": 1024,
        "default_height": 1024,
        "default_guidance": 3.0,
        "supports_guidance": True,
        "supports_negative": True,
        "supports_loras": True,
        "supports_multi_reference": False,
        "supports_ref": False,
        "max_reference_images": 0,
        "supports_fast_vae": False,
        "supports_bounding_box": True,
        "lora_format": "MOSS",
        "samplers": ["flow_match_euler_discrete", "euler"],
        "default_sampler": "flow_match_euler_discrete",
        "presets": [
            {"id": "dmd4", "label": "⚡ MOSS DMD 4×512×512", "width": 512, "height": 512, "steps": 4},
            {"id": "dmd8", "label": "⚡ MOSS DMD 8×1024×1024", "width": 1024, "height": 1024, "steps": 8},
            {"id": "fast", "label": "⚡ Fast 512×512", "width": 512, "height": 512, "steps": 25},
            {"id": "balanced", "label": "✦ Balanced 1024×1024", "width": 1024, "height": 1024, "steps": 50},
            {"id": "quality", "label": "✦ Quality 1024×1024 (50st)", "width": 1024, "height": 1024, "steps": 50},
        ],
    },
    "moss-image-base": {
        "id": "moss-image-base",
        "engine": "moss",
        "label": "MOSS-Image Base 5B",
        "repo": "OpenMOSS-Team/MOSS-Image-Base",
        "ecosystem": "MOSS",
        "is_distilled": False,
        "default_steps": 50,
        "default_width": 1024,
        "default_height": 1024,
        "default_guidance": 3.0,
        "supports_guidance": True,
        "supports_negative": True,
        "supports_loras": True,
        "supports_multi_reference": False,
        "supports_ref": False,
        "max_reference_images": 0,
        "supports_fast_vae": False,
        "supports_bounding_box": True,
        "lora_format": "MOSS",
        "samplers": ["flow_match_euler_discrete", "euler"],
        "default_sampler": "flow_match_euler_discrete",
        "presets": [
            {"id": "fast", "label": "⚡ Fast 512×512", "width": 512, "height": 512, "steps": 25},
            {"id": "balanced", "label": "✦ Balanced 1024×1024", "width": 1024, "height": 1024, "steps": 50},
            {"id": "quality", "label": "✦ Quality 1024×1024 (50st)", "width": 1024, "height": 1024, "steps": 50},
        ],
    },
    "flux1-schnell": {
        "id": "flux1-schnell",
        "engine": "mflux",
        "label": "FLUX.1-schnell",
        "repo": "mflux-community/flux-1-schnell-mflux-q4",
        "ecosystem": "FLUX.1",
        "is_distilled": True,
        "default_steps": 4,
        "default_width": 512,
        "default_height": 768,
        "default_guidance": 1.0,
        "supports_guidance": True,
        "supports_negative": False,
        "supports_loras": True,
        "supports_multi_reference": False,
        "supports_ref": True,
        "max_reference_images": 1,
        "supports_fast_vae": True,
        "lora_format": "FLUX.1",
        "samplers": ["flow_match_euler_discrete", "euler"],
        "default_sampler": "flow_match_euler_discrete",
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft 512×768", "width": 512, "height": 768, "steps": 4},
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
            {"id": "draft", "label": "⚡ Hyper-SD 8-step Draft 512×768 (~45s)", "width": 512, "height": 768, "steps": 8, "sampler": "euler_trailing", "guidance": 2.0, "cache_interval": 1, "loras": [{"path": "Hyper-SDXL-8steps-CFG-lora.safetensors", "scale": 1.0}]},
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
            {"id": "draft", "label": "⚡ Hyper-SD 8-step Draft 512×768 (~49s)", "width": 512, "height": 768, "steps": 8, "sampler": "euler_trailing", "guidance": 2.0, "cache_interval": 1, "loras": [{"path": "Hyper-SDXL-8steps-CFG-lora.safetensors", "scale": 1.0}]},
            {"id": "photo", "label": "✦ Carré 1024×1024 (25 steps)", "width": 1024, "height": 1024, "steps": 25, "sampler": "dpmpp_2m_karras", "guidance": 4.0, "cache_interval": 1},
            {"id": "portrait", "label": "✦ Portrait 832×1216 (25 steps)", "width": 832, "height": 1216, "steps": 25, "sampler": "dpmpp_2m_karras", "guidance": 4.0, "cache_interval": 1},
        ],
    },
    "Pony": {
        "id": "Pony",
        "label": "Pony Diffusion V6 XL",
        "repo": "AstraliteHeart/pony-diffusion-v6-xl",
        "model_dir": ASSET_DIR / "models" / "Pony",
        "engine": "sdxl",
        "civitai_version_id": 290640,
        "civitai_model_id": 257749,
        "civitai_model_name": "Pony Diffusion V6 XL",
        "civitai_version_name": "V6 (3.0)",
        "sha256": "67AB2FD8EC",
        "ecosystem": "SDXL",
        "default_steps": 25,
        "default_width": 512,
        "default_height": 768,
        "default_guidance": 5.0,
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
            {"id": "draft", "label": "⚡ Draft 512×768 (20 steps)", "width": 512, "height": 768, "steps": 20, "sampler": "dpmpp_2m_karras", "guidance": 5.0, "cache_interval": 1},
            {"id": "photo", "label": "✦ Carré 1024×1024 (25 steps)", "width": 1024, "height": 1024, "steps": 25, "sampler": "dpmpp_2m_karras", "guidance": 5.0, "cache_interval": 1},
        ],
    },
    "z-image-turbo": {
        "id": "z-image-turbo",
        "engine": "mflux",
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
        "samplers": ["linear", "flow_match_euler_discrete", "euler"],
        "default_sampler": "linear",
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft 512×768 (~1.5min)", "width": 512, "height": 768, "steps": 6},
            {"id": "turbo", "label": "✦ Turbo 1024×1024 (~4.9min)", "width": 1024, "height": 1024, "steps": 6},
            {"id": "wide", "label": "✦ Wide HD 1280×720 (~4.9min)", "width": 1280, "height": 720, "steps": 6},
        ],
    },
    "krea2-turbo": {
        "id": "krea2-turbo",
        "engine": "mflux",
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
        "samplers": ["er_sde", "euler"],
        "default_sampler": "er_sde",
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
        "engine": "qwen",
        "label": "Qwen-Image 2.1 (MLX q4, experimental)",
        "repo": "mlx-community/Qwen-Image-2.1-mflux-q4",
        # Must be the mflux pack, NOT mlx-community/Qwen-Image-2.1-MLX-4bit.
        # Measured 2026-10-22, same prompt/seed/512x512/25 steps/guidance 1.0:
        # the MLX-4bit pack renders waxy, grainy skin (median high-frequency noise
        # 16.2 vs 4.5-6.4 for SDXL/FLUX2 at the same size) because it pairs a
        # group-quantized q4 Qwen3-VL text encoder with the transformer; upstream
        # warns that quantizing the TE costs "significant semantic degradation".
        # The mflux pack keeps the TE dense (4.26GB bf16) and renders clean skin.
        # It is also much faster: 12.4s/step vs 85s/step at 512x512, and the whole
        # 8.9GB pack loads in 5.3s. Peak RSS 7.9GB, so 16GB is fine with the
        # engine's unbounded wired limit. Keep the installed dir at
        # models/qwen-image-2.1 so resolve_local_model_path finds it.
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
    "qwen-image-2.1-turbo": {
        "id": "qwen-image-2.1-turbo",
        "engine": "qwen",
        "label": "Qwen-Image 2.1 Turbo (MLX q4, experimental)",
        "repo": "Sawfwair/Qwen-Image-2.1-Turbo-MLX-Mixed-4bit",
        "ecosystem": "Qwen-Image 2.1",
        "is_distilled": True,
        "default_steps": 8,
        "default_width": 512,
        "default_height": 768,
        "default_guidance": 1.0,
        "supports_guidance": True,
        "supports_negative": True,
        "supports_loras": False,
        "supports_ref": True,
        "max_reference_images": 1,
        "supports_fast_vae": False,
        "samplers": ["linear", "euler"],
        "default_sampler": "linear",
        # No time estimates here. test_latency_cache.PresetTimeClaimTests requires a
        # preset to quote a time only when it has a measured median at that exact
        # (size, steps), and Turbo has none yet -- the only measurements are 512x768 and
        # 512x512 at 4 steps. These labels were carrying invented ~3-6min figures.
        # Measure each preset at its own size/steps and add them to MEASURED_PRESET_
        # SECONDS in test_latency_cache.py before quoting a time again.
        "presets": [
            {"id": "draft", "label": "⚡ Fast Draft 512×768 8 steps", "width": 512, "height": 768, "steps": 8},
            {"id": "wide", "label": "✦ Wide 768×768 8 steps", "width": 768, "height": 768, "steps": 8},
            {"id": "landscape", "label": "✦ Landscape 768×512 8 steps", "width": 768, "height": 512, "steps": 8},
            {"id": "quality", "label": "✦ Quality 1024×1024 10 steps", "width": 1024, "height": 1024, "steps": 10},
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
    # Custom registry fall-through: only ids no built-in owns reach here.
    import custom_models

    for entry in custom_models.read_custom_models():
        if entry.get("id") == model_id:
            return _translate_custom_entry(entry)
    return None


# A translated custom dict starts from a same-variant built-in entry and
# overrides identity, never literals: supports_* flags, defaults and presets
# ride along from the canonical entry, correct per family by construction.
_CUSTOM_CANONICAL_BY_VARIANT = {
    ("flux2", "4b"): "flux2-klein-4b",
    ("flux2", "9b"): "flux2-klein-9b",
    ("qwen", "base"): "qwen-image-2.1",
    ("qwen", "turbo"): "qwen-image-2.1-turbo",
}
_CUSTOM_FAMILY_REPRESENTATIVE = {
    "krea": "krea2-turbo",
    "zit": "z-image-turbo",
}

# A custom model must not inherit another model's Civitai identity: these are
# dropped from the canonical copy unless the custom entry supplies its own
# (as `civitai_ids`, stored verbatim under that key).
_CUSTOM_CIVITAI_IDENTITY_KEYS = (
    "civitai_model_id",
    "civitai_version_id",
    "civitai_model_name",
    "civitai_version_name",
    "sha256",
)


def _custom_canonical_id(entry: dict) -> str | None:
    canonical = _CUSTOM_CANONICAL_BY_VARIANT.get((entry.get("family"), entry.get("variant")))
    if canonical is not None:
        return canonical
    if entry.get("family") == "sdxl":
        if entry.get("is_distilled") is True:
            return "juggernaut-xl-lightning"
        # Standard SDXL (Pony, Illustrious, etc.) is non-distilled
        return "juggernaut-xi"
    return _CUSTOM_FAMILY_REPRESENTATIVE.get(entry.get("family"))


def _translate_custom_entry(entry: dict) -> dict | None:
    canonical_id = _custom_canonical_id(entry)
    canonical = MODELS.get(canonical_id) if canonical_id else None
    if canonical is None:
        return None
    info = copy.deepcopy(canonical)
    info["id"] = entry["id"]
    info["label"] = entry["label"]
    info["family"] = entry["family"]
    if entry.get("variant") is not None:
        info["variant"] = entry["variant"]
    else:
        info.pop("variant", None)
    if entry.get("local_path") is not None:
        info["local_path"] = entry["local_path"]
    else:
        info.pop("local_path", None)
    # A custom entry owns its directory via local_path: inheriting the
    # canonical model_dir would aim cached-checks, disk usage and uninstall
    # at a built-in's weights.
    info.pop("model_dir", None)
    info["repo"] = entry.get("hf_repo") or None
    if entry.get("hf_repo"):
        info["hf_repo"] = entry["hf_repo"]
    else:
        info.pop("hf_repo", None)
    for key in _CUSTOM_CIVITAI_IDENTITY_KEYS:
        info.pop(key, None)
    if entry.get("civitai_ids") is not None:
        info["civitai_ids"] = entry["civitai_ids"]

    # SDXL custom entries: unless explicitly flagged as distilled, they are
    # standard non-distilled models (e.g. Pony, Illustrious) allowing up to 50 steps.
    if entry.get("family") == "sdxl":
        is_distilled = bool(entry.get("is_distilled", False))
        info["is_distilled"] = is_distilled
        if not is_distilled:
            info["default_steps"] = int(entry.get("default_steps") or 25)
            info["default_guidance"] = float(entry.get("default_guidance") or 6.0)
            info["samplers"] = ["euler_a", "dpmpp_2m_karras", "euler", "ddim", "euler_trailing"]
            info["default_sampler"] = "euler_a"
            info["presets"] = [
                {"id": "draft", "label": "⚡ Fast Draft 512×768 (20 steps)", "width": 512, "height": 768, "steps": 20, "sampler": "euler_a", "guidance": 5.0, "cache_interval": 1},
                {"id": "photo", "label": "✦ Carré 1024×1024 (25 steps)", "width": 1024, "height": 1024, "steps": 25, "sampler": "dpmpp_2m_karras", "guidance": 6.0, "cache_interval": 1},
                {"id": "portrait", "label": "✦ Portrait 832×1216 (25 steps)", "width": 832, "height": 1216, "steps": 25, "sampler": "dpmpp_2m_karras", "guidance": 6.0, "cache_interval": 1},
                {"id": "cinematic", "label": "✦ Cinématique 1216×832 (25 steps)", "width": 1216, "height": 832, "steps": 25, "sampler": "dpmpp_2m_karras", "guidance": 6.0, "cache_interval": 1},
            ]
            info.pop("max_steps", None)
    elif "is_distilled" in entry and entry["is_distilled"] is not None:
        info["is_distilled"] = bool(entry["is_distilled"])

    return info

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
# p=1.00) while all 12 renders are pixel-identical, so the pin bought nothing.
def _env_wired_gb() -> float | None:
    return _WIRED_LIMIT_GB


def _wired_limit_gb() -> float | None:
    """Configured generic wired limit in GB. None means auto (derive from the device)."""
    v = app_settings.get_setting("memory_wired_limit_gb")
    return float(v) if v is not None else _env_wired_gb()


def _krea_wired_limit_gb() -> float | None:
    """Configured krea2 wired limit in GB. None means auto (derive from the device).

    Read the pin from settings.json itself, not from any note about it: an
    earlier explicit-pin experiment was reverted by changing an env default
    while a stale 9 GB sat in settings.json, so the cap stayed in force long
    after the note said it was gone. (That stale env plumbing,
    MLX_KREA_WIRED_LIMIT_GB, is deleted; this settings key is the only krea
    pin left.)
    """
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
    """Wired budget for krea2 (13B q4) only. Auto (settings None) means UNBOUNDED
    for krea2, not the derived ceiling: since the TE-quantise fix the pipeline sits
    at ~9.6GB, which fits *inside* the derived 10.88GB budget and therefore gets
    fully wired, leaving <1.3GB for text-encoder forward activations — and the
    first encode pass Metal-OOMs (BattlePlan Test D Hop 1, measured
    2026-10-18: crash in krea2 prompt_encoder mx.eval(embeds)). Unbounded, the
    same pipeline rendered fine (and the 2026-09-29 A/B had already shown a pin
    buys nothing: p=1.00, pixel-identical). An explicit positive pin is still
    honoured as a deliberate choice.
    """
    v = _krea_wired_limit_gb()
    if v is None:
        return 0
    return _resolve_wired_budget(v, 9)

_lock = threading.RLock()
_model_maintenance_lock = threading.RLock()
_cancel_event = threading.Event()
_pipeline = None
_current_pipeline_key = None
_current_pipeline_model = None
_current_sdxl_model = None
_prompt_cache: OrderedDict[str, tuple] = OrderedDict()
_PROMPT_CACHE_LOCK = threading.Lock()
_PROMPT_CACHE_MAX_SIZE = 32
_PROMPT_CACHE_MAX_BYTES = 256 * (1 << 20)

# Disk-backed prompt embedding cache (survives restarts)
# Key: sha256(prompt + model_id + negative_prompt) -> serialized MLX arrays
_PROMPT_CACHE_DB: sqlite3.Connection | None = None
_PROMPT_CACHE_DB_PATH = DATA_DIR / "prompt_cache.sqlite"
_PROMPT_CACHE_DB_LOCK = threading.Lock()
_PROMPT_CACHE_DB_MAX_ENTRIES = 1000
_PROMPT_CACHE_DB_MAX_AGE_DAYS = 30

_taef_models: dict = {}

# Quantized Krea 2 text encoder, reused across pipeline rebuilds. See
# _krea_quantized_text_encoder. Held outside the pipeline cache on purpose: the
# pipeline is dropped and rebuilt on every LoRA/step change, the encoder is not.
_krea_te_cache: dict = {}
_krea_te_lock = threading.RLock()
_krea_te_skip_installed = False
_krea_te_skip_bf16 = {"on": False}


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
    # Caller must hold _PROMPT_CACHE_LOCK
    total = sum(_value_bytes(v) for v in _prompt_cache.values())
    while _prompt_cache and (
        len(_prompt_cache) > _PROMPT_CACHE_MAX_SIZE or total > _PROMPT_CACHE_MAX_BYTES
    ):
        _, value = _prompt_cache.popitem(last=False)
        total -= _value_bytes(value)


def _cache_prompt(key, value):
    with _PROMPT_CACHE_LOCK:
        _prompt_cache[key] = value
        _prompt_cache.move_to_end(key)
        _trim_prompt_cache()


def _get_prompt_cache(key):
    with _PROMPT_CACHE_LOCK:
        if key in _prompt_cache:
            _prompt_cache.move_to_end(key)
            return _prompt_cache[key]
        return None


def _clear_prompt_cache():
    with _PROMPT_CACHE_LOCK:
        _prompt_cache.clear()


def _prompt_cache_len() -> int:
    with _PROMPT_CACHE_LOCK:
        return len(_prompt_cache)


# Disk-backed prompt cache functions
def _init_prompt_cache_db() -> None:
    global _PROMPT_CACHE_DB
    if _PROMPT_CACHE_DB is not None:
        return
    try:
        _PROMPT_CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PROMPT_CACHE_DB = sqlite3.connect(
            str(_PROMPT_CACHE_DB_PATH),
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        _PROMPT_CACHE_DB.execute("PRAGMA journal_mode=WAL")
        _PROMPT_CACHE_DB.execute("PRAGMA synchronous=NORMAL")
        _PROMPT_CACHE_DB.execute("""
            CREATE TABLE IF NOT EXISTS prompt_embeddings (
                key TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                prompt_embeds BLOB NOT NULL,
                text_ids BLOB NOT NULL,
                negative_embeds BLOB,
                negative_ids BLOB,
                created_at REAL NOT NULL,
                hit_count INTEGER DEFAULT 0
            )
        """)
        _PROMPT_CACHE_DB.execute("""
            CREATE INDEX IF NOT EXISTS idx_prompt_model ON prompt_embeddings(model_id)
        """)
        # Evict old entries
        cutoff = time.time() - (_PROMPT_CACHE_DB_MAX_AGE_DAYS * 86400)
        _PROMPT_CACHE_DB.execute(
            "DELETE FROM prompt_embeddings WHERE created_at < ?",
            (cutoff,)
        )
        # Enforce max entries
        _PROMPT_CACHE_DB.execute("""
            DELETE FROM prompt_embeddings
            WHERE key NOT IN (
                SELECT key FROM prompt_embeddings
                ORDER BY hit_count DESC, created_at DESC
                LIMIT ?
            )
        """, (_PROMPT_CACHE_DB_MAX_ENTRIES,))
        print(f"[generator] Prompt cache DB initialized at {_PROMPT_CACHE_DB_PATH}", flush=True)
    except Exception as e:
        print(f"[generator] WARN prompt cache DB init failed: {e}", flush=True)
        _PROMPT_CACHE_DB = None


def _disk_cache_prompt(key: str, model_id: str, prompt_hash: str,
                        prompt_embeds, text_ids,
                        negative_embeds=None, negative_ids=None) -> None:
    """Persist prompt embeddings to disk."""
    if _PROMPT_CACHE_DB is None:
        return
    try:
        import mlx.core as mx
        import tempfile
        import os

        def serialize(arr):
            if arr is None:
                return None
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
                fname = f.name
            try:
                mx.save(fname, arr)
                with open(fname, 'rb') as rf:
                    return rf.read()
            finally:
                if os.path.exists(fname):
                    try:
                        os.unlink(fname)
                    except OSError:
                        pass

        with _PROMPT_CACHE_DB_LOCK:
            _PROMPT_CACHE_DB.execute("""
                INSERT OR REPLACE INTO prompt_embeddings
                (key, model_id, prompt_hash, prompt_embeds, text_ids,
                 negative_embeds, negative_ids, created_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                key, model_id, prompt_hash,
                serialize(prompt_embeds), serialize(text_ids),
                serialize(negative_embeds), serialize(negative_ids),
                time.time()
            ))
    except Exception as e:
        print(f"[generator] WARN disk cache write failed: {e}", flush=True)


def _disk_get_prompt(key: str):
    """Retrieve prompt embeddings from disk."""
    if _PROMPT_CACHE_DB is None:
        return None
    try:
        import mlx.core as mx
        import tempfile
        import os

        def deserialize(blob):
            if blob is None:
                return None
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
                tmp.write(blob)
                fname = tmp.name
            try:
                return mx.load(fname)
            finally:
                if os.path.exists(fname):
                    try:
                        os.unlink(fname)
                    except OSError:
                        pass

        with _PROMPT_CACHE_DB_LOCK:
            row = _PROMPT_CACHE_DB.execute(
                "SELECT prompt_embeds, text_ids, negative_embeds, negative_ids, hit_count "
                "FROM prompt_embeddings WHERE key = ?",
                (key,)
            ).fetchone()
            if row is None:
                return None
            prompt_embeds, text_ids, neg_embeds, neg_ids, hit_count = row
            # Increment hit count
            _PROMPT_CACHE_DB.execute(
                "UPDATE prompt_embeddings SET hit_count = hit_count + 1 WHERE key = ?",
                (key,)
            )
            return (
                deserialize(prompt_embeds), deserialize(text_ids),
                deserialize(neg_embeds), deserialize(neg_ids)
            )
    except Exception as e:
        print(f"[generator] WARN disk cache read failed: {e}", flush=True)
        return None


def _disk_cache_prune() -> None:
    """Prune old/low-hit entries from disk cache."""
    if _PROMPT_CACHE_DB is None:
        return
    try:
        with _PROMPT_CACHE_DB_LOCK:
            # Remove old entries
            cutoff = time.time() - (_PROMPT_CACHE_DB_MAX_AGE_DAYS * 86400)
            _PROMPT_CACHE_DB.execute(
                "DELETE FROM prompt_embeddings WHERE created_at < ?",
                (cutoff,)
            )
            # Enforce max entries by hit count
            _PROMPT_CACHE_DB.execute("""
                DELETE FROM prompt_embeddings
                WHERE key NOT IN (
                    SELECT key FROM prompt_embeddings
                    ORDER BY hit_count DESC, created_at DESC
                    LIMIT ?
                )
            """, (_PROMPT_CACHE_DB_MAX_ENTRIES,))
    except Exception as e:
        print(f"[generator] WARN disk cache prune failed: {e}", flush=True)


# Initialize on import
_init_prompt_cache_db()


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

_moss_process = None
_moss_reader_messages = None
_moss_reader_done = None
_moss_reader_thread = None
_moss_stderr_thread = None
_moss_stderr_done = None
_moss_stderr_tail = deque(maxlen=200)
_moss_watchdog = None
_moss_idle_since = None
MOSS_IDLE_KILL_S = 300


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


def _kill_moss_process():
    global _moss_process, _moss_reader_messages, _moss_reader_done, _moss_reader_thread, _moss_stderr_thread, _moss_stderr_done
    _cancel_moss_watchdog()
    proc = _moss_process
    if proc is not None:
        _terminate_process(proc)
        _remove_moss_pid(proc.pid)
    _join_reader(_moss_stderr_thread)
    _join_reader(_moss_reader_thread)
    if _moss_process is proc:
        _moss_process = None
    _moss_reader_messages = None
    _moss_reader_done = None
    _moss_reader_thread = None
    _moss_stderr_thread = None
    _moss_stderr_done = None
    if proc is not None:
        gc.collect()
        try:
            import torch
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def _get_taef_decoder(variant: str):
    """Lazy loader and in-memory cache for fast TAEF mini-VAEs."""
    global _taef_models
    if variant not in _taef_models:
        if variant == "taef2":
            from mlx_taef import TAEF2
            m = TAEF2.from_pretrained(include_encoder=False)
        elif variant == "taef1":
            from mlx_taef import TAEF1
            m = TAEF1.from_pretrained(include_encoder=False)
        elif variant == "krea2":
            from mlx_taef import Krea2
            m = Krea2.from_pretrained(include_encoder=False)
        else:
            raise ValueError(f"unknown TAEF variant: {variant}")
        try:
            import mlx.core as mx
            if hasattr(m, "decoder"):
                m.decoder = mx.compile(m.decoder)
        except Exception:
            pass
        _taef_models[variant] = m
    return _taef_models[variant]


class GenerationCancelled(Exception):
    pass


def cancel_current():
    _cancel_event.set()
    for proc in (_qwen_process, _moss_process, globals().get("_sdxl_daemon")):
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
            if candidate.is_file()
        )
    )
    if ok and value:
        return True
    try:
        # For HF cache snapshots (inside snapshots/<hash>), also inspect sibling blobs/
        if path.parent.name == "snapshots":
            blobs_dir = path.parent.parent / "blobs"
            if blobs_dir.is_dir():
                return any(
                    candidate.name.endswith((".incomplete", ".part"))
                    for candidate in blobs_dir.iterdir()
                    if candidate.is_file()
                )
    except OSError:
        pass
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


def _directory_declares_model(directory: Path) -> set[str]:
    """Identity strings an installed model directory carries about itself.

    Sources, in order: the marker file the download worker writes
    (``DIFFUSIONBEAR_MODEL_ID``, content = registry id), the export manifests
    zimageapp-style packages ship at their root (``manifest.json``,
    ``package-manifest.json``, ``source-manifest.json``) and the root
    ``config.json`` ``model`` field. Only a small key vocabulary is scanned,
    and names are compared case-sensitively; a random directory yields an
    empty set.
    """
    import json as _json

    tokens: set[str] = set()
    marker = directory / "DIFFUSIONBEAR_MODEL_ID"
    try:
        if marker.is_file():
            value = marker.read_text("utf-8", errors="replace").strip()
            if value:
                tokens.add(value)
    except OSError:
        pass
    keys = ("model", "repo", "repo_id", "source", "source_repo", "model_name")
    for manifest_name in ("manifest.json", "package-manifest.json", "source-manifest.json", "config.json"):
        manifest = directory / manifest_name
        try:
            if not manifest.is_file():
                continue
            data = _json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A truncated or non-JSON manifest is a half-written download.
            continue
        if not isinstance(data, dict):
            continue
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value:
                tokens.add(value)
    return tokens


def _auto_bind_unbound_model(model_id: str) -> Path | None:
    """Detect an unbound install of ``model_id`` in the model store and bind it.

    Fires only when the settings binding AND the default ``models/<id>``
    layout both miss a validated install. The trigger was real: FLUX.3 was
    installed green on a test machine, the first generate shipped the weights
    from the repo again (~6 GB), and the directory sat listed as a standalone
    "local model" needing a manual attach. zimageapp packages carry no root
    ``model_index.json``, so nothing could name the directory for the
    detected-models heuristics. This scan identifies the directory by what it
    declares about itself, writes the settings binding (so the model manager
    stops listing it as standalone) and returns the path for immediate use --
    current installs self-heal on the next generation, no attach required.
    """
    import app_settings as _app_settings

    minfo = get_model_info(model_id)
    if not minfo or minfo.get("engine") == "sdxl":
        return None
    root = ASSET_DIR / "models"
    try:
        if not root.is_dir():
            return None
        candidates = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return None
    # The default layout is already checked by the caller; binding again is
    # pointless and would mask a corpse-directory with a validated one only
    # if one exists elsewhere.
    bound = set()
    for entry in (app_settings.get_settings().get("model_paths") or {}).values():
        try:
            bound.add(Path(entry).expanduser().resolve())
        except OSError:
            continue
    repo = model_download_repo(model_id, minfo) or ""
    identity = {str(model_id)}
    if "/" in repo:
        identity.add(repo)
    # Export manifests name the architecture, not the registry id
    # (zimageapp uses "flux-3-image-klein"), so a registry entry that knows
    # its declared names must carry them explicitly.
    identity.update(minfo.get("identity") or [])
    for directory in candidates:
        try:
            if directory.name == model_id or directory.resolve() in bound:
                continue
            if not _directory_declares_model(directory) & identity:
                continue
        except OSError:
            continue
        ok, _why = validate_local_model_dir(model_id, directory)
        if not ok:
            continue
        try:
            _app_settings.update_settings({"model_paths": {model_id: str(directory.resolve())}})
        except Exception as exc:
            print(f"[generator] WARN auto-bind of {model_id} to {directory} failed: {exc}", flush=True)
            continue
        print(
            f"[generator] auto-bound {model_id} -> {directory} "
            f"(installed without a model_paths entry)",
            flush=True,
        )
        return directory
    return None


def resolve_local_model_path(model_id: str, path: Path | str | None = None) -> Path | None:
    """Normalized local weights directory registered for a model, or None.

    HF-cache repo roots are resolved to their snapshot directory so the returned
    path can be handed straight to mflux's ``model_path`` / the SDXL ``model_dir``.
    """
    if path is None:
        explicit_path = False
        raw = app_settings.model_local_path(model_id)
        if not raw:
            minfo = get_model_info(model_id)
            if minfo and minfo.get("local_path"):
                raw = minfo.get("local_path")
        if not raw:
            # Fall back to default model directory: ASSET_DIR / "models" / <model_id>
            path = ASSET_DIR / "models" / model_id
        else:
            path = Path(raw).expanduser()
    else:
        explicit_path = True
        p = Path(path).expanduser()
        if not p.exists():
            return None
        return _resolve_hf_cache_snapshot(p)
    p = Path(path).expanduser()
    if p.exists():
        # A settings override (or a half-installed default dir) can point at a
        # corpse: the directory exists but holds no weights, e.g. an
        # interrupted download whose snapshot kept only config.json. Trusting
        # it hands the engine an unloadable path, so a weightless directory
        # falls through to the HF-cache resolution below, which finds the
        # complete copy. validate_local_model_dir is the existing gate, reused
        # rather than reimplemented. (Real case 2026-10-14: a stale override
        # aimed flux2-klein-4b at a dead black-forest-labs snapshot while the
        # complete mlx-community one sat unused in the same cache.)
        ok, _why = validate_local_model_dir(model_id, p)
    else:
        ok = False
    if not ok:
        if not explicit_path:
            # A validated install may exist under a name the registry knows only
            # by its identity (manifest exports are named after the architecture,
            # not the registry id). Bind before reaching for the network --
            # finding it here is the difference between "generate" and another
            # ~6 GB download (real case: FLUX.3 on a test machine, 2026-10-19).
            bound_path = _auto_bind_unbound_model(model_id)
            if bound_path is not None:
                return _resolve_hf_cache_snapshot(bound_path)
        # For mflux models, try HF cache if default path doesn't exist
        minfo = get_model_info(model_id)
        is_mflux = (
            minfo
            and (
                # Every registry entry now carries an explicit engine, so this is the
                # real discriminator. The id tuple stays only as a safety net for a
                # model added without one -- it is not load-bearing and should not be.
                minfo.get("engine") in ("mflux", "qwen")
                or model_id in ("flux2-klein-4b", "flux2-klein-9b", "z-image-turbo", "krea2-turbo", "qwen-image-2.1", "qwen-image-2.1-turbo")
            )
        )
        if minfo and is_mflux:
            repo = model_download_repo(model_id, minfo) or ""
            if "/" in repo:
                # Deliberately the SAME completeness test _is_model_cached_uncached uses
                # (:1296). Accepting a snapshot on "has any .safetensors" alone means an
                # interrupted download that finished one shard reads as installed, and
                # the pipeline then loads half a model instead of triggering a re-fetch.
                snapshot = _newest_complete_hf_snapshot(repo)
                if snapshot is not None:
                    return snapshot
        return None
    return _resolve_hf_cache_snapshot(p)


def _newest_complete_hf_snapshot(repo_id: str) -> Path | None:
    """Newest HF-cache snapshot for ``repo_id`` that has weights and no partial files."""
    if "/" not in str(repo_id or ""):
        return None
    snapshots = _hf_repo_cache_dir(repo_id) / "snapshots"
    if not snapshots.is_dir():
        return None
    try:
        candidates = sorted(
            (p for p in snapshots.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    for candidate in candidates:
        if _model_path_has_incomplete(candidate):
            continue
        if _model_path_has_weights(candidate):
            return candidate
    return None


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
    # A component can still be missing while others downloaded: the failure
    # mode is an interrupted pull that finished text_encoder/ but never wrote
    # transformer/. The engine loads the transformer first and dies scanning
    # an empty directory, so a present-but-weightless transformer/ is invalid
    # even when shards exist elsewhere. (Layouts without transformer/ at all
    # -- krea2-turbo-q4 keeps shards at the root -- skip this rule and keep
    # the any-weights verdict above.)
    transformer_dir = resolved / "transformer"
    if transformer_dir.is_dir() and not any(transformer_dir.glob("*.safetensors")):
        return False, "transformer/ has no .safetensors weights in directory"
    text_encoder_dir = resolved / "text_encoder"
    if text_encoder_dir.is_dir() and not any(text_encoder_dir.glob("*.safetensors")) and not any(text_encoder_dir.glob("*.safetensors.index.json")):
        return False, "text_encoder/ has no .safetensors weights in directory"
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
        model_dir = _resolve_sdxl_model_dir(model_id, minfo)
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
        # Custom entries own their directory via local_path, not via the
        # settings override resolve_local_model_path reads: without this a
        # verified local-source custom reports zero bytes and no home.
        entry_path = (minfo or {}).get("local_path")
        if entry_path:
            try:
                candidate = Path(entry_path).expanduser()
            except (OSError, RuntimeError):
                candidate = None
            if candidate is not None:
                try:
                    if candidate.is_dir():
                        return _dir_size(candidate)
                except (OSError, ValueError):
                    pass
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


def uninstall_model(model_id: str, force: bool = False) -> tuple[bool, str | None]:
    with _model_maintenance_lock:
        return _uninstall_model(model_id, force=force)


def _uninstall_model(model_id: str, force: bool = False) -> tuple[bool, str | None]:
    """Remove a model's weights from disk. Refuses (False, reason) while the model
    is queued/generating or a weights download is in-flight for it."""
    minfo = get_model_info(model_id)
    if not minfo:
        return False, "unknown model"
    if model_id not in MODELS:
        # Custom registry ids resolve through get_model_info above, but they
        # own no removable weights here: a translated sdxl entry reaching the
        # model_dir branch below would rmtree a built-in's weights. (The
        # DELETE route already 404s these ids; this is the second lock.)
        return False, "custom models cannot be uninstalled"
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
                    if not force:
                        return False, "model is in the shared asset store; remove it from the source application (use force=true to override)"
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
                    if not force:
                        return False, "model is in the shared asset store; remove it from the source application (use force=true to override)"
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
        "prompt_cache_size": _prompt_cache_len(),
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


class _MemoryLifecycleCallback:
    """Invariant 2: Inter-stage evaluation and Metal graph clearing.
    Clears scratch buffers after prompt encoding (before_loop) and forces
    latent evaluation, garbage collection, and cache clearing before VAE decode (after_loop).
    """
    def call_before_loop(self, **kwargs):
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

    def call_after_loop(self, latents=None, **kwargs):
        try:
            import mlx.core as mx
            import gc
            if latents is not None:
                mx.eval(latents)
            gc.collect()
            mx.clear_cache()
        except Exception:
            pass


def _normalize_lora_path(path: str) -> str:
    raw = str(path)
    candidate = Path(raw).expanduser()
    if candidate.exists() or raw.startswith(("/", "~", "./", "../")):
        return str(candidate.resolve())
    for folder in (
        ASSET_DIR / "SDXL",
        ASSET_DIR / "lora_files",
        DATA_DIR / "SDXL",
        DATA_DIR / "lora_files",
    ):
        p = folder / raw
        if p.is_file():
            return str(p.resolve())
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
    """Cache text-encoder output keyed by prompt with true LRU eviction + disk persistence.
    Encoding runs a Qwen3 forward pass which is a significant fraction of a short generation."""
    import mlx.core as mx
    import hashlib

    orig_encode = pipe._encode_prompt_pair

    def cached_encode(*, prompt, negative_prompt, guidance):
        # Disk cache key includes model + prompt + negative + guidance
        model_id = getattr(pipe, "model_config", None)
        model_id = getattr(model_id, "model_name", "unknown") if model_id else "unknown"
        key = f"{model_id}:{prompt}"
        neg_key = f"{model_id}:\x00neg:{negative_prompt}" if negative_prompt else None

        # Try in-memory first
        mem_prompt = _get_prompt_cache(prompt)
        if mem_prompt is not None:
            prompt_embeds, text_ids = mem_prompt
        else:
            # Try disk cache
            disk_key = hashlib.sha256(key.encode()).hexdigest()
            disk_result = _disk_get_prompt(disk_key)
            if disk_result is not None:
                prompt_embeds, text_ids, neg_embeds, neg_ids = disk_result
                _cache_prompt(prompt, (prompt_embeds, text_ids))
                if neg_key and neg_embeds is not None:
                    _cache_prompt(neg_key, (neg_embeds, neg_ids))
            else:
                prompt_embeds, text_ids = orig_encode(
                    prompt=prompt,
                    negative_prompt=None,
                    guidance=1.0,
                )[0:2]
                mx.eval(prompt_embeds, text_ids)
                _cache_prompt(prompt, (prompt_embeds, text_ids))
                _disk_cache_prompt(disk_key, model_id, key, prompt_embeds, text_ids)

        negative_embeds, negative_ids = None, None
        if guidance is not None and guidance > 1.0 and negative_prompt is not None:
            mem_neg = _get_prompt_cache(neg_key)
            if mem_neg is not None:
                negative_embeds, negative_ids = mem_neg
            else:
                disk_neg_key = hashlib.sha256(neg_key.encode()).hexdigest()
                disk_result = _disk_get_prompt(disk_neg_key)
                if disk_result is not None:
                    neg_embeds, neg_ids, _, _ = disk_result
                    negative_embeds, negative_ids = neg_embeds, neg_ids
                    _cache_prompt(neg_key, (negative_embeds, negative_ids))
                else:
                    negative_embeds, negative_ids = orig_encode(
                        prompt=negative_prompt,
                        negative_prompt=None,
                        guidance=1.0,
                    )[0:2]
                    mx.eval(negative_embeds, negative_ids)
                    _cache_prompt(neg_key, (negative_embeds, negative_ids))
                    _disk_cache_prompt(disk_neg_key, model_id, neg_key, negative_embeds, negative_ids)
        return prompt_embeds, text_ids, negative_embeds, negative_ids

    pipe._encode_prompt_pair = cached_encode


def _apply_prompt_cache_txt(pipe):
    """Shared LRU for Z-Image / Krea2 text encoders with disk persistence.

    Both mflux engines expose ``_encode_prompts(prompt, negative_prompt,
    guidance) -> (embeds, neg_embeds)`` and re-encode on every generate_image
    call even though the pipe (with its TextEncoder) is cached in memory.
    prompt/negative/guidance fully determine the embeddings, so memoizing the
    raw arrays (never mutated) is bit-identical. Uses _PROMPT_CACHE_MAX_SIZE;
    tuple keys cannot collide with the string keys used by the FLUX wrapper.
    """
    import hashlib
    import mlx.core as mx

    orig_encode = pipe._encode_prompts

    def cached_encode(*, prompt, negative_prompt=None, guidance=1.0):
        model_id = getattr(pipe, "model_config", None)
        model_id = getattr(model_id, "model_name", "unknown") if model_id else "unknown"
        key = (prompt, negative_prompt or "", float(guidance))
        cache_key = f"{model_id}:{prompt}|{negative_prompt or ''}|{guidance}"
        disk_key = hashlib.sha256(cache_key.encode()).hexdigest()

        # Try in-memory first
        mem = _get_prompt_cache(key)
        if mem is not None:
            embeds, neg = mem
            return embeds, neg

        # Try disk cache
        disk_result = _disk_get_prompt(disk_key)
        if disk_result is not None:
            embeds, neg, _, _ = disk_result
            _cache_prompt(key, (embeds, neg))
            return embeds, neg

        # Compute and cache
        embeds, neg = orig_encode(prompt=prompt, negative_prompt=negative_prompt, guidance=guidance)
        mx.eval(embeds)
        if neg is not None:
            mx.eval(neg)
        _cache_prompt(key, (embeds, neg))
        _disk_cache_prompt(disk_key, model_id, cache_key, embeds, None, neg, None)
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
        from mflux.models.krea2.model.krea2_sampler import ErSdeStepper, EulerStepper

        if not getattr(EulerStepper, "_mlxdiffusion_compiled_step", False):
            @mx.compile
            def _compiled_krea_euler_step(x: mx.array, v: mx.array, dt: mx.array) -> mx.array:
                return x + dt * v

            def _fast_krea_euler_step(self, i: int, x: mx.array, v: mx.array, denoised: mx.array) -> mx.array:
                dts = getattr(self, "_dts", None)
                if dts is None:
                    self._dts = self.sigmas[1:] - self.sigmas[:-1]
                    dts = self._dts
                return _compiled_krea_euler_step(x, v, dts[i])

            EulerStepper.step = _fast_krea_euler_step
            EulerStepper._mlxdiffusion_compiled_step = True

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


def _install_flux3_compiled_scheduler_step():
    """JIT-compile the FLUX.3 Euler flow map step and eliminate the .item() sync stall.

    The stock Flux3FlowMapScheduler.step() calls `.item()` on every step, forcing
    an unnecessary GPU-CPU synchronization barrier. This replaces it with an
    in-graph compiled step operating on precomputed dt tensors (6x faster step update,
    0.0 max diff).
    """
    try:
        import mlx.core as mx
        from mflux.models.flux3.model.flux3_scheduler.flux3_flow_map_scheduler import Flux3FlowMapScheduler

        if getattr(Flux3FlowMapScheduler, "_mlxdiffusion_compiled_step", False):
            return

        @mx.compile
        def _compiled_flux3_step(latents: mx.array, velocity: mx.array, dt: mx.array) -> mx.array:
            return latents + dt * velocity

        def _fast_step(self, latents: mx.array, velocity: mx.array, step: int) -> mx.array:
            dts = getattr(self, "_dts", None)
            if dts is None:
                self._dts = self.timesteps[1:] - self.timesteps[:-1]
                dts = self._dts
            return _compiled_flux3_step(latents, velocity, dts[step])

        Flux3FlowMapScheduler.step = _fast_step
        Flux3FlowMapScheduler._mlxdiffusion_compiled_step = True
    except Exception:
        pass


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
        "flux3-klein": ("mflux.models.flux3.model.flux3_transformer.flux3_pos_embed", "Flux3PosEmbed"),
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
    _install_rope_cache_on_class(cls)
    _ROPE_CACHE_INSTALLED.add(spec)
    print(f"[generator] installed rope memo cache for {model_id} ({spec[1]}, lossless; ids constant across steps)", flush=True)


def _install_rope_cache_on_class(cls):
    """Patch one rope class in place. Split out of `_install_rope_cache` so the
    trace-safety behaviour can be tested against a throwaway class."""
    orig = cls.__call__
    cache = OrderedDict()

    def _call(self, ids):
        if ids.size == 0 or (hasattr(ids, "shape") and len(ids.shape) > 0 and ids.shape[0] == 0):
            return orig(self, ids)
        # Building the key calls `.tolist()`, which forces an evaluation. Inside an
        # mx.compile trace MLX refuses that outright:
        #   ValueError: [eval] Attempting to eval an array during function
        #   transformations like compile or vmap is not allowed.
        # That is what broke FLUX 4B *edit*: the reference-image path reaches this
        # rope while the compiled graph is being traced, and the whole generation died
        # at denoise step 3/4. It is a catchable Python exception, so fall through to
        # the uncached call instead of dying.
        #
        # Nothing is lost by skipping the cache during tracing: a compiled graph traces
        # once and then reuses its own fused result, so the memo would never be hit for
        # those steps anyway. The cache still does its job on the eager paths.
        try:
            key = (ids.shape, str(ids.dtype), tuple(ids.reshape(-1).tolist()))
        except ValueError:
            return orig(self, ids)
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


_ROPE_CACHE_INSTALLED = set()


def _resolve_sdxl_model_dir(model: str, minfo: dict | None) -> str:
    """Daemon identity for an SDXL id: settings override, then a custom
    entry's own directory, then the built-in model_dir."""
    local = resolve_local_model_path(model)
    if local is not None:
        return str(local)
    entry_path = (minfo or {}).get("local_path")
    if entry_path:
        return str(Path(entry_path).expanduser())
    return str((minfo or {}).get("model_dir") or (ASSET_DIR / "models" / "juggernaut-xl-lightning"))


def _krea_te_disk_path(model_path, bits: int = 4, group_size: int = 64) -> Path:
    """Where the quantised Krea 2 text encoder is persisted between processes."""
    import hashlib

    digest = hashlib.sha256(str(model_path).encode()).hexdigest()[:10]
    name = Path(str(model_path)).name or "krea2"
    return ASSET_DIR / "models" / f"{name}-te-q{bits}-g{group_size}-{digest}.safetensors"


def _install_krea_te_bf16_skip() -> None:
    """Stop mflux loading the 7.5GB bf16 encoder when we already have a q4 copy on disk."""
    global _krea_te_skip_installed
    if _krea_te_skip_installed:
        return
    try:
        from mflux.models.common.weights.loading.weight_applier import WeightApplier

        original = WeightApplier._set_weights

        def _patched(weights, models, components=None):
            if _krea_te_skip_bf16["on"] and any(
                type(m).__name__ == "Krea2TextEncoder" for m in models.values()
            ):
                models = {k: v for k, v in models.items() if k != "text_encoder"}
            return original(weights, models, components)

        WeightApplier._set_weights = staticmethod(_patched)
    except Exception as exc:
        print(f"[generator] WARN could not patch WeightApplier for Krea TE skip: {exc}", flush=True)

    try:
        from mflux.models.common.weights.loading.weight_loader import WeightLoader

        original_load = WeightLoader._load_component

        def _patched_load(root_path, component, raw_weights_cache=None, save_subdir=None):
            if _krea_te_skip_bf16["on"] and getattr(component, "name", None) == "text_encoder":
                return {}, 4, "0.18.0"
            return original_load(root_path, component, raw_weights_cache, save_subdir)

        WeightLoader._load_component = staticmethod(_patched_load)
    except Exception as exc:
        print(f"[generator] WARN could not patch WeightLoader for Krea TE skip: {exc}", flush=True)

    _krea_te_skip_installed = True


def _krea_te_from_disk(model_path, bits: int = 4, group_size: int = 64):
    """Rebuild the quantised encoder straight from disk, or None if there is no cache."""
    import mlx.core as mx
    import mlx.nn as nn
    from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder

    path = _krea_te_disk_path(model_path, bits, group_size)
    if not path.exists():
        return None
    try:
        module = Krea2TextEncoder()
        nn.quantize(module, bits=bits, group_size=group_size)
        module.load_weights(str(path))
        mx.eval(module)
        return module
    except Exception as exc:
        print(f"[generator] WARN Krea 2 q4 text-encoder cache unusable ({exc}); falling back", flush=True)
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def _krea_te_save_disk(module, model_path, bits: int = 4, group_size: int = 64) -> Path | None:
    """Persist the quantised encoder so later processes never load bf16 at all."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    path = _krea_te_disk_path(model_path, bits, group_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.safetensors")
    try:
        mx.save_safetensors(str(tmp), dict(tree_flatten(module.parameters())))
        os.replace(tmp, path)
        return path
    except Exception as exc:
        print(f"[generator] WARN could not persist Krea 2 q4 text encoder ({exc})", flush=True)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def _krea_quantized_text_encoder(pipe, model_path, bits: int = 4, group_size: int = 64):
    """Return a 4-bit Krea 2 text encoder, reusing the in-memory cache or disk if available."""
    import mlx.core as mx
    import mlx.nn as nn

    key = (str(model_path), bits, group_size)
    disabled = os.environ.get("MLX_DISABLE_KREA_TE_CACHE") == "1"
    with _krea_te_lock:
        cached = None if disabled else (_krea_te_cache.get("module") if _krea_te_cache.get("key") == key else None)
        fresh = getattr(pipe, "text_encoder", None)
        if cached is not None:
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


def _clear_krea_te_cache():
    """Release the cached quantized Krea 2 text encoder (~2.1GB q4)."""
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


def ensure_krea_text_encoder_quantized(model_dir: Path | str, task_progress_cb=None) -> Path | None:
    """Ensure Krea 2 text encoder is quantized on disk (2.1GB q4) and prune 7.5GB bf16 shards."""
    import gc
    model_dir = Path(model_dir)
    q4_path = _krea_te_disk_path(model_dir, 4, 64)
    te_dir = model_dir / "text_encoder"

    if not q4_path.exists() and te_dir.is_dir():
        shards = [f for f in te_dir.glob("*.safetensors") if not f.name.startswith("._")]
        if shards:
            print(f"[generator] Quantizing Krea 2 text encoder at {te_dir} to 4-bit...", flush=True)
            if task_progress_cb:
                task_progress_cb("Quantizing text encoder to 4-bit...")
            try:
                import mlx.core as mx
                import mlx.nn as nn
                from mflux.models.common.weights.loading.weight_loader import WeightLoader
                from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
                from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

                comp = next(c for c in Krea2WeightDefinition.get_components() if c.name == "text_encoder")
                raw_weights, _, _ = WeightLoader._load_component(model_dir, comp)

                te_mod = Krea2TextEncoder()
                te_mod.update(raw_weights)
                del raw_weights
                nn.quantize(te_mod, bits=4, group_size=64)
                mx.eval(te_mod)
                saved_path = _krea_te_save_disk(te_mod, model_dir, 4, 64)
                del te_mod
                gc.collect()
                mx.clear_cache()
                print(f"[generator] Successfully quantized Krea 2 TE to {saved_path}", flush=True)
            except Exception as exc:
                print(f"[generator] Failed to quantize Krea 2 TE during setup: {exc}", flush=True)

    # Prune unneeded BF16 shards if q4_path exists and is valid
    if q4_path.exists() and q4_path.stat().st_size > 1 << 30 and te_dir.is_dir():
        shards = [f for f in te_dir.glob("*.safetensors") if not f.name.startswith("._")]
        total_bytes = sum(f.stat().st_size for f in shards)
        if total_bytes > 3 << 30:  # > 3GB unquantized shards
            for f in shards:
                try:
                    f.unlink()
                except OSError:
                    pass
            print(f"[generator] Pruned {total_bytes // (1 << 20)} MB of unneeded BF16 text encoder shards", flush=True)

    return q4_path if q4_path.exists() else None


def _build_custom_pipeline(model_id: str, info: dict, quantization: int, loras: list[dict],
                           scales: list, variant: str, local_arg: str | None, phase_cb=None):
    """In-process pipeline for a custom registry id. Custom ids only; the
    built-in model_id == chain in _get_pipeline is untouched.

    Delegates to the existing per-family builders with entry-derived args:
    Flux2Klein with the ModelConfig matching the verified variant, the SDXL
    daemon is keyed by the entry's directory in _generate_sdxl (never built
    here), and Qwen ids never reach here (separate subprocess). A vanished
    directory fails with the existing friendly validation error, never a
    traceback. Custom entries run family defaults: no Lightning-style
    auto-injection, no tuning flags.
    """
    label = info.get("label", model_id)
    family = info.get("family")
    entry_path = info.get("local_path") or local_arg
    if entry_path is None:
        raise ValueError(f"{label}: no local weights directory is registered for this model")
    entry_dir = Path(entry_path).expanduser()
    if family in ("sdxl", "qwen"):
        # Never built in-process: SDXL runs in the daemon keyed by the
        # entry's directory (_resolve_sdxl_model_dir), Qwen in its
        # subprocess. Reaching here means the dispatch regressed — except
        # a vanished directory, which still deserves the friendly layout
        # reason instead of a bare unknown-model refusal.
        if family == "sdxl":
            ok, why = sdxl_layout_report(entry_dir)
            if not ok:
                raise ValueError(f"SDXL cannot load this model: {why}")
        else:
            ok, why = validate_local_model_dir(model_id, entry_dir)
            if not ok:
                raise ValueError(f"{label}: {why}")
        raise ValueError(f"unknown model: {model_id}")
    if family == "flux2":
        ok, why = validate_local_model_dir(model_id, entry_dir)
        if not ok:
            raise ValueError(f"{label}: {why}")
        model_path = str(entry_dir)
        lora_paths = [l["path"] for l in loras] or None
        lora_scales = scales or None
        if variant == "edit":
            from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

            if info.get("variant") == "9b":
                from mflux.models.common.config.model_config import ModelConfig

                return Flux2KleinEdit(
                    model_config=ModelConfig.flux2_klein_9b(),
                    model_path=model_path,
                    quantize=quantization,
                    lora_paths=lora_paths,
                    lora_scales=lora_scales,
                    bake_lora=False,
                )
            return Flux2KleinEdit(
                model_path=model_path,
                quantize=quantization,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
                bake_lora=False,
            )
        from mflux.models.flux2 import Flux2Klein

        if info.get("variant") == "9b":
            from mflux.models.common.config.model_config import ModelConfig

            return Flux2Klein(
                model_config=ModelConfig.flux2_klein_9b(),
                model_path=model_path,
                quantize=quantization,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
                bake_lora=False,
            )
        return Flux2Klein(
            model_path=model_path,
            quantize=quantization,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=False,
        )
    if family == "zit":
        ok, why = validate_local_model_dir(model_id, entry_dir)
        if not ok:
            raise ValueError(f"{label}: {why}")
        from mflux.models.z_image import ZImageTurbo

        pipe = ZImageTurbo(
            quantize=quantization,
            model_path=str(entry_dir),
            lora_paths=[l["path"] for l in loras] or None,
            lora_scales=scales or None,
            bake_lora=False,  # runtime fp16 adapters — baking erodes deltas in q4 weights
        )
        _install_zit_dynamic_padding(pipe)
        return pipe
    if family == "krea":
        ok, why = validate_local_model_dir(model_id, entry_dir)
        if not ok:
            raise ValueError(f"{label}: {why}")
        from mflux.models.krea2 import Krea2
        import mlx.core as mx

        _install_krea_sampler_scalar_cache()
        krea_source = str(entry_dir)
        disk_te = None
        if os.environ.get("MLX_DISABLE_KREA_TE_CACHE") != "1":
            _install_krea_te_bf16_skip()
            disk_te = _krea_te_from_disk(krea_source, bits=4, group_size=64)
            if disk_te is None:
                ensure_krea_text_encoder_quantized(krea_source)
                disk_te = _krea_te_from_disk(krea_source, bits=4, group_size=64)
            _krea_te_skip_bf16["on"] = disk_te is not None
        try:
            pipe = Krea2(
                quantize=4,
                model_path=krea_source,
                lora_paths=[l["path"] for l in loras] or None,
                lora_scales=scales or None,
                bake_lora=False,
            )
        finally:
            _krea_te_skip_bf16["on"] = False

        if disk_te is not None:
            stale = getattr(pipe, "text_encoder", None)
            if stale is not None:
                pipe.text_encoder = None
                del stale
                gc.collect()
                mx.clear_cache()
            pipe.text_encoder = disk_te
        elif hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            _krea_quantized_text_encoder(pipe, krea_source, bits=4, group_size=64)

        _apply_krea_vae_tiling(pipe)
        return pipe
    raise ValueError(f"unknown model: {model_id}")


_MACHINE_MEM_GB: int | None = None


def _unified_memory_gb() -> int:
    global _MACHINE_MEM_GB
    if _MACHINE_MEM_GB is None:
        try:
            page = os.sysconf("SC_PAGE_SIZE")
            phys = os.sysconf("SC_PHYS_PAGES")
            _MACHINE_MEM_GB = max(1, int(page * phys / (1 << 30)))
        except (ValueError, OSError):
            _MACHINE_MEM_GB = 16
    return _MACHINE_MEM_GB


# On 32 GB and up a moss pipeline (~10 GB) can sit beside an SDXL q4 daemon
# (~6.5 GB) inside the auto wired budget without thrashing, so keeping both
# warm beats paying reload + Metal kernel recompile on every switch. Under
# 32 GB the shared budget is smaller than the pair, measured this afternoon:
# 4-step SDXL 26.9s -> 124-146s with 14.6/15.4 GB of swap committed.
KEEP_ALIVE_MIN_UNIFIED_GB = 32


def _evict_other_engines(keep: str) -> None:
    """Free every engine the requested one does not own, immediately.

    The 5-minute idle watchdogs are per engine, so switching engines left the
    outgoing engine's weights resident through the whole incoming run. On the
    16 GB reference machine that ended in committed swap and 4x-longer
    denoise times (measured 2026-10-21). Eviction costs a cold reload plus a
    Metal kernel recompile on the next switch, so on machines that can hold
    the pair in wired memory we keep the idle watchdogs in charge instead.
    """
    if _unified_memory_gb() >= KEEP_ALIVE_MIN_UNIFIED_GB:
        return
    if keep != "mflux":
        _drop_mflux_pipeline()
        _clear_krea_te_cache()
    if keep != "sdxl":
        _kill_sdxl_daemon()
    if keep != "moss":
        _kill_moss_process()
    if keep != "qwen":
        _kill_qwen_process()


def _get_pipeline(model_id: str, quantization: int, loras: list[dict], variant: str = "standard", phase_cb=None):
    global _pipeline, _current_pipeline_key, _current_pipeline_model
    loras = sorted(_normalize_loras(loras), key=lambda item: item["path"])
    key = _make_pipeline_key(model_id, quantization, loras, variant)
    if _pipeline is not None and _current_pipeline_key == key:
        _evict_other_engines("mflux")
        _ensure_tokenizer_chat_template(_pipeline)
        return _pipeline
    info = get_model_info(model_id)
    if not info:
        raise ValueError(f"unknown model: {model_id}")
    if phase_cb is not None:
        if is_model_cached(model_id):
            phase_cb("loading_model", f"Loading {info['label']} weights into Apple Silicon unified memory...")
        else:
            phase_cb("downloading", f"Downloading {info['label']} weights from repository...")
    _evict_other_engines("mflux")
    _drop_mflux_pipeline()
    scales = [l.get("scale", 1.0) for l in loras]
    local_path = resolve_local_model_path(model_id)
    local_arg = str(local_path) if local_path is not None else None
    if model_id in ("flux2-klein-4b", "flux2-klein-9b", "flux3-klein"):
        _ensure_flux_tokenizer_available(local_arg)
    if model_id == "flux2-klein-4b":
        if variant == "edit":
            from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

            _pipeline = Flux2KleinEdit(
                # Same reason as the txt2img branch below: with model_path=None, mflux
                # falls back to its own default repo (black-forest-labs/FLUX.2-klein-4B,
                # 22.1 GB of weights this app never loads) instead of the 4.3 GB
                # quantized repo the app downloads.
                model_path=local_arg or info["repo"],
                quantize=quantization,
                lora_paths=[l["path"] for l in loras] or None,
                lora_scales=scales or None,
                bake_lora=False,
            )
        else:
            from mflux.models.flux2 import Flux2Klein

            _pipeline = Flux2Klein(
                model_path=local_arg or info["repo"],
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
    elif model_id == "flux3-klein":
        # FLUX.3 Image Klein (5.6B) flow-matching pipeline
        _install_flux3_compiled_scheduler_step()
        if variant == "edit":
            from mflux.models.flux3.variants.edit.flux3_klein_edit import Flux3KleinEdit as pipe_cls
        else:
            from mflux.models.flux3 import Flux3Klein as pipe_cls
        try:
            _pipeline = pipe_cls(
                model_path=local_arg or info["repo"],
                quantize=quantization,
            )
        except Exception as exc:
            if "target_timestep_embedder" in str(exc) or "flux3-image-klein-base" in str(exc):
                from mflux.models.common.config.model_config import ModelConfig
                _pipeline = pipe_cls(
                    model_path=local_arg or info["repo"],
                    quantize=quantization,
                    model_config=ModelConfig.flux3_image_klein_base(),
                )
            else:
                raise
        _apply_vae_tiling(_pipeline, default_size=512, model_label="flux3-klein")
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

        disk_te = None
        if os.environ.get("MLX_DISABLE_KREA_TE_CACHE") != "1":
            _install_krea_te_bf16_skip()
            disk_te = _krea_te_from_disk(krea_source, bits=4, group_size=64)
            if disk_te is None:
                ensure_krea_text_encoder_quantized(krea_source)
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
            stale = getattr(_pipeline, "text_encoder", None)
            if stale is not None:
                _pipeline.text_encoder = None
                del stale
                gc.collect()
                import mlx.core as _mx
                _mx.clear_cache()
            _pipeline.text_encoder = disk_te
        elif hasattr(_pipeline, "text_encoder") and _pipeline.text_encoder is not None:
            _krea_quantized_text_encoder(_pipeline, krea_source, bits=4, group_size=64)
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
    elif model_id == "flux1-schnell":
        from mflux.models.flux.variants.txt2img.flux import Flux1

        # model_config defaults to ModelConfig.schnell() in the class itself;
        # passing nothing keeps the single source of truth there, not here.
        _pipeline = Flux1(
            model_path=local_arg or info["repo"],
            quantize=quantization,
            lora_paths=[l["path"] for l in loras] or None,
            lora_scales=scales or None,
            bake_lora=False,
        )
    # NOTE: qwen-image-2.1 and qwen-image-2.1-turbo deliberately have NO branch here.
    # generate() dispatches both to _generate_qwen_subprocess, which runs qwen_engine.py
    # in a separate interpreter -- that is where QwenImage21 is constructed and where the
    # text_encoder layout patch lives. Reaching this function with a Qwen id would mean
    # the dispatch regressed; raise rather than silently loading in the wrong process.
    else:
        if model_id not in MODELS and info.get("family"):
            _pipeline = _build_custom_pipeline(
                model_id, info, quantization, loras, scales, variant, local_arg, phase_cb
            )
        else:
            raise ValueError(f"unknown model: {model_id}")
    _current_pipeline_key = key
    _current_pipeline_model = model_id
    _clear_prompt_cache()
    if model_id in ("flux2-klein-4b", "flux2-klein-9b"):
        _install_flux_text_encoder_tail_trim(_pipeline)
    if model_id in ("flux2-klein-4b", "flux2-klein-9b", "flux3-klein"):
        if os.environ.get("MLX_DISABLE_ROPE_CACHE") != "1":
            _install_rope_cache(model_id)
            if model_id in ("flux2-klein-4b", "flux2-klein-9b"):
                _apply_prompt_cache(_pipeline)
    elif model_id in ("z-image-turbo", "krea2-turbo"):
        _install_rope_cache(model_id)
        if os.environ.get("MLX_DISABLE_ROPE_CACHE") != "1":
            _apply_prompt_cache_txt(_pipeline)
    _apply_vae_tiling(_pipeline, default_size=512, model_label=model_id)
    _ensure_tokenizer_chat_template(_pipeline)
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


def _custom_sampler_ids(minfo):
    """custom:* ids the given model may run. krea2's engine whitelist has no
    external-scheduler support, so custom samplers stay off it."""
    engine = str(minfo.get("engine") or "").lower()
    if minfo.get("id") == "krea2-turbo" or (minfo.get("family") and not minfo.get("engine")):
        return []
    if engine not in ("mflux", "sdxl", "moss"):
        return []
    return custom_ids_for_engine(engine)


def _validate_sampler(sampler, minfo):
    allowed = list(minfo.get("samplers") or []) + _custom_sampler_ids(minfo)
    if not minfo.get("samplers"):
        model_id = minfo.get("id")
        sampler_sets = {
            "flux2-klein-4b": ["flow_match_euler_discrete", "euler"],
            "flux2-klein-9b": ["flow_match_euler_discrete", "euler"],
            "flux3-klein": ["flow_match_euler_discrete", "euler"],
            "moss-image-sft": ["flow_match_euler_discrete", "euler"],
            "moss-image-base": ["flow_match_euler_discrete", "euler"],
            "z-image-turbo": ["linear", "flow_match_euler_discrete", "euler"],
            "krea2-turbo": ["er_sde", "euler"],
        }
        fallback_list = list(sampler_sets.get(model_id, []))
        if not fallback_list and minfo.get("family"):
            fallback_list = list(sampler_sets.get(_custom_canonical_id(minfo) or "", []))
        allowed = fallback_list + _custom_sampler_ids(minfo)

    if not allowed:
        return "Euler"

    default_choice = minfo.get("default_sampler") or allowed[0]
    value = sampler or default_choice

    # Exact or case-insensitive match against allowed
    match = next((s for s in allowed if s.lower() == str(value).lower()), None)
    if match is not None:
        return match

    # If the user explicitly requested an external custom: sampler that is not installed for this engine
    if sampler_is_custom(value):
        raise ValueError(f"custom sampler {value!r} is not installed for {minfo.get('id')}")

    # A sampler picked on one model and carried over to another whose list differs is
    # a stale UI value, not a hostile one: fall back to the model's default instead.
    print(
        f"[generator] sampler {value!r} not offered by {minfo.get('id')}; "
        f"falling back to {default_choice!r}",
        flush=True,
    )
    return default_choice


def _effective_guidance(minfo, guidance, negative_prompt=""):
    if not minfo.get("supports_guidance") and minfo.get("id") != "flux3-klein":
        return 1.0
    value = minfo.get("default_guidance", 1.0) if guidance is None else guidance
    try:
        value = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError("guidance must be numeric") from e
    if not math.isfinite(value) or value < 0:
        raise ValueError("guidance must be a finite non-negative number")
    if (minfo.get("id") in ("qwen-image-2.1", "qwen-image-2.1-turbo") or minfo.get("family") == "qwen") and negative_prompt and value <= 1.0:
        return 3.0
    return value


def _effective_steps(minfo, steps, overrides=None, log_clamp=True):
    """Clamp the step count for guidance-distilled models.

    A distilled checkpoint (SDXL-Lightning, FLUX.2-klein, Z-Image Turbo) is
    trained for a handful of steps. The request schema and the settings UI both
    allow up to 50, and app_settings permits a persisted model_defaults.steps of
    50, so a stray override silently turned a 4-step run into a 12x slower one
    with no quality gain - the extra steps land on a distribution the model was
    never distilled for. Non-distilled models are left alone.

    ``log_clamp=False`` is for callers that only need the number (state.py's
    progress readout): the engine logs the clamp itself once per image, and a
    second identical line per job would just be noise.
    """
    value = int(minfo.get("default_steps", 1) if steps is None else steps)
    if not minfo.get("is_distilled"):
        return value
    cap = minfo.get("max_steps", _FLUX3_MAX_STEPS if minfo.get("id") == "flux3-klein" else _DISTILLED_MAX_STEPS)
    override = (overrides or {}).get("steps")
    if override is not None:
        try:
            cap = min(cap, max(1, int(override)))
        except (TypeError, ValueError):
            pass
    if value > cap and log_clamp:
        print(
            f"[generator] {minfo.get('label')}: {value} steps clamped to {cap} "
            "(guidance-distilled model)",
            flush=True,
        )
    return min(value, cap)


def detect_architecture(directory) -> str | None:
    """Which engine can execute this checkpoint, or None if we do not know.

    Read from what the checkpoint declares, never from its folder name. The old path
    substring-matched the directory -- "flux", "krea", "z-image" -- which is a guess
    with no failure mode: a folder called `flux-typo` was claimed as FLUX, and an
    unrecognised checkpoint silently borrowed an engine it did not belong to. That is
    how RealVisXL could show as installed next to "not runnable here".

    A diffusers repo states its architecture in `model_index.json` under
    `_class_name`, so that is the authority. Only architectures we actually have a
    pipeline for are returned; everything else is None, which means "unknown", and
    the caller must not guess on our behalf.
    """
    import json as _json

    try:
        root = Path(directory)
        index = root / "model_index.json"
        if not index.is_file():
            return None
        declared = _json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A truncated or non-JSON index is a half-written download, not a crash.
        return None
    if not isinstance(declared, dict):
        return None

    class_name = str(declared.get("_class_name") or "")
    architectures = declared.get("_class_name")
    names = [class_name]
    if isinstance(architectures, str):
        names = [architectures]

    for name in names:
        if not name:
            continue
        # SDXL, and every finetune of it -- PONY, RealVis, Juggernaut, anything a
        # user drops in -- declares the XLPipeline classes.
        if "StableDiffusionXL" in name or "SDXL" in name:
            return "sdxl"
        # FLUX.1 / FLUX.2 diffusers pipelines. mflux runs these natively.
        if name.startswith("Flux") or "Flux" in name:
            return "mflux"
    return None


def _effective_quantization(model_id, quantization):
    try:
        value = int(quantization)
    except (TypeError, ValueError) as e:
        raise ValueError("quantization must be an integer") from e
    if model_id in ("krea2-turbo", "qwen-image-2.1", "qwen-image-2.1-turbo"):
        if value != 4:
            raise ValueError(f"{model_id} supports only 4-bit quantization")
        return 4
    if value not in (2, 3, 4, 8):
        raise ValueError("quantization must be 2, 3, 4, or 8")
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


def _ensure_qwen_model_complete(model_path: str | Path | None) -> None:
    """Ensure a local Qwen model directory has all required components (processor, scheduler, text_encoder).

    If any auxiliary components or text_encoder weights are missing, links them from any other
    complete local Qwen installation with zero duplication.
    """
    if not model_path:
        return
    try:
        p = Path(model_path).resolve()
        if not p.is_dir():
            return

        te_dir = p / "text_encoder"
        has_te_weights = te_dir.is_dir() and (
            any(te_dir.glob("*.safetensors")) or any(te_dir.glob("*.safetensors.index.json"))
        )
        has_processor = (p / "processor").is_dir()
        has_scheduler = (p / "scheduler").is_dir()
        has_index = (p / "model_index.json").is_file()

        if has_te_weights and has_processor and has_scheduler and has_index:
            return

        candidates = []
        models_dir = ASSET_DIR / "models"
        if models_dir.is_dir():
            candidates.extend(d for d in models_dir.iterdir() if d.is_dir() and "qwen" in d.name.lower() and d.resolve() != p)
        hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_hub.is_dir():
            for repo_dir in hf_hub.iterdir():
                if "qwen" in repo_dir.name.lower() and repo_dir.is_dir():
                    snaps = repo_dir / "snapshots"
                    if snaps.is_dir():
                        candidates.extend(s for s in snaps.iterdir() if s.is_dir() and s.resolve() != p)

        source = None
        for cand in candidates:
            c_te = cand / "text_encoder"
            if (
                (cand / "processor").is_dir()
                and c_te.is_dir()
                and any(c_te.glob("*.safetensors"))
            ):
                source = cand
                break

        if not source:
            return

        for name in ("processor", "scheduler", "model_index.json"):
            target = p / name
            src_item = source / name
            if not target.exists() and not target.is_symlink() and src_item.exists():
                try:
                    target.symlink_to(src_item)
                except OSError:
                    pass

        if not has_te_weights and (source / "text_encoder").is_dir():
            te_dir.mkdir(parents=True, exist_ok=True)
            for fname in ("model.safetensors", "model.safetensors.index.json"):
                target = te_dir / fname
                src_file = source / "text_encoder" / fname
                if not target.exists() and not target.is_symlink() and src_file.exists():
                    try:
                        target.symlink_to(src_file)
                    except OSError:
                        pass
    except (OSError, ValueError):
        pass
    except Exception as e:
        print(f"[generator] WARN _ensure_qwen_model_complete: {e}", flush=True)


def _qwen_local_model(minfo):
    entry_path = (minfo or {}).get("local_path")
    if entry_path:
        p = str(Path(entry_path).expanduser())
        _ensure_qwen_model_complete(p)
        return p
    model_id = minfo.get("id")
    if model_id == "qwen-image-2.1-turbo":
        local = resolve_local_model_path("qwen-image-2.1-turbo")
        p = str(local.resolve()) if local is not None else str(minfo.get("repo") or "Qwen/Qwen-Image-2.1-Turbo")
    else:
        local = resolve_local_model_path("qwen-image-2.1")
        p = str(local.resolve()) if local is not None else str(minfo.get("repo") or "mlx-community/Qwen-Image-2.1-mflux-q4")
    _ensure_qwen_model_complete(p)
    return p


def _flow_scheduler_id(model_id):
    return {
        "flux2-klein-4b": "flow_match_euler_discrete",
        "flux2-klein-9b": "flow_match_euler_discrete",
        "flux3-klein": "flow_match_euler_discrete",
        "moss-image-sft": "flow_match_euler_discrete",
        "moss-image-base": "flow_match_euler_discrete",
        "z-image-turbo": "linear",
        "krea2-turbo": "er_sde",
    }.get(model_id)


def _sidecar_model_value(minfo: dict | None, model: str) -> str:
    """Model key recorded in image sidecars.

    Built-ins record their HF repo. A local-source custom has repo None, and
    persisting None makes the gallery fall back to a wrong built-in label.
    The entry id resolves back through get_model_info and displays as-is.
    """
    if isinstance(minfo, dict):
        repo = minfo.get("repo")
        if repo:
            return repo
        entry_id = minfo.get("id")
        if entry_id:
            return entry_id
    return model


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
    use_teacache: bool = True,
    teacache_threshold: float | None = None,
    use_kv_cache: bool = True,
    clip_skip: int = 2,
    bounding_boxes: list[list[int]] | None = None,
    spatial_boxes: list[dict] | None = None,
    moss_distill: str = "auto",
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
    # Custom entries borrow their family's engine behavior: the id-keyed
    # rules below (quantization pin, Qwen ceilings, scheduler, edit variant)
    # resolve through the same-variant canonical built-in. Built-in ids map
    # to themselves, so their behavior is unchanged.
    rule_id = minfo["id"] if minfo["id"] in MODELS else (_custom_canonical_id(minfo) or minfo["id"])
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
    quantization = _effective_quantization(rule_id, quantization)
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
    if rule_id == "qwen-image-2.1-turbo" and width * height > 768 * 768:
        min_steps = _qwen_turbo_min_steps(width, height)
        if steps < min_steps:
            raise ValueError(
                f"{minfo['label']}: {steps} steps at {width}x{height} leaves faded duplicate "
                f"objects. Use at least {min_steps} steps above 768x768, or render at <= 768x768."
            )
    if rule_id in ("qwen-image-2.1", "qwen-image-2.1-turbo") and steps > _QWEN_MAX_STEPS:
        raise ValueError(
            f"{minfo['label']}: {steps} steps exceeds the {_QWEN_MAX_STEPS}-step ceiling for this "
            f"machine. Measured median at 25 steps is 790.8s vs 910s at 40 steps, so the extra "
            f"steps are not worth the time. Use ≤ {_QWEN_MAX_STEPS} steps, or change the "
            f"per-model default in Parameters > Defaults."
        )
    # The 589k-pixel ceiling was removed 2026-10-23: measured 1024x1024 at 25 steps
    # on the 16GB M1 completed in 1446s with the tiled VAE decode and no OOM.

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

    with _lock:
        # Invariant 1: Unconditional Prompt Enhancer Unload
        # Must execute at the very start of EVERY generation path (including SDXL and Qwen subprocesses),
        # so the parent process never holds the 1.2 GB Qwen-1.5B weights in Unified Memory.
        try:
            import prompt_enhancer
            if hasattr(prompt_enhancer, "unload_model"):
                prompt_enhancer.unload_model()
        except Exception:
            pass

        if minfo.get("engine") == "sdxl":
            _evict_other_engines("sdxl")
            return _generate_sdxl(
                prompt=prompt, width=width, height=height, steps=steps,
                guidance=effective_guidance, seed=seed, loras=loras, progress_cb=progress_cb,
                phase_cb=phase_cb, model=model, cancel_event=cancel_event,
                negative_prompt=negative_prompt, sampler=chosen_sampler,
                cache_interval=cache_interval, quantization=quantization,
                output_format=output_format, stealth=stealth, fast_vae=fast_vae,
                clip_skip=clip_skip,
            )
        if rule_id in ("qwen-image-2.1", "qwen-image-2.1-turbo"):
            _evict_other_engines("qwen")
            return _generate_qwen_subprocess(
                prompt=prompt, width=width, height=height, steps=steps,
                guidance=effective_guidance, seed=seed, progress_cb=progress_cb,
                phase_cb=phase_cb, model=model, cancel_event=cancel_event,
                negative_prompt=negative_prompt, sampler=chosen_sampler,
                ref_paths=ref_paths, image_strength=image_strength,
                quantization=quantization, output_format=output_format,
                stealth=stealth, fast_vae=fast_vae,
            )
        if minfo.get("engine") == "moss" or rule_id in ("moss-image-sft", "moss-image-base"):
            _evict_other_engines("moss")
            return _generate_moss_subprocess(
                prompt=prompt, width=width, height=height, steps=steps,
                guidance=effective_guidance, seed=seed, progress_cb=progress_cb,
                phase_cb=phase_cb, model=model, cancel_event=cancel_event,
                negative_prompt=negative_prompt, sampler=chosen_sampler,
                ref_paths=ref_paths, image_strength=image_strength,
                quantization=quantization, output_format=output_format,
                stealth=stealth, fast_vae=fast_vae, spatial_boxes=spatial_boxes,
                loras=loras, moss_distill=moss_distill,
            )
        _cancel_event.clear()
        _cancel_mflux_watchdog()
        seed = seed if seed is not None else int(time.time())

        variant = "standard"
        scheduler_id = chosen_sampler if chosen_sampler in ("er_sde", "euler", "linear", "flow_match_euler_discrete") else _flow_scheduler_id(rule_id)
        if sampler_is_custom(chosen_sampler) and chosen_sampler != "custom:":
            # Custom mflux schedulers resolve through mflux's own registry
            # (registered under their custom:<name> id); krea2 never gets here,
            # its engine whitelist rejects external ids.
            if ensure_mflux_registered():
                scheduler_id = chosen_sampler
        is_flux2 = rule_id in ("flux2-klein-4b", "flux2-klein-9b")
        is_flux3 = rule_id == "flux3-klein"
        # MOSS models are dispatched to the subprocess engine before this block,
        # so is_moss stays False here; the flag exists because the shared dispatch
        # below references it for the FLUX3-family kwargs filtering (line ~4248).
        is_moss = rule_id.startswith("moss")
        if (is_flux2 or is_flux3) and ref_paths:
            variant = "edit"

        if is_flux3 and spatial_boxes:
            from spatial_boxes import compose_flux3_bbox_prompt
            if "[{" not in prompt:
                prompt = compose_flux3_bbox_prompt(prompt, spatial_boxes, has_reference=bool(ref_paths))

        if is_flux3 and bounding_boxes is None:
            from flux3_spatial import parse_bounding_boxes
            boxes, _ = parse_bounding_boxes(prompt)
            if boxes:
                bounding_boxes = boxes

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
        pipe.callbacks.register(_MemoryLifecycleCallback())
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

            elif is_flux3:
                if hasattr(pipe, "vae") and hasattr(pipe.vae, "decoder"):
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

            # Acceleration Hooks: TeaCache step-skipping (mlx-teacache)
            # Supported: flux2-klein-4b, flux2-klein-9b, z-image-turbo
            # DISABLED for FLUX2 img2img (KV cache handles caching instead)
            teacache_handle = None
            use_teacache_for_this_run = use_teacache and _TEACACHE_AVAILABLE and model in ("flux2-klein-4b", "flux2-klein-9b", "z-image-turbo")
            # Disable TeaCache for FLUX2 img2img (KV cache handles caching)
            is_flux2_img2img = model in ("flux2-klein-4b", "flux2-klein-9b") and variant == "edit" and ref_paths
            if use_teacache_for_this_run and not is_flux2_img2img:
                # Conservative thresholds: 0.12 for FLUX.2 and Z-Image guarantees zero visual artifacts
                # and prevents over-skipping on distilled schedules.
                if teacache_threshold is None:
                    if model in ("flux2-klein-4b", "flux2-klein-9b"):
                        teacache_threshold = 0.12
                    elif model == "z-image-turbo":
                        teacache_threshold = 0.12
                try:
                    teacache_handle = apply_teacache(pipe, rel_l1_thresh=teacache_threshold)
                    print(f"[generator] TeaCache enabled for {model} with threshold={teacache_threshold}", flush=True)
                except Exception as _teacache_err:
                    print(f"[generator] TeaCache failed for {model}: {_teacache_err}", flush=True)
                    teacache_handle = None

            t_infer_start = time.time()
            sampler_label = chosen_sampler
            is_flux3_family = is_flux3 or is_moss or pipe.__class__.__name__.startswith("Flux3")
            flux3_guidance = effective_guidance if (effective_guidance is not None and effective_guidance > 1.0) else None

            try:
                import inspect
                sig_params = inspect.signature(pipe.generate_image).parameters
                has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
            except Exception:
                sig_params = {}
                has_var_kwargs = True

            if variant == "edit":
                # NOTE: image_strength is deliberately NOT passed here.
                # The FLUX edit variant starts from PURE NOISE (create_noise) and feeds
                # the reference as concatenated conditioning tokens -- it is not classic
                # img2img. image_strength only feeds Config.init_time_step, so on this
                # path it would not make the result more faithful to the reference; it
                # would just skip the first `int(steps * strength)` denoise steps and
                # degrade the image. Verified: passing the app's 0.6 default skips 60%
                # of the steps. Fidelity here is controlled by the prompt and guidance,
                # not by a strength slider. `reference_strength` is therefore accepted
                # and recorded but has no effect on FLUX -- see RELEASE_BLOCKERS.md.
                edit_kwargs = {
                    "seed": seed,
                    "prompt": prompt,
                    "num_inference_steps": steps,
                    "height": height,
                    "width": width,
                    "guidance": flux3_guidance if is_flux3_family else effective_guidance,
                }
                if "image_paths" in sig_params or has_var_kwargs:
                    edit_kwargs["image_paths"] = [Path(p) for p in ref_paths]
                elif "image_path" in sig_params:
                    edit_kwargs["image_path"] = ref_paths[0] if ref_paths else None

                if not is_flux3_family and scheduler_id is not None and ("scheduler" in sig_params or has_var_kwargs):
                    edit_kwargs["scheduler"] = scheduler_id
                if bounding_boxes and ("bounding_boxes" in sig_params or has_var_kwargs):
                    edit_kwargs["bounding_boxes"] = bounding_boxes

                if not has_var_kwargs and sig_params:
                    edit_kwargs = {k: v for k, v in edit_kwargs.items() if k in sig_params}

                out = pipe.generate_image(**edit_kwargs)
            else:
                gen_kwargs = {
                    "seed": seed,
                    "prompt": prompt,
                    "num_inference_steps": steps,
                    "height": height,
                    "width": width,
                    "guidance": flux3_guidance if is_flux3_family else effective_guidance,
                }
                if not is_flux3_family:
                    if ref_paths:
                        if "image_path" in sig_params or has_var_kwargs:
                            gen_kwargs["image_path"] = ref_paths[0]
                            if "image_strength" in sig_params or has_var_kwargs:
                                gen_kwargs["image_strength"] = image_strength
                    elif "image_path" in sig_params:
                        gen_kwargs["image_path"] = None
                        if "image_strength" in sig_params:
                            gen_kwargs["image_strength"] = image_strength

                    if minfo.get("supports_negative") and ("negative_prompt" in sig_params or has_var_kwargs):
                        gen_kwargs["negative_prompt"] = negative_prompt or None
                    if scheduler_id is not None and ("scheduler" in sig_params or has_var_kwargs):
                        gen_kwargs["scheduler"] = scheduler_id

                # KV Cache for FLUX2 img2img (enabled by default when reference image provided, unless explicitly disabled)
                if model in ("flux2-klein-4b", "flux2-klein-9b") and ref_paths and use_kv_cache:
                    if "use_kv_cache" in sig_params or has_var_kwargs:
                        gen_kwargs["use_kv_cache"] = True

                if bounding_boxes and ("bounding_boxes" in sig_params or has_var_kwargs):
                    gen_kwargs["bounding_boxes"] = bounding_boxes

                if not has_var_kwargs and sig_params:
                    gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in sig_params}

                sampler_label = chosen_sampler
                try:
                    out = pipe.generate_image(**gen_kwargs)
                except Exception as exc:
                    # A failed Metal command buffer keeps its allocations until the
                    # pipeline and MLX cache are released; without this the next
                    # model in the same process starts from a fragmented heap and
                    # OOMs at sizes it already rendered. Measured 2026-10-22:
                    # flux3-klein OOMed at 512x512 after a plain 9B run, then the
                    # same request succeeded in a fresh process.
                    message = str(exc).lower()
                    if ("insufficient memory" in message or "out of memory" in message
                            or "metal" in message):
                        _drop_mflux_pipeline()
                    raise
            infer_time = round(time.time() - t_infer_start, 2)
        finally:
            for _cb in reversed(restore_callbacks):
                try:
                    _cb()
                except Exception:
                    pass
            pipe.callbacks.in_loop = [
                c for c in pipe.callbacks.in_loop
                if not isinstance(c, (_ProgressCallback, _MemoryLifecycleCallback))
            ]
            pipe.callbacks.before_loop = [
                c for c in getattr(pipe.callbacks, "before_loop", [])
                if not isinstance(c, (_ProgressCallback, _MemoryLifecycleCallback))
            ]
            pipe.callbacks.after_loop = [
                c for c in getattr(pipe.callbacks, "after_loop", [])
                if not isinstance(c, (_ProgressCallback, _MemoryLifecycleCallback))
            ]
            if prev_wired is not None:
                try:
                    set_limit = getattr(mx, "set_wired_limit", None) or mx.metal.set_wired_limit
                    set_limit(prev_wired)
                except Exception:
                    pass
            # Restore TeaCache
            if teacache_handle is not None:
                try:
                    teacache_handle.restore()
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
            "model": _sidecar_model_value(minfo, model),
            "reference_images": [Path(p).name for p in ref_paths],
            "stealth": stealth,
            "format": fmt,
            "fast_vae": bool(fast_vae) and bool(minfo.get("supports_fast_vae")),
        }
        if scheduler_id is not None:
            meta["scheduler"] = scheduler_id
        if bounding_boxes is not None:
            meta["bounding_boxes"] = bounding_boxes
        if spatial_boxes is not None:
            meta["spatial_boxes"] = spatial_boxes
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
_FLUX3_MAX_STEPS = 24

# Hard ceiling for qwen-image-2.1 / qwen-image-2.1-turbo, by owner decision (2026-09-28): 40 steps is
# never worth running on a 16GB M1. Its own measured medians are 790.8s at
# 512x768 / 25 steps and 910s at 512x768 / 40 steps, so the extra 15 steps buy
# almost nothing for ~15% more time. Refused loudly rather than silently
# clamped, so a saved model_defaults of 40 cannot quietly keep costing that.
_QWEN_MAX_STEPS = 25


def _qwen_turbo_min_steps(width, height) -> int:
    """Turbo's distilled schedule needs more steps once the grid grows.

    Measured on the 16 GB M1 with the Sawfwair mixed-4bit pack, same prompt and
    seed 42: 768x768 / 8 steps is clean; 1024x1024 / 8 steps leaves a faded
    duplicate apple; 1024x1024 / 10 and 12 steps are clean. Keep 8 through
    768², require 10 above it.
    """
    return 10 if int(width) * int(height) > 768 * 768 else 8


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


def _moss_idle_kill_s() -> int:
    """Effective MOSS idle-kill seconds: persisted setting > built-in 300. 0 = never release."""
    v = app_settings.get_setting("idle_kill_s_moss")
    if v is None:
        return MOSS_IDLE_KILL_S
    return max(0, int(v))


def _moss_process_is_healthy() -> bool:
    proc = _moss_process
    if proc is None or proc.stdin is None:
        return False
    try:
        if proc.poll() is not None:
            return False
    except Exception:
        return False
    return _moss_reader_messages is not None and not _moss_reader_done.is_set()


def _arm_moss_watchdog():
    global _moss_watchdog, _moss_idle_since
    idle_s = _moss_idle_kill_s()
    proc = _moss_process
    if idle_s <= 0 or proc is None or proc.poll() is not None:
        _cancel_moss_watchdog()
        return
    if _moss_watchdog is not None:
        _moss_watchdog.cancel()
    _moss_idle_since = time.time()
    timer_ref = [None]

    def _kill():
        global _moss_watchdog, _moss_idle_since
        with _lock:
            if _moss_watchdog is not timer_ref[0] or _moss_process is not proc:
                return
            _kill_moss_process()
            print("[moss] watchdog: idle daemon terminated", flush=True)
            if _moss_watchdog is timer_ref[0]:
                _moss_watchdog = None
                _moss_idle_since = None

    timer = threading.Timer(idle_s, _kill)
    timer.daemon = True
    timer_ref[0] = timer
    _moss_watchdog = timer
    timer.start()


def _cancel_moss_watchdog():
    global _moss_watchdog, _moss_idle_since
    if _moss_watchdog is not None:
        _moss_watchdog.cancel()
        _moss_watchdog = None
    _moss_idle_since = None


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


def _apply_vae_tiling(pipe, default_size: int = 512, model_label: str = "model") -> None:
    """Give a pipeline a TilingConfig so large VAE decodes (>512px) tile into manageable chunks.

    mflux only tiles when tiles_per_dim > 1, and the tile size is a cap on each tile's
    working set. A value of 0 restores untiled decoding.
    """
    size = _krea_vae_tile_size() if model_label == "krea2" else default_size
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
        print(f"[generator] {model_label} VAE tiling unavailable: {exc}", flush=True)


def _apply_krea_vae_tiling(pipe) -> None:
    size = _krea_vae_tile_size()
    if size <= 0:
        return
    _apply_vae_tiling(pipe, default_size=size, model_label="krea2")


def _drop_mflux_pipeline():
    global _pipeline, _current_pipeline_key, _current_pipeline_model, _prompt_cache
    _cancel_mflux_watchdog()
    _pipeline = None
    _current_pipeline_key = None
    _current_pipeline_model = None
    _clear_prompt_cache()
    _clear_krea_te_cache()
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
        try:
            os.kill(pid, 0)
            return False
        except ProcessLookupError:
            _remove_qwen_pid(pid)
            return True
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


_MOSS_PID_FILE = DATA_DIR / ".tmp" / "moss_daemon.pid"


def _read_moss_pid() -> int | None:
    try:
        value = int(_MOSS_PID_FILE.read_text("utf-8").strip())
        return value if value > 1 else None
    except (OSError, UnicodeError, ValueError):
        return None


def _write_moss_pid(pid: int):
    _MOSS_PID_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = _MOSS_PID_FILE.with_name(f".{_MOSS_PID_FILE.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(str(pid), encoding="utf-8")
        os.replace(tmp, _MOSS_PID_FILE)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass


def _remove_moss_pid(pid: int | None = None):
    current = _read_moss_pid()
    if current is not None and (pid is None or current == pid):
        try:
            _MOSS_PID_FILE.unlink()
        except OSError:
            pass


def _cleanup_stale_moss_daemon() -> bool:
    """Reap a MOSS daemon left behind by a crashed parent."""
    pid = _read_moss_pid()
    if pid is None or pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        _remove_moss_pid(pid)
        return True
    except PermissionError:
        return False
    try:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=2)
        command = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    if not command:
        try:
            os.kill(pid, 0)
            return False
        except ProcessLookupError:
            _remove_moss_pid(pid)
            return True
    if "moss_engine.py" not in command or "--serve" not in command:
        _remove_moss_pid(pid)
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _remove_moss_pid(pid)
        return True
    except PermissionError:
        return False
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            _remove_moss_pid(pid)
            return True
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _remove_moss_pid(pid)
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
        try:
            os.kill(pid, 0)
            return False
        except ProcessLookupError:
            _remove_sdxl_pid(pid)
            return True
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
_MAX_SDXL_DIAGNOSTIC_CHARS = 8000


def _first_actionable_line(text):
    """Return the first non-traceback line from engine output."""
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "Traceback (most recent call last):":
            continue
        if line.startswith("File ") and ", line " in raw_line:
            continue
        return _redact_local_paths(line)
    return ""


def _redact_local_paths(text):
    """Keep filenames useful without exposing local absolute paths."""
    cleaned = str(text or "")
    replacements = (
        (str(ASSET_DIR), "<asset-dir>"),
        (str(DATA_DIR), "<data-dir>"),
        (str(Path(__file__).resolve().parent), "<backend>"),
        (os.path.expanduser("~"), "~"),
    )
    for root, label in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if root and root != "/" and root in cleaned:
            cleaned = cleaned.replace(root, label)
    return cleaned


def _format_sdxl_engine_failure(engine_error, stderr_tail, job_ref):
    """Split rich operator diagnostics from the narrow user-facing error."""
    first_line = _first_actionable_line(engine_error) or "SDXL engine failed"
    tail = str(stderr_tail or "").strip()
    if not tail:
        return first_line, ""
    user_message = f"{first_line} (engine diagnostics logged for generation {job_ref})"
    diagnostic = (
        f"[generator] SDXL generation {job_ref} failed: {first_line}\n"
        f"[generator] engine stderr:\n{tail[-_MAX_SDXL_DIAGNOSTIC_CHARS:]}"
    )
    return user_message, diagnostic


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
        [str(engine), str(script), "--serve",
         "--parent-pid", str(os.getpid())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
                    start_new_session=True,
                    cwd=str(Path(__file__).resolve().parent),
                    env={**os.environ,
                         "MOSS_TE_EMBED_CACHE_DIR": str(DATA_DIR / "te_embed_cache")},
                )
    proc._mlx_process_group = True
    _write_sdxl_pid(proc.pid)
    _sdxl_daemon = proc
    _sdxl_reader_messages, _sdxl_reader_done, _sdxl_reader_thread = _start_json_reader(proc)
    _sdxl_stderr_done, _sdxl_stderr_thread = _start_stderr_reader(proc, _sdxl_stderr_tail)
    return proc


# A step-distillation adapter, by filename or registry name. Two of them stacked
# (our auto-injected 4-step Lightning against a user-picked Hyper-SD 8-step, say)
# fight over the same timestep schedule and the result is worse than either alone,
# so anything that auto-adds one has to ask whether another is already active.
_DISTILL_LORA_RE = re.compile(r"lightning|hyper[-_ ]?sd|distill|lcm", re.IGNORECASE)


def _is_distill_lora_name(name) -> bool:
    return bool(_DISTILL_LORA_RE.search(str(name or "")))


def _distill_lora_active(loras) -> bool:
    for lora in loras or []:
        if _is_distill_lora_name(Path(str(lora.get("path", ""))).name):
            return True
        if _is_distill_lora_name(lora.get("name")):
            return True
    return False


def _sdxl_lightning_injection(minfo, steps, sampler, loras):
    """Path of the 4-step Lightning LoRA to auto-add for a fast run, or None.

    Factored out of _generate_sdxl so the decision (distilled model? short run?
    already-a-distill-LoRA-active?) is testable without spawning the engine.

    Custom entries run family defaults: no auto-injection for custom ids.
    """
    if minfo.get("family"):
        return None
    if minfo.get("is_distilled"):
        return None
    if not (steps <= 4 or sampler in ("euler_trailing", "trailing")):
        return None
    lightning_path = str(ASSET_DIR / "SDXL" / "sdxl_lightning_4step_lora.safetensors")
    if not os.path.exists(lightning_path):
        return None
    if _distill_lora_active(loras):
        return None
    if len(loras or []) >= 16:
        raise ValueError("cannot auto-add the SDXL Lightning LoRA when 16 LoRAs are already active")
    return lightning_path


def _generate_sdxl(prompt, width, height, steps, guidance, seed, loras,
                   progress_cb, model, cancel_event, negative_prompt, sampler,
                   cache_interval=1, quantization=4, output_format="png", stealth=False,
                   fast_vae=True, phase_cb=None, clip_skip=2):
    global _current_sdxl_model
    raw_image_path = None
    with _lock:
        minfo = get_model_info(model) or {}
        model_dir = _resolve_sdxl_model_dir(model, minfo)
        # Reuse a live daemon for the same model instead of paying a ~6.5 GB
        # reload on every run. The engine keys its own pipeline on
        # (model_dir, quantize_unet) and applies LoRAs per request, so only a
        # changed model_dir needs a fresh process: quantize_unet is fixed at 4
        # in the request below, and wired_limit_bytes is applied per request by
        # the engine, so neither belongs in this identity.
        if (
            _sdxl_daemon is None
            or _sdxl_daemon.poll() is not None
            or _current_sdxl_model != model_dir
        ):
            _kill_sdxl_daemon()
        _cancel_mflux_watchdog()
        _cancel_qwen_watchdog()
        try:
            _drop_mflux_pipeline()
            sampler = sampler or (minfo.get("samplers")[0] if minfo.get("samplers") else "euler_trailing")
            if sampler in ("euler", None) and (steps <= 8 or "lightning" in str(model).lower() or minfo.get("is_distilled")):
                sampler = "euler_trailing"
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
            lightning_path = _sdxl_lightning_injection(minfo, steps, sampler, actual_loras)
            if lightning_path:
                actual_loras.append({"path": lightning_path, "scale": 1.0})
            actual_loras = _enrich_loras_with_registry(actual_loras)
            sdxl_wired_limit = _wired_limit_bytes()
            tile_vae = width * height >= 768 * 768
            # Check the diffusers layout HERE, before handing the path to the engine.
            # Without this the owner gets a forty-line traceback ending in
            # "No .safetensors files in .../unet" and learns nothing; it was reported
            # from a second machine whose download had been interrupted, leaving an
            # empty unet/ behind an INSTALLED badge. This is the last point at which
            # the app can still explain itself.
            ok_sdxl, why_sdxl = sdxl_layout_report(Path(model_dir))
            if not ok_sdxl:
                raise ValueError(f"SDXL cannot load this model: {why_sdxl}")
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
                "clip_skip": int(clip_skip or 2),
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
            result = None
            try:
                _write_process_request(
                    proc,
                    (json.dumps(req, separators=(",", ":")) + "\n").encode("utf-8"),
                    (cancel_event, _cancel_event),
                    _engine_timeout("SDXL_ENGINE_TIMEOUT_S"),
                )
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
                    elif "diag" in payload:
                        print(f"[sdxl-engine diag] {json.dumps(payload)}", flush=True)
                        continue
                    elif payload.get("status") == "success" or "error" in payload or "width" in payload or "generation_time" in payload or "dest_size" in payload:
                        result = payload
                        break
                    else:
                        print(f"[sdxl-engine unhandled] {json.dumps(payload)}", flush=True)
                        continue
            except BaseException:
                _kill_sdxl_daemon()
                raise

            if result is None:
                _kill_sdxl_daemon()
                raise RuntimeError("SDXL engine returned no result")
            if result.get("error"):
                tail = "".join(_sdxl_stderr_tail).strip()
                message, diagnostic = _format_sdxl_engine_failure(
                    result.get("error"), tail, image_id
                )
                if diagnostic:
                    print(diagnostic, flush=True)
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
                "model": _sidecar_model_value(minfo, model),
                "engine": "sdxl",
                "stealth": stealth,
                "format": fmt,
                "fast_vae": bool(result.get("fast_vae", fast_vae)),
                "cache_interval": int(result.get("cache_interval", cache_interval)),
                "cache_final_step": bool(result.get("cache_final_step", cache_interval > 1)),
                "tile_vae": bool(result.get("tile_vae", tile_vae)),
                "clip_skip": int(result.get("clip_skip", clip_skip or 2)),
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
            # We cancelled the Qwen watchdog on the way in (a live timer would fire
            # mid-render). Re-arm it here: a warm Qwen daemon left with no idle timer
            # holds ~10.5 GB until the next Qwen job or app quit, which on a 16 GB machine
            # is the documented OOM-adjacent state. _arm_qwen_watchdog self-guards -- it
            # no-ops when there is no live daemon.
            _arm_qwen_watchdog()


_qwen_stderr_tail = collections.deque(maxlen=100)
_QWEN_ENGINE_LOCK = Path(tempfile.gettempdir()) / "diffusionbear-qwen-engine.lock"
_METAL_FAILURE_MARKERS = (
    "MTLCompilerService",
    "Unable to load kernel",
    "Reentrancy avoided",
    "metal::Device",
)


def _is_metal_compiler_failure(text: str) -> bool:
    """True for the Metal errors that kill the compiler service, not the model.

    Retrying these in a fresh interpreter is safe: the weights are unchanged and
    the failure is process-global Metal state. Ordinary errors (bad prompt, lock
    conflict, weight load) must not be retried.
    """
    haystack = str(text or "")
    return any(marker in haystack for marker in _METAL_FAILURE_MARKERS)


def _run_with_metal_retry(attempt, retry, attempts: int = 2):
    """Run ``attempt`` once, and one more time after a Metal compiler failure.

    ``retry`` performs the side effects that make a second attempt viable
    (kill the wedged engine, remove a partial output). Exceptions that do not
    look like Metal failures propagate unchanged.
    """
    for index in range(attempts):
        try:
            return attempt()
        except Exception as exc:
            if index + 1 >= attempts or not _is_metal_compiler_failure(str(exc)):
                raise
            retry()
    raise RuntimeError("Metal retry loop exhausted without a result")


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
    # model arg is one of "qwen-image-2.1" or "qwen-image-2.1-turbo"
    global _qwen_process, _qwen_reader_messages, _qwen_reader_done, _qwen_reader_thread
    global _qwen_stderr_thread, _qwen_stderr_done
    raw_image_path = None
    with _lock:
        _kill_sdxl_daemon()
        _cancel_mflux_watchdog()
        try:
            _drop_mflux_pipeline()
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
            if minfo.get("local_path"):
                # Custom entry: a vanished directory must fail here with the
                # existing friendly validation error, not inside the engine
                # subprocess as a traceback.
                ok_qwen, why_qwen = validate_local_model_dir(model, model_path)
                if not ok_qwen:
                    raise ValueError(f"{minfo.get('label', model)}: {why_qwen}")
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
                "wired_limit_bytes": _wired_limit_bytes(),
            }
            if phase_cb is not None:
                phase_cb("preparing", "Preparing prompt & Qwen-Image 2.1 subprocess...")
            _cancel_event.clear()
            engine = Path(__file__).parent / "qwen_engine.py"
            repo_python = Path(__file__).resolve().parent.parent / "venv" / "bin" / "python"
            python_exe = str(repo_python) if repo_python.exists() else sys.executable

            def _attempt():
                global _qwen_process, _qwen_reader_messages, _qwen_reader_done
                global _qwen_reader_thread, _qwen_stderr_done, _qwen_stderr_thread
                proc = _qwen_process
                if proc is not None and not _qwen_process_is_healthy():
                    _kill_qwen_process()
                    proc = None
                if proc is None:
                    if not _cleanup_stale_qwen_daemon():
                        raise RuntimeError("could not verify the previous Qwen daemon")
                    _qwen_stderr_tail.clear()
                    env = os.environ.copy()
                    env["DIFFUSIONBEAR_QWEN_LOCK"] = str(_QWEN_ENGINE_LOCK)
                    proc = subprocess.Popen(
                        [python_exe, str(engine), "--serve",
                         "--parent-pid", str(os.getpid())],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=False,
                        bufsize=0,
                        start_new_session=True,
                        cwd=str(Path(__file__).resolve().parent),
                        env=env,
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
                payload_result = None
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
                        payload_result = payload
                        break
                if payload_result is None:
                    err = "".join(_qwen_stderr_tail)[-2000:] or "no result received"
                    raise RuntimeError(f"Qwen engine returned no result: {err}")
                if payload_result.get("error"):
                    raise RuntimeError(str(payload_result["error"]))
                return payload_result

            def _restart_after_metal_failure():
                _kill_qwen_process()
                if raw_image_path is not None:
                    try:
                        raw_image_path.unlink()
                    except OSError:
                        pass
                print(
                    "[qwen] Metal compiler failure; restarting engine and retrying once",
                    flush=True,
                )
                time.sleep(1.0)

            result = _run_with_metal_retry(_attempt, _restart_after_metal_failure)
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
                "model": _sidecar_model_value(minfo, model),
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



MOSS_ADAPTER_FILES = ("lora.safetensors", "time_embedding.safetensors", "dataspace.safetensors")

# Step-distillation adapters, in ascending NFE order. Short runs on
# moss-image-sft auto-select one (krea2-turbo precedent), because a 4- or
# 8-step stock run is strictly worse than its DMD adapter: less compute for
# the same schedule family. Base has no distributed adapter yet. Third field
# is the public HF repo a missing kit can be fetched from: both repos ship
# exactly the three kit files under these names. A missing kit used to make
# auto-selection silently fall back to the stock schedule, which at 4-8 steps
# with CFG 3 produces garbage — the kinds of ugly 4- and 8-step outputs from
# the test-MC session of 2026-10-20.
MOSS_AUTO_DMD = (
    (4, "moss_distill/darask0-4step", "darask0/MOSS-Image_4-step_adapter_lora"),
    (8, "moss_distill/lodestones-8step", "lodestones/MOSS-Image-SFT-8-steps-dmd"),
)

MOSS_DMD_ADAPTER_REPOS = {rel: repo for _, rel, repo in MOSS_AUTO_DMD}


def _ensure_moss_dmd_adapter(target_dir, repo, phase_cb=None) -> str | None:
    """Download a missing MOSS step-distillation adapter kit into target_dir.

    Both public kits ship exactly the three files the engine expects
    (lora.safetensors + time_embedding.safetensors + dataspace.safetensors),
    so a missing kit is fetched as a unit (~1.24 GB for the 1.19 GB lora).
    hf_hub_download resumes and skips files that are already there, so a
    half-installed kit completes instead of re-downloading. Returns the
    directory on success, None on any failure (offline, repo gone) so the run
    falls back to the stock schedule instead of dying.
    """
    if not repo:
        return None
    if phase_cb is not None:
        phase_cb("download", f"MOSS: fetching DMD adapter from {repo}...")
    try:
        from huggingface_hub import hf_hub_download
        import hf_service
        token = hf_service.get_hf_token()
        d = Path(target_dir)
        d.mkdir(parents=True, exist_ok=True, mode=0o755)
        for name in MOSS_ADAPTER_FILES:
            hf_hub_download(
                repo_id=repo,
                filename=name,
                local_dir=str(d),
                token=token,
            )
        if not all((d / name).is_file() for name in MOSS_ADAPTER_FILES):
            raise RuntimeError("missing adapter files after download")
        return str(d)
    except Exception as exc:
        print(f"[moss-dmd] adapter download failed ({repo}): {exc}", flush=True)
        if phase_cb is not None:
            phase_cb("download", f"MOSS adapter download failed: {exc}")
        return None


def _auto_dmd_adapter(model_id: str, steps, mode: str = "auto",
                      download_missing: bool = False, phase_cb=None) -> str | None:
    """Adapter dir to select for a moss-image-sft run, or None.

    Testable without spawning the engine (same rationale as
    _sdxl_lightning_injection). mode="auto" follows the run length (small =
    smallest sufficient NFE); mode="dmd4"/"dmd8" forces that adapter regardless
    of steps (the engine clamps steps to the adapter's code count); mode="stock"
    never auto-selects. None when the adapter files are missing on disk, the
    model is not the SFT checkpoint, or nothing applies. With
    download_missing=True a missing kit is fetched from its public repo first
    (kept off the default so the pure selection logic stays unit-testable
    without network).
    """
    if model_id != "moss-image-sft":
        return None
    if mode == "stock":
        return None
    if mode in ("auto", "dmd4", "dmd8"):
        pass
    else:
        raise ValueError(f"unknown moss distillation mode: {mode!r}")
    wanted: int | None
    if mode == "auto":
        if not steps:
            return None
        try:
            requested = int(steps)
        except (TypeError, ValueError):
            return None
        if requested > 8:
            return None
        wanted = next((n for n, _, _ in MOSS_AUTO_DMD if requested <= n), None)
        if wanted is None:
            return None
    else:
        wanted = 4 if mode == "dmd4" else 8
    for nfe, rel, repo in MOSS_AUTO_DMD:
        if nfe == wanted:
            d = ASSET_DIR / "lora_files" / rel
            if all((d / name).is_file() for name in MOSS_ADAPTER_FILES):
                return str(d)
            if download_missing and repo:
                fetched = _ensure_moss_dmd_adapter(d, repo, phase_cb)
                if fetched and all((Path(fetched) / name).is_file() for name in MOSS_ADAPTER_FILES):
                    return fetched
            break
    return None


def _moss_adapter_dir(loras: list[dict] | None) -> str:
    """Distilled-adapter directory for a MOSS request, or "".

    The 8-step DMD adapter is a *directory* (lora.safetensors +
    time_embedding.safetensors + dataspace.safetensors), not a bare LoRA file:
    the dataspace sampler and the fp32 step embedder live beside the adapter
    weights. The presence of those siblings next to a selected LoRA is what
    marks a request as distilled, so the engine can switch sampler and embedder.
    """
    for lora in loras or []:
        parent = Path(str(lora.get("path", ""))).expanduser().parent
        if all((parent / name).is_file() for name in MOSS_ADAPTER_FILES):
            return str(parent)
    return ""


def _generate_moss_subprocess(prompt, width, height, steps, guidance, seed,
                              progress_cb=None, phase_cb=None, model="moss-image-sft",
                              cancel_event=None, negative_prompt="", sampler=None,
                              ref_paths=None, image_strength=None, quantization=4,
                              output_format="png", stealth=False, fast_vae=False,
                              spatial_boxes=None, loras=None, moss_distill="auto"):
    global _moss_process, _moss_reader_messages, _moss_reader_done, _moss_reader_thread
    global _moss_stderr_thread, _moss_stderr_done
    raw_image_path = None
    with _lock:
        _kill_sdxl_daemon()
        _kill_qwen_process()
        _cancel_mflux_watchdog()
        _cancel_moss_watchdog()
        try:
            _drop_mflux_pipeline()
            minfo = get_model_info(model) or {}
            seed = seed if seed is not None else int(time.time())
            eff_guidance = guidance if guidance is not None else (minfo.get("default_guidance") or 3.0)
            eff_steps = steps if steps is not None else (minfo.get("default_steps") or 50)

            image_id = uuid.uuid4().hex
            fmt = "jpeg" if str(output_format).lower() in ("jpeg", "jpg") else "png"
            raw_image_path = GENERATED_DIR / f"{image_id}.raw.png"
            final_image_path = GENERATED_DIR / f"{image_id}.{fmt}"

            # Resolve local model path or repo
            local = resolve_local_model_path(model)
            model_path = str(local) if local is not None else (minfo.get("repo") or "OpenMOSS-Team/MOSS-Image-SFT")

            if ref_paths:
                print(f"[generator] MOSS-Image is a text/layout diffusion model and does not accept image references; ignoring {len(ref_paths)} reference path(s)", flush=True)

            effective_prompt = prompt
            if spatial_boxes and '"bbox_format"' not in prompt:
                from spatial_boxes import compose_moss_bbox_prompt
                effective_prompt = compose_moss_bbox_prompt(prompt, spatial_boxes, width=width, height=height)

            moss_loras = [
                {"path": str(l["path"]), "scale": float(l.get("scale", 1.0) or 1.0)}
                for l in (loras or [])
                if isinstance(l, dict) and l.get("path") and str(l.get("format", "MOSS")).upper() in ("MOSS", "")
            ]
            moss_adapter_dir = _moss_adapter_dir(moss_loras)
            if not moss_adapter_dir:
                auto_dmd = _auto_dmd_adapter(model, eff_steps, moss_distill,
                                             download_missing=True, phase_cb=phase_cb)
                if auto_dmd:
                    # Short-run auto-selection (krea2-turbo precedent): a
                    # 4-8 step stock run is strictly worse than its DMD
                    # adapter, so the adapter wins the hide.
                    moss_adapter_dir = auto_dmd
                    moss_loras = moss_loras + [
                        {"path": str(auto_dmd + "/lora.safetensors"), "scale": 1.0}
                    ]
            if moss_adapter_dir:
                # DMD/DMAD adapters sample CFG-free: the t=1 vector is folded
                # into the step embedder and the dataspace codes replace the
                # schedule, so a guidance > 1 would just add a wasted pass.
                eff_guidance = 1.0

            req = {
                "prompt": effective_prompt,
                "negative_prompt": negative_prompt or "",
                "width": width,
                "height": height,
                "steps": eff_steps,
                "guidance": eff_guidance,
                "seed": seed,
                "dest": str(raw_image_path),
                "model_path": model_path,
                "model_id": model,
                "loras": moss_loras,
                "adapter_dir": moss_adapter_dir,
                "sampler": sampler if sampler_is_custom(sampler) else None,
                "phase_cb": phase_cb is not None,
            }

            if phase_cb is not None:
                phase_cb("preparing", "Preparing prompt & MOSS-Image engine...")

            _cancel_event.clear()
            engine_script = Path(__file__).parent / "moss_engine.py"
            _moss_stderr_tail.clear()
            repo_python = Path(__file__).resolve().parent.parent / "venv" / "bin" / "python"
            python_exe = str(repo_python) if repo_python.exists() else sys.executable

            proc = _moss_process
            if proc is None or proc.poll() is not None:
                if not _cleanup_stale_moss_daemon():
                    raise RuntimeError("could not verify the previous MOSS daemon")
                proc = subprocess.Popen(
                    [python_exe, str(engine_script), "--serve",
                     "--parent-pid", str(os.getpid())],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                    bufsize=0,
                    start_new_session=True,
                    cwd=str(Path(__file__).resolve().parent),
                )
                proc._mlx_process_group = True
                _moss_process = proc
                _write_moss_pid(proc.pid)
                _moss_reader_messages, _moss_reader_done, _moss_reader_thread = _start_json_reader(proc)
                _moss_stderr_done, _moss_stderr_thread = _start_stderr_reader(proc, _moss_stderr_tail)

            _write_process_request(
                proc,
                (json.dumps(req, separators=(",", ":")) + "\n").encode("utf-8"),
                (cancel_event, _cancel_event),
                _engine_timeout("MOSS_ENGINE_TIMEOUT_S"),
            )
            result = None
            while True:
                line = _read_engine_message(
                    proc,
                    _moss_reader_messages,
                    _moss_reader_done,
                    (cancel_event, _cancel_event),
                    _engine_timeout("MOSS_ENGINE_TIMEOUT_S"),
                    "MOSS engine",
                    _moss_stderr_tail,
                )
                if line is None:
                    err = "".join(_moss_stderr_tail)[-2000:] or "EOF on stdout"
                    raise RuntimeError(f"MOSS engine died: {err}")
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                # The daemon announces itself with {"status": "ready"} (and answers
                # ping with "pong"); without this skip the parent accepted that
                # message as the final result -- before any image existed -- and
                # crashed opening the .raw.png with FileNotFoundError.
                if payload.get("status") in ("ready", "pong"):
                    continue
                if "progress" in payload:
                    step = payload["progress"].get("step")
                    if progress_cb is not None and isinstance(step, int):
                        progress_cb(max(0, step - 1))
                elif "phase" in payload:
                    if phase_cb is not None:
                        phase_cb(str(payload["phase"]), str(payload.get("detail", "")))
                elif "diag" in payload:
                    print(f"[moss-engine diag] {json.dumps(payload)}", flush=True)
                    continue
                elif payload.get("status") == "success" or "error" in payload or "width" in payload or "generation_time" in payload or "dest_size" in payload:
                    result = payload
                    break
                else:
                    print(f"[moss-engine unhandled] {json.dumps(payload)}", flush=True)
                    continue

            if result is None:
                err = "".join(_moss_stderr_tail)[-2000:] or "no result received"
                raise RuntimeError(f"MOSS engine returned no result: {err}")
            if result.get("error"):
                raise RuntimeError(str(result["error"]))

            if phase_cb is not None:
                phase_cb("saving", "Finalizing image and metadata...")

            # Sounds cheap fails silent: the daemon sends success only after its
            # own save, so a missing raw here means the write did not survive
            # (external volume hiccup, half-write visible race). Surface the
            # engine's own stderr instead of a bare [Errno 2] at Image.open --
            # and the file name, so the user's bug report contains the answer.
            if not raw_image_path.exists() or raw_image_path.stat().st_size == 0:
                _engine_tail = "".join(_moss_stderr_tail)[-800:]
                raise RuntimeError(
                    "MOSS engine claimed success but never wrote the image to "
                    f"{raw_image_path.name} "
                    f"({raw_image_path}, engine stderr: "
                    f"{_engine_tail or 'empty'}): "
                    f"result={json.dumps(result, default=str)}")

            with Image.open(raw_image_path) as source:
                image = source.copy()

            actual_w = int(result.get("width", image.width))
            actual_h = int(result.get("height", image.height))
            meta = {
                "id": image_id,
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "sampler": result.get("sampler", "flow_match_euler_discrete"),
                "scheduler": "flow_match_euler_discrete",
                "width": actual_w,
                "height": actual_h,
                "steps": int(result.get("steps", eff_steps)),
                "guidance": float(result.get("guidance", eff_guidance)),
                "seed": seed,
                "quantization": None,
                "loras": moss_loras,
                "generation_time": result.get("generation_time", 0.0),
                "load_time": result.get("load_time", 0.0),
                "created_at": time.time(),
                "software": "DiffusionBear",
                "generator": "DiffusionBear",
                "artist": app_settings.metadata_artist(),
                "tags": [],
                "file": final_image_path.name,
                "model": _sidecar_model_value(minfo, model),
                "engine": "moss",
                "reference_images": [Path(p).name for p in (ref_paths or [])],
                "stealth": stealth,
                "format": fmt,
                "fast_vae": False,
                "tiled_vae": False,
            }
            if spatial_boxes:
                meta["spatial_boxes"] = spatial_boxes
            if moss_adapter_dir:
                meta["adapter_dir"] = moss_adapter_dir
                if moss_distill and moss_distill != "auto":
                    meta["moss_distill"] = moss_distill

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
            _kill_moss_process()
            raise
        finally:
            if not _moss_process_is_healthy():
                _kill_moss_process()
            else:
                _arm_moss_watchdog()
            if raw_image_path is not None:
                try:
                    raw_image_path.unlink()
                except OSError:
                    pass


def _shutdown_engines():
    try:
        _kill_moss_process()
    except Exception:
        pass
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
    except Exception:
        pass


atexit.register(_shutdown_engines)


