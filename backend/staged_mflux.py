"""Three-stage memory lifecycle for mflux text-to-image pipelines.

The stock mflux pipeline owns text encoder, DiT and VAE for the whole call. On a
16 GB Mac the text encoder is dead weight after the prompt is encoded, and the
DiT is dead weight after the last denoise step; keeping both resident turns the
peak into a swap event. This module mirrors tools/fibo2/fibo2_q4_inference.py:

    Stage 1  text encoder -> embeddings -> del encoder -> gc -> clear_cache
    Stage 2  DiT          -> latents     -> del DiT     -> gc -> clear_cache
    Stage 3  VAE          -> pixels      -> del VAE     -> gc -> clear_cache

Only one model tree is resident at a time. The intermediate tensors are tiny
compared with the weights. Each stage loads one component through mflux's own
WeightLoader/WeightApplier so quantisation, LoRA maps and checkpoint layouts are
identical to the monolithic pipeline.

Supported now: FLUX.2 Klein 4B/9B, Krea 2 Turbo, Qwen-Image 2.1 / Turbo.
Reference-image and edit variants deliberately fall back to the stock pipeline:
they need the VAE during latent preparation and/or a KV cache, which is not a
three-stage txt2img lifecycle.
"""
from __future__ import annotations

import gc
import math
import os
import time
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


_STAGE_MODELS = {
    "flux1-schnell",
    "flux2-klein-4b",
    "flux2-klein-9b",
    "flux3-klein",
    "krea2-turbo",
    "qwen-image-2.1",
    "qwen-image-2.1-turbo",
    "qwen-image-edit",
    "z-image-turbo",
}


_REFERENCE_MODELS = {
    "flux1-schnell",
    "flux2-klein-4b",
    "flux2-klein-9b",
    "flux3-klein",
    "krea2-turbo",
    "qwen-image-edit",
    "z-image-turbo",
}


_COMPACT_RESIDENT_MODELS = {
    "flux2-klein-4b",
    "flux3-klein",
}

_STAGED_PROMPT_CACHE: OrderedDict[tuple, tuple[dict[str, Any], int]] = OrderedDict()
_STAGED_PROMPT_CACHE_MAX = 16
# Entry count alone does not bound memory: Qwen-VL / Krea embeddings are far
# larger than FLUX.2's ~16 MB. Same byte budget as generator's prompt cache.
_STAGED_PROMPT_CACHE_MAX_BYTES = 256 * (1 << 20)


def _embeddings_nbytes(embeddings: dict[str, Any]) -> int:
    total = 0
    for value in embeddings.values():
        items = value if isinstance(value, (tuple, list)) else (value,)
        for item in items:
            total += int(getattr(item, "nbytes", 0) or 0)
    return total


def _staged_prompt_cache_bytes() -> int:
    return sum(_embeddings_nbytes(e) for e, _bits in _STAGED_PROMPT_CACHE.values())


def _staged_prompt_cache_put(cache_key: tuple, embeddings: dict[str, Any], bits: int) -> None:
    # Only one model's embeddings are ever useful: a switch makes the others dead weight.
    for key in [k for k in _STAGED_PROMPT_CACHE if k[0] != cache_key[0]]:
        _STAGED_PROMPT_CACHE.pop(key, None)
    _STAGED_PROMPT_CACHE[cache_key] = (dict(embeddings), bits)
    _STAGED_PROMPT_CACHE.move_to_end(cache_key)
    while len(_STAGED_PROMPT_CACHE) > 1 and (
        len(_STAGED_PROMPT_CACHE) > _STAGED_PROMPT_CACHE_MAX
        or _staged_prompt_cache_bytes() > _STAGED_PROMPT_CACHE_MAX_BYTES
    ):
        _STAGED_PROMPT_CACHE.popitem(last=False)


_staged_cache_put = _staged_prompt_cache_put


def clear_staged_prompt_cache(keep_model_id: str | None = None) -> int:
    """Drop cached prompt embeddings, except those of ``keep_model_id``; returns the count dropped."""
    doomed = [k for k in _STAGED_PROMPT_CACHE if keep_model_id is None or k[0] != str(keep_model_id)]
    for key in doomed:
        _STAGED_PROMPT_CACHE.pop(key, None)
    return len(doomed)


@contextmanager
def _compile_disabled():
    # mflux's predict closures capture the freshly loaded transformer, so once
    # mx.compile traces them the weights become graph constants that MLX keeps
    # after every Python reference is gone: ~2 GB of FLUX.2 DiT stayed active
    # after each staged run and the next run peaked 2 GB higher. A staged run
    # rebuilds the transformer anyway, so the compiled graph is never reused.
    if os.environ.get("DIFFUSIONBEAR_STAGED_COMPILE") == "1" or os.environ.get("MLX_DISABLE_COMPILE"):
        yield
        return
    mx.disable_compile()
    try:
        yield
    finally:
        mx.enable_compile()
# Encoded reference images keyed by file identity and encode arguments, so a
# re-roll of the same edit skips Stage 1/4 (reference VAE load + encode).
_REFERENCE_CACHE: OrderedDict[tuple, dict[str, Any]] = OrderedDict()
_REFERENCE_CACHE_MAX = 4


def _staged_embeds_to_numpy(embeddings: dict[str, Any], bits: int) -> dict[str, Any] | None:
    """Serialize staged embeddings dict losslessly (preserving bfloat16) for PromptEmbeddingCache."""
    try:
        import json
        import numpy as np

        payload: dict[str, Any] = {}
        entries: dict[str, Any] = {}
        for k, v in embeddings.items():
            if v is None:
                entries[k] = None
            elif isinstance(v, mx.array):
                mx.eval(v)
                raw = mx.contiguous(v).view(mx.uint8)
                mx.eval(raw)
                payload[f"t_{k}"] = np.ascontiguousarray(np.asarray(raw))
                entries[k] = {
                    "dtype": str(v.dtype).rsplit(".", 1)[-1],
                    "shape": [int(n) for n in v.shape],
                }
            else:
                return None
        meta = {"bits": int(bits), "entries": entries}
        payload["meta"] = np.frombuffer(
            json.dumps(meta, separators=(",", ":")).encode("utf-8"), dtype=np.uint8
        )
        return payload
    except Exception:
        return None


def _staged_embeds_from_numpy(data: dict[str, Any]) -> tuple[dict[str, Any], int] | None:
    """Rebuild staged embeddings dict and bits from PromptEmbeddingCache."""
    try:
        import json
        import numpy as np

        meta = json.loads(bytes(np.asarray(data["meta"], dtype=np.uint8)).decode("utf-8"))
        bits = int(meta["bits"])
        out: dict[str, Any] = {}
        for k, info in meta["entries"].items():
            if info is None:
                out[k] = None
            else:
                dtype = getattr(mx, str(info["dtype"]).rsplit(".", 1)[-1])
                raw = np.asarray(data[f"t_{k}"], dtype=np.uint8)
                shape = tuple(int(n) for n in info["shape"])
                out[k] = mx.array(raw).view(dtype).reshape(shape)
        return out, bits
    except Exception:
        return None


def supports_staged(
    model_id: str,
    *,
    variant: str = "standard",
    ref_paths: list[str] | None = None,
    image_strength: float | None = None,
) -> bool:
    """True for txt2img, and for reference models with their 4-stage lifecycle."""
    if variant == "controlnet":
        return False
    if model_id not in _STAGE_MODELS:
        return False
    # mflux >= 0.23 loads Qwen-Image-2.1 packs through its own normalizer; the
    # per-component staged loader predates it and scrambles those weights (measured
    # 2026-11: mosaic garbage at 2 and 4 steps). The Qwen subprocess is the proven
    # path on 0.23, so staged stays off for the qwen21 family there.
    if model_id.startswith("qwen-image-2.1") and _mflux_at_least(0, 23):
        return False
    if ref_paths:
        return model_id in _REFERENCE_MODELS
    return variant == "standard"


def _vendor_version() -> str:
    try:
        from importlib.metadata import version

        return version("mflux")
    except Exception:
        return "unknown"


def _mflux_at_least(major: int, minor: int) -> bool:
    try:
        parts = _vendor_version().split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except Exception:
        return False


def should_use_staged_default(
    model_id: str,
    *,
    variant: str = "standard",
    width: int = 1024,
    height: int = 1024,
    has_refs: bool = False,
    ref_paths: list[str] | None = None,
) -> bool:
    """Smart tiered staging default for 16 GB Apple Silicon.

    Compact models (FLUX.2 Klein 4B, FLUX.3 Klein) without reference images at
    <= 1024x1024 stay on the warm resident pipeline so consecutive runs pay 0.0s
    reload. Heavy models (FLUX.1 Schnell 12B + T5-XXL, Krea 2, FLUX.2 Klein 9B,
    Z-Image Turbo, Qwen-Image 2.1 / Edit), reference edits, and > 1 MP outputs
    use staged eviction to avoid 16 GB swap/OOM.
    """
    eff_refs = ref_paths if ref_paths is not None else (["__ref__"] if has_refs else None)
    if not supports_staged(model_id, variant=variant, ref_paths=eff_refs):
        return False
    if model_id in _COMPACT_RESIDENT_MODELS and not has_refs and not ref_paths and (width * height) <= (1024 * 1024):
        return False
    return True


def _release_cache() -> None:
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def _stage_memory() -> dict[str, float]:
    """MLX peak/active/cache and process footprint in MiB; resets the MLX peak."""
    from engine_metrics import footprint_mb

    mib = 1024 * 1024
    out: dict[str, float] = {}
    for key, name in (("mlx_peak_mb", "get_peak_memory"), ("mlx_active_mb", "get_active_memory"),
                      ("mlx_cache_mb", "get_cache_memory")):
        fn = getattr(mx, name, None)
        if fn is not None:
            try:
                out[key] = round(fn() / mib, 1)
            except Exception:
                pass
    footprint = footprint_mb()
    if footprint is not None:
        out["footprint_mb"] = round(footprint, 1)
    try:
        mx.reset_peak_memory()
    except Exception:
        pass
    return out


def apply_mlx_cache_limit() -> int | None:
    """Cap the MLX buffer cache per engine_memory policy; returns the previous cap."""
    from engine_memory import mlx_cache_limit_bytes

    limit = mlx_cache_limit_bytes()
    setter = getattr(mx, "set_cache_limit", None)
    if setter is None:
        return None
    if limit is None:
        import hardware_limits

        hardware_limits.apply_metal_cache_limit()
        return None
    try:
        return setter(limit)
    except Exception:
        return None


def restore_mlx_cache_limit(previous: int | None) -> None:
    setter = getattr(mx, "set_cache_limit", None)
    if previous is None or setter is None:
        return
    try:
        setter(previous)
    except Exception:
        pass


def _reference_cache_key(adapter, model_id, ref_paths, width, height, image_strength, seed, steps) -> tuple | None:
    files = []
    for path in ref_paths:
        try:
            resolved = Path(path).expanduser().resolve()
            stat = resolved.stat()
        except OSError:
            return None
        files.append((str(resolved), stat.st_mtime_ns, stat.st_size))
    noise_args = () if getattr(adapter, "reference_encode_is_deterministic", False) else (
        image_strength, int(seed), int(steps), getattr(adapter, "sampler_name", None),
    )
    return (
        str(model_id), str(getattr(adapter, "root", "")), int(getattr(adapter, "quantization", 0) or 0),
        tuple(files), int(width), int(height), noise_args,
    )


def _eval(*arrays: mx.array) -> None:
    live = [a for a in arrays if a is not None]
    if live:
        mx.eval(*live)


def _tiling_for(width: int, height: int) -> TilingConfig | None:
    # Match the app's 16 GB policy: tile only when the decode is large enough to
    # matter. 512x768 fits without tiling.
    if width * height <= 512 * 768:
        return None
    return TilingConfig(
        vae_decode_tiles_per_dim=2,
        vae_decode_tile_size=512,
        vae_decode_overlap=8,
    )


def _get_component(definition, name: str):
    for component in definition.get_components():
        if component.name == name:
            return component
    raise KeyError(f"mflux weight definition has no component {name!r}")


def _infer_stored_quantization(component_weights, group_size: int = 64) -> int | None:
    """Bit width of a pre-quantized component, read from packed weight/scales shapes.

    A packed weight is ``input_dims * bits / 32`` wide and its scales are
    ``input_dims / group_size`` wide, so ``bits = 32 * w / (s * group_size)``.
    The most common width wins; mflux's WeightApplier still structures each
    layer at its own stored width.
    """
    from mlx.utils import tree_flatten

    try:
        shapes = {str(key): getattr(value, "shape", None) for key, value in tree_flatten(component_weights)}
    except Exception:
        return None
    widths: Counter[int] = Counter()
    for key, scales_shape in shapes.items():
        if not key.endswith(".scales") or not scales_shape:
            continue
        weight_shape = shapes.get(key[: -len(".scales")] + ".weight")
        if not weight_shape or not scales_shape[-1] or not group_size:
            continue
        bits = weight_shape[-1] * 32 / (scales_shape[-1] * group_size)
        if bits in (2, 3, 4, 5, 6, 8):
            widths[int(bits)] += 1
    if widths:
        return widths.most_common(1)[0][0]
    if any(key.endswith(".scales") or ".scales." in key for key in shapes):
        return 4
    return None


def _apply_loaded(definition, name: str, loaded: LoadedWeights, quantize: int, model: nn.Module):
    component_weights = loaded.components.get(name)
    if component_weights is None:
        raise RuntimeError(f"no {name} weights in the loaded component set")
    # HF conversions often omit mflux's quantization_level metadata but still
    # carry packed q4 weights + scales. Without this, WeightApplier tries to
    # quantize uint32 packed tensors and raises "Only real floating types...".
    if loaded.meta_data.quantization_level is None:
        inferred = _infer_stored_quantization(
            component_weights, int(getattr(definition, "quantization_group_size", 64) or 64)
        )
        if inferred is not None:
            loaded.meta_data.quantization_level = inferred
    one = LoadedWeights(components={name: component_weights}, meta_data=loaded.meta_data)
    return WeightApplier.apply_and_quantize(
        weights=one,
        models={name: model},
        quantize_arg=quantize,
        weight_definition=definition,
    )


def _load_one(definition, name: str, root: Path, quantize: int, model: nn.Module):
    """Load exactly one mflux component into ``model`` and return the bit width."""
    component = _get_component(definition, name)
    loaded = WeightLoader.load_single_local(component=component, root_path=root)
    if getattr(definition, "__name__", "") == "Qwen21WeightDefinition":
        from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer

        if hasattr(Qwen21Initializer, "_rebuild_quantized_layers"):
            # mflux <= 0.20 fork only; >= 0.23 handles these packs natively and
            # staged qwen is disabled in supports_staged() anyway.
            Qwen21Initializer._rebuild_quantized_layers(model, loaded.components[name])
    return _apply_loaded(definition, name, loaded, quantize, model)


# Tokenizers are a few MB of host memory but took 0.5-0.85 s to rebuild per prompt.
_TOKENIZER_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_TOKENIZER_CACHE_MAX = 2


def _tokenizer_fingerprint(root: Path) -> tuple:
    # A repair rewrites tokenizer files in place; the mtimes make that a cache miss.
    stamps = []
    for path in sorted(Path(root).glob("tokenizer*/*")):
        try:
            stamps.append((path.relative_to(root).as_posix(), path.stat().st_mtime_ns))
        except OSError:
            pass
    return tuple(stamps)


def _load_tokenizers(definition, root: Path):
    from mflux.models.common.tokenizer import TokenizerLoader

    key = (
        str(root),
        getattr(definition, "__name__", None) or type(definition).__name__,
        _tokenizer_fingerprint(root),
    )
    cached = _TOKENIZER_CACHE.get(key)
    if cached is not None:
        _TOKENIZER_CACHE.move_to_end(key)
        return cached
    tokenizers = TokenizerLoader.load_all(definitions=definition.get_tokenizers(), model_path=str(root))
    _TOKENIZER_CACHE[key] = tokenizers
    while len(_TOKENIZER_CACHE) > _TOKENIZER_CACHE_MAX:
        _TOKENIZER_CACHE.popitem(last=False)
    return tokenizers


def clear_reference_cache() -> None:
    _REFERENCE_CACHE.clear()


@dataclass
class StageResult:
    image: Image.Image
    model_config: ModelConfig
    quantization: int
    seed: int
    prompt: str
    steps: int
    guidance: float
    negative_prompt: str
    width: int
    height: int
    lora_paths: list[str]
    lora_scales: list[float]
    load_time: float
    inference_time: float
    stage_times: dict[str, float] = field(default_factory=dict)
    stage_memory: dict[str, dict[str, float]] = field(default_factory=dict)


class _Adapter:
    definition = None
    loras_affect_text_encoder = True
    model_config: ModelConfig

    def __init__(self, model_id: str, model_path: str | Path, quantization: int,
                 loras: list[dict] | None, phase_cb: Callable | None = None,
                 progress_cb: Callable | None = None, cancel_check: Callable | None = None):
        self.model_id = model_id
        self.root = Path(model_path).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"staged mflux needs a local model directory: {self.root}")
        self.quantization = int(quantization)
        self.loras = list(loras or [])
        self.phase_cb = phase_cb
        self.progress_cb = progress_cb
        self.cancel_check = cancel_check
        self.lora_paths = [str(item["path"]) for item in self.loras]
        self.lora_scales = [float(item.get("scale", 1.0)) for item in self.loras]
        self.fast_vae = False

    def _phase(self, name: str, detail: str) -> None:
        if self.phase_cb is not None:
            try:
                self.phase_cb(name, detail)
            except Exception:
                pass

    def _check_cancel(self) -> None:
        if self.cancel_check is not None:
            self.cancel_check()

    def _step(self, t: int) -> None:
        self._check_cancel()
        if self.progress_cb is not None:
            self.progress_cb(int(t))

    def _apply_transformer_lora(self, transformer: nn.Module, mapping) -> None:
        if not self.loras:
            return
        from mflux.models.common.lora.mapping.lora_loader import LoRALoader

        LoRALoader.load_and_apply_lora(
            lora_mapping=mapping,
            transformer=transformer,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            bake_lora=False,
        )
        mx.eval(transformer.parameters())

    def encode_references(self, ref_paths: list[str], width: int, height: int,
                          image_strength: float | None, seed: int, steps: int) -> dict[str, Any]:
        return {}

    def encode(self, prompt: str, negative_prompt: str, guidance: float) -> dict[str, Any]:
        raise NotImplementedError

    def denoise(self, prompt: str, negative_prompt: str, width: int, height: int,
                steps: int, guidance: float, seed: int, embeddings: dict[str, Any],
                reference: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def decode(self, latents: mx.array, width: int, height: int, meta: dict[str, Any]) -> Image.Image:
        raise NotImplementedError


def _decode_with_taef(variant: str, nchw_latents: mx.array) -> Image.Image | None:
    """Fast TAEF mini-VAE decode when ``fast_vae`` is enabled; returns None on fallback."""
    try:
        from generator import _get_taef_decoder

        dec = _get_taef_decoder(variant)
        nhwc = mx.transpose(nchw_latents, (0, 2, 3, 1))
        decoded_nhwc = dec.decode(nhwc)
        decoded = mx.transpose(decoded_nhwc, (0, 3, 1, 2)) * 2.0 - 1.0
        mx.eval(decoded)
        return ImageUtil.to_pil(decoded)
    except Exception:
        return None


def _install_rope_hook(model_id: str, extra_hook: str | None = None) -> None:
    """Install compiled RoPE/sampler hooks from generator when available."""
    try:
        import generator

        if extra_hook == "krea2":
            generator._install_krea_sampler_scalar_cache()
            generator._install_krea_euler_a_trailing()
        elif extra_hook == "flux3":
            generator._install_flux3_compiled_scheduler_step()
        generator._install_rope_cache(model_id)
    except Exception:
        pass


_QWEN3VL_MROPE_PATCHED = False


def _install_qwen3vl_mlx_mrope() -> None:
    """Replace Qwen3VL NumPy interleaved mrope with pure-MLX ops (avoids np.array guard sync)."""
    global _QWEN3VL_MROPE_PATCHED
    if _QWEN3VL_MROPE_PATCHED:
        return
    try:
        from mflux.models.common_models.qwen3_vl.qwen3_vl_rope import Qwen3VLRotaryEmbedding

        @staticmethod
        def _mlx_interleaved_mrope(freqs: mx.array, mrope_section: list[int]) -> mx.array:
            idx = mx.arange(freqs.shape[-1])
            out = freqs[0]
            for dim, offset in enumerate((1, 2), start=1):
                length = int(mrope_section[dim]) * 3
                mask = (idx < length) & ((idx % 3) == offset)
                out = mx.where(mask, freqs[dim], out)
            return out

        Qwen3VLRotaryEmbedding._apply_interleaved_mrope = _mlx_interleaved_mrope
        _QWEN3VL_MROPE_PATCHED = True
    except Exception:
        pass


def _mlx_quantized_dir(path: Path) -> bool:
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        return False
    try:
        import json
        weight_map = json.loads(index.read_text("utf-8")).get("weight_map") or {}
        return any(str(key).endswith(".scales") for key in weight_map)
    except Exception:
        return False


def _load_mlx_shards(path: Path) -> dict:
    import json
    index = path / "model.safetensors.index.json"
    shards = []
    if index.is_file():
        try:
            shards = sorted(set((json.loads(index.read_text("utf-8")).get("weight_map") or {}).values()))
        except Exception:
            shards = []
    if not shards:
        shards = [p.name for p in sorted(path.glob("*.safetensors"))]
    weights = {}
    for name in shards:
        weights.update(mx.load(str(path / name)))
    return weights


class Flux2Adapter(_Adapter):
    reference_encode_is_deterministic = True
    loras_affect_text_encoder = False

    def __init__(self, model_id: str, model_path, quantization, loras, phase_cb=None):
        from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

        super().__init__(model_id, model_path, quantization, loras, phase_cb)
        self.definition = Flux2KleinWeightDefinition
        self.model_config = (
            ModelConfig.flux2_klein_9b() if model_id.endswith("-9b")
            else ModelConfig.flux2_klein_4b()
        )
        self.bits = quantization

    def encode(self, prompt: str, negative_prompt: str, guidance: float) -> dict[str, Any]:
        from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
        from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder

        self._phase("loading_text_encoder", "Stage 1/3: loading FLUX.2 text encoder")
        t0 = time.perf_counter()
        encoder = Qwen3TextEncoder(**(getattr(self.model_config, "text_encoder_overrides", None) or {}))
        self.bits = _load_one(self.definition, "text_encoder", self.root, self.quantization, encoder) or self.quantization
        tokenizers = _load_tokenizers(self.definition, self.root)
        tokenizer = tokenizers["qwen3"]

        self._phase("encoding", "Stage 1/3: encoding prompt")
        prompt_embeds, text_ids = Flux2PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=tokenizer,
            text_encoder=encoder,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        negative_embeds = None
        negative_ids = None
        if guidance is not None and guidance > 1.0:
            negative_embeds, negative_ids = Flux2PromptEncoder.encode_prompt(
                prompt=negative_prompt or " ",
                tokenizer=tokenizer,
                text_encoder=encoder,
                num_images_per_prompt=1,
                max_sequence_length=512,
                text_encoder_out_layers=(9, 18, 27),
            )
        _eval(prompt_embeds, text_ids, negative_embeds, negative_ids)
        load_time = time.perf_counter() - t0
        del encoder, tokenizer, tokenizers
        self._phase("evicting_text_encoder", "Stage 1/3: evicting text encoder")
        _release_cache()
        return {
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
            "negative_prompt_embeds": negative_embeds,
            "negative_text_ids": negative_ids,
            "load_time": load_time,
        }

    def encode_references(self, ref_paths, width, height, image_strength, seed, steps):
        from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
        from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers

        self._phase("loading_reference_vae", "Stage 1/4: loading FLUX.2 VAE for reference")
        t0 = time.perf_counter()
        vae = Flux2VAE()
        self.bits = _load_one(self.definition, "vae", self.root, self.quantization, vae) or self.bits
        image_latents, image_latent_ids = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
            vae=vae,
            tiling_config=_tiling_for(width, height),
            image_paths=ref_paths,
            batch_size=1,
        )
        _eval(image_latents, image_latent_ids)
        load_time = time.perf_counter() - t0
        # Kept for decode (stage 4): reloading it cost a second full VAE read.
        self._vae = vae
        del vae
        return {
            "image_latents": image_latents,
            "image_latent_ids": image_latent_ids,
            "load_time": load_time,
        }

    def denoise(self, prompt, negative_prompt, width, height, steps, guidance, seed, embeddings, reference=None):
        from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
        from mflux.models.flux2.model.flux2_transformer.flux2_kv_cache import Flux2KVCache
        from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
        from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
        from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        from mflux.models.flux2.weights.flux2_lora_mapping import Flux2LoRAMapping

        self._phase("loading_transformer", "Stage 2/3: loading FLUX.2 DiT")
        _install_rope_hook(self.model_id)
        transformer = Flux2Transformer(**(getattr(self.model_config, "transformer_overrides", None) or {}))
        self.bits = _load_one(self.definition, "transformer", self.root, self.quantization, transformer) or self.bits
        self._apply_transformer_lora(transformer, Flux2LoRAMapping.get_mapping())
        config = Config(
            model_config=self.model_config,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            scheduler=self.sampler_name or "flow_match_euler_discrete",
        )
        use_reference = bool(reference and reference.get("image_latents") is not None)
        if use_reference:
            latents, latent_ids, latent_height, latent_width = _Flux2KleinEditHelpers.prepare_generation_latents(
                seed=seed, height=height, width=width,
            )
            image_latents = reference["image_latents"]
            image_latent_ids = reference["image_latent_ids"]
            cache_enabled = (
                bool(getattr(self, "use_kv_cache", True))
                and bool(self.model_config.supports_kv_cache)
                and image_latents.shape[1] > 0
            )
            kv_cache = None
            negative_kv_cache = None
            if cache_enabled:
                kv_cache = Flux2KVCache(
                    num_double_layers=len(transformer.transformer_blocks),
                    num_single_layers=len(transformer.single_transformer_blocks),
                )
                if embeddings.get("negative_prompt_embeds") is not None:
                    negative_kv_cache = Flux2KVCache(
                        num_double_layers=len(transformer.transformer_blocks),
                        num_single_layers=len(transformer.single_transformer_blocks),
                    )
            predict = Flux2KleinEdit._predict(None, transformer)
            cached_predict = Flux2KleinEdit._cached_predict(None, transformer) if cache_enabled else None
            self._phase("generating", f"Stage 3/4: denoising {steps} steps with reference")
            for step_idx, t in enumerate(config.time_steps):
                self._check_cancel()
                if cache_enabled and step_idx == 0:
                    Flux2KleinEdit._configure_kv_caches(
                        kv_cache=kv_cache, negative_kv_cache=negative_kv_cache,
                        mode="extract", num_ref_tokens=image_latents.shape[1],
                    )
                    noise = predict(
                        latents=latents, image_latents=image_latents, latent_ids=latent_ids,
                        image_latent_ids=image_latent_ids,
                        prompt_embeds=embeddings["prompt_embeds"], text_ids=embeddings["text_ids"],
                        negative_prompt_embeds=embeddings.get("negative_prompt_embeds"),
                        negative_text_ids=embeddings.get("negative_text_ids"),
                        guidance=guidance, timestep=config.scheduler.timesteps[t],
                        kv_cache=kv_cache, negative_kv_cache=negative_kv_cache,
                    )
                elif cache_enabled:
                    Flux2KleinEdit._configure_kv_caches(
                        kv_cache=kv_cache, negative_kv_cache=negative_kv_cache,
                        mode="cached", num_ref_tokens=image_latents.shape[1],
                    )
                    noise = cached_predict(
                        latents=latents, latent_ids=latent_ids,
                        prompt_embeds=embeddings["prompt_embeds"], text_ids=embeddings["text_ids"],
                        negative_prompt_embeds=embeddings.get("negative_prompt_embeds"),
                        negative_text_ids=embeddings.get("negative_text_ids"),
                        guidance=guidance, timestep=config.scheduler.timesteps[t],
                        kv_cache=kv_cache, negative_kv_cache=negative_kv_cache,
                    )
                else:
                    noise = predict(
                        latents=latents, image_latents=image_latents, latent_ids=latent_ids,
                        image_latent_ids=image_latent_ids,
                        prompt_embeds=embeddings["prompt_embeds"], text_ids=embeddings["text_ids"],
                        negative_prompt_embeds=embeddings.get("negative_prompt_embeds"),
                        negative_text_ids=embeddings.get("negative_text_ids"),
                        guidance=guidance, timestep=config.scheduler.timesteps[t],
                    )
                latents = config.scheduler.step(
                    noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas,
                )
                mx.eval(latents)
                self._step(int(t))
            del kv_cache, negative_kv_cache, cached_predict, predict
            phase_label = "Stage 3/4: evicting FLUX.2 DiT"
        else:
            latents, latent_ids, latent_height, latent_width = Flux2LatentCreator.prepare_packed_latents(
                seed=seed, height=height, width=width, batch_size=1,
            )
            # mflux reads transformer.model_config off M1/M2 but never sets it.
            transformer.model_config = self.model_config
            predict = Flux2Klein._predict(transformer)
            self._phase("generating", f"Stage 2/3: denoising {steps} steps")
            for t in config.time_steps:
                self._check_cancel()
                noise = predict(
                    latents=latents,
                    latent_ids=latent_ids,
                    prompt_embeds=embeddings["prompt_embeds"],
                    text_ids=embeddings["text_ids"],
                    negative_prompt_embeds=embeddings.get("negative_prompt_embeds"),
                    negative_text_ids=embeddings.get("negative_text_ids"),
                    guidance=guidance,
                    timestep=config.scheduler.timesteps[t],
                )
                latents = config.scheduler.step(
                    noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas,
                )
                mx.eval(latents)
                self._step(int(t))
            del predict
            phase_label = "Stage 2/3: evicting FLUX.2 DiT"
        _eval(latents)
        del transformer, config
        self._phase("evicting_transformer", phase_label)
        _release_cache()
        return {
            "latents": latents,
            "latent_ids": latent_ids,
            "latent_height": latent_height,
            "latent_width": latent_width,
        }

    def decode(self, latents, width, height, meta):
        from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE

        self._phase("loading_vae", "Stage 3/3: loading FLUX.2 VAE")
        packed = latents.reshape(
            latents.shape[0], meta["latent_height"], meta["latent_width"], latents.shape[-1],
        ).transpose(0, 3, 1, 2)
        image = None
        if getattr(self, "fast_vae", False):
            image = _decode_with_taef("taef2", Flux2VAE._unpatchify_latents(packed))
        vae = getattr(self, "_vae", None)
        self._vae = None
        if image is None:
            if vae is None:
                vae = Flux2VAE()
                self.bits = _load_one(self.definition, "vae", self.root, self.quantization, vae) or self.bits
            decoded = vae.decode_packed_latents(packed, tiling_config=_tiling_for(width, height))
            mx.eval(decoded)
            image = ImageUtil.to_pil(decoded)
            del decoded
        del vae, packed, latents
        self._phase("evicting_vae", "Stage 3/3: evicting FLUX.2 VAE")
        _release_cache()
        return image

    sampler_name = None


class Krea2Adapter(_Adapter):
    loras_affect_text_encoder = False

    def __init__(self, model_id, model_path, quantization, loras, phase_cb=None):
        from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition

        super().__init__(model_id, model_path, quantization, loras, phase_cb)
        self.definition = Krea2WeightDefinition
        self.model_config = ModelConfig.krea2()
        self.bits = quantization
        self.sampler_name = None

    def encode(self, prompt, negative_prompt, guidance):
        from mflux.models.krea2.model.krea2_text_encoder.prompt_encoder import Krea2PromptEncoder
        from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder

        self._phase("loading_text_encoder", "Stage 1/3: loading Krea 2 text encoder")
        t0 = time.perf_counter()
        encoder = Krea2TextEncoder()
        te_dir = self.root / "text_encoder"
        if _mlx_quantized_dir(te_dir):
            # The Ouinche full pack ships the encoder already MLX-quantized with
            # metadata format=mlx, not mflux's quantization_level key. WeightApplier
            # cannot see that metadata and would try to quantize uint32 weights.
            self.bits = 4
            nn.quantize(encoder, bits=4, group_size=64)
            raw = _load_mlx_shards(te_dir)
            from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
            stripped = {}
            for key, value in raw.items():
                mapped = Krea2WeightDefinition.strip_te_prefix(key)
                if mapped is not None:
                    stripped[mapped] = value
            encoder.load_weights(list(stripped.items()))
            mx.eval(encoder.parameters())
        else:
            self.bits = _load_one(
                self.definition, "text_encoder", self.root, self.quantization, encoder,
            ) or self.quantization
        tokenizers = _load_tokenizers(self.definition, self.root)
        tokenizer = tokenizers["qwen3vl"]
        embeds, neg_embeds = Krea2PromptEncoder.encode_prompt_pair(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=guidance,
            tokenizer=tokenizer,
            text_encoder=encoder,
            prompt_cache={},
        )
        _eval(embeds, neg_embeds)
        load_time = time.perf_counter() - t0
        del encoder, tokenizer, tokenizers
        self._phase("evicting_text_encoder", "Stage 1/3: evicting Krea 2 text encoder")
        _release_cache()
        return {"prompt_embeds": embeds, "negative_prompt_embeds": neg_embeds, "load_time": load_time}

    def encode_references(self, ref_paths, width, height, image_strength, seed, steps):
        from mflux.models.common.latent_creator.latent_creator import LatentCreator
        from mflux.models.krea2.latent_creator.krea2_latent_creator import Krea2LatentCreator
        from mflux.models.krea2.variants.txt2img.krea2 import Krea2
        from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE

        self._phase("loading_reference_vae", "Stage 1/4: loading Krea 2 VAE for reference")
        t0 = time.perf_counter()
        vae = QwenVAE()
        self.bits = _load_one(self.definition, "vae", self.root, self.quantization, vae) or self.bits
        config = Config(
            model_config=self.model_config,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=1.0,
            image_path=ref_paths[0],
            image_strength=image_strength if image_strength is not None else 1.0,
            scheduler=Krea2._resolve_scheduler(self.sampler_name),
        )
        encoded = LatentCreator.encode_image(
            vae=vae, image_path=ref_paths[0], height=height, width=width,
            tiling_config=_tiling_for(width, height),
        )
        clean = Krea2LatentCreator.pack_latents(encoded, height, width)
        noise = Krea2LatentCreator.create_noise(seed, height, width)
        sigma = config.scheduler.sigmas[config.init_time_step]
        latents = LatentCreator.add_noise_by_interpolation(clean=clean, noise=noise, sigma=sigma)
        _eval(latents)
        load_time = time.perf_counter() - t0
        del vae, encoded, clean, noise
        self._phase("evicting_reference_vae", "Stage 1/4: evicting Krea 2 VAE")
        _release_cache()
        return {"latents": latents, "load_time": load_time}

    def denoise(self, prompt, negative_prompt, width, height, steps, guidance, seed, embeddings, reference=None):
        from mflux.models.krea2.latent_creator.krea2_latent_creator import Krea2LatentCreator
        from mflux.models.krea2.model.krea2_sampler import Krea2Sampler
        from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
        from mflux.models.krea2.variants.txt2img.krea2 import Krea2
        from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping

        self._phase("loading_transformer", "Stage 2/3: loading Krea 2 DiT")
        _install_rope_hook("krea2-turbo", extra_hook="krea2")
        transformer = Krea2Transformer(**(getattr(self.model_config, "transformer_overrides", None) or {}))
        self.bits = _load_one(self.definition, "transformer", self.root, self.quantization, transformer) or self.bits
        self._apply_transformer_lora(transformer, Krea2LoRAMapping.get_mapping())
        scheduler_name = Krea2._resolve_scheduler(self.sampler_name)
        config = Config(
            model_config=self.model_config,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            scheduler=scheduler_name,
        )
        sigmas = config.scheduler.sigmas
        latents = reference["latents"] if reference and reference.get("latents") is not None else Krea2LatentCreator.create_noise(seed, height, width)
        stepper = Krea2Sampler.make_stepper(scheduler_name, sigmas, seed)
        predict = Krea2._predict(
            transformer, embeddings["prompt_embeds"], embeddings.get("negative_prompt_embeds"), guidance,
        )
        self._phase("generating", f"Stage 2/3: denoising {steps} steps")
        for t in config.time_steps:
            self._check_cancel()
            ts = sigmas[t].reshape(1)
            v = predict(latents=latents, timestep=ts)
            denoised = latents - sigmas[t] * v
            latents = stepper.step(t, latents, v, denoised)
            mx.eval(latents)
            self._step(int(t))
        _eval(latents)
        del predict, transformer, stepper, config
        self._phase("evicting_transformer", "Stage 2/3: evicting Krea 2 DiT")
        _release_cache()
        return {"latents": latents}

    def decode(self, latents, width, height, meta):
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE

        self._phase("loading_vae", "Stage 3/3: loading Krea 2 VAE")
        image = _decode_with_taef("krea2", latents) if getattr(self, "fast_vae", False) else None
        if image is None:
            vae = QwenVAE()
            self.bits = _load_one(self.definition, "vae", self.root, self.quantization, vae) or self.bits
            decoded = VAEUtil.decode(vae=vae, latent=latents, tiling_config=_tiling_for(width, height))
            mx.eval(decoded)
            image = ImageUtil.to_pil(decoded)
            del vae, decoded
        del latents
        self._phase("evicting_vae", "Stage 3/3: evicting Krea 2 VAE")
        _release_cache()
        return image


class Qwen21Adapter(_Adapter):
    loras_affect_text_encoder = False

    def __init__(self, model_id, model_path, quantization, loras, phase_cb=None):
        from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition

        super().__init__(model_id, model_path, quantization, loras, phase_cb)
        self.definition = Qwen21WeightDefinition
        self.model_config = ModelConfig.qwen_image_21()
        self.bits = quantization
        self.sampler_name = None

    def encode(self, prompt, negative_prompt, guidance):
        _install_qwen3vl_mlx_mrope()
        # Importing qwen_engine installs its single-file text-encoder selector.
        try:
            import qwen_engine  # noqa: F401
        except Exception:
            pass
        from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
        from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder

        self._phase("loading_text_encoder", "Stage 1/3: loading Qwen text encoder")
        t0 = time.perf_counter()
        encoder = Qwen21TextEncoder()
        self.bits = _load_one(
            self.definition, "text_encoder", self.root, self.quantization, encoder,
        ) or self.quantization
        tokenizers = _load_tokenizers(self.definition, self.root)
        tokenizer = tokenizers["qwen21"]
        cache: dict = {}
        prompt_embeds, prompt_mask = Qwen21PromptEncoder.encode_prompt(
            prompt=prompt, prompt_cache=cache, tokenizer=tokenizer, text_encoder=encoder,
        )
        negative_embeds = negative_mask = None
        if guidance is not None and guidance > 1.0 and negative_prompt:
            negative_embeds, negative_mask = Qwen21PromptEncoder.encode_prompt(
                prompt=negative_prompt, prompt_cache=cache, tokenizer=tokenizer, text_encoder=encoder,
            )
        _eval(prompt_embeds, prompt_mask, negative_embeds, negative_mask)
        load_time = time.perf_counter() - t0
        del encoder, tokenizer, tokenizers, cache
        self._phase("evicting_text_encoder", "Stage 1/3: evicting Qwen text encoder")
        _release_cache()
        return {
            "prompt_embeds": prompt_embeds,
            "prompt_mask": prompt_mask,
            "negative_prompt_embeds": negative_embeds,
            "negative_prompt_mask": negative_mask,
            "load_time": load_time,
        }

    def denoise(self, prompt, negative_prompt, width, height, steps, guidance, seed, embeddings, reference=None):
        from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
        from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

        self._phase("loading_transformer", "Stage 2/3: loading Qwen DiT")
        transformer = Qwen21Transformer()
        self.bits = _load_one(self.definition, "transformer", self.root, self.quantization, transformer) or self.bits
        if self.loras:
            try:
                import qwen_engine
                qwen_engine._apply_qwen_loras(type("_QwenHost", (), {"transformer": transformer})(), self.loras)
            except Exception:
                from mflux.models.common.lora.mapping.lora_loader import LoRALoader
                from lora_compat import _load_qwen_mapping
                LoRALoader.load_and_apply_lora(
                    lora_mapping=_load_qwen_mapping().get_mapping(),
                    transformer=transformer,
                    lora_paths=self.lora_paths,
                    lora_scales=self.lora_scales,
                    role="target",
                    bake_lora=False,
                )
                mx.eval(transformer.parameters())
        is_dmd2 = int(steps) <= 2 and any("dmd2" in p.lower() for p in self.lora_paths)
        if is_dmd2:
            import copy
            model_cfg = copy.copy(self.model_config)
            model_cfg.requires_sigma_shift = False
            effective_guidance = 1.0
            scheduler_name = "linear"
        else:
            model_cfg = self.model_config
            effective_guidance = guidance
            scheduler_name = self.sampler_name or "linear"
        config = Config(
            model_config=model_cfg,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=effective_guidance,
            scheduler=scheduler_name,
        )
        latents = Qwen21LatentCreator.create_noise(seed, height, width)
        do_cfg = bool(effective_guidance and effective_guidance > 1.0 and embeddings.get("negative_prompt_embeds") is not None)
        self._phase("generating", f"Stage 2/3: denoising {steps} steps")
        for t in config.time_steps:
            self._check_cancel()
            latents = config.scheduler.scale_model_input(latents, t)
            noise = transformer(
                t=t,
                config=config,
                hidden_states=latents,
                encoder_hidden_states=embeddings["prompt_embeds"],
                encoder_hidden_states_mask=embeddings["prompt_mask"],
            )
            if do_cfg:
                noise_negative = transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    encoder_hidden_states=embeddings["negative_prompt_embeds"],
                    encoder_hidden_states_mask=embeddings["negative_prompt_mask"],
                )
                noise = noise_negative + guidance * (noise - noise_negative)
            latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
            mx.eval(latents)
            self._step(int(t))
        _eval(latents)
        del transformer, config
        self._phase("evicting_transformer", "Stage 2/3: evicting Qwen DiT")
        _release_cache()
        return {"latents": latents}

    def decode(self, latents, width, height, meta):
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
        from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE

        self._phase("loading_vae", "Stage 3/3: loading Qwen VAE")
        vae = Qwen21VAE()
        self.bits = _load_one(self.definition, "vae", self.root, self.quantization, vae) or self.bits
        unpacked = Qwen21LatentCreator.unpack_latents(latents=latents, height=height, width=width)
        decoded = VAEUtil.decode(vae=vae, latent=unpacked, tiling_config=_tiling_for(width, height))
        mx.eval(decoded)
        image = ImageUtil.to_pil(decoded)
        del vae, decoded, unpacked, latents
        self._phase("evicting_vae", "Stage 3/3: evicting Qwen VAE")
        _release_cache()
        return image




class Flux1Adapter(_Adapter):
    loras_affect_text_encoder = False

    def __init__(self, model_id, model_path, quantization, loras, phase_cb=None):
        from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition

        super().__init__(model_id, model_path, quantization, loras, phase_cb)
        self.definition = FluxWeightDefinition
        self.model_config = ModelConfig.schnell()
        self.bits = quantization
        self.sampler_name = None

    def encode(self, prompt, negative_prompt, guidance):
        from mflux.models.flux.model.flux_text_encoder.clip_encoder.clip_encoder import CLIPEncoder
        from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
        from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder

        self._phase("loading_text_encoder", "Stage 1/3: loading FLUX.1 T5 + CLIP")
        t0 = time.perf_counter()
        t5 = T5Encoder()
        clip = CLIPEncoder()
        self.bits = _load_one(self.definition, "t5_encoder", self.root, self.quantization, t5) or self.quantization
        self.bits = _load_one(self.definition, "clip_encoder", self.root, self.quantization, clip) or self.bits
        tok = _load_tokenizers(self.definition, self.root)
        embeds, pooled = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache={},
            t5_tokenizer=tok["t5"],
            clip_tokenizer=tok["clip"],
            t5_text_encoder=t5,
            clip_text_encoder=clip,
        )
        _eval(embeds, pooled)
        load_time = time.perf_counter() - t0
        del t5, clip, tok
        self._phase("evicting_text_encoder", "Stage 1/3: evicting FLUX.1 text encoders")
        _release_cache()
        return {"prompt_embeds": embeds, "pooled_prompt_embeds": pooled, "load_time": load_time}

    def encode_references(self, ref_paths, width, height, image_strength, seed, steps):
        from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
        from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
        from mflux.models.flux.model.flux_vae.vae import VAE

        self._phase("loading_reference_vae", "Stage 1/4: loading FLUX.1 VAE for reference")
        t0 = time.perf_counter()
        vae = VAE()
        self.bits = _load_one(self.definition, "vae", self.root, self.quantization, vae) or self.bits
        config = Config(
            model_config=self.model_config,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=1.0,
            image_path=ref_paths[0],
            image_strength=image_strength if image_strength is not None else 0.6,
            scheduler=self.sampler_name or "linear",
        )
        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            width=width,
            height=height,
            img2img=Img2Img(
                vae=vae,
                latent_creator=FluxLatentCreator,
                image_path=config.image_path,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
            ),
        )
        _eval(latents)
        load_time = time.perf_counter() - t0
        del vae
        self._phase("evicting_reference_vae", "Stage 1/4: evicting FLUX.1 VAE")
        _release_cache()
        return {"latents": latents, "load_time": load_time}

    def denoise(self, prompt, negative_prompt, width, height, steps, guidance, seed, embeddings, reference=None):
        from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
        from mflux.models.flux.model.flux_transformer.transformer import Transformer
        from mflux.models.flux.weights.flux_lora_mapping import FluxLoRAMapping

        self._phase("loading_transformer", "Stage 2/3: loading FLUX.1 DiT")
        transformer = Transformer(model_config=self.model_config)
        self.bits = _load_one(self.definition, "transformer", self.root, self.quantization, transformer) or self.bits
        self._apply_transformer_lora(transformer, FluxLoRAMapping.get_mapping())
        config = Config(
            model_config=self.model_config,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            scheduler=self.sampler_name or "linear",
        )
        latents = reference["latents"] if reference and reference.get("latents") is not None else FluxLatentCreator.create_noise(seed, height, width)
        self._phase("generating", f"Stage 2/3: denoising {steps} steps")
        for t in config.time_steps:
            self._check_cancel()
            latents = config.scheduler.scale_model_input(latents, t)
            noise = transformer(
                t=t,
                config=config,
                hidden_states=latents,
                prompt_embeds=embeddings["prompt_embeds"],
                pooled_prompt_embeds=embeddings["pooled_prompt_embeds"],
            )
            latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
            mx.eval(latents)
            self._step(int(t))
        _eval(latents)
        del transformer, config
        self._phase("evicting_transformer", "Stage 2/3: evicting FLUX.1 DiT")
        _release_cache()
        return {"latents": latents}

    def decode(self, latents, width, height, meta):
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
        from mflux.models.flux.model.flux_vae.vae import VAE

        self._phase("loading_vae", "Stage 3/3: loading FLUX.1 VAE")
        unpacked = FluxLatentCreator.unpack_latents(latents, height, width)
        image = _decode_with_taef("taef1", unpacked) if getattr(self, "fast_vae", False) else None
        if image is None:
            vae = VAE()
            self.bits = _load_one(self.definition, "vae", self.root, self.quantization, vae) or self.bits
            decoded = VAEUtil.decode(vae=vae, latent=unpacked, tiling_config=_tiling_for(width, height))
            mx.eval(decoded)
            image = ImageUtil.to_pil(decoded)
            del vae, decoded
        del unpacked, latents
        self._phase("evicting_vae", "Stage 3/3: evicting FLUX.1 VAE")
        _release_cache()
        return image


class Flux3Adapter(_Adapter):
    reference_encode_is_deterministic = True
    loras_affect_text_encoder = False

    def __init__(self, model_id, model_path, quantization, loras, phase_cb=None):
        from mflux.models.flux3.flux3_initializer import Flux3Initializer

        super().__init__(model_id, model_path, quantization, loras, phase_cb)
        self.model_config = ModelConfig.flux3_image_klein()
        self.definition = Flux3Initializer.weight_definition(self.model_config)
        self.bits = quantization
        self.sampler_name = None

    def encode(self, prompt, negative_prompt, guidance):
        from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
        from mflux.models.flux3.model.flux3_text_encoder.prompt_encoder import Flux3PromptEncoder

        self._phase("loading_text_encoder", "Stage 1/3: loading FLUX.3 text encoder")
        t0 = time.perf_counter()
        encoder = Qwen3TextEncoder(**self.model_config.text_encoder_overrides)
        self.bits = _load_one(self.definition, "text_encoder", self.root, self.quantization, encoder) or self.quantization
        tok = _load_tokenizers(self.definition, self.root)["qwen3_vl"]
        text_tokens, text_ids = Flux3PromptEncoder.encode_prompt(prompt=prompt, tokenizer=tok, text_encoder=encoder)
        uncond_tokens = uncond_ids = None
        if not self.model_config.transformer_overrides.get("flow_map", True):
            uncond_tokens, uncond_ids = Flux3PromptEncoder.encode_prompt(prompt="", tokenizer=tok, text_encoder=encoder)
        _eval(text_tokens, text_ids, uncond_tokens, uncond_ids)
        load_time = time.perf_counter() - t0
        del encoder, tok
        self._phase("evicting_text_encoder", "Stage 1/3: evicting FLUX.3 text encoder")
        _release_cache()
        return {
            "text_tokens": text_tokens,
            "text_ids": text_ids,
            "uncond_tokens": uncond_tokens,
            "uncond_ids": uncond_ids,
            "load_time": load_time,
        }

    def encode_references(self, ref_paths, width, height, image_strength, seed, steps):
        from PIL import Image
        from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
        from mflux.models.flux3.latent_creator.flux3_latent_creator import Flux3LatentCreator
        from mflux.models.flux3.latent_creator.flux3_reference_images import Flux3ReferenceImages

        self._phase("loading_reference_vae", "Stage 1/4: loading FLUX.3 VAE for reference")
        t0 = time.perf_counter()
        vae = Flux2VAE()
        self.bits = _load_one(self.definition, "vae", self.root, self.quantization, vae) or self.bits
        images = [Image.open(path).convert("RGB") for path in ref_paths]
        references, _box = Flux3ReferenceImages.prepare(images)
        tokens, ids = [], []
        for index, reference_image in enumerate(references):
            image = mx.array(Flux3ReferenceImages.to_array(reference_image))
            encoded = Flux3LatentCreator.encode_tokens(
                vae, image, tiling_config=_tiling_for(width, height),
            )
            h, w = Flux3LatentCreator.latent_shape(reference_image.height, reference_image.width)
            tokens.append(encoded.astype(mx.bfloat16))
            ids.append(Flux3LatentCreator.prepare_ids(h, w, layer=-(1 + index)))
        cond_tokens = mx.concatenate(tokens, axis=1)
        cond_ids = mx.concatenate(ids, axis=0)
        _eval(cond_tokens, cond_ids)
        load_time = time.perf_counter() - t0
        del vae, images, references
        self._phase("evicting_reference_vae", "Stage 1/4: evicting FLUX.3 VAE")
        _release_cache()
        return {"cond_tokens": cond_tokens, "cond_ids": cond_ids, "load_time": load_time}

    def denoise(self, prompt, negative_prompt, width, height, steps, guidance, seed, embeddings, reference=None):
        from mflux.models.flux3.latent_creator.flux3_latent_creator import Flux3LatentCreator
        from mflux.models.flux3.model.flux3_scheduler.flux3_flow_map_scheduler import Flux3FlowMapScheduler
        from mflux.models.flux3.model.flux3_transformer.flux3_transformer import Flux3Transformer
        from mflux.models.flux3.variants.flux3_klein_core import _Flux3KleinCore

        self._phase("loading_transformer", "Stage 2/3: loading FLUX.3 DiT")
        _install_rope_hook("flux3-klein", extra_hook="flux3")
        transformer = Flux3Transformer(**self.model_config.transformer_overrides)
        self.bits = _load_one(self.definition, "transformer", self.root, self.quantization, transformer) or self.bits
        config = Config(
            model_config=self.model_config,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
        )
        h, w = Flux3LatentCreator.latent_shape(height, width)
        latents = Flux3LatentCreator.create_noise(seed, height, width)
        latent_ids = Flux3LatentCreator.prepare_ids(h, w)
        scheduler = Flux3FlowMapScheduler(num_inference_steps=config.num_inference_steps)
        predict = _Flux3KleinCore._predict(transformer)
        flow_map = transformer.flow_map
        self._phase("generating", f"Stage 2/3: denoising {steps} steps")
        for t in config.time_steps:
            self._check_cancel()
            velocity = predict(
                latents=latents,
                latent_ids=latent_ids,
                text_tokens=embeddings["text_tokens"],
                text_ids=embeddings["text_ids"],
                uncond_tokens=embeddings.get("uncond_tokens"),
                uncond_ids=embeddings.get("uncond_ids"),
                cond_tokens=reference.get("cond_tokens") if reference else None,
                cond_ids=reference.get("cond_ids") if reference else None,
                guidance=guidance,
                timestep=scheduler.timesteps[t:t + 1],
                target_timestep=scheduler.timesteps[t + 1:t + 2] if flow_map else None,
            )
            latents = scheduler.step(latents=latents, velocity=velocity, step=t)
            mx.eval(latents)
            self._step(int(t))
        _eval(latents)
        del predict, transformer, scheduler, config
        self._phase("evicting_transformer", "Stage 2/3: evicting FLUX.3 DiT")
        _release_cache()
        return {"latents": latents, "latent_height": h, "latent_width": w}

    def decode(self, latents, width, height, meta):
        from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
        from mflux.models.flux3.latent_creator.flux3_latent_creator import Flux3LatentCreator

        self._phase("loading_vae", "Stage 3/3: loading FLUX.3 VAE")
        vae = Flux2VAE()
        self.bits = _load_one(self.definition, "vae", self.root, self.quantization, vae) or self.bits
        decoded = Flux3LatentCreator.decode_tokens(
            vae, latents, meta["latent_height"], meta["latent_width"],
            tiling_config=_tiling_for(width, height),
        )
        mx.eval(decoded)
        image = ImageUtil.to_pil(decoded)
        del vae, decoded, latents
        self._phase("evicting_vae", "Stage 3/3: evicting FLUX.3 VAE")
        _release_cache()
        return image


class ZImageAdapter(_Adapter):
    loras_affect_text_encoder = False

    def __init__(self, model_id, model_path, quantization, loras, phase_cb=None):
        from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition

        super().__init__(model_id, model_path, quantization, loras, phase_cb)
        self.definition = ZImageWeightDefinition
        self.model_config = ModelConfig.z_image_turbo()
        self.bits = quantization
        self.sampler_name = None

    def encode(self, prompt, negative_prompt, guidance):
        from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
        from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder

        self._phase("loading_text_encoder", "Stage 1/3: loading Z-Image text encoder")
        t0 = time.perf_counter()
        encoder = TextEncoder()
        self.bits = _load_one(self.definition, "text_encoder", self.root, self.quantization, encoder) or self.quantization
        tok = _load_tokenizers(self.definition, self.root)["z_image"]
        text_encodings = PromptEncoder.encode_prompt(prompt=prompt, tokenizer=tok, text_encoder=encoder)
        negative_encodings = None
        if guidance is not None and guidance > 1.0:
            negative_encodings = PromptEncoder.encode_prompt(
                prompt=negative_prompt or " ", tokenizer=tok, text_encoder=encoder,
            )
        _eval(text_encodings, negative_encodings)
        load_time = time.perf_counter() - t0
        del encoder, tok
        self._phase("evicting_text_encoder", "Stage 1/3: evicting Z-Image text encoder")
        _release_cache()
        return {
            "text_encodings": text_encodings,
            "negative_encodings": negative_encodings,
            "load_time": load_time,
        }

    def encode_references(self, ref_paths, width, height, image_strength, seed, steps):
        from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
        from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
        from mflux.models.z_image.model.z_image_vae.vae import VAE

        self._phase("loading_reference_vae", "Stage 1/4: loading Z-Image VAE for reference")
        t0 = time.perf_counter()
        vae = VAE()
        self.bits = _load_one(self.definition, "vae", self.root, self.quantization, vae) or self.bits
        config = Config(
            model_config=self.model_config,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=1.0,
            image_path=ref_paths[0],
            image_strength=image_strength if image_strength is not None else 0.6,
            scheduler=self.sampler_name or "linear",
        )
        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            width=width,
            height=height,
            img2img=Img2Img(
                vae=vae,
                latent_creator=ZImageLatentCreator,
                image_path=config.image_path,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
                tiling_config=_tiling_for(width, height),
            ),
        )
        _eval(latents)
        load_time = time.perf_counter() - t0
        del vae
        self._phase("evicting_reference_vae", "Stage 1/4: evicting Z-Image VAE")
        _release_cache()
        return {"latents": latents, "load_time": load_time}

    def denoise(self, prompt, negative_prompt, width, height, steps, guidance, seed, embeddings, reference=None):
        from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
        from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
        from mflux.models.z_image.variants.z_image import ZImage
        from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping

        self._phase("loading_transformer", "Stage 2/3: loading Z-Image DiT")
        _install_rope_hook("z-image-turbo")
        transformer = ZImageTransformer()
        self.bits = _load_one(self.definition, "transformer", self.root, self.quantization, transformer) or self.bits
        self._apply_transformer_lora(transformer, ZImageLoRAMapping.get_mapping())
        config = Config(
            model_config=self.model_config,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            scheduler=self.sampler_name or "linear",
        )
        latents = reference["latents"] if reference and reference.get("latents") is not None else ZImageLatentCreator.create_noise(seed, height, width)
        predict = ZImage._predict(transformer)
        self._phase("generating", f"Stage 2/3: denoising {steps} steps")
        for t in config.time_steps:
            self._check_cancel()
            sigma_t = config.scheduler.sigmas[t].reshape((1,))
            timestep = mx.ones_like(sigma_t) - sigma_t
            noise = predict(
                latents=latents,
                timestep=timestep,
                sigmas=config.scheduler.sigmas,
                text_encodings=embeddings["text_encodings"],
                negative_encodings=embeddings.get("negative_encodings"),
                guidance=config.guidance,
            )
            latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
            mx.eval(latents)
            self._step(int(t))
        _eval(latents)
        del predict, transformer, config
        self._phase("evicting_transformer", "Stage 2/3: evicting Z-Image DiT")
        _release_cache()
        return {"latents": latents}

    def decode(self, latents, width, height, meta):
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
        from mflux.models.z_image.model.z_image_vae.vae import VAE

        self._phase("loading_vae", "Stage 3/3: loading Z-Image VAE")
        unpacked = ZImageLatentCreator.unpack_latents(latents, height, width)
        image = _decode_with_taef("taef1", unpacked) if getattr(self, "fast_vae", False) else None
        if image is None:
            vae = VAE()
            self.bits = _load_one(self.definition, "vae", self.root, self.quantization, vae) or self.bits
            decoded = VAEUtil.decode(vae=vae, latent=unpacked, tiling_config=_tiling_for(width, height))
            mx.eval(decoded)
            image = ImageUtil.to_pil(decoded)
            del vae, decoded
        del unpacked, latents
        self._phase("evicting_vae", "Stage 3/3: evicting Z-Image VAE")
        _release_cache()
        return image




class QwenEditAdapter(_Adapter):
    reference_encode_is_deterministic = True
    # encode()/denoise() read these, so a reference cache hit must restore them.
    reference_state_attrs = ("_ref_paths", "_vl_width", "_vl_height", "_cond_h", "_cond_w", "_num_images")
    loras_affect_text_encoder = False

    def __init__(self, model_id, model_path, quantization, loras, phase_cb=None):
        from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

        super().__init__(model_id, model_path, quantization, loras, phase_cb)
        self.definition = QwenWeightDefinition
        self.model_config = ModelConfig.qwen_image_edit()
        self.bits = quantization
        self.sampler_name = None
        self._ref_paths = []
        self._vl_width = None
        self._vl_height = None
        self._cond_h = None
        self._cond_w = None
        self._num_images = 1
        self.use_kv_cache = False

    def encode_references(self, ref_paths, width, height, image_strength, seed, steps):
        from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
        from mflux.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil
        from mflux.utils.image_util import ImageUtil

        self._ref_paths = list(ref_paths)
        image = ImageUtil.load_image(self._ref_paths[0]).convert("RGB")
        ratio = image.width / image.height
        vl_width = round(math.sqrt(384 * 384 * ratio) / 32) * 32
        vl_height = round((math.sqrt(384 * 384 * ratio) / ratio) / 32) * 32
        self._vl_width, self._vl_height = int(vl_width), int(vl_height)

        self._phase("loading_reference_vae", "Stage 1/4: loading Qwen-Image-Edit VAE")
        t0 = time.perf_counter()
        vae = QwenVAE()
        self.bits = _load_one(self.definition, "vae", self.root, self.quantization, vae) or self.bits
        image_latents, qwen_image_ids, cond_h, cond_w, num_images = QwenEditUtil.create_image_conditioning_latents(
            vae=vae,
            width=width,
            height=height,
            image_paths=self._ref_paths,
            tiling_config=_tiling_for(width, height),
        )
        self._cond_h, self._cond_w, self._num_images = cond_h, cond_w, num_images
        _eval(image_latents, qwen_image_ids)
        load_time = time.perf_counter() - t0
        del vae
        self._phase("evicting_reference_vae", "Stage 1/4: evicting Qwen-Image-Edit VAE")
        _release_cache()
        return {
            "image_latents": image_latents,
            "qwen_image_ids": qwen_image_ids,
            "cond_h": cond_h,
            "cond_w": cond_w,
            "num_images": num_images,
            "load_time": load_time,
        }

    def encode(self, prompt, negative_prompt, guidance):
        from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
        from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_language_encoder import QwenVisionLanguageEncoder
        from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_transformer import VisionTransformer
        from mflux.models.qwen.tokenizer.qwen_vision_language_processor import QwenVisionLanguageProcessor
        from mflux.models.qwen.tokenizer.qwen_vision_language_tokenizer import QwenVisionLanguageTokenizer

        self._phase("loading_text_encoder", "Stage 2/4: loading Qwen-Image-Edit VLM encoder")
        t0 = time.perf_counter()
        encoder = QwenTextEncoder()
        encoder.encoder.visual = VisionTransformer()
        self.bits = _load_one(self.definition, "text_encoder", self.root, self.quantization, encoder) or self.quantization
        base_tok = _load_tokenizers(self.definition, self.root)["qwen"]
        vl_tok = QwenVisionLanguageTokenizer(
            processor=QwenVisionLanguageProcessor(tokenizer=base_tok.tokenizer),
            max_length=1024,
            use_picture_prefix=False,
        )
        vl_encoder = QwenVisionLanguageEncoder(encoder=encoder.encoder)
        ids, mask, pixels, grid = vl_tok.tokenize_with_image(
            prompt, self._ref_paths, vl_width=self._vl_width, vl_height=self._vl_height,
        )
        hidden = vl_encoder(
            input_ids=ids, attention_mask=mask, pixel_values=pixels, image_grid_thw=grid,
        )
        prompt_embeds = hidden[0].astype(mx.float16)
        prompt_mask = hidden[1].astype(mx.float16)
        negative_embeds = negative_mask = None
        if guidance is not None and guidance > 1.0 and negative_prompt:
            nids, nmask, npix, ngrid = vl_tok.tokenize_with_image(
                negative_prompt, self._ref_paths, vl_width=self._vl_width, vl_height=self._vl_height,
            )
            nhidden = vl_encoder(
                input_ids=nids, attention_mask=nmask, pixel_values=npix, image_grid_thw=ngrid,
            )
            negative_embeds = nhidden[0].astype(mx.float16)
            negative_mask = nhidden[1].astype(mx.float16)
        _eval(prompt_embeds, prompt_mask, negative_embeds, negative_mask)
        load_time = time.perf_counter() - t0
        del encoder, vl_encoder, base_tok, vl_tok
        self._phase("evicting_text_encoder", "Stage 2/4: evicting Qwen-Image-Edit VLM encoder")
        _release_cache()
        return {
            "prompt_embeds": prompt_embeds,
            "prompt_mask": prompt_mask,
            "negative_prompt_embeds": negative_embeds,
            "negative_prompt_mask": negative_mask,
            "load_time": load_time,
        }

    def denoise(self, prompt, negative_prompt, width, height, steps, guidance, seed, embeddings, reference=None):
        from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
        from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
        from mflux.models.qwen.weights.qwen_lora_mapping import QwenLoRAMapping

        self._phase("loading_transformer", "Stage 3/4: loading Qwen-Image-Edit DiT")
        transformer = QwenTransformer()
        self.bits = _load_one(self.definition, "transformer", self.root, self.quantization, transformer) or self.bits
        self._apply_transformer_lora(transformer, QwenLoRAMapping.get_mapping())
        config = Config(
            model_config=self.model_config,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=self._ref_paths[0] if self._ref_paths else None,
            scheduler=self.sampler_name or "linear",
        )
        latents = QwenLatentCreator.create_noise(seed=seed, width=width, height=height)
        image_latents = reference["image_latents"]
        qwen_image_ids = reference["qwen_image_ids"]
        cond_h, cond_w = reference["cond_h"], reference["cond_w"]
        num_images = reference["num_images"]
        cond_grid = (1, cond_h, cond_w)
        if num_images > 1:
            cond_grid = [(1, cond_h, cond_w) for _ in range(num_images)]
        self._phase("generating", f"Stage 3/4: denoising {steps} steps with reference")
        for t in range(len(config.scheduler.timesteps)):
            self._check_cancel()
            hidden_states = mx.concatenate([latents, image_latents], axis=1)
            noise = transformer(
                t=t,
                config=config,
                hidden_states=hidden_states,
                encoder_hidden_states=embeddings["prompt_embeds"],
                encoder_hidden_states_mask=embeddings["prompt_mask"],
                qwen_image_ids=qwen_image_ids,
                cond_image_grid=cond_grid,
            )[:, : latents.shape[1]]
            if config.guidance > 1.0 and embeddings.get("negative_prompt_embeds") is not None:
                noise_negative = transformer(
                    t=t,
                    config=config,
                    hidden_states=hidden_states,
                    encoder_hidden_states=embeddings["negative_prompt_embeds"],
                    encoder_hidden_states_mask=embeddings["negative_prompt_mask"],
                    qwen_image_ids=qwen_image_ids,
                    cond_image_grid=cond_grid,
                )[:, : latents.shape[1]]
                noise = QwenImage.compute_guided_noise(noise, noise_negative, config.guidance)
            latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
            mx.eval(latents)
            self._step(int(t))
        _eval(latents)
        del transformer, config
        self._phase("evicting_transformer", "Stage 3/4: evicting Qwen-Image-Edit DiT")
        _release_cache()
        return {"latents": latents}

    def decode(self, latents, width, height, meta):
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
        from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE

        self._phase("loading_vae", "Stage 4/4: loading Qwen-Image-Edit VAE")
        vae = QwenVAE()
        self.bits = _load_one(self.definition, "vae", self.root, self.quantization, vae) or self.bits
        unpacked = QwenLatentCreator.unpack_latents(latents=latents, height=height, width=width)
        decoded = VAEUtil.decode(vae=vae, latent=unpacked, tiling_config=_tiling_for(width, height))
        mx.eval(decoded)
        image = ImageUtil.to_pil(decoded)
        del vae, decoded, unpacked, latents
        self._phase("evicting_vae", "Stage 4/4: evicting Qwen-Image-Edit VAE")
        _release_cache()
        return image


def _make_adapter(model_id, model_path, quantization, loras, phase_cb, progress_cb=None, cancel_check=None):
    if model_id == "flux1-schnell":
        adapter = Flux1Adapter(model_id, model_path, quantization, loras, phase_cb)
    elif model_id in ("flux2-klein-4b", "flux2-klein-9b"):
        adapter = Flux2Adapter(model_id, model_path, quantization, loras, phase_cb)
    elif model_id == "flux3-klein":
        adapter = Flux3Adapter(model_id, model_path, quantization, loras, phase_cb)
    elif model_id == "krea2-turbo":
        adapter = Krea2Adapter(model_id, model_path, quantization, loras, phase_cb)
    elif model_id in ("qwen-image-2.1", "qwen-image-2.1-turbo"):
        adapter = Qwen21Adapter(model_id, model_path, quantization, loras, phase_cb)
    elif model_id == "qwen-image-edit":
        adapter = QwenEditAdapter(model_id, model_path, quantization, loras, phase_cb)
    elif model_id == "z-image-turbo":
        adapter = ZImageAdapter(model_id, model_path, quantization, loras, phase_cb)
    else:
        raise ValueError(f"no staged mflux adapter for {model_id}")
    adapter.progress_cb = progress_cb
    adapter.cancel_check = cancel_check
    return adapter


def generate_staged(
    *,
    model_id: str,
    model_path: str | Path,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    quantization: int = 4,
    loras: list[dict] | None = None,
    negative_prompt: str = "",
    sampler: str | None = None,
    ref_paths: list[str] | None = None,
    image_strength: float | None = None,
    use_kv_cache: bool = True,
    fast_vae: bool = False,
    phase_cb: Callable | None = None,
    progress_cb: Callable | None = None,
    cancel_check: Callable | None = None,
) -> StageResult:
    """Run one generation with exactly one component tree resident."""
    adapter = _make_adapter(model_id, model_path, quantization, loras, phase_cb, progress_cb, cancel_check)
    adapter.progress_cb = progress_cb
    adapter.cancel_check = cancel_check
    adapter.sampler_name = sampler
    adapter.use_kv_cache = bool(use_kv_cache)
    adapter.fast_vae = bool(fast_vae)
    stage_times: dict[str, float] = {}
    stage_memory: dict[str, dict[str, float]] = {}
    reference = None
    prev_cache_limit = apply_mlx_cache_limit()
    try:
        mx.reset_peak_memory()
    except Exception:
        pass

    def _check_cancel():
        if cancel_check is not None:
            cancel_check()

    try:
        _check_cancel()
        if ref_paths:
            ref_key = _reference_cache_key(
                adapter, model_id, ref_paths, width, height, image_strength, seed, steps,
            )
            cached_ref = _REFERENCE_CACHE.get(ref_key) if ref_key is not None else None
            if cached_ref is not None:
                _REFERENCE_CACHE.move_to_end(ref_key)
                reference = dict(cached_ref["reference"])
                for attr, value in cached_ref["state"].items():
                    setattr(adapter, attr, list(value) if isinstance(value, list) else value)
                stage_times["reference_vae"] = 0.0
                reference_load = 0.0
            else:
                t0 = time.perf_counter()
                reference = adapter.encode_references(
                    ref_paths, width, height, image_strength, seed, steps,
                )
                stage_times["reference_vae"] = time.perf_counter() - t0
                reference_load = float(reference.pop("load_time", stage_times["reference_vae"]))
                if ref_key is not None:
                    _REFERENCE_CACHE[ref_key] = {
                        "reference": dict(reference),
                        "state": {
                            attr: getattr(adapter, attr)
                            for attr in getattr(adapter, "reference_state_attrs", ())
                        },
                    }
                    while len(_REFERENCE_CACHE) > _REFERENCE_CACHE_MAX:
                        _REFERENCE_CACHE.popitem(last=False)
                stage_memory["reference_vae"] = _stage_memory()
        else:
            reference_load = 0.0

        use_cache = isinstance(adapter, _Adapter)
        uses_vlm_refs = bool(ref_paths) and model_id == "qwen-image-edit"
        cache_key = None
        if use_cache:
            lora_key = (
                tuple(zip(adapter.lora_paths, adapter.lora_scales))
                if adapter.loras_affect_text_encoder else ()
            )
            ref_key = tuple(str(p) for p in ref_paths) if uses_vlm_refs else ()
            size_key = (int(width), int(height)) if uses_vlm_refs else ()
            cache_key = (
                str(model_id),
                str(adapter.root),
                int(quantization),
                str(prompt),
                str(negative_prompt or ""),
                float(guidance or 0.0),
                lora_key,
                ref_key,
                size_key,
            )

        cached_entry = _STAGED_PROMPT_CACHE.get(cache_key) if cache_key is not None else None
        if cached_entry is not None:
            _STAGED_PROMPT_CACHE.move_to_end(cache_key)
            cached_embeds, cached_bits = cached_entry
            embeddings = dict(cached_embeds)
            adapter.bits = cached_bits
            stage_times["text_encoder"] = 0.0
            load_time = reference_load
        else:
            disk_cache = None
            disk_key = None
            disk_hit = None
            if cache_key is not None and not uses_vlm_refs:
                try:
                    from prompt_embedding_cache import PromptEmbeddingCache

                    disk_cache = PromptEmbeddingCache()
                    if not disk_cache.disabled:
                        disk_key = disk_cache.key(
                            engine="staged_mflux",
                            model_fingerprint=f"{model_id}:{adapter.root}:{int(quantization)}:mflux-{_vendor_version()}",
                            prompt=prompt,
                            negative_prompt=negative_prompt or "",
                            clip_skip=2,
                            te_signature=[f"{p}:{s}" for p, s in lora_key],
                            extra={"guidance": float(guidance or 0.0)},
                        )
                        raw_disk = disk_cache.get(disk_key)
                        if raw_disk is not None:
                            disk_hit = _staged_embeds_from_numpy(raw_disk)
                except Exception:
                    disk_hit = None

            if disk_hit is not None:
                disk_embeds, disk_bits = disk_hit
                embeddings = dict(disk_embeds)
                adapter.bits = disk_bits
                stage_times["text_encoder"] = 0.0
                load_time = reference_load
                _staged_cache_put(cache_key, embeddings, disk_bits)
            else:
                _check_cancel()
                t0 = time.perf_counter()
                embeddings = adapter.encode(prompt, negative_prompt, guidance)
                stage_times["text_encoder"] = time.perf_counter() - t0
                stage_memory["text_encoder"] = _stage_memory()
                te_load = float(embeddings.pop("load_time", stage_times["text_encoder"]))
                load_time = reference_load + te_load
                bits_used = int(getattr(adapter, "bits", quantization) or quantization)
                if cache_key is not None:
                    _staged_cache_put(cache_key, embeddings, bits_used)
                    if disk_cache is not None and disk_key is not None:
                        np_payload = _staged_embeds_to_numpy(embeddings, bits_used)
                        if np_payload is not None:
                            disk_cache.put(disk_key, np_payload)

        _check_cancel()
        t0 = time.perf_counter()
        with _compile_disabled():
            latents = adapter.denoise(
                prompt, negative_prompt, width, height, steps, guidance, seed, embeddings,
                reference=reference,
            )
        stage_times["denoise"] = time.perf_counter() - t0
        stage_memory["denoise"] = _stage_memory()
        del embeddings
        _release_cache()

        _check_cancel()
        t0 = time.perf_counter()
        image = adapter.decode(latents["latents"], width, height, latents)
        stage_times["vae"] = time.perf_counter() - t0
        stage_memory["vae"] = _stage_memory()
    except BaseException:
        _release_cache()
        raise
    finally:
        restore_mlx_cache_limit(prev_cache_limit)

    return StageResult(
        image=image,
        model_config=adapter.model_config,
        quantization=int(adapter.bits or quantization),
        seed=seed,
        prompt=prompt,
        steps=steps,
        guidance=guidance,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        lora_paths=adapter.lora_paths,
        lora_scales=adapter.lora_scales,
        load_time=load_time,
        inference_time=sum(
            value for key, value in stage_times.items()
            if key in ("reference_vae", "denoise", "vae")
        ),
        stage_times=stage_times,
        stage_memory=stage_memory,
    )


def to_generated_image(result: StageResult) -> GeneratedImage:
    """Convert the stage result into mflux's GeneratedImage contract."""
    generated = GeneratedImage(
        image=result.image,
        model_config=result.model_config,
        seed=result.seed,
        prompt=result.prompt,
        steps=result.steps,
        guidance=result.guidance,
        precision=getattr(ModelConfig, "precision", mx.bfloat16),
        quantization=result.quantization,
        generation_time=result.inference_time,
        lora_paths=result.lora_paths,
        lora_scales=result.lora_scales,
        height=result.height,
        width=result.width,
        negative_prompt=result.negative_prompt,
    )
    generated.load_time = result.load_time
    generated.inference_time = result.inference_time
    generated.stage_times = result.stage_times
    generated.stage_memory = result.stage_memory
    return generated
