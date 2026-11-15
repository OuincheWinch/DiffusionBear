"""Central home for runtime patches applied to the vendored mflux runtime.

The patch functions used to live in ``generator.py``. They are moved here so the
mflux surface we depend on is one module, with a registry that names the model
families and the vendor release each patch was written against. ``generator``
keeps thin wrappers for compatibility; no behaviour changed in the move.

Every installer is a monkey-patch on a vendored class, so each one either guards
itself with a class-level flag (idempotent) or applies a plain assignment that a
second call harmlessly repeats. Functions that need generator state read it from
the module at call time, which keeps the existing tests that monkeypatch
``generator`` working unchanged.
"""

import importlib
import math
import os


def _vendor_version(package: str = "mflux") -> str:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return "unknown"


MFLUX_VENDOR_VERSION = _vendor_version("mflux")

# Original vendor callables and the wrappers we installed, captured before the
# first install so the patch can be rolled back without clobbering a foreign
# installer. Keyed by stepper name; the restore function pops each entry.
_KREA_SAMPLER_ORIGINALS: dict[str, dict[str, object]] = {}


def _generator():
    import generator

    return generator


def apply_prompt_cache(
    pipe,
    lora_signature=None,
    quantization=None,
    model_fingerprint=None,
    te_signature=None,
):
    """Cache Flux2 text-encoder output with the bounded memory LRU in front.

    The disk key includes model id, prompt, negative prompt, LoRA signature,
    guidance, engine version, quantization, model fingerprint and text-encoder
    identity, so a pipeline rebuild or restart can reuse the embeddings without
    an encode pass.
    """
    import mlx.core as mx

    gen = _generator()
    orig_encode = pipe._encode_prompt_pair
    signature = (
        gen._prompt_cache_lora_signature(pipe)
        if lora_signature is None
        else tuple(lora_signature)
    )
    version = gen._prompt_cache_version()
    quant = (
        gen._prompt_cache_quantization(pipe)
        if quantization is None
        else int(quantization)
    )
    fingerprint = model_fingerprint
    te_signature = (
        gen._prompt_cache_te_signature(pipe)
        if te_signature is None
        else tuple(te_signature)
    )

    def cached_encode(*, prompt, negative_prompt, guidance):
        key = gen._prompt_cache_key(
            pipe, prompt, negative_prompt, guidance, signature, version,
            quant, fingerprint, te_signature,
        )
        cached = gen._get_prompt_cache(key)
        if cached is not None:
            return cached
        cached = gen._PROMPT_DISK.get(key)
        if cached is not None:
            gen._cache_prompt(key, cached)
            return cached

        value = orig_encode(
            prompt=prompt, negative_prompt=negative_prompt, guidance=guidance
        )
        for item in value:
            if item is not None:
                mx.eval(item)
        gen._cache_prompt(key, value)
        gen._PROMPT_DISK[key] = value
        return value

    if not hasattr(pipe, "_mlxdiffusion_original_encode_prompt_pair"):
        pipe._mlxdiffusion_original_encode_prompt_pair = orig_encode
    pipe._mlxdiffusion_patched_encode_prompt_pair = cached_encode
    pipe._encode_prompt_pair = cached_encode


def restore_prompt_cache(pipe) -> bool:
    """Restore the pipe's original ``_encode_prompt_pair`` if we still own it."""
    original = getattr(pipe, "_mlxdiffusion_original_encode_prompt_pair", None)
    patched = getattr(pipe, "_mlxdiffusion_patched_encode_prompt_pair", None)
    if original is None or patched is None or pipe._encode_prompt_pair is not patched:
        return False
    pipe._encode_prompt_pair = original
    for attr in (
        "_mlxdiffusion_original_encode_prompt_pair",
        "_mlxdiffusion_patched_encode_prompt_pair",
    ):
        try:
            delattr(pipe, attr)
        except AttributeError:
            pass
    return True


def apply_prompt_cache_txt(
    pipe,
    lora_signature=None,
    quantization=None,
    model_fingerprint=None,
    te_signature=None,
):
    """Shared LRU for Z-Image / Krea2 text encoders with disk persistence.

    Both mflux engines expose ``_encode_prompts(prompt, negative_prompt,
    guidance) -> (embeds, neg_embeds)`` and re-encode on every generate_image
    call even though the pipe (with its TextEncoder) is cached in memory.
    prompt/negative/guidance fully determine the embeddings, so memoizing the
    raw arrays (never mutated) is bit-identical.
    """
    import mlx.core as mx

    gen = _generator()
    orig_encode = pipe._encode_prompts
    signature = (
        gen._prompt_cache_lora_signature(pipe)
        if lora_signature is None
        else tuple(lora_signature)
    )
    version = gen._prompt_cache_version()
    quant = (
        gen._prompt_cache_quantization(pipe)
        if quantization is None
        else int(quantization)
    )
    fingerprint = model_fingerprint
    te_signature = (
        gen._prompt_cache_te_signature(pipe)
        if te_signature is None
        else tuple(te_signature)
    )

    def cached_encode(*, prompt, negative_prompt=None, guidance=1.0):
        key = gen._prompt_cache_key(
            pipe, prompt, negative_prompt, guidance, signature, version,
            quant, fingerprint, te_signature,
        )
        cached = gen._get_prompt_cache(key)
        if cached is not None:
            return cached
        cached = gen._PROMPT_DISK.get(key)
        if cached is not None:
            gen._cache_prompt(key, cached)
            return cached

        value = orig_encode(
            prompt=prompt, negative_prompt=negative_prompt, guidance=guidance
        )
        for item in value:
            if item is not None:
                mx.eval(item)
        gen._cache_prompt(key, value)
        gen._PROMPT_DISK[key] = value
        return value

    if not hasattr(pipe, "_mlxdiffusion_original_encode_prompts"):
        pipe._mlxdiffusion_original_encode_prompts = orig_encode
    pipe._mlxdiffusion_patched_encode_prompts = cached_encode
    pipe._encode_prompts = cached_encode


def restore_prompt_cache_txt(pipe) -> bool:
    """Restore the pipe's original ``_encode_prompts`` if we still own it."""
    original = getattr(pipe, "_mlxdiffusion_original_encode_prompts", None)
    patched = getattr(pipe, "_mlxdiffusion_patched_encode_prompts", None)
    if original is None or patched is None or pipe._encode_prompts is not patched:
        return False
    pipe._encode_prompts = original
    for attr in (
        "_mlxdiffusion_original_encode_prompts",
        "_mlxdiffusion_patched_encode_prompts",
    ):
        try:
            delattr(pipe, attr)
        except AttributeError:
            pass
    return True


def install_flux_text_encoder_tail_trim(pipe):
    if os.environ.get("MLX_DISABLE_FLUX_TE_TRIM") == "1":
        return
    text_encoder = getattr(pipe, "text_encoder", None)
    layers = getattr(text_encoder, "layers", None)
    if text_encoder is None or layers is None or len(layers) != 36:
        return
    if getattr(text_encoder, "num_hidden_layers", None) != 36:
        return
    # MLX's Module.__setattr__ routes lists into the module dict, where
    # `named_modules()`/`parameters()` would walk the trimmed layers a second
    # time and hold them in the model tree. Keep rollback state in the object
    # __dict__ instead so the dropped layers leave the module graph.
    if not hasattr(text_encoder, "_mlxdiffusion_original_layers"):
        object.__setattr__(text_encoder, "_mlxdiffusion_original_layers", layers)
        object.__setattr__(
            text_encoder,
            "_mlxdiffusion_original_num_hidden_layers",
            text_encoder.num_hidden_layers,
        )
    trimmed = layers[:28]
    text_encoder.layers = trimmed
    object.__setattr__(text_encoder, "_mlxdiffusion_patched_layers", trimmed)
    text_encoder.num_hidden_layers = 28
    text_encoder._mlxdiffusion_te_tail_trimmed = True


def restore_flux_text_encoder_tail_trim(pipe) -> bool:
    """Restore the full 36-layer FLUX text encoder on one pipeline.

    Acts only while the pipeline still holds the exact trimmed list this module
    installed; a foreign layer list is left untouched.
    """
    text_encoder = getattr(pipe, "text_encoder", None)
    original = getattr(text_encoder, "_mlxdiffusion_original_layers", None)
    patched = getattr(text_encoder, "_mlxdiffusion_patched_layers", None)
    if original is None or patched is None or text_encoder.layers is not patched:
        return False
    text_encoder.layers = original
    text_encoder.num_hidden_layers = getattr(
        text_encoder, "_mlxdiffusion_original_num_hidden_layers", 36
    )
    for attr in (
        "_mlxdiffusion_original_layers",
        "_mlxdiffusion_patched_layers",
        "_mlxdiffusion_original_num_hidden_layers",
        "_mlxdiffusion_te_tail_trimmed",
    ):
        try:
            object.__delattr__(text_encoder, attr)
        except AttributeError:
            pass
    return True


def install_krea_sampler_scalar_cache():
    if os.environ.get("MLX_DISABLE_KREA_SIGMA_CACHE") == "1":
        return
    try:
        import mlx.core as mx
        from mflux.models.krea2.model.krea2_sampler import ErSdeStepper, EulerStepper

        if not getattr(EulerStepper, "_mlxdiffusion_compiled_step", False):
            entry = _KREA_SAMPLER_ORIGINALS.setdefault("euler", {})
            entry.setdefault("original", EulerStepper.step)

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
            entry["patched"] = _fast_krea_euler_step
            EulerStepper._mlxdiffusion_compiled_step = True

        if getattr(ErSdeStepper, "_mlxdiffusion_sigma_cache", False):
            return

        entry = _KREA_SAMPLER_ORIGINALS.setdefault("er_sde", {})
        entry.setdefault("original", ErSdeStepper.step)

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
        entry["patched"] = step
        ErSdeStepper._mlxdiffusion_sigma_cache = True
    except Exception:
        return


def restore_krea_sampler_scalar_cache() -> bool:
    """Roll back ``install_krea_sampler_scalar_cache``.

    Each stepper is restored only when the class still holds the wrapper this
    module installed; a foreign or later installer is left untouched. The
    matching class flag is deleted only with that stepper, and a call before
    any install (no captured original) is a no-op. Returns True when at least
    one callable was restored.
    """
    try:
        from mflux.models.krea2.model.krea2_sampler import ErSdeStepper, EulerStepper
    except Exception:
        return False

    restored = False
    for key, cls, flag in (
        ("euler", EulerStepper, "_mlxdiffusion_compiled_step"),
        ("er_sde", ErSdeStepper, "_mlxdiffusion_sigma_cache"),
    ):
        entry = _KREA_SAMPLER_ORIGINALS.get(key)
        if not entry:
            continue
        original = entry.get("original")
        patched = entry.get("patched")
        if original is None or patched is None:
            continue
        if cls.step is not patched:
            continue
        cls.step = original
        try:
            delattr(cls, flag)
        except AttributeError:
            pass
        _KREA_SAMPLER_ORIGINALS.pop(key, None)
        restored = True

    return restored


def install_krea_euler_a_trailing():
    """Add the Draw Things-style Euler A Trailing sampler to Krea 2.

    The upstream Krea2Sampler only knows deterministic Euler and ErSde. Euler A
    Trailing is an ancestral Euler with trailing flow scheduling; this stepper
    follows ComfyUI's ancestral Euler math (sigma_up/sigma_down) so the sampler
    is usable and reproducible in this stack.
    """
    try:
        import mlx.core as mx
        from mflux.models.krea2.model.krea2_sampler import Krea2Sampler
        from mflux.models.krea2.variants.txt2img.krea2 import Krea2
    except Exception:
        return
    if getattr(Krea2Sampler, "_mlxdiffusion_euler_a_trailing", False):
        return

    class EulerATrailingStepper:
        def __init__(self, sigmas, seed: int, s_noise: float = 1.0):
            self.sigmas = sigmas
            self.s_noise = s_noise
            self._key = mx.random.key(int(seed) ^ 0xA7E)

        def step(self, i: int, x: mx.array, v: mx.array, denoised: mx.array) -> mx.array:
            s0 = float(self.sigmas[i])
            s1 = float(self.sigmas[i + 1])
            if s1 <= 1e-6:
                return x + (s1 - s0) * v
            ratio = min(1.0, max(0.0, s1 / s0)) if s0 > 0 else 0.0
            sigma_up = s1 * math.sqrt(max(0.0, 1.0 - ratio * ratio))
            sigma_down = s1 * ratio
            x = x + (sigma_down - s0) * v
            self._key, sub = mx.random.split(self._key)
            noise = mx.random.normal(x.shape, key=sub)
            return x + noise * (sigma_up * self.s_noise)

    original_make = Krea2Sampler.make_stepper
    original_resolve = Krea2._resolve_scheduler

    def make_stepper(name: str, sigmas, seed: int):
        if name in ("euler_a_trailing", "euler_trailing"):
            return EulerATrailingStepper(sigmas, seed)
        return original_make(name, sigmas, seed)

    def resolve_scheduler(scheduler: str | None) -> str:
        if scheduler in ("euler_a_trailing", "euler_trailing"):
            return "euler_a_trailing"
        return original_resolve(scheduler)

    Krea2Sampler._mlxdiffusion_original_make_stepper = original_make
    Krea2Sampler._mlxdiffusion_patched_make_stepper = make_stepper
    Krea2._mlxdiffusion_original_resolve_scheduler = original_resolve
    Krea2._mlxdiffusion_patched_resolve_scheduler = resolve_scheduler
    Krea2Sampler.make_stepper = staticmethod(make_stepper)
    Krea2._resolve_scheduler = staticmethod(resolve_scheduler)
    Krea2Sampler._mlxdiffusion_euler_a_trailing = True


def restore_krea_euler_a_trailing() -> bool:
    """Remove the Euler A Trailing sampler patch if this module still owns it."""
    try:
        from mflux.models.krea2.model.krea2_sampler import Krea2Sampler
        from mflux.models.krea2.variants.txt2img.krea2 import Krea2
    except Exception:
        return False

    restored = False
    make_original = getattr(
        Krea2Sampler, "_mlxdiffusion_original_make_stepper", None
    )
    make_patched = getattr(Krea2Sampler, "_mlxdiffusion_patched_make_stepper", None)
    if (
        make_original is not None
        and make_patched is not None
        and Krea2Sampler.make_stepper is make_patched
    ):
        Krea2Sampler.make_stepper = staticmethod(make_original)
        restored = True
    resolve_original = getattr(
        Krea2, "_mlxdiffusion_original_resolve_scheduler", None
    )
    resolve_patched = getattr(
        Krea2, "_mlxdiffusion_patched_resolve_scheduler", None
    )
    if (
        resolve_original is not None
        and resolve_patched is not None
        and Krea2._resolve_scheduler is resolve_patched
    ):
        Krea2._resolve_scheduler = staticmethod(resolve_original)
        restored = True

    for cls, attrs in (
        (
            Krea2Sampler,
            (
                "_mlxdiffusion_original_make_stepper",
                "_mlxdiffusion_patched_make_stepper",
                "_mlxdiffusion_euler_a_trailing",
            ),
        ),
        (
            Krea2,
            (
                "_mlxdiffusion_original_resolve_scheduler",
                "_mlxdiffusion_patched_resolve_scheduler",
            ),
        ),
    ):
        for attr in attrs:
            try:
                delattr(cls, attr)
            except AttributeError:
                pass
    return restored


def install_zit_dynamic_padding(pipe):
    if os.environ.get("MLX_DISABLE_ZIT_DYNAMIC_PADDING") == "1":
        return
    tokenizers = getattr(pipe, "tokenizers", None)
    tokenizer = tokenizers.get("z_image") if hasattr(tokenizers, "get") else None
    if tokenizer is not None and getattr(tokenizer, "padding", None) == "max_length":
        tokenizer._mlxdiffusion_original_padding = tokenizer.padding
        tokenizer._mlxdiffusion_patched_padding = "longest"
        tokenizer.padding = "longest"


def restore_zit_dynamic_padding(pipe) -> bool:
    """Restore the z_image tokenizer padding only while this module owns it."""
    tokenizers = getattr(pipe, "tokenizers", None)
    tokenizer = tokenizers.get("z_image") if hasattr(tokenizers, "get") else None
    original = getattr(tokenizer, "_mlxdiffusion_original_padding", None)
    patched = getattr(tokenizer, "_mlxdiffusion_patched_padding", None)
    if (
        original is None
        or patched is None
        or getattr(tokenizer, "padding", None) != patched
    ):
        return False
    tokenizer.padding = original
    for attr in (
        "_mlxdiffusion_original_padding",
        "_mlxdiffusion_patched_padding",
    ):
        try:
            delattr(tokenizer, attr)
        except AttributeError:
            pass
    return True


def install_flux3_compiled_scheduler_step():
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

        Flux3FlowMapScheduler._mlxdiffusion_original_step = Flux3FlowMapScheduler.step
        Flux3FlowMapScheduler._mlxdiffusion_patched_step = _fast_step
        Flux3FlowMapScheduler.step = _fast_step
        Flux3FlowMapScheduler._mlxdiffusion_compiled_step = True
    except Exception:
        pass


def restore_flux3_compiled_scheduler_step() -> bool:
    """Restore the vendor FLUX.3 scheduler step if this module still owns it."""
    try:
        from mflux.models.flux3.model.flux3_scheduler.flux3_flow_map_scheduler import Flux3FlowMapScheduler
    except Exception:
        return False

    original = getattr(Flux3FlowMapScheduler, "_mlxdiffusion_original_step", None)
    patched = getattr(Flux3FlowMapScheduler, "_mlxdiffusion_patched_step", None)
    if (
        original is None
        or patched is None
        or Flux3FlowMapScheduler.step is not patched
    ):
        return False
    Flux3FlowMapScheduler.step = original
    for attr in (
        "_mlxdiffusion_original_step",
        "_mlxdiffusion_patched_step",
        "_mlxdiffusion_compiled_step",
    ):
        try:
            delattr(Flux3FlowMapScheduler, attr)
        except AttributeError:
            pass
    return True


def _rope_spec(model_id: str):
    return {
        "flux2-klein-4b": ("mflux.models.flux2.model.flux2_transformer.pos_embed", "Flux2PosEmbed"),
        "flux2-klein-9b": ("mflux.models.flux2.model.flux2_transformer.pos_embed", "Flux2PosEmbed"),
        "flux3-klein": ("mflux.models.flux3.model.flux3_transformer.flux3_pos_embed", "Flux3PosEmbed"),
        "z-image-turbo": ("mflux.models.z_image.model.z_image_transformer.rope_embedder", "RopeEmbedder"),
        "krea2-turbo": ("mflux.models.krea2.model.krea2_transformer.rope_embedder", "Krea2RopeEmbedder"),
    }.get(model_id)


_ROPE_CACHE_ORIGINALS: dict[tuple, tuple] = {}


def install_rope_cache(model_id: str):
    """Memoize each engine's position-embedding cos/sin across denoise steps.

    cos/sin depend only on position ids, which are constant for one generation
    (FLUX.2 img_ids/txt_ids; Z-Image x_pos_ids/cap_pos_ids from patchify;
    Krea2 pos from image/text lengths) but every mflux engine recomputes them on
    each forward. ids are never mutated after creation, so caching per
    (shape, dtype, flat values) is bit-identical while skipping the per-step
    trig work. Class-level patch — mlx dispatches __call__ via the class, so
    instance monkeypatching never fires. Idempotent per rope class.
    """
    if os.environ.get("MLX_DISABLE_ROPE_CACHE") == "1":
        return
    spec = _rope_spec(model_id)
    if spec is None:
        return
    gen = _generator()
    try:
        mod = importlib.import_module(spec[0])
        cls = getattr(mod, spec[1])
    except Exception as e:
        print(f"[generator] rope cache unavailable for {model_id}: {e}", flush=True)
        return
    if spec in gen._ROPE_CACHE_INSTALLED:
        return
    original = cls.__call__
    gen._install_rope_cache_on_class(cls)
    _ROPE_CACHE_ORIGINALS[spec] = (original, cls.__call__)
    gen._ROPE_CACHE_INSTALLED.add(spec)
    print(f"[generator] installed rope memo cache for {model_id} ({spec[1]}, lossless; ids constant across steps)", flush=True)


def restore_rope_cache(model_id: str) -> bool:
    """Restore a rope class patched by ``install_rope_cache``."""
    spec = _rope_spec(model_id)
    if spec is None:
        return False
    original, patched = _ROPE_CACHE_ORIGINALS.get(spec, (None, None))
    gen = _generator()
    if original is None or patched is None:
        gen._ROPE_CACHE_INSTALLED.discard(spec)
        _ROPE_CACHE_ORIGINALS.pop(spec, None)
        return False
    try:
        mod = importlib.import_module(spec[0])
        cls = getattr(mod, spec[1])
    except Exception:
        return False
    restored = False
    if cls.__call__ is patched:
        cls.__call__ = original
        restored = True
    gen._ROPE_CACHE_INSTALLED.discard(spec)
    _ROPE_CACHE_ORIGINALS.pop(spec, None)
    return restored


def install_krea_te_bf16_skip() -> bool:
    """Stop mflux loading the 7.5GB bf16 encoder when we already have a q4 copy on disk.

    Returns True only when both vendor hooks installed. A partial patch must not
    set ``_krea_te_skip_installed``: the caller would then preload q4 and let
    ``Krea2()`` load bf16 as well, putting both encoders in memory on 16 GB.
    """
    gen = _generator()
    if gen._krea_te_skip_installed:
        return True

    installed = True
    try:
        from mflux.models.common.weights.loading.weight_applier import WeightApplier

        original = WeightApplier._set_weights

        def _patched(weights, models, components=None):
            if gen._krea_te_skip_bf16["on"] and any(
                type(m).__name__ == "Krea2TextEncoder" for m in models.values()
            ):
                models = {k: v for k, v in models.items() if k != "text_encoder"}
            return original(weights, models, components)

        WeightApplier._mlxdiffusion_original_set_weights = original
        WeightApplier._mlxdiffusion_patched_set_weights = _patched
        WeightApplier._set_weights = staticmethod(_patched)
    except Exception as exc:
        installed = False
        print(f"[generator] WARN could not patch WeightApplier for Krea TE skip: {exc}", flush=True)

    try:
        from mflux.models.common.weights.loading.weight_loader import WeightLoader

        original_load = WeightLoader._load_component

        def _patched_load(root_path, component, raw_weights_cache=None, save_subdir=None):
            if gen._krea_te_skip_bf16["on"] and getattr(component, "name", None) == "text_encoder":
                return {}, 4, "0.18.0"
            return original_load(root_path, component, raw_weights_cache, save_subdir)

        WeightLoader._mlxdiffusion_original_load_component = original_load
        WeightLoader._mlxdiffusion_patched_load_component = _patched_load
        WeightLoader._load_component = staticmethod(_patched_load)
    except Exception as exc:
        installed = False
        print(f"[generator] WARN could not patch WeightLoader for Krea TE skip: {exc}", flush=True)

    gen._krea_te_skip_installed = installed
    return installed


def restore_krea_te_bf16_skip() -> bool:
    """Restore the vendor WeightApplier/WeightLoader methods if still ours."""
    gen = _generator()
    restored = False
    try:
        from mflux.models.common.weights.loading.weight_applier import WeightApplier

        original = getattr(
            WeightApplier, "_mlxdiffusion_original_set_weights", None
        )
        patched = getattr(
            WeightApplier, "_mlxdiffusion_patched_set_weights", None
        )
        if original is not None and patched is not None and WeightApplier._set_weights is patched:
            WeightApplier._set_weights = staticmethod(original)
            restored = True
        for attr in (
            "_mlxdiffusion_original_set_weights",
            "_mlxdiffusion_patched_set_weights",
        ):
            try:
                delattr(WeightApplier, attr)
            except AttributeError:
                pass
    except Exception:
        pass

    try:
        from mflux.models.common.weights.loading.weight_loader import WeightLoader

        original = getattr(
            WeightLoader, "_mlxdiffusion_original_load_component", None
        )
        patched = getattr(
            WeightLoader, "_mlxdiffusion_patched_load_component", None
        )
        if (
            original is not None
            and patched is not None
            and WeightLoader._load_component is patched
        ):
            WeightLoader._load_component = staticmethod(original)
            restored = True
        for attr in (
            "_mlxdiffusion_original_load_component",
            "_mlxdiffusion_patched_load_component",
        ):
            try:
                delattr(WeightLoader, attr)
            except AttributeError:
                pass
    except Exception:
        pass

    if restored:
        gen._krea_te_skip_installed = False
    return restored


def apply_vae_tiling(pipe, default_size: int = 512, model_label: str = "model") -> None:
    """Give a pipeline a TilingConfig so large VAE decodes (>512px) tile into manageable chunks.

    mflux only tiles when tiles_per_dim > 1, and the tile size is a cap on each tile's
    working set. A value of 0 restores untiled decoding.
    """
    size = _generator()._krea_vae_tile_size() if model_label == "krea2" else default_size
    if size <= 0:
        return
    try:
        from mflux.models.common.vae.tiling_config import TilingConfig

        if not hasattr(pipe, "_mlxdiffusion_original_tiling_config"):
            pipe._mlxdiffusion_original_tiling_config = getattr(
                pipe, "tiling_config", None
            )
            pipe._mlxdiffusion_had_tiling_config = hasattr(pipe, "tiling_config")
        config = TilingConfig(
            vae_decode_tile_size=size,
            vae_decode_tiles_per_dim=8,
            vae_decode_overlap=8,
        )
        pipe.tiling_config = config
        pipe._mlxdiffusion_patched_tiling_config = config
    except Exception as exc:  # pragma: no cover - depends on the mflux build
        print(f"[generator] {model_label} VAE tiling unavailable: {exc}", flush=True)


def restore_vae_tiling(pipe) -> bool:
    """Restore the pipeline tiling only while this module still owns it."""
    if not hasattr(pipe, "_mlxdiffusion_original_tiling_config"):
        return False
    patched = getattr(pipe, "_mlxdiffusion_patched_tiling_config", None)
    if patched is None or getattr(pipe, "tiling_config", None) is not patched:
        return False
    if pipe._mlxdiffusion_had_tiling_config:
        pipe.tiling_config = pipe._mlxdiffusion_original_tiling_config
    else:
        try:
            delattr(pipe, "tiling_config")
        except AttributeError:
            pass
    for attr in (
        "_mlxdiffusion_original_tiling_config",
        "_mlxdiffusion_patched_tiling_config",
        "_mlxdiffusion_had_tiling_config",
    ):
        try:
            delattr(pipe, attr)
        except AttributeError:
            pass
    return True


PATCHES = {
    "apply_prompt_cache": {
        "families": ("flux2",),
        "vendor_version": MFLUX_VENDOR_VERSION,
        "purpose": "Reuse FLUX.2 prompt embeddings across generations instead of re-encoding.",
        "restore": "restore_prompt_cache",
        "rollback": "Call `restore_prompt_cache(pipe)`: it reinstates the captured `_encode_prompt_pair` when the installed wrapper is still current, otherwise it is a no-op.",
    },
    "apply_prompt_cache_txt": {
        "families": ("zit", "krea"),
        "vendor_version": MFLUX_VENDOR_VERSION,
        "purpose": "Reuse Z-Image/Krea text-encoder embeddings across generations.",
        "restore": "restore_prompt_cache_txt",
        "rollback": "Call `restore_prompt_cache_txt(pipe)`: it reinstates the captured `_encode_prompts` when the installed wrapper is still current, otherwise it is a no-op.",
    },
    "install_rope_cache": {
        "families": ("flux2", "flux3", "zit", "krea"),
        "vendor_version": MFLUX_VENDOR_VERSION,
        "purpose": "Memoize constant per-generation RoPE cos/sin instead of recomputing every step.",
        "restore": "restore_rope_cache",
        "rollback": "Call `restore_rope_cache(model_id)`: it restores the captured class `__call__` when the wrapper is still current and drops the install marker.",
    },
    "install_flux3_compiled_scheduler_step": {
        "families": ("flux3",),
        "vendor_version": MFLUX_VENDOR_VERSION,
        "purpose": "Remove FLUX.3's per-step `.item()` barrier by compiling the Euler update on a precomputed dt array.",
        "restore": "restore_flux3_compiled_scheduler_step",
        "rollback": "Call `restore_flux3_compiled_scheduler_step()`: it reinstates the captured vendor step when the wrapper is still current, otherwise it is a no-op.",
    },
    "install_zit_dynamic_padding": {
        "families": ("zit",),
        "vendor_version": MFLUX_VENDOR_VERSION,
        "purpose": "Pad Z-Image tokenizer output to the longest sequence instead of a fixed max_length.",
        "restore": "restore_zit_dynamic_padding",
        "rollback": "Call `restore_zit_dynamic_padding(pipe)`: it reinstates the captured tokenizer padding mode.",
    },
    "install_krea_sampler_scalar_cache": {
        "families": ("krea",),
        "vendor_version": MFLUX_VENDOR_VERSION,
        "purpose": "Cache ErSde zero-sigma flags once and compile the Euler step, removing the per-step sigma host read.",
        "restore": "restore_krea_sampler_scalar_cache",
        "rollback": "Call `restore_krea_sampler_scalar_cache()`: it reinstates each captured vendor step only while this module still owns that class method and never touches a foreign installer.",
    },
    "install_krea_euler_a_trailing": {
        "families": ("krea",),
        "vendor_version": MFLUX_VENDOR_VERSION,
        "purpose": "Add the Euler A Trailing sampler (ComfyUI ancestral Euler math) to Krea 2.",
        "restore": "restore_krea_euler_a_trailing",
        "rollback": "Call `restore_krea_euler_a_trailing()`: it reinstates the captured `make_stepper`/`_resolve_scheduler` only when this module still owns them.",
    },
    "install_flux_text_encoder_tail_trim": {
        "families": ("flux2",),
        "vendor_version": MFLUX_VENDOR_VERSION,
        "purpose": "Drop the last 8 FLUX text-encoder layers from the transformer traversal to cut encode compute.",
        "restore": "restore_flux_text_encoder_tail_trim",
        "rollback": "Call `restore_flux_text_encoder_tail_trim(pipe)`: it reinstates the captured layer list and hidden-layer count. The rollback list keeps the 8 layers resident, so this patch lowers traversal compute, not steady-state memory.",
    },
    "install_krea_te_bf16_skip": {
        "families": ("krea",),
        "vendor_version": MFLUX_VENDOR_VERSION,
        "purpose": "Skip loading the 7.5 GB bf16 Krea text encoder when a local quantised copy is available.",
        "restore": "restore_krea_te_bf16_skip",
        "rollback": "Call `restore_krea_te_bf16_skip()`: it reinstates the captured `WeightApplier._set_weights` and `WeightLoader._load_component` only when this module still owns them, then clears the installed flag.",
    },
    "apply_vae_tiling": {
        "families": ("flux2", "flux3", "zit", "krea"),
        "vendor_version": MFLUX_VENDOR_VERSION,
        "purpose": "Give large VAE decodes a tiling config so they fit the wired memory budget.",
        "restore": "restore_vae_tiling",
        "rollback": "Call `restore_vae_tiling(pipe)`: it reinstates the captured pre-patch tiling config, or removes the attribute when the pipe had none.",
    },
}
