from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.pid_decoder.pid_decoder import pid_decode_latents
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_transformer.flux2_kv_cache import CacheMode, Flux2KVCache
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Flux2Klein(nn.Module):
    vae: Flux2VAE
    transformer: Flux2Transformer
    text_encoder: Qwen3TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
        model_config: ModelConfig | None = None,
    ):
        super().__init__()
        Flux2Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            model_config=model_config or ModelConfig.flux2_klein_4b(),
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "flow_match_euler_discrete",
        pid_decode: bool = False,
        pid_degrade_sigma: float = 0.0,
        use_kv_cache: bool | None = None,
    ) -> GeneratedImage:
        # 0. Create a new config based on the model type and input parameters
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=image_path,
            image_strength=image_strength,
            scheduler=scheduler,
        )
        # 1. Encode prompt(s)
        prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids = self._encode_prompt_pair(
            prompt=prompt,
            negative_prompt=" ",
            guidance=guidance,
        )

        # 2. Prepare latents (txt2img or img2img)
        latents, latent_ids, latent_height, latent_width = self._prepare_generation_latents(
            seed=seed,
            config=config,
        )

        # 3. KV Cache setup (only for img2img with reference image)
        cache_enabled = (
            (use_kv_cache if use_kv_cache is not None else self.model_config.supports_kv_cache)
            and config.image_path is not None
            and config.image_strength is not None
            and config.image_strength > 0.0
        )
        kv_cache, negative_kv_cache = self._create_kv_caches(
            cache_enabled=cache_enabled,
            needs_negative_cache=negative_prompt_embeds is not None,
        )

        # 3. Denoising loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._predict(self.transformer)
        cached_predict = self._cached_predict(self.transformer) if cache_enabled else None

        # If cache enabled and img2img, extract reference tokens on first step
        if cache_enabled and config.image_path is not None:
            # Encode reference image for KV cache
            encoded = LatentCreator.encode_image(
                vae=self.vae,
                image_path=config.image_path,
                height=config.height,
                width=config.width,
                tiling_config=self.tiling_config,
            )
            encoded = _Flux2KleinEditHelpers.ensure_4d_latents(encoded)
            encoded = _Flux2KleinEditHelpers.crop_to_even_spatial(encoded)
            encoded = self._match_latent_spatial_size(
                encoded=encoded,
                target_height=latent_height * 2,
                target_width=latent_width * 2,
            )
            encoded = Flux2LatentCreator.patchify_latents(encoded)
            encoded = _Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents(encoded, vae=self.vae)
            ref_latents = Flux2LatentCreator.pack_latents(encoded)
            num_ref_tokens = ref_latents.shape[1]
            self._configure_kv_caches(
                kv_cache=kv_cache,
                negative_kv_cache=negative_kv_cache,
                mode="extract",
                num_ref_tokens=num_ref_tokens,
            )
            # Concatenate reference latents for first step
            latents = mx.concatenate([latents, ref_latents], axis=1)
            latent_ids = mx.concatenate([latent_ids, mx.zeros_like(ref_latents[:, :, :2])], axis=1)

        for step_idx, t in enumerate(config.time_steps):
            try:
                if cache_enabled and step_idx == 0:
                    # First step with reference tokens
                    noise = predict(
                        latents=latents,
                        latent_ids=latent_ids,
                        prompt_embeds=prompt_embeds,
                        text_ids=text_ids,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_text_ids=negative_text_ids,
                        guidance=guidance,
                        timestep=config.scheduler.timesteps[t],
                        kv_cache=kv_cache,
                        negative_kv_cache=negative_kv_cache,
                    )
                    # After first step, remove reference tokens and switch to cached predict
                    if cache_enabled:
                        latents = latents[:, :latent_ids.shape[1] - num_ref_tokens]
                        latent_ids = latent_ids[:, :latent_ids.shape[1] - num_ref_tokens]
                        self._configure_kv_caches(
                            kv_cache=kv_cache,
                            negative_kv_cache=negative_kv_cache,
                            mode="cached",
                            num_ref_tokens=num_ref_tokens,
                        )
                elif cache_enabled:
                    self._configure_kv_caches(
                        kv_cache=kv_cache,
                        negative_kv_cache=negative_kv_cache,
                        mode="cached",
                        num_ref_tokens=num_ref_tokens,
                    )
                    assert cached_predict is not None
                    noise = cached_predict(
                        latents=latents,
                        latent_ids=latent_ids,
                        prompt_embeds=prompt_embeds,
                        text_ids=text_ids,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_text_ids=negative_text_ids,
                        guidance=guidance,
                        timestep=config.scheduler.timesteps[t],
                        kv_cache=kv_cache,
                        negative_kv_cache=negative_kv_cache,
                    )
                else:
                    noise = predict(
                        latents=latents,
                        latent_ids=latent_ids,
                        prompt_embeds=prompt_embeds,
                        text_ids=text_ids,
                        negative_prompt_embeds=negative_prompt_embeds,
                        negative_text_ids=negative_text_ids,
                        guidance=guidance,
                        timestep=config.scheduler.timesteps[t],
                    )

                # 4.t Take one denoise step
                latents = config.scheduler.step(
                    noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas
                )

                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)

        # 5. Decode latents
        packed_latents = latents.reshape(latents.shape[0], latent_height, latent_width, latents.shape[-1]).transpose(0, 3, 1, 2)  # fmt: off
        if pid_decode:
            lq_latent = self.vae.unpack_packed_latents(packed_latents)
            decoded = pid_decode_latents(
                vae=self.vae, latent=lq_latent, caption=prompt, seed=seed, degrade_sigma=pid_degrade_sigma
            )
        else:
            decoded = self.vae.decode_packed_latents(packed_latents, tiling_config=self.tiling_config)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            negative_prompt=None,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=config.image_path,
            image_strength=config.image_strength,
            generation_time=config.time_steps.format_dict["elapsed"],
            pid_decode=pid_decode,
            pid_degrade_sigma=pid_degrade_sigma,
        )

    def _encode_prompt_pair(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]:
        prompt_embeds, text_ids = Flux2PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizers["qwen3"],
            text_encoder=self.text_encoder,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        negative_prompt_embeds = None
        negative_text_ids = None
        if guidance is not None and guidance > 1.0 and negative_prompt is not None:
            negative_prompt_embeds, negative_text_ids = Flux2PromptEncoder.encode_prompt(
                prompt=negative_prompt,
                tokenizer=self.tokenizers["qwen3"],
                text_encoder=self.text_encoder,
                num_images_per_prompt=1,
                max_sequence_length=512,
                text_encoder_out_layers=(9, 18, 27),
            )
        return prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids

    def _prepare_generation_latents(
        self,
        *,
        seed: int,
        config: Config,
    ) -> tuple[mx.array, mx.array, int, int]:
        if config.image_path is None or config.image_strength is None or config.image_strength <= 0.0:
            return Flux2LatentCreator.prepare_packed_latents(
                seed=seed,
                height=config.height,
                width=config.width,
                batch_size=1,
            )
        return self._prepare_img2img_latents(seed=seed, config=config)

    def _prepare_img2img_latents(
        self,
        *,
        seed: int,
        config: Config,
    ) -> tuple[mx.array, mx.array, int, int]:
        noise_latents, latent_ids, latent_height, latent_width = Flux2LatentCreator.prepare_packed_latents(
            seed=seed,
            height=config.height,
            width=config.width,
            batch_size=1,
        )

        encoded = LatentCreator.encode_image(
            vae=self.vae,
            image_path=config.image_path,
            height=config.height,
            width=config.width,
            tiling_config=self.tiling_config,
        )
        encoded = _Flux2KleinEditHelpers.ensure_4d_latents(encoded)
        encoded = _Flux2KleinEditHelpers.crop_to_even_spatial(encoded)
        encoded = self._match_latent_spatial_size(
            encoded=encoded,
            target_height=latent_height * 2,
            target_width=latent_width * 2,
        )
        encoded = Flux2LatentCreator.patchify_latents(encoded)
        encoded = _Flux2KleinEditHelpers.bn_normalize_vae_encoded_latents(encoded, vae=self.vae)
        clean_latents = Flux2LatentCreator.pack_latents(encoded)

        sigma = config.scheduler.sigmas[config.init_time_step]
        latents = LatentCreator.add_noise_by_interpolation(clean=clean_latents, noise=noise_latents, sigma=sigma)
        return latents, latent_ids, latent_height, latent_width

    @staticmethod
    def _match_latent_spatial_size(
        *,
        encoded: mx.array,
        target_height: int,
        target_width: int,
    ) -> mx.array:
        _, _, height, width = encoded.shape
        if height != target_height:
            if height > target_height:
                offset = (height - target_height) // 2
                encoded = encoded[:, :, offset : offset + target_height, :]
            else:
                pad_total = target_height - height
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                encoded = mx.pad(encoded, ((0, 0), (0, 0), (pad_before, pad_after), (0, 0)))
        if width != target_width:
            if width > target_width:
                offset = (width - target_width) // 2
                encoded = encoded[:, :, :, offset : offset + target_width]
            else:
                pad_total = target_width - width
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                encoded = mx.pad(encoded, ((0, 0), (0, 0), (0, 0), (pad_before, pad_after)))
        return encoded

    def _create_kv_caches(
        self,
        *,
        cache_enabled: bool,
        needs_negative_cache: bool,
    ) -> tuple[Flux2KVCache | None, Flux2KVCache | None]:
        if not cache_enabled:
            return None, None

        kv_cache = self._new_kv_cache()
        negative_kv_cache = self._new_kv_cache() if needs_negative_cache else None
        return kv_cache, negative_kv_cache

    def _new_kv_cache(self) -> Flux2KVCache:
        return Flux2KVCache(
            num_double_layers=len(self.transformer.transformer_blocks),
            num_single_layers=len(self.transformer.single_transformer_blocks),
        )

    @staticmethod
    def _configure_kv_caches(
        *,
        kv_cache: Flux2KVCache | None,
        negative_kv_cache: Flux2KVCache | None,
        mode: CacheMode,
        num_ref_tokens: int,
    ) -> None:
        assert kv_cache is not None
        kv_cache.configure(mode=mode, num_ref_tokens=num_ref_tokens)
        if negative_kv_cache is not None:
            negative_kv_cache.configure(mode=mode, num_ref_tokens=num_ref_tokens)

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Flux2KleinWeightDefinition,
        )

    @staticmethod
    def _predict(transformer):
        def predict(
            latents: mx.array,
            latent_ids: mx.array,
            prompt_embeds: mx.array,
            text_ids: mx.array,
            negative_prompt_embeds: mx.array | None,
            negative_text_ids: mx.array | None,
            guidance: float,
            timestep: mx.array,
            kv_cache: Flux2KVCache | None = None,
            negative_kv_cache: Flux2KVCache | None = None,
        ):
            noise = transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=None,
                kv_cache=kv_cache,
            )
            if negative_prompt_embeds is not None and negative_text_ids is not None:
                negative_noise = transformer(
                    hidden_states=latents,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=timestep,
                    img_ids=latent_ids,
                    txt_ids=negative_text_ids,
                    guidance=None,
                    kv_cache=negative_kv_cache or kv_cache,
                )
                noise = negative_noise + guidance * (noise - negative_noise)
            return noise

        # Warmup + shapeless compilation for M1/M2 (MLX 0.32+ improved compilation)
        # Run first 2 steps eagerly to establish shapes, then compile with shapeless=True
        warmup_steps = 2
        step_counter = [0]
        predict_ref = [predict]  # mutable container for function reference

        def warmup_predict(
            latents: mx.array,
            latent_ids: mx.array,
            prompt_embeds: mx.array,
            text_ids: mx.array,
            negative_prompt_embeds: mx.array | None,
            negative_text_ids: mx.array | None,
            guidance: float,
            timestep: mx.array,
            kv_cache: Flux2KVCache | None = None,
            negative_kv_cache: Flux2KVCache | None = None,
        ):
            result = predict_ref[0](
                latents, latent_ids, prompt_embeds, text_ids,
                negative_prompt_embeds, negative_text_ids,
                guidance, timestep, kv_cache, negative_kv_cache
            )
            step_counter[0] += 1
            if step_counter[0] == warmup_steps:
                # After warmup, replace with compiled version
                predict_ref[0] = mx.compile(predict_ref[0], shapeless=True)
            return result

        # Use warmup_predict initially; it will swap to compiled after warmup
        if AppleSiliconUtil.is_m1_or_m2() or transformer.model_config.supports_kv_cache:
            return warmup_predict
        # Non-M1/M2 without KV cache: compile immediately with shapeless
        return mx.compile(predict, shapeless=True)

    @staticmethod
    def _cached_predict(transformer):
        def predict(
            latents: mx.array,
            latent_ids: mx.array,
            prompt_embeds: mx.array,
            text_ids: mx.array,
            negative_prompt_embeds: mx.array | None,
            negative_text_ids: mx.array | None,
            guidance: float,
            timestep: mx.array,
            kv_cache: Flux2KVCache,
            negative_kv_cache: Flux2KVCache | None = None,
        ):
            noise = transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=None,
                kv_cache=kv_cache,
            )
            if negative_prompt_embeds is not None and negative_text_ids is not None:
                negative_noise = transformer(
                    hidden_states=latents,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=timestep,
                    img_ids=latent_ids,
                    txt_ids=negative_text_ids,
                    guidance=None,
                    kv_cache=negative_kv_cache or kv_cache,
                )
                negative_noise = negative_noise + guidance * (noise - negative_noise)
            return noise

        return predict