from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.pid_decoder.pid_decoder import pid_decode_latents
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.krea2.krea2_initializer import Krea2Initializer
from mflux.models.krea2.latent_creator.krea2_latent_creator import Krea2LatentCreator
from mflux.models.krea2.model.krea2_sampler import Krea2Sampler
from mflux.models.krea2.model.krea2_text_encoder.prompt_encoder import Krea2PromptEncoder
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Krea2(nn.Module):
    vae: QwenVAE
    transformer: Krea2Transformer
    text_encoder: Krea2TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ):
        super().__init__()
        Krea2Initializer.init(
            model=self,
            model_config=model_config or ModelConfig.krea2(),
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 8,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        negative_prompt: str | None = None,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str | None = None,
        pid_decode: bool = False,
        pid_degrade_sigma: float = 0.0,
    ) -> GeneratedImage:
        resolved_scheduler = Krea2._resolve_scheduler(scheduler)

        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=image_path,
            image_strength=image_strength,
            scheduler=resolved_scheduler,
        )

        sigmas = config.scheduler.sigmas
        latents = self._prepare_latents(seed=seed, config=config)
        embeds, neg_embeds = self._encode_prompts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=guidance,
        )
        mx.eval(latents, embeds)
        if neg_embeds is not None:
            mx.eval(neg_embeds)

        stepper = Krea2Sampler.make_stepper(resolved_scheduler, sigmas, seed)
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._predict(self.transformer, embeds, neg_embeds, guidance)

        for t in config.time_steps:
            try:
                ts = sigmas[t].reshape(1)
                v = predict(latents=latents, timestep=ts)
                denoised = latents - sigmas[t] * v
                latents = stepper.step(t, latents, v, denoised)
                ctx.in_loop(t, latents, denoised=denoised)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )
        ctx.after_loop(latents)

        decoded = self._decode_latents(
            latents=latents,
            prompt=prompt,
            seed=seed,
            pid_decode=pid_decode,
            degrade_sigma=pid_degrade_sigma,
        )
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=config.time_steps.format_dict["elapsed"],
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            negative_prompt=negative_prompt,
            image_path=config.image_path,
            image_strength=config.image_strength,
            pid_decode=pid_decode,
            pid_degrade_sigma=pid_degrade_sigma,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Krea2WeightDefinition,
        )

    def _encode_prompts(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[mx.array, mx.array | None]:
        return Krea2PromptEncoder.encode_prompt_pair(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=guidance,
            tokenizer=self.tokenizers["qwen3vl"],
            text_encoder=self.text_encoder,
            prompt_cache=self.prompt_cache,
        )

    def _prepare_latents(self, *, seed: int, config: Config) -> mx.array:
        if config.image_path is None or config.image_strength is None or config.image_strength <= 0.0:
            return Krea2LatentCreator.create_noise(seed, config.height, config.width)

        pure_noise = Krea2LatentCreator.create_noise(seed, config.height, config.width)
        encoded = LatentCreator.encode_image(
            vae=self.vae,
            image_path=config.image_path,
            height=config.height,
            width=config.width,
            tiling_config=self.tiling_config,
        )
        clean_latents = Krea2LatentCreator.pack_latents(encoded, config.height, config.width)
        sigma = float(config.scheduler.sigmas[config.init_time_step])
        return LatentCreator.add_noise_by_interpolation(clean=clean_latents, noise=pure_noise, sigma=sigma)

    def _decode_latents(
        self,
        *,
        latents: mx.array,
        prompt: str,
        seed: int,
        pid_decode: bool = False,
        degrade_sigma: float = 0.0,
    ) -> mx.array:
        if pid_decode:
            return pid_decode_latents(
                vae=self.vae, latent=latents, caption=prompt, seed=seed, degrade_sigma=degrade_sigma
            )
        return VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)

    @staticmethod
    def _predict(
        transformer: Krea2Transformer,
        embeds: mx.array,
        neg_embeds: mx.array | None,
        guidance: float,
    ):
        def predict(latents: mx.array, timestep: mx.array) -> mx.array:
            v = transformer(latents, timestep, embeds)
            if neg_embeds is not None:
                v_neg = transformer(latents, timestep, neg_embeds)
                v = v_neg + guidance * (v - v_neg)
            return v

        # Warmup + shapeless compilation for M1/M2 (MLX 0.32+ improved compilation)
        # Run first 2 steps eagerly to establish shapes, then compile with shapeless=True
        warmup_steps = 2
        step_counter = [0]
        predict_ref = [predict]

        def warmup_predict(latents: mx.array, timestep: mx.array) -> mx.array:
            result = predict_ref[0](latents, timestep)
            step_counter[0] += 1
            if step_counter[0] == warmup_steps:
                predict_ref[0] = mx.compile(predict_ref[0], shapeless=True)
            return result

        # Use warmup_predict initially; it will swap to compiled after warmup
        if AppleSiliconUtil.is_m1_or_m2():
            return warmup_predict
        return mx.compile(predict, shapeless=True)

    @staticmethod
    def _resolve_scheduler(scheduler: str | None) -> str:
        if scheduler is None or scheduler == "linear":
            return "er_sde"
        if scheduler in ("er_sde", "euler"):
            return scheduler
        raise ValueError(f"Unknown Krea-2 scheduler {scheduler!r}. Expected 'er_sde' or 'euler'.")
