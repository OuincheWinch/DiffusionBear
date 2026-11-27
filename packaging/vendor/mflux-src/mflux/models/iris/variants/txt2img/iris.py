import copy

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.iris.iris_initializer import IrisInitializer
from mflux.models.iris.latent_creator.iris_latent_creator import IrisLatentCreator
from mflux.models.iris.model.iris_scheduler.iris_few_step_scheduler import IrisFewStepScheduler
from mflux.models.iris.model.iris_scheduler.iris_flow_dpm_solver import IrisFlowDPMSolver
from mflux.models.iris.model.iris_scheduler.iris_turbo_scheduler import IrisTurboScheduler
from mflux.models.iris.model.iris_text_encoder.prompt_encoder import IrisPromptEncoder
from mflux.models.iris.model.iris_transformer.iris_transformer import IrisTransformer
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Iris(nn.Module):
    transformer: IrisTransformer
    text_encoder: Qwen3TextEncoder

    DEFAULT_STEPS = 100
    DEFAULT_GUIDANCE = 3.0
    SHIFT = 4.0
    # A LoRA stays a separate layer by default: merged into the bfloat16 weights, a rank-64 few-step LoRA rounds enough
    # to move the composition (22 dB from the unmerged image, against 30 dB for -q 8, which merges in float32)
    DEFAULT_BAKE_LORA = False

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = DEFAULT_BAKE_LORA,
    ):
        super().__init__()
        IrisInitializer.init(
            model=self,
            model_config=model_config or ModelConfig.iris_3b(),
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )

    @property
    def is_turbo(self) -> bool:
        return not self.model_config.supports_guidance

    # The weights would keep the LoRA but the saved model has no record of its sampler, so it would run the base
    # model's 100 steps with CFG
    SAVE_REFUSAL = "A few-step LoRA changes the sampler, which a saved model cannot record. Save the model without it and pass --lora when you generate."  # fmt: off

    @staticmethod
    def check_save(model_config: ModelConfig, lora_paths: list[str] | None, lora_scales: list[float] | None) -> None:
        # The same refusal as save_model, from the LoRA file headers only, so mflux-save stops before the load
        if model_config.supports_guidance and IrisFewStepScheduler.from_lora_files(lora_paths, lora_scales):
            raise ValueError(Iris.SAVE_REFUSAL)

    def save_model(self, base_path: str) -> None:
        if self.few_step is not None:
            raise ValueError(Iris.SAVE_REFUSAL)
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=IrisInitializer.weight_definition(self.model_config),
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None = None,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = None,
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        IrisLatentCreator.check_size(height, width)
        # Turbo has a fixed 4-step schedule with distilled guidance; steps, guidance and negative prompt are for Base.
        if self.is_turbo:
            num_inference_steps, guidance, negative_prompt = IrisTurboScheduler().num_inference_steps, None, None
        elif self.few_step is not None:
            # A few-step LoRA on the base model has its guidance distilled in: its own step count, no CFG
            num_inference_steps, guidance, negative_prompt = num_inference_steps or self.few_step[0], None, None
        else:
            num_inference_steps = num_inference_steps or Iris.DEFAULT_STEPS
            guidance = Iris.DEFAULT_GUIDANCE if guidance is None else float(guidance)
            negative_prompt = negative_prompt or ""
        config = Config(
            model_config=self._metadata_model_config(),
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
        )

        # 1. Encode the prompt (and the null prompt for CFG)
        cond, cond_mask = self._encode_prompt(prompt)
        use_cfg = guidance is not None and guidance != 1.0
        uncond, uncond_mask = self._encode_prompt(negative_prompt) if use_cfg else (None, None)

        # 2. Noise
        latents = IrisLatentCreator.create_noise(seed, config.height, config.width)

        # 3. Denoising loop; the integrator state stays float32, the network runs in the model precision.
        scheduler = self._scheduler(num_inference_steps, seed)
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        clean = latents
        for t in config.time_steps:
            try:
                if use_cfg:
                    output = self._predict_cfg(
                        latents, scheduler.model_time(t), cond, cond_mask, uncond, uncond_mask, guidance
                    )
                else:
                    output = self._predict(latents, scheduler.model_time(t), cond, cond_mask)
                if self.is_turbo or self.few_step is not None:
                    latents, clean = scheduler.step(latents, output, t)
                else:
                    latents = clean = scheduler.step(latents, output, t)
                # Callbacks get the solver state as latents, as with the other models. Turbo noises its next state
                # again, so its clean prediction goes to the callbacks that declare `denoised`.
                ctx.in_loop(t, latents, denoised=clean)
                mx.eval(latents, clean)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)

        # 4. No VAE: the sample is already an RGB image in [-1, 1].
        decoded = IrisLatentCreator.unpack_latents(mx.clip(clean, -1, 1), config.height, config.width)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            negative_prompt=negative_prompt,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    def _scheduler(self, num_inference_steps: int, seed: int):
        if self.is_turbo:
            return IrisTurboScheduler()
        if self.few_step is not None:
            return IrisFewStepScheduler(num_inference_steps, self.few_step[1], seed)
        return IrisFlowDPMSolver(num_inference_steps, Iris.SHIFT)

    def _metadata_model_config(self) -> ModelConfig:
        # Turbo and base share one repo id, which is all a metadata sidecar records as the model. Naming the base
        # entry as base_model lets --config-from-conf replay a base run as base instead of as the default Turbo.
        if self.is_turbo:
            return self.model_config
        model_config = copy.copy(self.model_config)
        model_config.base_model = model_config.aliases[0]
        return model_config

    def _encode_prompt(self, prompt: str) -> tuple[mx.array, mx.array]:
        if prompt not in self.prompt_cache:
            self.prompt_cache[prompt] = IrisPromptEncoder.encode_prompt(
                prompt=prompt,
                tokenizer=self.tokenizers["qwen3_vl"],
                text_encoder=self.text_encoder,
            )
        return self.prompt_cache[prompt]

    def _predict(self, latents: mx.array, model_time: mx.array, cond: mx.array, mask: mx.array) -> mx.array:
        return self.transformer(latents, model_time, cond, mask).astype(mx.float32)

    def _predict_cfg(
        self,
        latents: mx.array,
        model_time: mx.array,
        cond: mx.array,
        cond_mask: mx.array,
        uncond: mx.array,
        uncond_mask: mx.array,
        guidance: float,
    ) -> mx.array:
        output = self.transformer(
            mx.concatenate([latents, latents], axis=0),
            mx.concatenate([model_time, model_time], axis=0),
            mx.concatenate([uncond, cond], axis=0),
            mx.concatenate([uncond_mask, cond_mask], axis=0),
        ).astype(mx.float32)
        out_uncond, out_cond = output[:1], output[1:]
        return out_uncond + guidance * (out_cond - out_uncond)
