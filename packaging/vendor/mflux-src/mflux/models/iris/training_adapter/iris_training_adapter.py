from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.training.adapters.base import TrainingAdapter
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.utils import TrainingUtil
from mflux.models.iris.model.iris_scheduler.iris_training_schedule import IrisTrainingSchedule
from mflux.models.iris.model.iris_text_encoder.prompt_encoder import IrisPromptEncoder
from mflux.models.iris.variants.txt2img.iris import Iris
from mflux.models.iris.weights.iris_lora_mapping import IrisLoRAMapping
from mflux.utils.image_util import ImageUtil
from mflux.utils.version_util import VersionUtil


class IrisTrainingAdapter(TrainingAdapter):
    # Iris generates in pixel space, so the trainer's "latents" are the image itself in [-1, 1], (1, 3, H, W). The
    # transformer takes 1000 * sigma and returns the velocity noise - clean, the target of the trainer's loss. LoRAs
    # train on the base model (Turbo is distilled) and are saved with the reference checkpoint's names under
    # diffusion_model., the keys and shapes ai-toolkit writes for Iris.
    SCHEDULE = f"{IrisTrainingSchedule.__module__}.{IrisTrainingSchedule.__name__}"

    def __init__(self, *, model_config: ModelConfig, quantize: int | None, model_path: str | None = None):
        if not model_config.supports_guidance:
            raise ValueError('Iris LoRAs are trained on the base model: set "model": "iris-3b-base" (Turbo is distilled).')  # fmt: off
        self._model_config = model_config
        self._iris = Iris(quantize=quantize, model_config=model_config, model_path=model_path, bake_lora=False)
        self._guidance: float | None = None
        # The preview prompts' encodings once the text encoder is gone (release_text_encoder)
        self._preview_encodings: dict[str, tuple[mx.array, mx.array]] = {}

    def model(self):
        return self._iris

    def transformer(self):
        return self._iris.transformer

    def create_config(self, training_spec: TrainingSpec, *, width: int, height: int) -> Config:
        # A config without guidance gives 0.0; the preview then uses the base model's default
        self._guidance = training_spec.guidance or None
        # The step count is the size of the training grid, 1000 levels as in the reference trainer. The previews get
        # training_spec.steps on their own
        return Config(
            model_config=self._model_config,
            num_inference_steps=IrisTrainingSchedule.NUM_TRAIN_TIMESTEPS,
            width=width,
            height=height,
            guidance=training_spec.guidance,
            scheduler=IrisTrainingAdapter.SCHEDULE,
        )

    def freeze_base(self) -> None:
        self._iris.transformer.freeze()
        if self._iris.text_encoder is not None:  # gone after release_text_encoder
            self._iris.text_encoder.freeze()

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self._iris.transformer.gradient_checkpointing = enabled

    def encode_data(
        self,
        *,
        data_id: int,  # noqa: ARG002
        image_path: Path,
        prompt: str,
        width: int,
        height: int,
        input_image_path: Path | None = None,  # noqa: ARG002
    ) -> tuple[mx.array, Any]:
        image = ImageUtil.scale_to_dimensions(
            image=ImageUtil.load_image(image_path).convert("RGB"), target_width=width, target_height=height
        )
        clean = ImageUtil.to_array(image).astype(mx.float32)
        embeddings, mask = IrisPromptEncoder.encode_prompt(
            prompt=prompt, tokenizer=self._iris.tokenizers["qwen3_vl"], text_encoder=self._iris.text_encoder
        )
        mx.eval(clean, embeddings, mask)
        # A dict of arrays, the form the low_ram data cache stores
        return clean, {"embeddings": embeddings, "mask": mask}

    def predict_noise(self, *, t: int, latents_t: mx.array, sigmas: mx.array, cond: Any, config: Config) -> mx.array:  # noqa: ARG002
        model_time = mx.reshape(sigmas[t], (1,)) * IrisTrainingSchedule.NUM_TRAIN_TIMESTEPS
        output = self._iris.transformer(latents_t, model_time, cond["embeddings"], cond["mask"])
        return output.astype(latents_t.dtype)

    def generate_preview_image(
        self,
        *,
        seed: int,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        image_paths: list[Path | str] | None = None,  # noqa: ARG002
    ):
        # After release_text_encoder, the prompts come from the encodings kept then
        self._iris.prompt_cache = dict(self._preview_encodings)
        with TrainingUtil.assistant_disabled(self._iris.transformer):
            image = self._iris.generate_image(
                seed=seed, prompt=prompt, num_inference_steps=steps, height=height, width=width, guidance=self._guidance
            )
        # The cache lives on the module, so its encodings would become trainable parameters of model()
        self._iris.prompt_cache = {}
        return image.image

    def release_text_encoder(self, training_spec: TrainingSpec) -> bool:
        # Encodes the preview prompts and the empty negative of CFG now, keeps them on the adapter, and drops the 8.9 GB
        # Qwen3-VL text encoder, which training never runs again. The encodings live here, not in the model's prompt
        # cache, which training would treat as parameters.
        monitoring = training_spec.monitoring
        prompts = list(monitoring.preview_prompts) if monitoring is not None else []
        for prompt in [*prompts, ""]:
            self._preview_encodings[prompt] = self._iris._encode_prompt(prompt)
        mx.eval(list(self._preview_encodings.values()))
        self._iris.prompt_cache = {}
        self._iris.text_encoder = None
        return True

    def save_lora_adapter(self, *, path: Path, training_spec: TrainingSpec) -> None:
        weights: dict[str, mx.array] = {}
        for target in training_spec.lora_layers.targets:
            blocks = target.blocks.get_blocks() if target.blocks is not None else [None]
            for block in blocks:
                module_path = target.module_path if block is None else target.module_path.format(block=block)
                IrisTrainingAdapter._append_train_lora_weights(weights, self._iris.transformer, module_path)
        mx.save_safetensors(
            str(path),
            weights,
            metadata={"mflux_version": VersionUtil.get_mflux_version(), "model": training_spec.model},
        )

    def load_lora_adapter(self, *, path: str | Path) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=IrisLoRAMapping.get_mapping(),
            transformer=self._iris.transformer,
            lora_paths=[str(path)],
            lora_scales=[1.0],
            role="train",
            bake_lora=False,
        )

    def load_training_adapter(self, *, path: str | Path, scale: float = 1.0) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=IrisLoRAMapping.get_mapping(),
            transformer=self._iris.transformer,
            lora_paths=[str(path)],
            lora_scales=[float(scale)],
            role="assistant",
            bake_lora=False,
        )

    @staticmethod
    def _append_train_lora_weights(weights: dict[str, mx.array], transformer, module_path: str) -> None:
        train_lora = TrainingUtil.get_train_lora(transformer, module_path)
        weights.update(
            TrainingUtil.lora_weights(f"diffusion_model.{IrisLoRAMapping.reference_name(module_path)}", train_lora)
        )
