from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from mflux.models.anima.latent_creator.anima_latent_creator import AnimaLatentCreator
from mflux.models.anima.model.anima_text_encoder.anima_prompt_encoder import AnimaPromptEncoder
from mflux.models.anima.variants.txt2img.anima import Anima
from mflux.models.anima.weights.anima_lora_mapping import AnimaLoRAMapping
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.training.adapters.base import TrainingAdapter
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.utils import TrainingUtil
from mflux.utils.version_util import VersionUtil


class AnimaTrainingAdapter(TrainingAdapter):
    # Anima is flow matching on the Qwen-Image VAE latents: the transformer takes sigma and returns the velocity
    # (noise - clean), which is what the trainer's loss expects. LoRAs are trained on Base, like ai-toolkit and the
    # model card advise, and saved with ComfyUI's keys (diffusion_model.blocks.N...), which ComfyUI, ai-toolkit and
    # mflux all read.

    def __init__(self, *, model_config: ModelConfig, quantize: int | None, model_path: str | None = None):
        if model_config.aliases[0] == "anima-turbo":
            raise ValueError("Anima LoRAs are trained on Base: set \"model\": \"anima-base\" (Turbo is distilled, and \"anima\" is Turbo).")  # fmt: off
        self._model_config = model_config
        self._anima = Anima(quantize=quantize, model_config=model_config, model_path=model_path, bake_lora=False)
        self._guidance: float | None = None

    def model(self):
        return self._anima

    def transformer(self):
        return self._anima.transformer

    def create_config(self, training_spec: TrainingSpec, *, width: int, height: int) -> Config:
        # The LLM adapter runs once per caption in encode_data, before training, so a LoRA on it would never learn
        adapter_targets = [t.module_path for t in training_spec.lora_layers.targets if t.module_path.startswith("llm_adapter")]  # fmt: off
        if adapter_targets:
            raise ValueError(f"Anima trains LoRAs on the transformer blocks only, not on {adapter_targets[0]}: the LLM adapter's output is computed once per caption.")  # fmt: off
        # A config without guidance gives 0.0, which would make the preview the negative prompt's image: use the default
        self._guidance = training_spec.guidance or None
        return Config(
            model_config=self._model_config,
            num_inference_steps=training_spec.steps,
            width=width,
            height=height,
            guidance=training_spec.guidance,
            scheduler="anima",
        )

    def freeze_base(self) -> None:
        self._anima.vae.freeze()
        self._anima.transformer.freeze()
        self._anima.text_encoder.freeze()

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self._anima.transformer.gradient_checkpointing = enabled

    def encode_data(
        self,
        *,
        data_id: int,
        image_path: Path,
        prompt: str,
        width: int,
        height: int,
        input_image_path: Path | None = None,
    ) -> tuple[mx.array, Any]:
        # low_ram sets a tiling config on the model, which the encode of a large image needs as much as the decode
        encoded = LatentCreator.encode_image(
            vae=self._anima.vae,
            image_path=image_path,
            height=height,
            width=width,
            tiling_config=self._anima.tiling_config,
        )
        clean_latents = AnimaLatentCreator.pack_latents(encoded, height, width)
        # Captions are literal text, as ai-toolkit reads them: "hatsune miku (vocaloid)" is a Danbooru tag, not a weight
        context = AnimaPromptEncoder.encode(
            prompt=prompt,
            qwen_tokenizer=self._anima.tokenizers["qwen3"],
            t5_tokenizer=self._anima.tokenizers["t5"],
            text_encoder=self._anima.text_encoder,
            transformer=self._anima.transformer,
            weighted=False,
        )
        mx.eval(clean_latents, context)
        return clean_latents, context

    def predict_noise(self, *, t: int, latents_t: mx.array, sigmas: mx.array, cond: Any, config: Config) -> mx.array:  # noqa: ARG002
        sigma = mx.reshape(sigmas[t], (1,))
        return self._anima.transformer(latents_t, sigma, cond).astype(latents_t.dtype)

    def generate_preview_image(
        self,
        *,
        seed: int,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        image_paths: list[Path | str] | None = None,
    ):
        with self._assistant_disabled():
            image = self._anima.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=self._guidance,
            )
        # The cache lives on the module, so its contexts would become trainable parameters of model()
        self._anima.prompt_cache = {}
        return image

    def save_lora_adapter(self, *, path: Path, training_spec: TrainingSpec) -> None:
        weights: dict[str, mx.array] = {}
        for target in training_spec.lora_layers.targets:
            if target.blocks is not None:
                for b in target.blocks.get_blocks():
                    AnimaTrainingAdapter._append_train_lora_weights(weights, self._anima.transformer, target.module_path.format(block=b))  # fmt: off
            else:
                AnimaTrainingAdapter._append_train_lora_weights(weights, self._anima.transformer, target.module_path)
        mx.save_safetensors(
            str(path),
            weights,
            metadata={"mflux_version": VersionUtil.get_mflux_version(), "model": training_spec.model},
        )

    def load_lora_adapter(self, *, path: str | Path) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=AnimaLoRAMapping.get_mapping(),
            transformer=self._anima.transformer,
            lora_paths=[str(path)],
            lora_scales=[1.0],
            role="train",
            bake_lora=False,
        )

    def load_training_adapter(self, *, path: str | Path, scale: float = 1.0) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=AnimaLoRAMapping.get_mapping(),
            transformer=self._anima.transformer,
            lora_paths=[str(path)],
            lora_scales=[float(scale)],
            role="assistant",
            bake_lora=False,
        )

    @staticmethod
    def _append_train_lora_weights(weights: dict[str, mx.array], transformer, module_path: str) -> None:
        train_lora = TrainingUtil.get_train_lora(transformer, module_path)
        weights[f"diffusion_model.{module_path}.lora_A.weight"] = mx.transpose(train_lora.lora_A)
        weights[f"diffusion_model.{module_path}.lora_B.weight"] = mx.transpose(train_lora.lora_B)
        # An alpha other than the rank trains at alpha / rank; the file keeps it so the LoRA loads at that strength.
        # With the default (alpha = rank) the file has no alpha keys, like ai-toolkit's.
        if train_lora.scale != 1.0:
            weights[f"diffusion_model.{module_path}.alpha"] = mx.array(train_lora.scale * train_lora.lora_A.shape[1])

    def _assistant_disabled(self):
        return TrainingUtil.assistant_disabled(self._anima.transformer)
