from dataclasses import replace
from pathlib import Path
from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping
from mflux.models.flux3.weights.flux3_weight_mapping import Flux3WeightMapping


class Flux3KleinWeightDefinition:
    # The text encoder is the Qwen3-VL-4B language tower; layers beyond the last output layer (32) are never read.
    TEXT_ENCODER_LAYERS = 32

    @staticmethod
    def _text_encoder_key(key: str) -> str | None:
        for prefix in ("model.language_model.", "language_model.", "model."):
            if key.startswith(prefix):
                name = key[len(prefix) :]
                if name.startswith("layers."):
                    try:
                        if int(name.split(".")[1]) >= Flux3KleinWeightDefinition.TEXT_ENCODER_LAYERS:
                            return None
                    except (IndexError, ValueError):
                        pass
                return "model." + name
        return None

    @classmethod
    def _transformer_mapping_getter(cls):
        return Flux3WeightMapping.get_transformer_mapping

    @classmethod
    def _select_transformer_variant(cls, root_path: Path) -> ComponentDefinition:
        mapping_fn = cls._transformer_mapping_getter()
        for native_file in ("model.safetensors", "transformer.safetensors"):
            if (root_path / native_file).is_file():
                return ComponentDefinition(
                    name="transformer",
                    hf_subdir="",
                    weight_files=[native_file],
                    precision=ModelConfig.precision,
                    mapping_getter=mapping_fn,
                )
        return ComponentDefinition(
            name="transformer",
            hf_subdir="transformer",
            precision=ModelConfig.precision,
            mapping_getter=mapping_fn,
        )

    @staticmethod
    def _select_vae_variant(root_path: Path) -> ComponentDefinition:
        for native_file in ("ae.safetensors", "vae.safetensors"):
            if (root_path / native_file).is_file():
                return ComponentDefinition(
                    name="vae",
                    hf_subdir="",
                    weight_files=[native_file],
                    precision=ModelConfig.precision,
                    mapping_getter=Flux3WeightMapping.get_vae_mapping,
                )
        return ComponentDefinition(
            name="vae",
            hf_subdir="vae",
            precision=ModelConfig.precision,
            mapping_getter=Flux3WeightMapping.get_vae_mapping,
        )

    @classmethod
    def get_components(cls) -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=Flux3WeightMapping.get_vae_mapping,
                variant_selector=cls._select_vae_variant,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=cls._transformer_mapping_getter(),
                variant_selector=cls._select_transformer_variant,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                precision=ModelConfig.precision,
                mapping_getter=Flux2WeightMapping.get_text_encoder_mapping,
                num_layers=Flux3KleinWeightDefinition.TEXT_ENCODER_LAYERS,
                key_transform=Flux3KleinWeightDefinition._text_encoder_key,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen3_vl",
                hf_subdir="text_encoder",
                fallback_subdirs=["tokenizer", "processor", "."],
                tokenizer_class="AutoTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=4096,
                use_chat_template=True,
                download_patterns=[
                    "text_encoder/tokenizer*",
                    "text_encoder/vocab.json",
                    "text_encoder/merges.txt",
                    "text_encoder/chat_template.json",
                    "text_encoder/chat_template.jinja",
                    "tokenizer*",
                    "vocab.json",
                    "merges.txt",
                ],  # fmt: skip
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "*.safetensors",
            "*.json",
            "vae/*.safetensors",
            "vae/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "text_encoder/merges.txt",
            "text_encoder/chat_template.jinja",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")


class Flux3KleinBaseWeightDefinition(Flux3KleinWeightDefinition):
    # Klein Base is undistilled: same layout as the distilled checkpoint minus target_timestep_embedder.
    @classmethod
    def _transformer_mapping_getter(cls):
        return Flux3WeightMapping.get_base_transformer_mapping

    @classmethod
    def get_components(cls) -> List[ComponentDefinition]:
        return [
            replace(
                component,
                mapping_getter=Flux3WeightMapping.get_base_transformer_mapping,
                variant_selector=cls._select_transformer_variant,
            )
            if component.name == "transformer"
            else component
            for component in Flux3KleinWeightDefinition.get_components()
        ]