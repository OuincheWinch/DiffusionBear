from typing import List

import mlx.core as mx

from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import Qwen21PromptEncoder
from mflux.models.qwen21.weights.qwen21_weight_mapping import Qwen21WeightMapping


class Qwen21WeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                loading_mode="single",
                mapping_getter=Qwen21WeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                loading_mode="multi_glob",
                mapping_getter=Qwen21WeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                loading_mode="multi_json",
                # NOTE (16GB build): upstream forces the Qwen3-VL text encoder to bf16
                # (~17.5 GB resident), which is not feasible on 16 GB machines and makes
                # pre-quantized repos (e.g. mlx-community/Qwen-Image-2.1-MLX-4bit) unloadable.
                # Removing precision/skip_quantization lets the encoder follow the same
                # quantization path as the transformer and VAE, so `-q 4` and native q4
                # checkpoints both fit. Quality tradeoff vs. bf16: upstream warns of
                # "significant semantic degradation" - user opted in.
                mapping_getter=Qwen21WeightMapping.get_text_encoder_mapping,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen21",
                hf_subdir="processor",
                tokenizer_class="AutoTokenizer",
                encoder_class=LanguageTokenizer,
                max_length=2048,
                padding="longest",
                template=Qwen21PromptEncoder.PROMPT_TEMPLATE_T2I,
                download_patterns=["processor/**"],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "vae/*.safetensors",
            "vae/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
        ]

    @staticmethod
    def quantization_predicate(path: str, module, bits: int | None = None):
        if not hasattr(module, "to_quantized"):
            return False
        return True
