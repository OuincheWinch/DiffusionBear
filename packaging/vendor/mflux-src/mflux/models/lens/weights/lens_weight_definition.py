from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition

# The single-file native checkpoint inside the Comfy-Org mirror (the Microsoft
# originals were withdrawn). Its tensor names already match LensTransformer's module
# paths, so the component loads in passthrough mode, the same arrangement as the
# ERNIE and Krea 2 native checkpoints.
TURBO_WEIGHTS_PATTERN = "diffusion_models/lens_turbo_bf16.safetensors"


class LensWeightDefinition:
    """Lens Turbo assembles from three repositories, and this definition covers the one
    the model config names: the DiT repo. The FLUX.2 VAE loads through
    Flux2KleinWeightDefinition's vae component from the klein repo, and the GPT-OSS 20B
    text encoder is a pre-quantized mlx-format checkpoint whose config carries its own
    quantization recipe (mxfp4 experts + q8 elsewhere), so it self-describes through
    LensGptOssEncoder and never passes through WeightApplier."""

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir="diffusion_models",
                loading_mode="mlx_native",
                weight_files=["lens_turbo_bf16.safetensors"],
                precision=ModelConfig.precision,
                mapping_getter=None,  # direct load; checkpoint keys are the module paths
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [TURBO_WEIGHTS_PATTERN]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")
