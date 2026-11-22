from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget

PREFIXES = ("", "diffusion_model.", "transformer.", "base_model.model.")


class MOSSImageLoRAMapping(LoRAMapping):
    # MOSS-Image LoRAs train the six projections of each transformer layer. The MLX modules have the names of the
    # official checkpoint, so the formats differ only in the prefix and the matrix names: kohya files write
    # diffusion_model.layers.N.proj.q.lora_down.weight with an alpha, PEFT files write layers.N.proj.q.lora_A.
    PROJECTIONS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        return [
            MOSSImageLoRAMapping._target(f"layers.{{block}}.proj.{name}") for name in MOSSImageLoRAMapping.PROJECTIONS
        ]

    @staticmethod
    def _target(model_path: str) -> LoRATarget:
        paths = [f"{prefix}{model_path}" for prefix in PREFIXES]
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=MOSSImageLoRAMapping._matrix_patterns(paths, ("lora_B", "lora_up")),
            possible_down_patterns=MOSSImageLoRAMapping._matrix_patterns(paths, ("lora_A", "lora_down")),
            possible_alpha_patterns=[f"{path}.alpha" for path in paths],
        )

    @staticmethod
    def _matrix_patterns(paths: list[str], names: tuple[str, str]) -> list[str]:
        # The loader also matches each pattern without its .weight suffix, which is how PEFT names the matrices.
        return [f"{path}.{name}{default}.weight" for path in paths for name in names for default in ("", ".default")]
