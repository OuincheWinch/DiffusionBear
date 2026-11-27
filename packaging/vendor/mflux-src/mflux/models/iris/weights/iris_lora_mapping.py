import re

from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


class IrisLoRAMapping(LoRAMapping):
    # The transformer's module paths are the reference checkpoint's, so a LoRA names them directly: the reference repo
    # and its forks save kohya names (blocks.0.q_proj.lora_down) with no prefix, ai-toolkit saves PEFT names under
    # diffusion_model. Only two Sequential MLPs have MLX names of their own.
    PREFIXES = ("", "diffusion_model.", "transformer.", "base_model.model.")
    REFERENCE_NAMES = {
        "t_embedder.linear_1": "t_embedder.mlp.0",
        "t_embedder.linear_2": "t_embedder.mlp.2",
        "y_embedder.layer_blocks.{block}.mlp.fc1": "y_embedder.layer_blocks.{block}.mlp.0",
        "y_embedder.layer_blocks.{block}.mlp.fc2": "y_embedder.layer_blocks.{block}.mlp.2",
    }

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        return [IrisLoRAMapping._target(path) for path in IrisLoRAMapping.module_paths()]

    @staticmethod
    def reference_name(module_path: str) -> str:
        # The reference checkpoint's name for a concrete MLX module path, e.g. y_embedder.layer_blocks.1.mlp.fc1
        for mlx_name, reference in IrisLoRAMapping.REFERENCE_NAMES.items():
            match = re.fullmatch(re.escape(mlx_name).replace(re.escape("{block}"), r"(\d+)"), module_path)
            if match:
                return reference.format(block=match.group(1)) if match.groups() else reference
        return module_path

    @staticmethod
    def module_paths() -> list[str]:
        # Every linear layer. Blocks 0-7 are dual-stream (_x image, _y text) and the rest single-stream, so a file only
        # has keys for the dual names in the first blocks and for the single names in the others
        block = "blocks.{block}"
        dual = [f"{block}.attn.{proj}_{stream}" for proj in ("q_proj", "k_proj", "v_proj", "proj") for stream in "xy"]
        dual += [f"{block}.attn_gate_{stream}" for stream in "xy"]
        dual += [f"{block}.mlp_{stream}.{w}" for stream in "xy" for w in ("w1", "w2", "w3")]
        single = [f"{block}.{name}" for name in ("q_proj", "k_proj", "v_proj", "attn_gate", "attn_proj")]
        single += [f"{block}.mlp.{w}" for w in ("w1", "w2", "w3")]
        pixel = [f"pixel_blocks.{{block}}.{name}" for name in ("adaln", "attn.qkv", "attn.proj", "compress", "expand", "mlp.fc1", "mlp.fc2")]  # fmt: off
        text = [f"y_embedder.layer_blocks.{{block}}.{name}" for name in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")]  # fmt: off
        text += [f"y_embedder.refiner.blocks.{{block}}.{name}" for name in ("attn.qkv", "attn.proj", "mlp.w1", "mlp.w2", "mlp.w3")]  # fmt: off
        top = [
            "s_embedder.proj",
            "pixel_embedder.proj",
            "t_embedder.linear_1",
            "t_embedder.linear_2",
            "modulation_cores.adaln_img",
            "modulation_cores.adaln_txt",
            "final_layer.linear",
            "y_embedder.layer_pool",
            "y_embedder.refiner.proj",
        ]
        return top + dual + single + pixel + text

    @staticmethod
    def _target(path: str) -> LoRATarget:
        source = IrisLoRAMapping.REFERENCE_NAMES.get(path, path)
        names = [f"{prefix}{source}" for prefix in IrisLoRAMapping.PREFIXES]
        flat = [f"lora_unet_{source.replace('.', '_')}"]
        return LoRATarget(
            model_path=path,
            possible_up_patterns=IrisLoRAMapping._matrix(names, ("lora_up", "lora_B"))
            + [f"{n}.lora_up.weight" for n in flat],  # fmt: off
            possible_down_patterns=IrisLoRAMapping._matrix(names, ("lora_down", "lora_A"))
            + [f"{n}.lora_down.weight" for n in flat],  # fmt: off
            possible_alpha_patterns=[f"{name}.alpha" for name in names + flat],
            possible_lokr_w1_patterns=[f"{name}.lokr_w1" for name in names + flat],
            possible_lokr_w2_patterns=[f"{name}.lokr_w2" for name in names + flat],
        )

    @staticmethod
    def _matrix(names: list[str], matrices: tuple[str, ...]) -> list[str]:
        # The reference repo saves bare matrices (blocks.0.q_proj.lora_down); PEFT adds .weight, or .default.weight
        suffixes = ("", ".weight", ".default.weight")
        return [f"{name}.{matrix}{suffix}" for name in names for matrix in matrices for suffix in suffixes]
