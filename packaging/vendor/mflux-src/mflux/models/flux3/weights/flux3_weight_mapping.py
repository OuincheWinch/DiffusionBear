from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget
from mflux.models.common.weights.mapping.weight_transforms import WeightTransforms
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping


class Flux3WeightMapping(WeightMapping):
    # Supports both diffusers-style checkpoints and official Black Forest Labs (BFL) native checkpoints
    # in full precision (float32, bfloat16, float16) and quantized (weight, scales, biases).
    _BASE_BLOCK_PARTS = [
        ("attn.to_q", "attn.to_q", "q_proj"),
        ("attn.to_k", "attn.to_k", "k_proj"),
        ("attn.to_v", "attn.to_v", "v_proj"),
        ("attn.to_out", "attn.to_out.0", "attn_out"),
        ("ff.linear_in", "ff.linear_in", "mlp_in"),
        ("ff.linear_out", "ff.linear_out", "mlp_out"),
    ]

    @classmethod
    def _get_mode_block_parts(cls):
        parts = []
        for to_p, diff_p, bfl_p in cls._BASE_BLOCK_PARTS:
            for suffix in (".weight", ".scales", ".biases"):
                parts.append((to_p + suffix, diff_p + suffix, bfl_p + suffix))
        parts.append(("attn.norm_q.weight", "attn.norm_q.weight", "norm.query_norm.scale"))
        parts.append(("attn.norm_k.weight", "attn.norm_k.weight", "norm.key_norm.scale"))
        return parts

    @staticmethod
    def get_transformer_mapping(flow_map: bool = True) -> List[WeightTarget]:
        named_bases = [
            ("context_embedder", ["context_embedder", "txt_in"]),
            ("timestep_embedder.linear_1", ["timestep_embedder.linear_1", "time_in.in_layer"]),
            ("timestep_embedder.linear_2", ["timestep_embedder.linear_2", "time_in.out_layer"]),
            (
                "norm_out.linear",
                ["norm_out.linear", "final_layer.image.adaLN_modulation.1", "final_layer.adaLN_modulation.1"],
            ),
            ("proj_out", ["proj_out", "final_layer.image.linear", "final_layer.linear"]),
        ]
        target_time_bases = [
            ("target_timestep_embedder.linear_1", ["target_timestep_embedder.linear_1", "ref_time_in.in_layer"]),
            ("target_timestep_embedder.linear_2", ["target_timestep_embedder.linear_2", "ref_time_in.out_layer"]),
        ]

        named_pairs = []
        for to_b, from_bs in named_bases:
            for suffix in (".weight", ".scales", ".biases"):
                named_pairs.append((to_b + suffix, [f + suffix for f in from_bs]))

        target_time_pairs = []
        for to_b, from_bs in target_time_bases:
            for suffix in (".weight", ".scales", ".biases"):
                target_time_pairs.append((to_b + suffix, [f + suffix for f in from_bs]))

        if flow_map:
            named_pairs += target_time_pairs

        for mode in ("image", "image_cond"):
            for suffix in (".weight", ".scales", ".biases"):
                named_pairs.append((f"x_embedders.{mode}{suffix}", [f"x_embedders.{mode}{suffix}", f"emb_in.{mode}{suffix}"]))

        for mode in ("image", "image_cond", "txt"):
            for suffix in (".weight", ".scales", ".biases"):
                named_pairs.append((
                    f"early_stream_modulations.{mode}.linear{suffix}",
                    [f"early_stream_modulations.{mode}.linear{suffix}", f"early_stream_modulations.{mode}.lin{suffix}"],
                ))
                named_pairs.append((
                    f"single_stream_modulations.{mode}.linear{suffix}",
                    [f"single_stream_modulations.{mode}.linear{suffix}", f"single_stream_modulations.{mode}.lin{suffix}"],
                ))

        targets = [WeightTarget(to_pattern=to_p, from_pattern=from_ps, required=False) for to_p, from_ps in named_pairs]
        if not flow_map:
            targets += [WeightTarget(to_pattern=to_p, from_pattern=from_ps, required=False) for to_p, from_ps in target_time_pairs]

        prefix_map = [
            ("mode_blocks.image", "content_mode_blocks.image", 5),
            ("mode_blocks.image_cond", "content_mode_blocks.image_cond", 5),
            ("txt_mode_blocks", "txt_mode_blocks", 5),
            ("single_transformer_blocks", "single_blocks", 28),
        ]
        for prefix, bfl_prefix, max_blocks in prefix_map:
            targets += Flux3WeightMapping._block_targets(prefix, bfl_prefix, max_blocks=max_blocks)
        return targets

    @staticmethod
    def get_base_transformer_mapping() -> List[WeightTarget]:
        return Flux3WeightMapping.get_transformer_mapping(flow_map=False)

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        # Include Diffusers FLUX.2 VAE targets
        targets = list(Flux2WeightMapping.get_vae_mapping())

        # Native Black Forest Labs (BFL) AutoEncoder targets (ae.safetensors layout)
        targets.extend([
            WeightTarget(to_pattern="bn.running_mean", from_pattern=["bn.running_mean"]),
            WeightTarget(to_pattern="bn.running_var", from_pattern=["bn.running_var"]),
            WeightTarget(
                to_pattern="post_quant_conv.weight",
                from_pattern=["decoder.post_quant_conv.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
                required=False,
            ),
            WeightTarget(
                to_pattern="post_quant_conv.bias",
                from_pattern=["decoder.post_quant_conv.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="quant_conv.weight",
                from_pattern=["encoder.quant_conv.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
                required=False,
            ),
            WeightTarget(
                to_pattern="quant_conv.bias",
                from_pattern=["encoder.quant_conv.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="decoder.conv_in.weight",
                from_pattern=["decoder.conv_in.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(to_pattern="decoder.conv_in.bias", from_pattern=["decoder.conv_in.bias"]),
            WeightTarget(
                to_pattern="decoder.conv_out.weight",
                from_pattern=["decoder.conv_out.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(to_pattern="decoder.conv_out.bias", from_pattern=["decoder.conv_out.bias"]),
            WeightTarget(
                to_pattern="decoder.conv_norm_out.weight",
                from_pattern=["decoder.norm_out.weight", "decoder.conv_norm_out.weight"],
            ),
            WeightTarget(
                to_pattern="decoder.conv_norm_out.bias",
                from_pattern=["decoder.norm_out.bias", "decoder.conv_norm_out.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.conv_in.weight",
                from_pattern=["encoder.conv_in.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(to_pattern="encoder.conv_in.bias", from_pattern=["encoder.conv_in.bias"]),
            WeightTarget(
                to_pattern="encoder.conv_out.weight",
                from_pattern=["encoder.conv_out.weight"],
                transform=WeightTransforms.transpose_conv2d_weight,
            ),
            WeightTarget(to_pattern="encoder.conv_out.bias", from_pattern=["encoder.conv_out.bias"]),
            WeightTarget(
                to_pattern="encoder.conv_norm_out.weight",
                from_pattern=["encoder.norm_out.weight", "encoder.conv_norm_out.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.conv_norm_out.bias",
                from_pattern=["encoder.norm_out.bias", "encoder.conv_norm_out.bias"],
            ),
        ])

        # Decoder Mid Block
        for suffix in ("weight", "bias"):
            targets.append(
                WeightTarget(
                    to_pattern=f"decoder.mid_block.resnets.0.norm1.{suffix}",
                    from_pattern=[f"decoder.mid.block_1.norm1.{suffix}"],
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"decoder.mid_block.resnets.0.norm2.{suffix}",
                    from_pattern=[f"decoder.mid.block_1.norm2.{suffix}"],
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"decoder.mid_block.resnets.1.norm1.{suffix}",
                    from_pattern=[f"decoder.mid.block_2.norm1.{suffix}"],
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"decoder.mid_block.resnets.1.norm2.{suffix}",
                    from_pattern=[f"decoder.mid.block_2.norm2.{suffix}"],
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"decoder.mid_block.attentions.0.group_norm.{suffix}",
                    from_pattern=[f"decoder.mid.attn_1.norm.{suffix}"],
                )
            )

        for conv in ("conv1", "conv2"):
            targets.append(
                WeightTarget(
                    to_pattern=f"decoder.mid_block.resnets.0.{conv}.weight",
                    from_pattern=[f"decoder.mid.block_1.{conv}.weight"],
                    transform=WeightTransforms.transpose_conv2d_weight,
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"decoder.mid_block.resnets.0.{conv}.bias",
                    from_pattern=[f"decoder.mid.block_1.{conv}.bias"],
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"decoder.mid_block.resnets.1.{conv}.weight",
                    from_pattern=[f"decoder.mid.block_2.{conv}.weight"],
                    transform=WeightTransforms.transpose_conv2d_weight,
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"decoder.mid_block.resnets.1.{conv}.bias",
                    from_pattern=[f"decoder.mid.block_2.{conv}.bias"],
                )
            )

        for proj, bfl_proj in [("to_q", "q"), ("to_k", "k"), ("to_v", "v"), ("to_out", "proj_out")]:
            targets.append(
                WeightTarget(
                    to_pattern=f"decoder.mid_block.attentions.0.{proj}.weight",
                    from_pattern=[f"decoder.mid.attn_1.{bfl_proj}.weight"],
                    transform=WeightTransforms.squeeze_to_2d,
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"decoder.mid_block.attentions.0.{proj}.bias",
                    from_pattern=[f"decoder.mid.attn_1.{bfl_proj}.bias"],
                )
            )

        # Decoder Up Blocks (BFL level 0..3 maps to Diffusers up_blocks 3..0)
        for level in range(4):
            up_idx = 3 - level
            for blk in range(3):
                prefix_to = f"decoder.up_blocks.{up_idx}.resnets.{blk}"
                prefix_from = f"decoder.up.{level}.block.{blk}"
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.norm1.weight",
                        from_pattern=[f"{prefix_from}.norm1.weight"],
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.norm1.bias",
                        from_pattern=[f"{prefix_from}.norm1.bias"],
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.norm2.weight",
                        from_pattern=[f"{prefix_from}.norm2.weight"],
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.norm2.bias",
                        from_pattern=[f"{prefix_from}.norm2.bias"],
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv1.weight",
                        from_pattern=[f"{prefix_from}.conv1.weight"],
                        transform=WeightTransforms.transpose_conv2d_weight,
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv1.bias",
                        from_pattern=[f"{prefix_from}.conv1.bias"],
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv2.weight",
                        from_pattern=[f"{prefix_from}.conv2.weight"],
                        transform=WeightTransforms.transpose_conv2d_weight,
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv2.bias",
                        from_pattern=[f"{prefix_from}.conv2.bias"],
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv_shortcut.weight",
                        from_pattern=[f"{prefix_from}.nin_shortcut.weight"],
                        transform=WeightTransforms.transpose_conv2d_weight,
                        required=False,
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv_shortcut.bias",
                        from_pattern=[f"{prefix_from}.nin_shortcut.bias"],
                        required=False,
                    )
                )

            if level != 0:
                prefix_to = f"decoder.up_blocks.{up_idx}.upsamplers.0"
                prefix_from = f"decoder.up.{level}.upsample"
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv.weight",
                        from_pattern=[f"{prefix_from}.conv.weight"],
                        transform=WeightTransforms.transpose_conv2d_weight,
                        required=False,
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv.bias",
                        from_pattern=[f"{prefix_from}.conv.bias"],
                        required=False,
                    )
                )

        # Encoder Down Blocks
        for level in range(4):
            for blk in range(2):
                prefix_to = f"encoder.down_blocks.{level}.resnets.{blk}"
                prefix_from = f"encoder.down.{level}.block.{blk}"
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.norm1.weight",
                        from_pattern=[f"{prefix_from}.norm1.weight"],
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.norm1.bias",
                        from_pattern=[f"{prefix_from}.norm1.bias"],
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.norm2.weight",
                        from_pattern=[f"{prefix_from}.norm2.weight"],
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.norm2.bias",
                        from_pattern=[f"{prefix_from}.norm2.bias"],
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv1.weight",
                        from_pattern=[f"{prefix_from}.conv1.weight"],
                        transform=WeightTransforms.transpose_conv2d_weight,
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv1.bias",
                        from_pattern=[f"{prefix_from}.conv1.bias"],
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv2.weight",
                        from_pattern=[f"{prefix_from}.conv2.weight"],
                        transform=WeightTransforms.transpose_conv2d_weight,
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv2.bias",
                        from_pattern=[f"{prefix_from}.conv2.bias"],
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv_shortcut.weight",
                        from_pattern=[f"{prefix_from}.nin_shortcut.weight"],
                        transform=WeightTransforms.transpose_conv2d_weight,
                        required=False,
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv_shortcut.bias",
                        from_pattern=[f"{prefix_from}.nin_shortcut.bias"],
                        required=False,
                    )
                )

            if level != 3:
                prefix_to = f"encoder.down_blocks.{level}.downsamplers.0"
                prefix_from = f"encoder.down.{level}.downsample"
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv.weight",
                        from_pattern=[f"{prefix_from}.conv.weight"],
                        transform=WeightTransforms.transpose_conv2d_weight,
                        required=False,
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix_to}.conv.bias",
                        from_pattern=[f"{prefix_from}.conv.bias"],
                        required=False,
                    )
                )

        # Encoder Mid Block
        for suffix in ("weight", "bias"):
            targets.append(
                WeightTarget(
                    to_pattern=f"encoder.mid_block.resnets.0.norm1.{suffix}",
                    from_pattern=[f"encoder.mid.block_1.norm1.{suffix}"],
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"encoder.mid_block.resnets.0.norm2.{suffix}",
                    from_pattern=[f"encoder.mid.block_1.norm2.{suffix}"],
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"encoder.mid_block.resnets.1.norm1.{suffix}",
                    from_pattern=[f"encoder.mid.block_2.norm1.{suffix}"],
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"encoder.mid_block.resnets.1.norm2.{suffix}",
                    from_pattern=[f"encoder.mid.block_2.norm2.{suffix}"],
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"encoder.mid_block.attentions.0.group_norm.{suffix}",
                    from_pattern=[f"encoder.mid.attn_1.norm.{suffix}"],
                )
            )

        for conv in ("conv1", "conv2"):
            targets.append(
                WeightTarget(
                    to_pattern=f"encoder.mid_block.resnets.0.{conv}.weight",
                    from_pattern=[f"encoder.mid.block_1.{conv}.weight"],
                    transform=WeightTransforms.transpose_conv2d_weight,
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"encoder.mid_block.resnets.0.{conv}.bias",
                    from_pattern=[f"encoder.mid.block_1.{conv}.bias"],
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"encoder.mid_block.resnets.1.{conv}.weight",
                    from_pattern=[f"encoder.mid.block_2.{conv}.weight"],
                    transform=WeightTransforms.transpose_conv2d_weight,
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"encoder.mid_block.resnets.1.{conv}.bias",
                    from_pattern=[f"encoder.mid.block_2.{conv}.bias"],
                )
            )

        for proj, bfl_proj in [("to_q", "q"), ("to_k", "k"), ("to_v", "v"), ("to_out", "proj_out")]:
            targets.append(
                WeightTarget(
                    to_pattern=f"encoder.mid_block.attentions.0.{proj}.weight",
                    from_pattern=[f"encoder.mid.attn_1.{bfl_proj}.weight"],
                    transform=WeightTransforms.squeeze_to_2d,
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"encoder.mid_block.attentions.0.{proj}.bias",
                    from_pattern=[f"encoder.mid.attn_1.{bfl_proj}.bias"],
                )
            )

        return targets

    @staticmethod
    def _block_targets(prefix: str, bfl_prefix: str, max_blocks: int) -> List[WeightTarget]:
        return [
            WeightTarget(
                to_pattern=f"{prefix}.{{block}}.{to_part}",
                from_pattern=[f"{prefix}.{{block}}.{diffusers_part}", f"{bfl_prefix}.{{block}}.{bfl_part}"],
                max_blocks=max_blocks,
                required=False,
            )
            for to_part, diffusers_part, bfl_part in Flux3WeightMapping._get_mode_block_parts()
        ]