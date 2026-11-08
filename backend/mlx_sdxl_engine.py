"""Self-contained MLX SDXL engine for DiffusionBear.

Public contract:
- ``class MLXSDXLEngine`` with ``__init__(model_dir=None, quantize_unet=4)``
  and ``generate(req: dict) -> dict``.
- ``get_pipeline(model_dir=None, quantize_unet=4) -> MLXSDXLEngine`` returns
  the resident engine, keyed on the resolved model directory and quantization.
- ``generate(req: dict) -> dict`` is the module-level daemon entry point. It
  validates the request, selects or creates the resident engine, and delegates.
- ``_validate_request(req: dict) -> dict`` applies every bound check and returns
  a normalized request with defaults filled in.
- ``_write_ipc_json(obj: dict) -> None`` writes one JSON line for the daemon.

Runs under the main venv (Python 3.10). The inference path imports no ``torch``
or ``diffusers``, and none of the removed second-runtime packages. Base models
are diffusers directories; only LoRA adapters are single-file ``.safetensors``.
Output written to ``dest`` is raw RGB bytes, not a PNG. One generation runs at a
time.
"""

import dataclasses
import gc
import json
import math
import os
import re
import struct
import sys
import threading
import time

from collections import OrderedDict

from pathlib import Path

import numpy as np

# Ensure backend directory is in sys.path for local imports like taesd_mlx
_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import mlx.core as mx
import mlx.nn as nn

_ipc_stdout_lock = threading.Lock()

_UNET_INDEX = "diffusion_pytorch_model.safetensors.index.json"


def _component_shards(folder: Path) -> list:
    """Resolve a component's safetensors shards, refusing path escapes."""
    folder = Path(folder)
    root = folder.resolve()
    index_path = folder / _UNET_INDEX
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ValueError(f"unreadable safetensors index: {index_path}") from e
        names = sorted(set((index.get("weight_map") or {}).values()))
        if not names:
            raise FileNotFoundError(f"no safetensors shards named in {index_path}")
        files = []
        missing = []
        for name in names:
            candidate = folder / name
            try:
                resolved = candidate.resolve()
            except OSError as e:
                raise ValueError(f"unresolvable safetensors shard: {name}") from e
            try:
                resolved.relative_to(root)
            except ValueError as e:
                raise ValueError(
                    f"safetensors shard escapes component folder: {name}"
                ) from e
            if resolved.suffix.lower() != ".safetensors":
                raise ValueError(
                    f"safetensors shard must end in .safetensors: {name}"
                )
            if not resolved.is_file():
                missing.append(str(candidate))
            files.append(resolved)
        if missing:
            raise FileNotFoundError(
                "missing safetensors shards: " + ", ".join(missing)
            )
        return files
    files = []
    seen = set()
    for candidate in sorted(folder.glob("*.safetensors")):
        resolved = candidate.resolve()
        if resolved in seen:
            # A canonical symlink plus its target both match the glob. Loading
            # both yields duplicate parameter keys and makes MLX's
            # tree_unflatten recurse forever.
            continue
        seen.add(resolved)
        files.append(resolved)
    if not files:
        raise FileNotFoundError(f"no safetensors files found in {folder}")
    return files


def _shard_metadata(path: Path) -> dict:
    """Read one shard's tensor names, dtypes and shapes without loading data."""
    path = Path(path)
    file_size = path.stat().st_size
    with open(path, "rb") as fh:
        header, _ = _read_safetensors_header(fh, path, file_size)
    return {name: info for name, info in header.items() if name != "__metadata__"}


def _component_metadata(files: list) -> dict:
    metadata: dict = {}
    for path in files:
        metadata.update(_shard_metadata(path))
    return metadata


def load_sharded_safetensors(folder: Path, chunk_size: int = 8) -> dict:
    """Merge an SDXL component's safetensors shards into one dict.

    Shards are evaluated in bounded chunks so a large component does not hold
    every lazy array plus every evaluated copy at once.
    """
    files = _component_shards(folder)
    tensors: dict = {}
    pending = []
    for path in files:
        shard = mx.load(str(path))
        try:
            for name, tensor in shard.items():
                tensors[name] = tensor
                pending.append(tensor)
                if len(pending) >= max(1, int(chunk_size)):
                    mx.eval(*pending)
                    pending.clear()
                    try:
                        mx.clear_cache()
                    except Exception:
                        pass
        finally:
            del shard
    if pending:
        mx.eval(*pending)
    return tensors


def convert_conv_weight(weight: mx.array) -> mx.array:
    """Convert PyTorch conv weights (O, I, ...) to MLX (O, ..., I)."""
    if weight.ndim == 4:
        return weight.transpose(0, 2, 3, 1)
    if weight.ndim == 5:
        return weight.transpose(0, 2, 3, 4, 1)
    return weight


def _timestep_embedding(
    timesteps: mx.array,
    dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
) -> mx.array:
    """Sinusoidal embedding matching diffusers' get_timestep_embedding."""
    half = dim // 2
    exponent = -math.log(10000.0) * (
        mx.arange(half).astype(mx.float32) / max(half - downscale_freq_shift, 1)
    )
    freqs = mx.exp(exponent)
    emb = timesteps.reshape(-1, 1).astype(mx.float32) * freqs[None, :]
    if flip_sin_to_cos:
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    else:
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if dim % 2:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


def _exact_gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x * 0.7071067811865476))


def _group_norm(groups: int, channels: int, eps: float) -> nn.GroupNorm:
    return nn.GroupNorm(
        groups, channels, eps=eps, affine=True, pytorch_compatible=True
    )


class _TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, sample: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(sample)))


class _ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        groups: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = _group_norm(groups, in_channels, eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(time_embed_dim, out_channels)
        self.norm2 = _group_norm(groups, out_channels, eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def __call__(self, x: mx.array, temb: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = h + self.time_emb_proj(nn.silu(temb))[:, None, None, :]
        h = self.conv2(nn.silu(self.norm2(h)))
        shortcut = self.conv_shortcut(x) if hasattr(self, "conv_shortcut") else x
        return shortcut + h


class _Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_dim: int,
        heads: int,
        dim_head: int,
    ) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_dim, inner_dim, bias=False)
        self.to_out = [nn.Linear(inner_dim, query_dim), nn.Dropout(0.0)]

    def _split_heads(self, x: mx.array) -> mx.array:
        batch, seq, _ = x.shape
        return x.reshape(batch, seq, self.heads, self.dim_head).transpose(
            0, 2, 1, 3
        )

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        batch, seq, _ = x.shape
        query = self._split_heads(self.to_q(x))
        key = self._split_heads(self.to_k(context))
        value = self._split_heads(self.to_v(context))
        out = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.dim_head ** -0.5,
        )
        out = out.transpose(0, 2, 1, 3).reshape(
            batch, seq, self.heads * self.dim_head
        )
        return self.to_out[0](out)


class _GEGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)

    def __call__(self, x: mx.array) -> mx.array:
        hidden, gate = mx.split(self.proj(x), 2, axis=-1)
        return hidden * _exact_gelu(gate)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        inner = dim * mult
        self.net = [_GEGLU(dim, inner), nn.Dropout(0.0), nn.Linear(inner, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.net[2](self.net[0](x))


class _BasicTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, cross_dim: int, heads: int, dim_head: int
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = _Attention(dim, dim, heads, dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = _Attention(dim, cross_dim, heads, dim_head)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        x = x + self.attn1(self.norm1(x), self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class _Transformer2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cross_dim: int,
        heads: int,
        dim_head: int,
        num_layers: int,
        groups: int,
        eps: float,
        use_linear_projection: bool,
    ) -> None:
        super().__init__()
        self.norm = _group_norm(groups, in_channels, eps)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, in_channels)
            self.proj_out = nn.Linear(in_channels, in_channels)
        else:
            self.proj_in = nn.Conv2d(in_channels, in_channels, 1)
            self.proj_out = nn.Conv2d(in_channels, in_channels, 1)
        self.transformer_blocks = [
            _BasicTransformerBlock(in_channels, cross_dim, heads, dim_head)
            for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        residual = x
        h = self.proj_in(self.norm(x))
        batch, height, width, channels = h.shape
        h = h.reshape(batch, height * width, channels)
        for block in self.transformer_blocks:
            h = block(h, context)
        h = self.proj_out(h).reshape(batch, height, width, channels)
        return h + residual


class _Downsample(nn.Module):
    def __init__(self, channels: int, padding: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, 3, stride=2, padding=padding
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.Upsample(scale_factor=2, mode="nearest")(x)
        return self.conv(x)


class _DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        time_embed_dim: int,
        groups: int,
        eps: float,
        has_attention: bool,
        cross_dim: int,
        heads: int,
        dim_head: int,
        transformer_layers: int,
        add_downsample: bool,
        downsample_padding: int,
        use_linear_projection: bool,
    ) -> None:
        super().__init__()
        self.resnets = []
        self.attentions = []
        for i in range(num_layers):
            resnet_in = in_channels if i == 0 else out_channels
            self.resnets.append(
                _ResnetBlock(
                    resnet_in, out_channels, time_embed_dim, groups, eps
                )
            )
            if has_attention:
                self.attentions.append(
                    _Transformer2D(
                        out_channels,
                        cross_dim,
                        heads,
                        dim_head,
                        transformer_layers,
                        groups,
                        eps,
                        use_linear_projection,
                    )
                )
        self.downsamplers = []
        if add_downsample:
            self.downsamplers.append(
                _Downsample(out_channels, downsample_padding)
            )

    def __call__(
        self, x: mx.array, temb: mx.array, context: mx.array
    ) -> tuple:
        output_states = []
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, temb)
            if self.attentions:
                x = self.attentions[i](x, context)
            output_states.append(x)
        if self.downsamplers:
            x = self.downsamplers[0](x)
            output_states.append(x)
        return x, output_states


class _UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        num_layers: int,
        time_embed_dim: int,
        groups: int,
        eps: float,
        has_attention: bool,
        cross_dim: int,
        heads: int,
        dim_head: int,
        transformer_layers: int,
        add_upsample: bool,
        use_linear_projection: bool,
    ) -> None:
        super().__init__()
        self.resnets = []
        self.attentions = []
        current = prev_output_channel
        for i in range(num_layers):
            res_skip_channels = (
                in_channels if i == num_layers - 1 else out_channels
            )
            self.resnets.append(
                _ResnetBlock(
                    current + res_skip_channels,
                    out_channels,
                    time_embed_dim,
                    groups,
                    eps,
                )
            )
            if has_attention:
                self.attentions.append(
                    _Transformer2D(
                        out_channels,
                        cross_dim,
                        heads,
                        dim_head,
                        transformer_layers,
                        groups,
                        eps,
                        use_linear_projection,
                    )
                )
            current = out_channels
        self.upsamplers = []
        if add_upsample:
            self.upsamplers.append(_Upsample(out_channels))

    def __call__(
        self,
        x: mx.array,
        res_samples: list,
        temb: mx.array,
        context: mx.array,
    ) -> mx.array:
        for i, resnet in enumerate(self.resnets):
            x = mx.concatenate([x, res_samples.pop()], axis=-1)
            x = resnet(x, temb)
            if self.attentions:
                x = self.attentions[i](x, context)
        if self.upsamplers:
            x = self.upsamplers[0](x)
        return x


class _MidBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        groups: int,
        eps: float,
        cross_dim: int,
        heads: int,
        dim_head: int,
        transformer_layers: int,
        use_linear_projection: bool,
    ) -> None:
        super().__init__()
        self.resnets = [
            _ResnetBlock(channels, channels, time_embed_dim, groups, eps),
            _ResnetBlock(channels, channels, time_embed_dim, groups, eps),
        ]
        self.attentions = [
            _Transformer2D(
                channels,
                cross_dim,
                heads,
                dim_head,
                transformer_layers,
                groups,
                eps,
                use_linear_projection,
            )
        ]

    def __call__(
        self, x: mx.array, temb: mx.array, context: mx.array
    ) -> mx.array:
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        return self.resnets[1](x, temb)


class SDXLUNet(nn.Module):
    """Channels-last SDXL UNet mirroring the diffusers module tree."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        block_out_channels = list(config["block_out_channels"])
        down_block_types = list(config["down_block_types"])
        up_block_types = list(config["up_block_types"])
        num_blocks = len(block_out_channels)
        layers_per_block = int(config.get("layers_per_block", 2))
        groups = int(
            config.get("norm_groups", config.get("norm_num_groups", 32))
        )
        eps = float(config.get("norm_eps", 1e-5))
        cross_dim = config.get("cross_attention_dim", 2048)
        if isinstance(cross_dim, (list, tuple)):
            cross_dims = list(cross_dim)
        else:
            cross_dims = [int(cross_dim)] * num_blocks
        time_embed_dim = int(
            config.get("time_embedding_dim") or block_out_channels[0] * 4
        )
        use_linear_projection = bool(
            config.get("use_linear_projection", True)
        )
        downsample_padding = int(config.get("downsample_padding", 1))
        in_channels = int(config.get("in_channels", 4))
        out_channels = int(config.get("out_channels", 4))
        conv_in_kernel = int(config.get("conv_in_kernel", 3))
        conv_out_kernel = int(config.get("conv_out_kernel", 3))

        raw_heads = config.get("num_attention_heads") or config.get(
            "attention_head_dim"
        )
        if raw_heads is None:
            heads = [8] * num_blocks
        elif isinstance(raw_heads, int):
            heads = [raw_heads] * num_blocks
        else:
            heads = [int(h) for h in raw_heads]

        raw_tlayers = config.get("transformer_layers_per_block", 1)
        if isinstance(raw_tlayers, int):
            tlayers = [raw_tlayers] * num_blocks
        else:
            tlayers = [int(v) for v in raw_tlayers]

        addition_time_embed_dim = int(
            config.get("addition_time_embed_dim", 256)
        )
        projection_dim = int(
            config.get("projection_class_embeddings_input_dim", 2816)
        )
        pooled_dim = projection_dim - 6 * addition_time_embed_dim
        if pooled_dim <= 0:
            pooled_dim = projection_dim
        add_input_dim = pooled_dim + 6 * addition_time_embed_dim

        self._add_time_embed_dim = addition_time_embed_dim

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            conv_in_kernel,
            padding=conv_in_kernel // 2,
        )
        self.time_embedding = _TimestepEmbedding(
            block_out_channels[0], time_embed_dim
        )
        self.add_embedding = _TimestepEmbedding(add_input_dim, time_embed_dim)

        self.down_blocks = []
        input_channel = block_out_channels[0]
        for i, block_type in enumerate(down_block_types):
            out_channel = block_out_channels[i]
            has_attention = "CrossAttn" in block_type
            self.down_blocks.append(
                _DownBlock(
                    input_channel,
                    out_channel,
                    layers_per_block,
                    time_embed_dim,
                    groups,
                    eps,
                    has_attention,
                    cross_dims[i],
                    heads[i],
                    out_channel // heads[i],
                    tlayers[i],
                    i != num_blocks - 1,
                    downsample_padding,
                    use_linear_projection,
                )
            )
            input_channel = out_channel

        self.mid_block = _MidBlock(
            block_out_channels[-1],
            time_embed_dim,
            groups,
            eps,
            cross_dims[-1],
            heads[-1],
            block_out_channels[-1] // heads[-1],
            tlayers[-1],
            use_linear_projection,
        )

        self.up_blocks = []
        reversed_channels = list(reversed(block_out_channels))
        reversed_heads = list(reversed(heads))
        reversed_tlayers = list(reversed(tlayers))
        reversed_cross = list(reversed(cross_dims))
        prev_output_channel = reversed_channels[0]
        for i, block_type in enumerate(up_block_types):
            out_channel = reversed_channels[i]
            input_channel = reversed_channels[
                min(i + 1, len(reversed_channels) - 1)
            ]
            has_attention = "CrossAttn" in block_type
            self.up_blocks.append(
                _UpBlock(
                    input_channel,
                    out_channel,
                    prev_output_channel,
                    layers_per_block + 1,
                    time_embed_dim,
                    groups,
                    eps,
                    has_attention,
                    reversed_cross[i],
                    reversed_heads[i],
                    out_channel // reversed_heads[i],
                    reversed_tlayers[i],
                    i != len(up_block_types) - 1,
                    use_linear_projection,
                )
            )
            prev_output_channel = out_channel

        self.conv_norm_out = _group_norm(groups, block_out_channels[0], eps)
        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            conv_out_kernel,
            padding=conv_out_kernel // 2,
        )

    def __call__(
        self,
        sample: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        text_embeds: mx.array,
        time_ids: mx.array,
        cache=None,
    ) -> mx.array:
        noise, _ = self.forward_with_cache(
            sample,
            timestep,
            encoder_hidden_states,
            text_embeds,
            time_ids,
            cache=cache,
        )
        return noise

    def forward_with_cache(
        self,
        sample: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        text_embeds: mx.array,
        time_ids: mx.array,
        cache=None,
    ) -> tuple:
        """Return ``(noise, deep)``; ``deep`` feeds DeepCache reuse steps.

        A ``cache`` that is a tuple of arrays is a stored deep state from an
        earlier full step: the down and mid paths are skipped and only the up
        path runs. Otherwise everything runs and the deep state is returned.
        """
        reuse = (
            isinstance(cache, (tuple, list))
            and len(cache) >= 1
            and all(isinstance(part, mx.array) for part in cache)
        )
        emb_dtype = sample.dtype
        temb = _timestep_embedding(
            timestep, self.conv_in.weight.shape[0]
        ).astype(emb_dtype)
        emb = self.time_embedding(temb)
        if self.add_embedding is not None:
            tids = time_ids.reshape(-1, time_ids.shape[-1])
            flat_ids = _timestep_embedding(
                tids.reshape(-1), self._add_time_embed_dim
            ).astype(emb_dtype)
            time_embeds = flat_ids.reshape(tids.shape[0], -1)
            aug = mx.concatenate(
                [text_embeds.astype(emb_dtype), time_embeds], axis=-1
            )
            emb = emb + self.add_embedding(aug)

        if reuse:
            x = cache[0]
            res_samples = list(cache[1:])
            deep = tuple(cache)
        else:
            x = self.conv_in(sample)
            res_samples = [x]
            for block in self.down_blocks:
                x, output_states = block(x, emb, encoder_hidden_states)
                res_samples.extend(output_states)
            x = self.mid_block(x, emb, encoder_hidden_states)
            deep = tuple([x] + list(res_samples))
        for block in self.up_blocks:
            x = block(x, res_samples, emb, encoder_hidden_states)
        x = nn.silu(self.conv_norm_out(x))
        return self.conv_out(x), deep


def _quantize_unet(model: nn.Module, bits: int, group_size: int = 64) -> nn.Module:
    """Quantize dense Linear/Embedding layers whose last dim divides the group."""
    if bits not in (4, 8):
        raise ValueError("SDXL UNet quantization must be 4 or 8")

    def predicate(path, module):
        if isinstance(module, nn.QuantizedLinear):
            return False
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        return module.weight.shape[-1] % group_size == 0

    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        class_predicate=predicate,
    )
    return model


_WEIGHT_BUFFERS = ("position_ids", "num_batches_tracked")
_QUANT_PARAM_SUFFIXES = (".scales", ".biases")


def _infer_quant_group_size(weights: dict, bits: int) -> int:
    """Recover a pre-quantised checkpoint's group size from packed shapes.

    MLX packs quantised weights into uint32 words holding ``32 // bits``
    values each, so the dense last dimension is
    ``packed_last * 32 // bits``. Dividing that by the number of scale groups
    gives the group size. Reading the packed dimension as dense computes
    ``32 // 4 = 8`` for a group-64 q4 layer and makes strict load fail.
    """
    if bits not in (4, 8):
        raise ValueError("SDXL UNet quantization must be 4 or 8")
    values_per_word = 32 // bits
    for name, scales in weights.items():
        if not name.endswith(".scales") or scales.ndim < 1:
            continue
        packed = weights.get(name[: -len(".scales")] + ".weight")
        if packed is None or packed.ndim < 1 or int(scales.shape[-1]) <= 0:
            continue
        if packed.dtype == mx.uint32:
            dense_last = int(packed.shape[-1]) * values_per_word
        else:
            dense_last = int(packed.shape[-1])
        groups = int(scales.shape[-1])
        if dense_last % groups == 0 and dense_last // groups > 0:
            return dense_last // groups
    return 64


def _prequantized_layer_spec(metadata: dict, base: str, bits: int) -> tuple:
    """Validate one quantised layer's metadata and return (out, in, group)."""
    weight = metadata.get(base + ".weight")
    scales = metadata.get(base + ".scales")
    biases = metadata.get(base + ".biases")
    if not all(isinstance(info, dict) for info in (weight, scales, biases)):
        raise ValueError(
            f"prequantized layer {base} is missing weight/scales/biases"
        )
    weight_shape = list(weight.get("shape") or [])
    scale_shape = list(scales.get("shape") or [])
    bias_shape = list(biases.get("shape") or [])
    if len(weight_shape) != 2 or len(scale_shape) != 2:
        raise ValueError(f"prequantized layer {base} has an invalid shape")
    if scale_shape != bias_shape:
        raise ValueError(
            f"prequantized layer {base} scales/biases shapes differ"
        )
    values_per_word = 32 // int(bits)
    dense_last = int(weight_shape[-1])
    if weight.get("dtype") == "U32":
        dense_last *= values_per_word
    groups = int(scale_shape[-1])
    if groups <= 0 or dense_last % groups:
        raise ValueError(
            f"prequantized layer {base} has an invalid group size"
        )
    return int(weight_shape[0]), dense_last, dense_last // groups


def _apply_prequantized_modules(model, metadata: dict, bits: int) -> set:
    """Build per-key QuantizedLinear/Embedding modules for a checkpoint.

    Constructing each layer from its own ``scales``/``biases`` supports
    checkpoints that mix quantised and dense layers, and layers with different
    group sizes. Only the named bases are replaced; every other weight loads
    densely. A base that cannot be recovered exactly fails before load_weights
    rather than silently dequantising or corrupting a layer.
    """
    quant_bases = {
        name[: -len(".scales")]
        for name in metadata
        if name.endswith(".scales")
    }
    for base in sorted(quant_bases):
        out_dim, in_dim, group_size = _prequantized_layer_spec(
            metadata, base, bits
        )
        parent, last, module = _resolve_module(model, base)
        if isinstance(module, nn.QuantizedLinear):
            continue
        if isinstance(module, nn.Linear):
            if tuple(module.weight.shape) != (out_dim, in_dim):
                raise ValueError(
                    f"prequantized layer {base} does not match the model "
                    f"shape {tuple(module.weight.shape)}"
                )
            bias = (base + ".bias") in metadata
            replacement = nn.QuantizedLinear(
                in_dim,
                out_dim,
                bias=bias,
                group_size=group_size,
                bits=bits,
            )
        elif isinstance(module, nn.Embedding):
            if tuple(module.weight.shape) != (out_dim, in_dim):
                raise ValueError(
                    f"prequantized embedding {base} does not match the model "
                    f"shape {tuple(module.weight.shape)}"
                )
            replacement = nn.QuantizedEmbedding(
                out_dim, in_dim, group_size=group_size, bits=bits
            )
        else:
            raise ValueError(
                f"prequantized parameter {base} is not a Linear or Embedding"
            )
        _replace_module(parent, last, replacement)
    return quant_bases


def load_unet(
    folder: Path,
    dtype=mx.float16,
    quantize: int | None = 4,
) -> SDXLUNet:
    """Build the UNet from a diffusers folder, load weights, then quantize."""
    folder = Path(folder)
    config_path = folder / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"missing UNet config.json in {folder}") from e
    except ValueError as e:
        raise ValueError(f"invalid UNet config.json in {folder}") from e

    model = SDXLUNet(config)
    files = _component_shards(folder)
    metadata = _component_metadata(files)
    prequantized = any(
        name.endswith(_QUANT_PARAM_SUFFIXES) for name in metadata
    )
    quant_bases = set()
    if prequantized:
        # A quantised checkpoint has packed ``weight`` arrays plus ``scales``
        # and ``biases``; those only load into QuantizedLinear modules. Build
        # them before load_weights, and do not quantise a second time after.
        if quantize is None:
            raise ValueError(
                "a prequantized UNet checkpoint needs an explicit "
                "quantization (4 or 8); bits cannot be recovered from the file"
            )
        quant_bases = _apply_prequantized_modules(
            model, metadata, int(quantize)
        )

    converted = []
    pending = []
    for path in files:
        shard = mx.load(str(path))
        try:
            for name, tensor in shard.items():
                if name.endswith(_WEIGHT_BUFFERS):
                    continue
                quant_param = name.endswith(_QUANT_PARAM_SUFFIXES) or (
                    name.endswith(".weight")
                    and name[: -len(".weight")] in quant_bases
                )
                if (
                    not quant_param
                    and name.endswith(".weight")
                    and tensor.ndim in (4, 5)
                ):
                    tensor = convert_conv_weight(tensor)
                if dtype is not None and not quant_param:
                    tensor = tensor.astype(dtype)
                converted.append((name, tensor))
                pending.append(tensor)
                if len(pending) >= 8:
                    mx.eval(*pending)
                    pending.clear()
                    try:
                        mx.clear_cache()
                    except Exception:
                        pass
        finally:
            del shard
    if pending:
        mx.eval(*pending)
    del metadata
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass

    model.load_weights(converted, strict=True)
    del converted
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass
    if dtype is not None and not prequantized:
        # Keep a prequantised checkpoint's scales/biases at their stored
        # precision. The conversion loop already cast every dense parameter;
        # casting quant params here would change dequantisation for no gain.
        model.set_dtype(dtype)
    if quantize is not None and not prequantized:
        # Free the dense source weights before allocating the quantised copy.
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass
        _quantize_unet(model, int(quantize))
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass
    return model


class _VAEResnet(nn.Module):
    """VAE residual block: no time embedding, matching diffusers ResnetBlock2D."""

    def __init__(
        self, in_channels: int, out_channels: int, groups: int, eps: float
    ) -> None:
        super().__init__()
        self.norm1 = _group_norm(groups, in_channels, eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = _group_norm(groups, out_channels, eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        shortcut = self.conv_shortcut(x) if hasattr(self, "conv_shortcut") else x
        return shortcut + h


class _VAEAttention(nn.Module):
    """Single-head spatial attention from the VAE mid block."""

    def __init__(self, channels: int, groups: int, eps: float) -> None:
        super().__init__()
        self.group_norm = _group_norm(groups, channels, eps)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, x: mx.array) -> mx.array:
        batch, height, width, channels = x.shape
        y = self.group_norm(x).reshape(batch, height * width, channels)
        query, key, value = (
            z[:, None] for z in (self.to_q(y), self.to_k(y), self.to_v(y))
        )
        out = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=channels ** -0.5
        )[:, 0]
        out = self.to_out[0](out).reshape(batch, height, width, channels)
        return x + out


class _VAEMidBlock(nn.Module):
    def __init__(self, channels: int, groups: int, eps: float) -> None:
        super().__init__()
        self.attentions = [_VAEAttention(channels, groups, eps)]
        self.resnets = [
            _VAEResnet(channels, channels, groups, eps),
            _VAEResnet(channels, channels, groups, eps),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


class _VAEUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        groups: int,
        eps: float,
        add_upsample: bool,
    ) -> None:
        super().__init__()
        self.resnets = []
        current = in_channels
        for _ in range(num_layers):
            self.resnets.append(
                _VAEResnet(current, out_channels, groups, eps)
            )
            current = out_channels
        self.upsamplers = [_Upsample(out_channels)] if add_upsample else []

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers:
            x = self.upsamplers[0](x)
        return x


class SDXLVAEDecoder(nn.Module):
    """Channels-last SDXL VAE decoder mirroring the diffusers module tree."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        block_out_channels = list(
            config.get("block_out_channels") or (128, 256, 512, 512)
        )
        reversed_channels = list(reversed(block_out_channels))
        groups = int(
            config.get("norm_num_groups", config.get("norm_groups", 32))
        )
        eps = float(config.get("norm_eps", 1e-6))
        layers_per_block = int(config.get("layers_per_block", 2))
        latent_channels = int(config.get("latent_channels", 4))
        out_channels = int(config.get("out_channels", 3))

        self.conv_in = nn.Conv2d(latent_channels, reversed_channels[0], 3, padding=1)
        self.mid_block = _VAEMidBlock(reversed_channels[0], groups, eps)
        self.up_blocks = []
        current = reversed_channels[0]
        for i, channels in enumerate(reversed_channels):
            self.up_blocks.append(
                _VAEUpBlock(
                    current,
                    channels,
                    layers_per_block + 1,
                    groups,
                    eps,
                    add_upsample=i != len(reversed_channels) - 1,
                )
            )
            current = channels
        self.conv_norm_out = _group_norm(groups, reversed_channels[-1], eps)
        self.conv_out = nn.Conv2d(
            reversed_channels[-1], out_channels, 3, padding=1
        )
        self.post_quant_conv = nn.Conv2d(
            latent_channels, latent_channels, 1
        )

    def __call__(self, z: mx.array) -> mx.array:
        x = self.conv_in(z)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))


def load_vae_decoder(folder: Path) -> SDXLVAEDecoder:
    """Load only the decoder and post_quant_conv from a diffusers VAE folder."""
    folder = Path(folder)
    config_path = folder / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"missing VAE config.json in {folder}") from e
    except ValueError as e:
        raise ValueError(f"invalid VAE config.json in {folder}") from e

    weights = load_sharded_safetensors(folder)
    model = SDXLVAEDecoder(config)
    converted = []
    for name, tensor in weights.items():
        if name.startswith("decoder."):
            target = name[len("decoder."):]
        elif name.startswith("post_quant_conv."):
            target = name
        else:
            continue
        if name.endswith(_WEIGHT_BUFFERS):
            continue
        if name.endswith(".weight") and tensor.ndim == 4:
            tensor = convert_conv_weight(tensor)
        converted.append((target, tensor.astype(mx.float32)))
    model.load_weights(converted, strict=True)
    model.set_dtype(mx.float32)
    return model


def _feather(length: int, overlap: int) -> mx.array:
    """1-D blend window: linear ramp up, flat middle, linear ramp down."""
    if overlap <= 0:
        return mx.ones((length,), dtype=mx.float32)
    overlap = min(overlap, length // 2)
    if overlap <= 0:
        return mx.ones((length,), dtype=mx.float32)
    ramp = (mx.arange(overlap).astype(mx.float32) + 1) / (overlap + 1)
    middle = mx.ones((length - 2 * overlap,), dtype=mx.float32)
    return mx.concatenate([ramp, middle, ramp[::-1]])


def _tiled_decode(decoder, z: mx.array, tile: int, overlap: int) -> mx.array:
    """Decode latents in overlapping tiles and feather-blend the result."""
    height, width = z.shape[1], z.shape[2]
    tile = max(1, min(int(tile), height, width))
    overlap = max(0, min(int(overlap), tile - 1))
    stride = tile - overlap
    ys = sorted(
        {*range(0, max(height - tile, 0) + 1, stride), max(height - tile, 0)}
    )
    xs = sorted(
        {*range(0, max(width - tile, 0) + 1, stride), max(width - tile, 0)}
    )
    first = decoder(z[:, :tile, :tile])
    scale = max(1, first.shape[1] // tile)
    out = mx.zeros(
        (z.shape[0], height * scale, width * scale, first.shape[-1])
    )
    weight_sum = mx.zeros((1, height * scale, width * scale, 1))
    for y in ys:
        for x in xs:
            if y == 0 and x == 0:
                decoded = first
            else:
                decoded = decoder(z[:, y:y + tile, x:x + tile])
            th, tw = decoded.shape[1], decoded.shape[2]
            window = (
                _feather(th, overlap * scale)[:, None]
                * _feather(tw, overlap * scale)[None]
            )[None, :, :, None]
            py, px = y * scale, x * scale
            out[:, py:py + th, px:px + tw] += decoded * window
            weight_sum[:, py:py + th, px:px + tw] += window
            mx.eval(out, weight_sum)
    result = out / weight_sum
    mx.eval(result)
    return result


# ---------------------------------------------------------------- schedulers
@dataclasses.dataclass
class EulerConfig:
    """Config object compatible with custom sampler constructors."""

    num_train_timesteps: int = 1000
    beta_schedule: str = "scaled_linear"
    beta_start: float = 0.00085
    beta_end: float = 0.012
    prediction_type: str = "epsilon"
    timestep_spacing: str = "leading"
    steps_offset: int = 1


@dataclasses.dataclass
class DDIMConfig:
    num_train_timesteps: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    prediction_type: str = "epsilon"
    clip_sample: bool = False
    clip_sample_range: float = 1.0
    eta: float = 0.0


def _make_betas(schedule, num_train_timesteps, beta_start, beta_end):
    if schedule == "linear":
        return mx.linspace(beta_start, beta_end, num_train_timesteps)
    if schedule == "scaled_linear":
        return (
            mx.linspace(
                beta_start**0.5, beta_end**0.5, num_train_timesteps
            )
            ** 2
        )
    if schedule == "cosine":
        steps = num_train_timesteps + 1
        s = 0.008
        x = mx.linspace(0, num_train_timesteps, steps)
        acp = (
            mx.cos(
                ((x / num_train_timesteps) + s) / (1 + s) * mx.pi * 0.5
            )
            ** 2
        )
        acp = acp / acp[0]
        betas = 1 - (acp[1:] / acp[:-1])
        return mx.clip(betas, 0, 0.999)
    raise ValueError(f"Unknown beta schedule {schedule!r}.")


def _evaluated_scalar(value) -> float:
    """Read one MLX scalar after evaluation; float() alone is not safe."""
    if isinstance(value, mx.array):
        mx.eval(value)
        return float(value.reshape(-1)[0].item())
    return float(value)


def _integer_timestep(t) -> int:
    return int(_evaluated_scalar(t))


class EulerDiscreteScheduler:
    """Euler and EulerTrailing in sigma space (k-diffusion / SDXL parity)."""

    def __init__(self, config=None):
        self.config = config or EulerConfig()
        betas = _make_betas(
            self.config.beta_schedule,
            self.config.num_train_timesteps,
            self.config.beta_start,
            self.config.beta_end,
        )
        self.alphas_cumprod = mx.cumprod(1.0 - betas)
        mx.eval(self.alphas_cumprod)
        self._train_sigmas = mx.sqrt(
            (1.0 - self.alphas_cumprod) / self.alphas_cumprod
        )
        mx.eval(self._train_sigmas)
        self.sigmas = None
        self.timesteps = None
        self.num_inference_steps = None
        self._step_index = 0

    def set_timesteps(self, num_inference_steps: int) -> None:
        T = self.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        train_sigmas = np.array(self._train_sigmas)
        if self.config.timestep_spacing == "leading":
            step_ratio = T // num_inference_steps
            ts = (
                np.arange(num_inference_steps) * step_ratio
            ).round()[::-1].astype(np.float64)
            ts = ts + self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = T / num_inference_steps
            ts = (
                (np.arange(1, num_inference_steps + 1) * step_ratio)
                .round()[::-1]
                .astype(np.float64)
                - 1
            )
        else:  # "linspace"
            ts = np.linspace(0, T - 1, num_inference_steps)[::-1].copy()
        sigmas = np.interp(ts, np.arange(T), train_sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = mx.array(sigmas)
        self.timesteps = mx.array(ts.astype(np.float32))
        mx.eval(self.sigmas, self.timesteps)
        self._step_index = 0

    @property
    def init_noise_sigma(self) -> float:
        assert self.sigmas is not None, "call set_timesteps() before sampling"
        sigma_max = _evaluated_scalar(self.sigmas[0])
        if self.config.timestep_spacing in ("linspace", "trailing"):
            return sigma_max
        return (sigma_max**2 + 1.0) ** 0.5

    def scale_model_input(self, sample: mx.array, t: mx.array) -> mx.array:
        assert self.sigmas is not None, "call set_timesteps() before sampling"
        sigma = self.sigmas[self._step_index]
        return sample / mx.sqrt(sigma**2 + 1.0)

    def _pred_original(
        self, model_output: mx.array, sigma: mx.array, sample: mx.array
    ) -> mx.array:
        prediction_type = self.config.prediction_type
        if prediction_type == "epsilon":
            return sample - sigma * model_output
        if prediction_type == "sample":
            return model_output
        if prediction_type == "v_prediction":
            return (
                model_output * (-sigma / mx.sqrt(sigma**2 + 1.0))
                + sample / (sigma**2 + 1.0)
            )
        raise ValueError(
            "EulerDiscreteScheduler does not support "
            f"prediction_type={prediction_type!r}."
        )

    def step(
        self,
        model_output: mx.array,
        t: mx.array,
        sample: mx.array,
        key: mx.array | None = None,
    ) -> mx.array:
        assert self.sigmas is not None, "call set_timesteps() before sampling"
        i = self._step_index
        sigma = self.sigmas[i]
        sigma_next = self.sigmas[i + 1]
        pred_original = self._pred_original(model_output, sigma, sample)
        derivative = (sample - pred_original) / sigma
        out = sample + derivative * (sigma_next - sigma)
        self._step_index += 1
        mx.eval(out)
        return out


class EulerAncestralScheduler:
    """Euler A (eta=1) with DrawThings-style k-diffusion ancestral update."""

    def __init__(self, config=None, substeps: int = 1):
        self._inner = EulerDiscreteScheduler(config or EulerConfig())
        self._substeps = max(1, int(substeps))
        self._sigmas = None
        self._timesteps = None
        self._timesteps_array = None
        self._step_index = 0

    def set_timesteps(self, num_inference_steps: int) -> None:
        self._inner.set_timesteps(num_inference_steps)
        base_sig = self._inner.sigmas.tolist()
        base_ts = self._inner.timesteps.tolist()
        if self._substeps == 1:
            sigmas = base_sig
            timesteps = base_ts
        else:
            eps = 1e-5
            sigmas = []
            timesteps = []
            for i in range(len(base_sig) - 1):
                sigma_0 = base_sig[i]
                sigma_1 = max(base_sig[i + 1], 0.0)
                t0 = base_ts[i]
                t1 = base_ts[i + 1] if i + 1 < len(base_ts) else 0.0
                for j in range(self._substeps):
                    f = j / self._substeps
                    sigma_j = math.exp(
                        (1 - f) * math.log(max(sigma_0, eps))
                        + f * math.log(max(sigma_1, eps))
                    )
                    sigmas.append(sigma_j)
                    timesteps.append(t0 + (t1 - t0) * f)
            sigmas.append(0.0)
        self._sigmas = sigmas
        self._timesteps = timesteps
        self._timesteps_array = mx.array(timesteps)
        mx.eval(self._timesteps_array)
        self._step_index = 0

    @property
    def timesteps(self):
        return self._timesteps_array

    @property
    def init_noise_sigma(self) -> float:
        return 1.0

    def scale_model_input(self, sample: mx.array, t: mx.array) -> mx.array:
        sigma = self._sigmas[self._step_index]
        return sample / mx.sqrt(sigma**2 + 1)

    def step(
        self,
        model_output: mx.array,
        t: mx.array,
        sample: mx.array,
        key: mx.array | None = None,
    ) -> mx.array:
        i = self._step_index
        sigma = self._sigmas[i]
        sigma_next = self._sigmas[i + 1]
        self._step_index += 1
        variance = max(
            sigma_next**2 - (sigma_next**4) / max(sigma**2, 1e-8), 0.0
        )
        sigma_up = min(variance**0.5, sigma_next)
        sigma_down = max(sigma_next**2 - sigma_up**2, 0.0) ** 0.5
        out = sample + model_output * (sigma_down - sigma)
        if sigma_next > 0 and sigma_up > 0:
            seed = getattr(self, "_seed", 0)
            noise_key = mx.random.key(
                int(seed) * 1000003 + _integer_timestep(t) * 7919 + i
            )
            out = out + mx.random.normal(out.shape, key=noise_key) * sigma_up
        mx.eval(out)
        return out


class DPMSolverMultistepKarrasScheduler:
    """DPM-Solver++(2M) in the Euler scheduler's sigma space."""

    def __init__(self, config=None):
        self._inner = EulerDiscreteScheduler(config or EulerConfig())
        self._old = None
        self._h_last = None

    def set_timesteps(self, num_inference_steps: int) -> None:
        self._inner.set_timesteps(num_inference_steps)
        self._old = None
        self._h_last = None

    @property
    def timesteps(self):
        return self._inner.timesteps

    @property
    def init_noise_sigma(self) -> float:
        return self._inner.init_noise_sigma

    def scale_model_input(self, sample: mx.array, t: mx.array) -> mx.array:
        return self._inner.scale_model_input(sample, t)

    def step(
        self,
        model_output: mx.array,
        t: mx.array,
        sample: mx.array,
        key: mx.array | None = None,
    ) -> mx.array:
        i = self._inner._step_index
        sig = self._inner.sigmas[i]
        sig_next = self._inner.sigmas[i + 1]
        self._inner._step_index += 1

        sig_value = _evaluated_scalar(sig)
        sig_next_value = _evaluated_scalar(sig_next)
        denoised = sample - sig * model_output
        if sig_next_value == 0.0:
            out = denoised
        elif self._old is None:
            out = (sig_next / sig) * sample - math.expm1(
                math.log(sig_next_value) - math.log(sig_value)
            ) * denoised
        else:
            h = math.log(sig_value) - math.log(sig_next_value)
            r = self._h_last / h
            denoised_mix = (1.0 + 1.0 / (2 * r)) * denoised - (
                1.0 / (2 * r)
            ) * self._old
            out = (sig_next / sig) * sample - math.expm1(-h) * denoised_mix

        if sig_next_value > 0:
            self._h_last = math.log(sig_value) - math.log(sig_next_value)
        self._old = denoised
        mx.eval(out)
        return out


class DDIMScheduler:
    """DDIM with linear beta schedule, stride ``T // steps``, and eta 0."""

    def __init__(self, config=None):
        self.config = config or DDIMConfig()
        betas = _make_betas(
            self.config.beta_schedule,
            self.config.num_train_timesteps,
            self.config.beta_start,
            self.config.beta_end,
        )
        self.alphas_cumprod = mx.cumprod(1.0 - betas)
        mx.eval(self.alphas_cumprod)
        self.timesteps = None
        self.num_inference_steps = None
        self._stride = 1
        self._step_index = 0

    def set_timesteps(self, num_inference_steps: int) -> None:
        T = self.config.num_train_timesteps
        if num_inference_steps > T:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) > "
                f"num_train_timesteps ({T})."
            )
        self.num_inference_steps = num_inference_steps
        self._stride = T // num_inference_steps
        self.timesteps = (
            mx.arange(num_inference_steps) * self._stride
        )[::-1]
        mx.eval(self.timesteps)
        self._step_index = 0

    @property
    def init_noise_sigma(self) -> float:
        return 1.0

    def scale_model_input(self, sample: mx.array, t: mx.array) -> mx.array:
        return sample

    def predict_x0(
        self, model_output: mx.array, t: mx.array, sample: mx.array
    ) -> mx.array:
        ti = _integer_timestep(t)
        acp = self.alphas_cumprod[ti]
        sqrt_acp = mx.sqrt(acp)
        sqrt_one_minus_acp = mx.sqrt(1.0 - acp)
        prediction_type = self.config.prediction_type
        if prediction_type == "epsilon":
            x0 = (sample - sqrt_one_minus_acp * model_output) / sqrt_acp
        elif prediction_type == "sample":
            x0 = model_output
        elif prediction_type == "v_prediction":
            x0 = sqrt_acp * sample - sqrt_one_minus_acp * model_output
        else:
            raise ValueError(prediction_type)
        if self.config.clip_sample:
            limit = self.config.clip_sample_range
            x0 = mx.clip(x0, -limit, limit)
        return x0

    def step(
        self,
        model_output: mx.array,
        t: mx.array,
        sample: mx.array,
        key: mx.array | None = None,
    ) -> mx.array:
        ti = _integer_timestep(t)
        prev = ti - self._stride
        acp_t = self.alphas_cumprod[ti]
        acp_prev = (
            self.alphas_cumprod[prev] if prev >= 0 else mx.array(1.0)
        )
        x0 = self.predict_x0(model_output, mx.array(ti), sample)
        pred_epsilon = (
            sample - mx.sqrt(acp_t) * x0
        ) / mx.sqrt(1.0 - acp_t)
        eta = self.config.eta
        sigma = (
            eta
            * mx.sqrt((1.0 - acp_prev) / (1.0 - acp_t))
            * mx.sqrt(1.0 - acp_t / acp_prev)
            if prev >= 0
            else mx.array(0.0)
        )
        direction = (
            mx.sqrt(mx.maximum(1.0 - acp_prev - sigma**2, 0.0))
            * pred_epsilon
        )
        out = mx.sqrt(acp_prev) * x0 + direction
        if eta > 0 and prev >= 0:
            noise_key = key if key is not None else mx.random.key(ti)
            out = out + sigma * mx.random.normal(sample.shape, key=noise_key)
        mx.eval(out)
        return out


def make_scheduler(name: str):
    """Map a sampler name to the scheduler used by the SDXL engine."""
    if str(name).startswith("custom:"):
        try:
            import samplers_store

            cls = samplers_store.get_class(name.split(":", 1)[1], "sdxl")
            if cls is None:
                raise ValueError("not declared for the sdxl engine or broken")
            print(f"[sdxl-engine] custom sampler {name}", flush=True)
            return cls(
                EulerConfig(
                    beta_schedule="scaled_linear",
                    beta_start=0.00085,
                    beta_end=0.012,
                    prediction_type="epsilon",
                    timestep_spacing="leading",
                    steps_offset=1,
                )
            )
        except Exception as exc:
            print(
                f"[sdxl-engine] custom sampler {name} unavailable: {exc}; "
                f"falling back to EulerDiscreteScheduler",
                flush=True,
            )
            name = "euler"
    if name in ("euler_a", "euler_a_substep"):
        return EulerAncestralScheduler(
            EulerConfig(
                beta_schedule="scaled_linear",
                beta_start=0.00085,
                beta_end=0.012,
                prediction_type="epsilon",
                timestep_spacing="leading",
                steps_offset=1,
            ),
            substeps=2 if name == "euler_a_substep" else 1,
        )
    if name in ("dpmpp_2m_karras", "dpmpp_2m_karras_trailing"):
        spacing = "trailing" if "trailing" in name else "leading"
        return DPMSolverMultistepKarrasScheduler(
            EulerConfig(
                beta_schedule="scaled_linear",
                beta_start=0.00085,
                beta_end=0.012,
                prediction_type="epsilon",
                timestep_spacing=spacing,
                steps_offset=1,
            )
        )
    if name == "ddim":
        return DDIMScheduler()
    spacing = (
        "trailing" if name in ("euler_trailing", "trailing") else "leading"
    )
    return EulerDiscreteScheduler(
        EulerConfig(
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            prediction_type="epsilon",
            timestep_spacing=spacing,
            steps_offset=1,
        )
    )


@mx.compile
def _compiled_euler_step(
    model_output: mx.array,
    sample: mx.array,
    sigma: mx.array,
    sigma_next: mx.array,
    prediction_type_code: int,
) -> mx.array:
    """Euler update in sigma space, compiled for the hot denoise loop.

    Only this elementwise update is compiled. Compiling the full UNet forward
    measured 17-82 s/step and produced a Metal OOM on this 16 GB M1, against
    1.1-1.5 s/step for the plain MLX graph. Keep the UNet uncompiled.
    """
    if prediction_type_code == 1:
        pred_original = model_output
    elif prediction_type_code == 2:
        pred_original = (
            model_output * (-sigma / mx.sqrt(sigma**2 + 1.0))
            + sample / (sigma**2 + 1.0)
        )
    else:
        pred_original = sample - sigma * model_output
    derivative = (sample - pred_original) / sigma
    return sample + derivative * (sigma_next - sigma)


_PREDICTION_TYPE_CODES = {"epsilon": 0, "sample": 1, "v_prediction": 2}


class DeepCache:
    """Periodically reuse a stored UNet deep state between full steps."""

    def __init__(self, interval: int = 1) -> None:
        self.interval = max(1, int(interval or 1))
        self.deep = None
        self.steps = 0
        self.skipped = 0
        self._force = False

    def reset(self) -> None:
        self.deep = None
        self.steps = 0
        self.skipped = 0
        self._force = False

    def force_final(self) -> None:
        self._force = True

    def should_reuse(self) -> bool:
        step = self.steps
        self.steps += 1
        if self._force:
            self._force = False
            return False
        if self.interval <= 1:
            return False
        if step % self.interval == 0:
            return False
        self.skipped += 1
        return True


def _quick_gelu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(1.702 * x)


def _clip_activation(name: str, x: mx.array) -> mx.array:
    if name == "quick_gelu":
        return _quick_gelu(x)
    return _exact_gelu(x)


class _CLIPAttention(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        self.num_heads = int(config["num_attention_heads"])
        self.head_dim = hidden // self.num_heads
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        batch, seq, _ = x.shape
        shape = (batch, seq, self.num_heads, self.head_dim)
        query = self.q_proj(x).reshape(shape).transpose(0, 2, 1, 3)
        key = self.k_proj(x).reshape(shape).transpose(0, 2, 1, 3)
        value = self.v_proj(x).reshape(shape).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.head_dim ** -0.5,
            mask=mask,
        )
        out = out.transpose(0, 2, 1, 3).reshape(
            batch, seq, self.num_heads * self.head_dim
        )
        return self.out_proj(out)


class _CLIPMLP(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        intermediate = int(config["intermediate_size"])
        self.activation = str(config.get("hidden_act", "quick_gelu"))
        self.fc1 = nn.Linear(hidden, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(_clip_activation(self.activation, self.fc1(x)))


class _CLIPEncoderLayer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        eps = float(config.get("layer_norm_eps", 1e-5))
        self.layer_norm1 = nn.LayerNorm(hidden, eps=eps)
        self.self_attn = _CLIPAttention(config)
        self.layer_norm2 = nn.LayerNorm(hidden, eps=eps)
        self.mlp = _CLIPMLP(config)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class _CLIPEncoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.layers = [
            _CLIPEncoderLayer(config)
            for _ in range(int(config["num_hidden_layers"]))
        ]


class _CLIPEmbeddings(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        vocab = int(config["vocab_size"])
        hidden = int(config["hidden_size"])
        max_positions = int(config["max_position_embeddings"])
        self.token_embedding = nn.Embedding(vocab, hidden)
        self.position_embedding = nn.Embedding(max_positions, hidden)

    def __call__(self, ids: mx.array) -> mx.array:
        seq = ids.shape[1]
        positions = mx.arange(seq)
        return self.token_embedding(ids) + self.position_embedding(positions)


class _CLIPTextModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        eps = float(config.get("layer_norm_eps", 1e-5))
        self.embeddings = _CLIPEmbeddings(config)
        self.encoder = _CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(hidden, eps=eps)

    def __call__(self, ids: mx.array) -> tuple:
        hidden = self.embeddings(ids)
        seq = ids.shape[1]
        mask = mx.triu(
            mx.full((seq, seq), mx.finfo(hidden.dtype).min, dtype=hidden.dtype),
            k=1,
        )
        states = [hidden]
        for layer in self.encoder.layers:
            hidden = layer(hidden, mask)
            states.append(hidden)
        last_hidden = self.final_layer_norm(hidden)
        positions = mx.argmax(ids, axis=-1)
        pooled = mx.take_along_axis(
            last_hidden, positions[:, None, None], axis=1
        ).reshape(ids.shape[0], -1)
        return states, pooled


class CLIPTextEncoder(nn.Module):
    """CLIP-L / OpenCLIP-bigG text encoder mirroring the transformers tree."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        config = dict(config)
        hidden = int(config["hidden_size"])
        projection_dim = int(config.get("projection_dim", hidden))
        self.with_projection = bool(config.get("with_projection", False))
        self.text_model = _CLIPTextModel(config)
        if self.with_projection:
            self.text_projection = nn.Linear(
                hidden, projection_dim, bias=False
            )

    def __call__(self, ids: mx.array) -> tuple:
        states, pooled = self.text_model(ids)
        if self.with_projection:
            pooled = self.text_projection(pooled)
        return states, pooled


def load_text_encoder(folder: Path, dtype=mx.float16) -> CLIPTextEncoder:
    """Build a CLIP text encoder from a diffusers folder and load weights."""
    folder = Path(folder)
    config_path = folder / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"missing text encoder config.json in {folder}"
        ) from e
    except ValueError as e:
        raise ValueError(f"invalid text encoder config.json in {folder}") from e

    weights = load_sharded_safetensors(folder)
    config = dict(config)
    config["with_projection"] = "text_projection.weight" in weights
    model = CLIPTextEncoder(config)
    converted = []
    for name, tensor in weights.items():
        if name.endswith(_WEIGHT_BUFFERS):
            continue
        if dtype is not None:
            tensor = tensor.astype(dtype)
        converted.append((name, tensor))
    model.load_weights(converted, strict=True)
    if dtype is not None:
        model.set_dtype(dtype)
    return model


def load_tokenizer(folder: Path):
    from transformers import CLIPTokenizer

    return CLIPTokenizer.from_pretrained(str(Path(folder)))


# ---------------------------------------------------------------- lora loading
_LORA_TE_COMPONENTS = ("text_encoder", "text_encoder_2")


def _map_kohya_key(key: str):
    """kohya/comfy/diffusers/peft LoRA key -> (component, mlx module path). None to skip."""
    if key.startswith("base_model.model."):
        key = key[len("base_model.model."):]

    if key.startswith("lora_unet_"):
        comp, rest = "unet", key[len("lora_unet_"):]
    elif key.startswith("lora_te1_"):
        comp, rest = "text_encoder", key[len("lora_te1_"):]
    elif key.startswith("lora_te2_"):
        comp, rest = "text_encoder_2", key[len("lora_te2_"):]
    elif key.startswith("unet."):
        comp, rest = "unet", key[len("unet."):]
    elif key.startswith("text_encoder."):
        comp, rest = "text_encoder", key[len("text_encoder."):]
    elif key.startswith("text_encoder_2."):
        comp, rest = "text_encoder_2", key[len("text_encoder_2."):]
    elif key.startswith("te1."):
        comp, rest = "text_encoder", key[len("te1."):]
    elif key.startswith("te2."):
        comp, rest = "text_encoder_2", key[len("te2."):]
    else:
        return None

    if "." in rest:
        return comp, rest

    if comp in ("text_encoder", "text_encoder_2"):
        m = re.match(r"text_model_encoder_layers_(\d+)_(.+)", rest)
        if m:
            layer_idx, tail = m.group(1), m.group(2)
            te_map = {
                "mlp_fc1": "mlp.fc1",
                "mlp_fc2": "mlp.fc2",
                "self_attn_q_proj": "self_attn.q_proj",
                "self_attn_k_proj": "self_attn.k_proj",
                "self_attn_v_proj": "self_attn.v_proj",
                "self_attn_out_proj": "self_attn.out_proj",
            }
            sub = te_map.get(tail)
            if sub:
                return comp, f"text_model.encoder.layers.{layer_idx}.{sub}"
        return None

    suffix_map = {
        "attn1_to_q": "attn1.to_q", "attn1_to_k": "attn1.to_k",
        "attn1_to_v": "attn1.to_v", "attn1_to_out_0": "attn1.to_out.0",
        "attn2_to_q": "attn2.to_q", "attn2_to_k": "attn2.to_k",
        "attn2_to_v": "attn2.to_v", "attn2_to_out_0": "attn2.to_out.0",
        "ff_net_0_proj": "ff.net.0.proj", "ff_net_2": "ff.net.2",
        "proj_in": "proj_in", "proj_out": "proj_out",
    }

    if rest.startswith("middle_block_1_"):
        tail = rest[len("middle_block_1_"):]
        mt = re.match(r"transformer_blocks_(\d+)_(.+)", tail)
        if mt:
            sub = suffix_map.get(mt.group(2))
            if not sub:
                return None
            tail = f"transformer_blocks.{mt.group(1)}.{sub}"
        else:
            tail = suffix_map.get(tail)
            if not tail:
                return None
        return comp, f"mid_block.attentions.0.{tail}"

    # Diffusers format: down_blocks / up_blocks / mid_block (SDXL-Lightning, official diffusers LoRAs)
    m_proj = re.match(r"(down_blocks|up_blocks)_(\d+)_attentions_(\d+)_(proj_in|proj_out)", rest)
    if m_proj:
        return comp, f"{m_proj.group(1)}.{m_proj.group(2)}.attentions.{m_proj.group(3)}.{m_proj.group(4)}"

    m_mid_proj = re.match(r"mid_block_attentions_0_(proj_in|proj_out)", rest)
    if m_mid_proj:
        return comp, f"mid_block.attentions.0.{m_mid_proj.group(1)}"

    md = re.match(r"(down_blocks|up_blocks)_(\d+)_attentions_(\d+)_transformer_blocks_(\d+)_(.+)", rest)
    if md:
        block, b_idx, a_idx, tb_idx, tail = md.group(1), md.group(2), md.group(3), md.group(4), md.group(5)
        sub = suffix_map.get(tail)
        if sub:
            return comp, f"{block}.{b_idx}.attentions.{a_idx}.transformer_blocks.{tb_idx}.{sub}"

    mm = re.match(r"mid_block_attentions_0_transformer_blocks_(\d+)_(.+)", rest)
    if mm:
        tb_idx, tail = mm.group(1), mm.group(2)
        sub = suffix_map.get(tail)
        if sub:
            return comp, f"mid_block.attentions.0.transformer_blocks.{tb_idx}.{sub}"

    m_time = re.match(r"(down_blocks|up_blocks)_(\d+)_resnets_(\d+)_time_emb_proj", rest)
    if m_time:
        return comp, f"{m_time.group(1)}.{m_time.group(2)}.resnets.{m_time.group(3)}.time_emb_proj"

    m_mid_time = re.match(r"mid_block_resnets_(\d+)_time_emb_proj", rest)
    if m_mid_time:
        return comp, f"mid_block.resnets.{m_mid_time.group(1)}.time_emb_proj"

    m = re.match(r"(input_blocks|output_blocks)_(\d+)_(\d+)_(.+)", rest)
    if not m:
        return None
    kind, a, _, tail = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    mt = re.match(r"transformer_blocks_(\d+)_(.+)", tail)
    if mt:
        sub = suffix_map.get(mt.group(2))
        if not sub:
            return None
        tail = f"transformer_blocks.{mt.group(1)}.{sub}"
    else:
        tail = suffix_map.get(tail)
        if not tail:
            return None
    if kind == "input_blocks":
        level, idx = {4: (1, 0), 5: (1, 1), 7: (2, 0), 8: (2, 1)}.get(a, (None, None))
        if level is None:
            return None
        path = f"down_blocks.{level}.attentions.{idx}.{tail}"
    elif kind == "output_blocks":
        path = f"up_blocks.{a // 3}.attentions.{a % 3}.{tail}"
    else:
        return None
    return comp, path


_LORA_PART_SUFFIXES = (
    (".alpha", "alpha"),
    (".lora_down.weight", "down"),
    (".lora_up.weight", "up"),
    (".lora_A.weight", "down"),
    (".lora_B.weight", "up"),
    (".down.weight", "down"),
    (".up.weight", "up"),
)


def _lora_key_part(key: str):
    for suffix, part in _LORA_PART_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], part
    return None


def _check_lora_file(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"LoRA file not found: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"LoRA must be a .safetensors file: {path.name}")
    file_size = path.stat().st_size
    if file_size < 16:
        raise ValueError(f"LoRA file too small ({file_size} bytes): {path.name}")
    return file_size


def _read_safetensors_header(fh, path: Path, file_size: int):
    header_len_bytes = fh.read(8)
    if len(header_len_bytes) < 8:
        raise ValueError(
            f"Could not read safetensors header length from {path.name}"
        )
    n = struct.unpack("<Q", header_len_bytes)[0]
    if n <= 0 or n > file_size - 8 or n > 100 * 1024 * 1024:
        raise ValueError(
            f"Invalid safetensors header length {n} (file size: {file_size}) "
            f"in {path.name}. The file may be corrupted or not a valid "
            f"safetensors archive."
        )
    header_bytes = fh.read(n)
    if len(header_bytes) < n:
        raise ValueError(f"Truncated safetensors header in {path.name}")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(
            f"Failed to parse safetensors JSON header in {path.name}: {e}"
        ) from e
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header is not an object: {path.name}")
    return header, 8 + n


def _read_safetensors_tensor(fh, info, offset_base, file_size=None):
    dtype = info.get("dtype") if isinstance(info, dict) else None
    shape = info.get("shape") if isinstance(info, dict) else None
    offsets = info.get("data_offsets") if isinstance(info, dict) else None
    if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError("invalid safetensors tensor metadata")
    if any(not isinstance(v, int) or v < 0 for v in shape):
        raise ValueError("invalid safetensors tensor shape")
    if any(not isinstance(v, int) for v in offsets) or offsets[1] < offsets[0]:
        raise ValueError("invalid safetensors tensor offsets")
    start, end = offsets
    absolute_end = offset_base + end
    if start < 0 or (file_size is not None and absolute_end > file_size):
        raise ValueError("safetensors tensor exceeds file bounds")
    fh.seek(offset_base + start)
    buf = fh.read(end - start)
    if len(buf) != end - start:
        raise ValueError("truncated safetensors tensor")
    try:
        if dtype == "F32":
            return np.frombuffer(buf, dtype="<f4").reshape(shape)
        if dtype == "F16":
            return np.frombuffer(buf, dtype="<f2").astype(np.float32).reshape(shape)
        if dtype == "BF16":
            u16 = np.frombuffer(buf, dtype="<u2")
            return (u16.astype(np.uint32) << 16).view("<f4").reshape(shape)
        if dtype == "F64":
            return np.frombuffer(buf, dtype="<f8").astype(np.float32).reshape(shape)
        if dtype in ("I64", "INT64"):
            return np.frombuffer(buf, dtype="<i8").reshape(shape)
        if dtype in ("I32", "INT32"):
            return np.frombuffer(buf, dtype="<i4").reshape(shape)
        if dtype in ("I16", "INT16"):
            return np.frombuffer(buf, dtype="<i2").reshape(shape)
        if dtype in ("I8", "INT8"):
            return np.frombuffer(buf, dtype="<i1").reshape(shape)
        if dtype in ("U64", "UINT64"):
            return np.frombuffer(buf, dtype="<u8").reshape(shape)
        if dtype in ("U32", "UINT32"):
            return np.frombuffer(buf, dtype="<u4").reshape(shape)
        if dtype in ("U16", "UINT16"):
            return np.frombuffer(buf, dtype="<u2").reshape(shape)
        if dtype in ("U8", "UINT8"):
            return np.frombuffer(buf, dtype="<u1").reshape(shape)
        if dtype == "BOOL":
            return np.frombuffer(buf, dtype="?").reshape(shape)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid {dtype} tensor: {e}") from e
    raise ValueError(f"unsupported dtype {dtype}")


def _parse_lora_entries(path: Path) -> list:
    """Parse one single-file LoRA into (component, path, down, up, alpha) rows."""
    path = Path(path)
    file_size = _check_lora_file(path)
    entries = {}
    with open(path, "rb") as fh:
        header, base_off = _read_safetensors_header(fh, path, file_size)
        for key, info in header.items():
            if key == "__metadata__":
                continue
            if not isinstance(info, dict):
                raise ValueError(
                    f"Invalid tensor metadata for {key} in {path.name}"
                )
            offsets = info.get("data_offsets")
            shape = info.get("shape")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(not isinstance(v, int) for v in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
                or base_off + offsets[1] > file_size
                or not isinstance(shape, list)
                or any(not isinstance(v, int) or v < 0 for v in shape)
                or math.prod(shape) <= 0
            ):
                raise ValueError(
                    f"Invalid tensor metadata for {key} in {path.name}"
                )
            classified = _lora_key_part(key)
            if classified is None:
                continue
            base, part = classified
            try:
                tensor = _read_safetensors_tensor(fh, info, base_off, file_size)
            except ValueError as e:
                raise ValueError(f"{e} for {key} in {path.name}") from e
            entries.setdefault(base, {})[part] = tensor

    parsed = []
    for base, parts in entries.items():
        mapped = _map_kohya_key(base)
        if mapped is None:
            continue
        if "down" not in parts or "up" not in parts:
            raise ValueError(f"Incomplete LoRA pair for {base} in {path.name}")
        down = parts["down"]
        up = parts["up"]
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] <= 0:
            raise ValueError(
                f"Invalid LoRA tensor rank for {base} in {path.name}"
            )
        if down.shape[1] <= 0 or up.shape[0] <= 0 or down.shape[0] != up.shape[1]:
            raise ValueError(
                f"LoRA A/B shape mismatch for {base} in {path.name}"
            )
        rank = int(down.shape[0])
        if "alpha" in parts:
            alpha_values = np.asarray(parts["alpha"]).reshape(-1)
            if alpha_values.size != 1:
                raise ValueError(f"Invalid LoRA alpha for {base} in {path.name}")
            alpha = float(alpha_values[0])
        else:
            alpha = float(rank)
        if not math.isfinite(alpha) or rank <= 0:
            raise ValueError(
                f"Invalid LoRA alpha/rank for {base} in {path.name}"
            )
        parsed.append(
            (
                mapped[0],
                mapped[1],
                mx.array(down.astype(np.float32)),
                mx.array(up.astype(np.float32)),
                alpha,
            )
        )
    if not parsed:
        raise ValueError(
            f"LoRA contains no supported SDXL layers: {path.name}"
        )
    return parsed


def parse_lora_file(path: Path) -> dict:
    """Return ``{module_path: (down, up, alpha)}`` for one single-file LoRA."""
    parsed = {}
    for _component, module_path, down, up, alpha in _parse_lora_entries(path):
        parsed[module_path] = (down, up, alpha)
    return parsed


def _scan_lora_components(path: Path) -> tuple:
    """Return the component names one LoRA targets, reading only the header."""
    path = Path(path)
    file_size = _check_lora_file(path)
    with open(path, "rb") as fh:
        header, _ = _read_safetensors_header(fh, path, file_size)
    components = set()
    for key in header:
        if key == "__metadata__":
            continue
        classified = _lora_key_part(key)
        if classified is None:
            continue
        mapped = _map_kohya_key(classified[0])
        if mapped is not None:
            components.add(mapped[0])
    return tuple(sorted(components))


def _lora_signature(path: str) -> tuple:
    resolved = Path(path).expanduser().resolve()
    try:
        stat = resolved.stat()
        return (str(resolved), stat.st_size, stat.st_mtime_ns)
    except OSError:
        return (str(resolved), None, None)


class LoRALinear(nn.Module):
    """A linear layer plus a low-rank LoRA delta."""

    def __init__(self, base, lora_a: mx.array, lora_b: mx.array, scale: float = 1.0) -> None:
        super().__init__()
        self.base = base
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.scale = float(scale)

    def __call__(self, x: mx.array) -> mx.array:
        delta = (x @ self.lora_a.T) @ self.lora_b.T
        return self.base(x) + self.scale * delta


def _replace_module(parent, last, value) -> None:
    if isinstance(parent, list):
        parent[int(last)] = value
    else:
        setattr(parent, last, value)


def _resolve_module(root, dotted: str):
    segments = dotted.split(".")
    parent = root
    for segment in segments[:-1]:
        if isinstance(parent, (list, tuple)):
            parent = parent[int(segment)]
        elif hasattr(parent, segment):
            parent = getattr(parent, segment)
        else:
            parent = parent[int(segment)] if segment.isdigit() else parent[segment]
    last = segments[-1]
    if isinstance(parent, (list, tuple)):
        module = parent[int(last)]
    elif hasattr(parent, last):
        module = getattr(parent, last)
    else:
        module = parent[last]
    return parent, last, module


def _linear_features(linear) -> tuple:
    if isinstance(linear, nn.QuantizedLinear):
        return (
            linear.weight.shape[0],
            linear.scales.shape[1] * linear.group_size,
        )
    return tuple(linear.weight.shape)


def _linear_dtype(linear):
    if isinstance(linear, nn.QuantizedLinear):
        return linear.scales.dtype
    return linear.weight.dtype


def load_multilora(components: dict, loras: list[dict], hooks: list | None = None) -> int:
    """Apply one or more single-file LoRAs, stacking ranks per target."""
    deltas = {}
    for lora in loras:
        if not isinstance(lora, dict) or not lora.get("path"):
            raise ValueError("each LoRA must contain a path")
        try:
            scale = float(lora.get("scale", 1.0))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"invalid LoRA scale in {lora.get('path')!r}"
            ) from e
        if not math.isfinite(scale) or scale < 0:
            raise ValueError(f"invalid LoRA scale in {lora.get('path')!r}")
        path = Path(str(lora["path"])).expanduser().resolve()
        for component, module_path, down, up, alpha in _parse_lora_entries(path):
            if component not in components:
                raise ValueError(
                    f"LoRA target {component}:{module_path} is unavailable"
                )
            rank = int(down.shape[0])
            effective = scale * alpha / rank
            deltas.setdefault(component, {}).setdefault(
                module_path, []
            ).append((down * effective, up))

    pending = []
    for component, paths in deltas.items():
        model = components[component]
        for module_path, stack in paths.items():
            try:
                parent, last, linear = _resolve_module(model, module_path)
            except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
                raise ValueError(
                    f"LoRA target {component}:{module_path} is unavailable"
                ) from e
            if isinstance(linear, LoRALinear):
                raise ValueError(
                    f"LoRA target {component}:{module_path} is already adapted"
                )
            if not isinstance(linear, (nn.Linear, nn.QuantizedLinear)):
                raise ValueError(
                    f"LoRA target {component}:{module_path} is "
                    f"{type(linear).__name__}"
                )
            out_features, in_features = _linear_features(linear)
            for down, up in stack:
                if down.shape[1] != in_features or up.shape[0] != out_features:
                    raise ValueError(
                        f"LoRA shape does not fit {component}:{module_path}"
                    )
            dtype = _linear_dtype(linear)
            down = mx.concatenate([item[0] for item in stack], axis=0).astype(dtype)
            up = mx.concatenate([item[1] for item in stack], axis=1).astype(dtype)
            adapter = LoRALinear(linear, down, up, 1.0)
            pending.append((component, parent, last, linear, adapter))

    staged = []
    try:
        for component, parent, last, linear, adapter in pending:
            _replace_module(parent, last, adapter)
            staged.append((component, parent, last, linear))
        for component in deltas:
            mx.eval(components[component].parameters())
    except Exception:
        for component, parent, last, original in reversed(staged):
            try:
                _replace_module(parent, last, original)
            except Exception:
                pass
        raise
    if hooks is not None:
        hooks.extend(staged)
    return len(pending)


class PromptEmbeddingCache:
    """Bounded LRU cache of encoded prompt embeddings."""

    def __init__(self, max_entries: int = 64, max_bytes: int = 128 * (1 << 20)):
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._entries: OrderedDict = OrderedDict()

    @staticmethod
    def _value_bytes(value) -> int:
        nbytes = getattr(value, "nbytes", None)
        if isinstance(nbytes, int):
            return nbytes
        if isinstance(value, dict):
            return sum(PromptEmbeddingCache._value_bytes(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(PromptEmbeddingCache._value_bytes(v) for v in value)
        return 0

    def get(self, key):
        value = self._entries.get(key)
        if value is None:
            return None
        self._entries.move_to_end(key)
        return value

    def put(self, key, value) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        total = sum(self._value_bytes(v) for v in self._entries.values())
        while self._entries and (
            len(self._entries) > self.max_entries or total > self.max_bytes
        ):
            _, dropped = self._entries.popitem(last=False)
            total -= self._value_bytes(dropped)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def _write_ipc_json(obj: dict) -> None:
    line = json.dumps(obj) + "\n"
    with _ipc_stdout_lock:
        try:
            sys.stdout.write(line)
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            sys.exit(0)


_configured_asset_dir = (
    os.environ.get("MLX_DIFFUSION_ASSET_DIR", "").strip()
    or os.environ.get("DIFFUSIONBEAR_ASSET_DIR", "").strip()
)
if _configured_asset_dir:
    _asset_dir = Path(_configured_asset_dir).expanduser()
    if not _asset_dir.is_absolute():
        raise ValueError(
            "MLX_DIFFUSION_ASSET_DIR/DIFFUSIONBEAR_ASSET_DIR must be an absolute path"
        )
    ASSET_DIR = _asset_dir.resolve()
else:
    ASSET_DIR = Path(__file__).resolve().parent.parent / "backend" / "data"
DATA_DIR = ASSET_DIR
DEFAULT_MODEL_DIR = ASSET_DIR / "models" / "juggernaut-xl-lightning"
_MAX_PROMPT_BYTES = 128 * (1 << 10)

_ALLOWED_SAMPLERS = {
    "euler",
    "euler_trailing",
    "trailing",
    "euler_a",
    "euler_a_substep",
    "dpmpp_2m_karras",
    "dpmpp_2m_karras_trailing",
    "ddim",
}


def _validate_request(req: dict) -> dict:
    """Validate request bounds and return a normalized dict with defaults."""
    if not isinstance(req, dict):
        raise ValueError("request must be an object")
    if req.get("reference_images") or req.get("image"):
        raise ValueError("SDXL reference-image conditioning is not supported")
    try:
        width = int(req["width"])
        height = int(req["height"])
        steps = int(req["steps"])
        seed = int(req.get("seed", 0))
        quantize_raw = req.get("quantize_unet", 4)
        quantize_unet = int(4 if quantize_raw is None else quantize_raw)
        cache_interval = int(req.get("cache_interval", 1))
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("invalid SDXL generation dimensions or steps") from e
    if (
        width < 128
        or height < 128
        or width > 2048
        or height > 2048
        or width % 8
        or height % 8
    ):
        raise ValueError("SDXL width and height must be multiples of 8 within 128-2048")
    if steps < 1 or steps > 50:
        raise ValueError("SDXL steps must be between 1 and 50")
    if quantize_unet not in (4, 8):
        raise ValueError("SDXL UNet quantization must be 4 or 8")
    if not 1 <= cache_interval <= 10:
        raise ValueError("SDXL cache_interval must be between 1 and 10")

    sampler = str(req.get("sampler") or "dpmpp_2m_karras")
    if sampler.startswith("custom:"):
        try:
            import samplers_store

            if samplers_store.get_class(sampler.split(":", 1)[1], "sdxl") is None:
                raise ValueError("not declared for the sdxl engine or broken")
        except ImportError:
            pass  # store unreadable: make_scheduler falls back and reports why
    elif sampler not in _ALLOWED_SAMPLERS:
        raise ValueError(f"unsupported SDXL sampler: {sampler}")

    loras = req.get("loras") or []
    if not isinstance(loras, list) or len(loras) > 16:
        raise ValueError("SDXL LoRA list must contain at most 16 entries")

    prompt_text = str(req.get("prompt") or "")
    negative_text = str(req.get("negative_prompt") or "")
    if not prompt_text.strip():
        raise ValueError("prompt is required")
    if not req.get("dest"):
        raise ValueError("dest is required")
    if len(prompt_text.encode("utf-8")) > _MAX_PROMPT_BYTES:
        raise ValueError("prompt exceeds the 128 KiB limit")
    if len(negative_text.encode("utf-8")) > _MAX_PROMPT_BYTES:
        raise ValueError("negative prompt exceeds the 128 KiB limit")

    model_dir = req.get("model_dir")
    if req.get("guidance") is None:
        guidance = (
            1.0
            if (steps <= 8 or "lightning" in str(model_dir or "").lower())
            else 2.4
        )
    else:
        try:
            guidance = float(req["guidance"])
        except (TypeError, ValueError) as e:
            raise ValueError(
                "SDXL guidance must be a finite non-negative number"
            ) from e
        if not math.isfinite(guidance) or guidance < 0:
            raise ValueError("SDXL guidance must be a finite non-negative number")

    try:
        clip_skip = int(req.get("clip_skip", 2) or 2)
    except (TypeError, ValueError) as e:
        raise ValueError("invalid SDXL clip_skip") from e

    wired_limit_bytes = req.get("wired_limit_bytes")
    if wired_limit_bytes is not None:
        try:
            wired_limit_bytes = max(0, int(wired_limit_bytes))
        except (TypeError, ValueError) as e:
            raise ValueError("invalid SDXL wired_limit_bytes") from e

    tile_default = width * height >= 768 * 768
    tile_value = req.get("tile_vae")
    tile_vae = tile_default if tile_value is None else bool(tile_value)
    normalized = dict(req)
    normalized.update(
        {
            "prompt": prompt_text,
            "negative_prompt": negative_text,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "guidance": guidance,
            "sampler": sampler,
            "image_id": str(req.get("image_id") or ""),
            "loras": loras,
            "cache_interval": cache_interval,
            "tile_vae": tile_vae,
            "fast_vae": bool(req.get("fast_vae", True)),
            "quantize_unet": quantize_unet,
            "clip_skip": clip_skip,
            "wired_limit_bytes": wired_limit_bytes,
            "model_dir": model_dir,
        }
    )
    return normalized


class MLXSDXLEngine:
    """Resident SDXL model, text encoders, and generation orchestration."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        quantize_unet: int | None = 4,
    ) -> None:
        if quantize_unet not in (None, 4, 8):
            raise ValueError("SDXL UNet quantization must be 4 or 8")
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR).resolve()
        self.quantize_unet = quantize_unet
        self.unet = None
        self._text_encoder = None
        self._text_encoder_2 = None
        self._tokenizer = None
        self._tokenizer_2 = None
        self._vae_decoder = None
        self._prompt_cache = PromptEmbeddingCache()
        self._loras = []
        self._lora_signature = ()
        self._te_lora_signature = ()
        self._applied_lora_hooks = []
        self._lora_component_cache = {}
        self._unet_generation = 0
        self._lora_component_ids = self._component_identity()
        self.cache_interval = 1
        self._load_time = 0.0
        self._forward_key = None
        self._unet_forward = None
        self._deep_forward = None

    def _component_identity(self) -> tuple:
        return (id(self.unet), id(self._text_encoder), id(self._text_encoder_2))

    def _restore_scoped_hooks(self, keep: tuple) -> tuple:
        """Restore hooks for a component set, keeping the other components."""
        restored_te = False
        kept = []
        for hook in self._applied_lora_hooks:
            if hook[0] in keep:
                try:
                    _replace_module(hook[1], hook[2], hook[3])
                except Exception:
                    pass
                if hook[0] in _LORA_TE_COMPONENTS:
                    restored_te = True
            else:
                kept.append(hook)
        self._applied_lora_hooks = kept
        return kept, restored_te

    def _invalidate_unet_lora_state(self) -> None:
        """Drop UNet adapters only; resident text-encoder adapters stay."""
        self._restore_scoped_hooks(("unet",))
        self._lora_signature = None
        self._unet_generation += 1

    def _invalidate_te_lora_state(self, clear_prompt_cache: bool = True) -> None:
        """Drop text-encoder adapters only; UNet adapters stay applied."""
        _kept, restored_te = self._restore_scoped_hooks(
            _LORA_TE_COMPONENTS
        )
        self._lora_signature = None
        self._te_lora_signature = ()
        if clear_prompt_cache or restored_te:
            self._prompt_cache.clear()

    def load_unet(self, folder=None, dtype=mx.float16):
        """Load and cache the resident UNet."""
        if self.unet is None:
            target = Path(folder) if folder is not None else self.model_dir / "unet"
            started = time.monotonic()
            self.unet = load_unet(
                target, dtype=dtype, quantize=self.quantize_unet
            )
            mx.eval(self.unet.parameters())
            self._load_time = round(time.monotonic() - started, 2)
            self._invalidate_unet_lora_state()
        else:
            # A reused resident UNet was not loaded by this request, matching
            # the old engine's per-generation load_time of zero.
            self._load_time = 0.0
        return self.unet

    def ensure_text_encoders_loaded(self) -> None:
        """Load both text encoders and tokenizers once; never unload them."""
        if (
            self._text_encoder is not None
            and self._text_encoder_2 is not None
            and self._tokenizer is not None
            and self._tokenizer_2 is not None
        ):
            return
        if self._text_encoder is None:
            self._text_encoder = load_text_encoder(
                self.model_dir / "text_encoder"
            )
        if self._text_encoder_2 is None:
            self._text_encoder_2 = load_text_encoder(
                self.model_dir / "text_encoder_2"
            )
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.model_dir / "tokenizer")
        if self._tokenizer_2 is None:
            self._tokenizer_2 = load_tokenizer(self.model_dir / "tokenizer_2")
        self._invalidate_te_lora_state(clear_prompt_cache=True)

    def _lora_components(self, resolved_path: str) -> tuple:
        signature = _lora_signature(resolved_path)
        cached = self._lora_component_cache.get(signature[0])
        if cached is not None and cached[0] == signature:
            return cached[1]
        try:
            components = _scan_lora_components(Path(signature[0]))
        except Exception:
            components = ()
        if len(self._lora_component_cache) >= 64:
            self._lora_component_cache.clear()
        self._lora_component_cache[signature[0]] = (signature, components)
        return components

    def _compute_lora_signatures(self, loras: list) -> tuple:
        full = []
        te_only = []
        targets = set()
        for lora in loras:
            if not isinstance(lora, dict) or not lora.get("path"):
                raise ValueError("each LoRA must contain a path")
            try:
                scale = float(lora.get("scale", 1.0))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"invalid LoRA scale in {lora.get('path')!r}"
                ) from e
            if not math.isfinite(scale) or scale < 0:
                raise ValueError(
                    f"invalid LoRA scale in {lora.get('path')!r}"
                )
            signature = _lora_signature(str(lora["path"]))
            full.append((*signature, scale))
            all_components = self._lora_components(signature[0])
            targets.update(all_components)
            components = tuple(
                name
                for name in all_components
                if name in _LORA_TE_COMPONENTS
            )
            if components:
                te_only.append((*signature, scale, components))
        return (
            tuple(sorted(full, key=repr)),
            tuple(sorted(te_only, key=repr)),
            tuple(sorted(targets)),
        )

    def _ensure_lora_components(self, targets: tuple) -> None:
        if "unet" in targets and self.unet is None:
            self.load_unet()
        if any(name in targets for name in _LORA_TE_COMPONENTS):
            self.ensure_text_encoders_loaded()

    def _restore_lora_hooks(self, hooks: list) -> bool:
        had_te = any(
            hook[0] in _LORA_TE_COMPONENTS for hook in hooks
        )
        for _component, parent, last, original in reversed(hooks):
            try:
                _replace_module(parent, last, original)
            except Exception:
                pass
        return had_te

    def ensure_loras(self, loras=None) -> int:
        """Apply the requested LoRAs, owning UNet invalidation and TE cache keys."""
        loras = [] if loras is None else list(loras)
        full_signature, te_signature, targets = self._compute_lora_signatures(
            loras
        )
        current_ids = self._component_identity()
        component_changed = current_ids != self._lora_component_ids
        if full_signature == self._lora_signature and not component_changed:
            return 0

        old_hooks = self._applied_lora_hooks
        self._applied_lora_hooks = []
        self._restore_lora_hooks(old_hooks)
        if component_changed:
            previous = self._lora_component_ids or (None, None, None)
            te_replaced = (
                previous[1] != current_ids[1]
                or previous[2] != current_ids[2]
            )
            if te_replaced:
                self._te_lora_signature = ()
                self._prompt_cache.clear()
            self._unet_generation += 1
        te_changed = te_signature != self._te_lora_signature
        if te_changed:
            self._prompt_cache.clear()

        self._ensure_lora_components(targets)

        components = {
            "unet": self.unet,
            "text_encoder": self._text_encoder,
            "text_encoder_2": self._text_encoder_2,
        }
        applied = []
        try:
            count = load_multilora(components, loras, hooks=applied) if loras else 0
            if loras and count <= 0:
                raise ValueError("no LoRA layers were applied")
        except Exception:
            self._loras = []
            self._lora_signature = None
            self._te_lora_signature = ()
            self._lora_component_ids = self._component_identity()
            raise

        self._applied_lora_hooks = applied
        self._loras = loras
        self._lora_signature = full_signature
        self._te_lora_signature = te_signature
        self._lora_component_ids = self._component_identity()
        self._unet_generation += 1
        return count

    def _prompt_cache_key(self, prompt: str, clip_skip: int) -> tuple:
        return (prompt, clip_skip, self._te_lora_signature)

    @staticmethod
    def _tokenize(tokenizer, prompts: list) -> mx.array:
        encoded = tokenizer(
            list(prompts),
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )
        ids = mx.array(encoded["input_ids"])
        return ids.astype(mx.int32)

    @staticmethod
    def _tokenize_chunks(tokenizer, text: str) -> mx.array:
        """Split a long prompt into 77-token CLIP windows.

        SDXL's CLIP positional table has 77 positions. A 272-token prompt used
        to lose 195 tokens before the UNet ever saw it. Each chunk is encoded
        with its own BOS/EOS and the per-token states are concatenated; the
        UNet cross-attention accepts the longer context.
        """
        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_tensors=None,
        )["input_ids"]
        ids = list(ids)
        bos = int(tokenizer.bos_token_id)
        eos = int(tokenizer.eos_token_id)
        content = 75
        chunks = []
        if not ids:
            ids = []
        for start in range(0, max(1, len(ids)), content):
            piece = ids[start:start + content]
            row = [bos] + piece + [eos]
            row = row[:77] + [eos] * max(0, 77 - len(row))
            chunks.append(row)
        if not chunks:
            chunks.append([bos, eos] + [eos] * 75)
        return mx.array(chunks, dtype=mx.int32)

    def _encode_prompt_batch(self, prompts, clip_skip: int) -> tuple:
        prompts = list(prompts)
        skip = max(1, int(clip_skip or 2))
        # Keep the existing fast batched path when every prompt fits the native
        # 77-token window.
        chunked = [
            self._tokenize_chunks(self._tokenizer, text).shape[0] > 1
            for text in prompts
        ]
        if not any(chunked):
            ids1 = self._tokenize(self._tokenizer, prompts)
            ids2 = self._tokenize(self._tokenizer_2, prompts)
            states1, _ = self._text_encoder(ids1)
            states2, pooled = self._text_encoder_2(ids2)
            return (
                mx.concatenate(
                    [
                        states1[-min(skip, len(states1))],
                        states2[-min(skip, len(states2))],
                    ],
                    axis=-1,
                ),
                pooled,
            )

        contexts, pools = [], []
        for text in prompts:
            ids1 = self._tokenize_chunks(self._tokenizer, text)
            ids2 = self._tokenize_chunks(self._tokenizer_2, text)
            states1, _ = self._text_encoder(ids1)
            states2, pooled = self._text_encoder_2(ids2)
            ctx = mx.concatenate(
                [
                    states1[-min(skip, len(states1))],
                    states2[-min(skip, len(states2))],
                ],
                axis=-1,
            )
            contexts.append(ctx.reshape(1, -1, ctx.shape[-1]))
            pools.append(pooled[0:1])
        common_len = max(c.shape[1] for c in contexts)
        contexts = [self._pad_context(c, common_len) for c in contexts]
        return mx.concatenate(contexts, axis=0), mx.concatenate(pools, axis=0)

    def encode_prompt(
        self, prompt: str, negative_prompt: str = "", clip_skip: int = 2
    ) -> tuple:
        positive = prompt or ""
        negative = negative_prompt or ""
        skip = max(1, int(clip_skip or 2))
        if (
            self._text_encoder is None
            or self._text_encoder_2 is None
            or self._tokenizer is None
            or self._tokenizer_2 is None
        ):
            self.ensure_text_encoders_loaded()
        self.ensure_loras(getattr(self, "_loras", []))
        entries = []
        missing = []
        for text in (positive, negative):
            entry = self._prompt_cache.get(self._prompt_cache_key(text, skip))
            entries.append(entry)
            if entry is None:
                missing.append(text)
        if missing:
            if (
                self._text_encoder is None
                or self._text_encoder_2 is None
                or self._tokenizer is None
                or self._tokenizer_2 is None
            ):
                self.ensure_text_encoders_loaded()
            context, pooled = self._encode_prompt_batch(missing, skip)
            mx.eval(context, pooled)
            offset = 0
            for index in range(len(entries)):
                if entries[index] is not None:
                    continue
                entries[index] = (
                    context[offset:offset + 1],
                    pooled[offset:offset + 1],
                )
                self._prompt_cache.put(
                    self._prompt_cache_key(missing[offset], skip),
                    entries[index],
                )
                offset += 1
        positive_context, positive_pooled = entries[0]
        negative_context, negative_pooled = entries[1]
        # Chunked prompts have a longer context than the 77-token negative.
        # CFG concatenates on the batch axis, so both sequences must match.
        common_len = max(
            positive_context.shape[1], negative_context.shape[1], 77
        )
        positive_context = self._pad_context(positive_context, common_len)
        negative_context = self._pad_context(negative_context, common_len)
        return (
            mx.concatenate([positive_context, negative_context], axis=0),
            mx.concatenate([positive_pooled, negative_pooled], axis=0),
        )

    @staticmethod
    def _pad_context(context, length: int):
        if context.shape[1] >= length:
            return context
        return mx.pad(
            context,
            [(0, 0), (0, length - context.shape[1]), (0, 0)],
        )

    def _ensure_vae_decoder(self, folder=None) -> SDXLVAEDecoder:
        if self._vae_decoder is None:
            target = Path(folder) if folder is not None else self.model_dir / "vae"
            self._vae_decoder = load_vae_decoder(target)
        return self._vae_decoder

    def _release_vae_decoder(self) -> None:
        self._vae_decoder = None
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    def decode(
        self,
        latents: mx.array,
        fast_vae: bool = True,
        tile_vae: bool = False,
        scaling_factor: float = 0.13025,
    ) -> mx.array:
        """Decode latents to ``[-1, 1]`` pixels, fast TAESD or full VAE."""
        if fast_vae:
            try:
                import taesd_mlx

                decoder = taesd_mlx.get_taesd_decoder()
                image = decoder(latents) * 2.0 - 1.0
                mx.eval(image)
                return image
            except Exception as err:
                print(
                    "[sdxl] WARN TAESD decode failed, falling back to full "
                    f"VAE: {err}",
                    file=sys.stderr,
                )

        decoder = self._ensure_vae_decoder()
        try:
            z = decoder.post_quant_conv(
                latents.astype(mx.float32)
                / float(scaling_factor or 0.13025)
            )
            if tile_vae:
                latent_size = max(z.shape[1], z.shape[2])
                effective_tile = 96 if latent_size > 128 else 64
                effective_overlap = min(16, effective_tile // 4)
                image = _tiled_decode(
                    decoder, z, effective_tile, effective_overlap
                )
            else:
                image = decoder(z)
            mx.eval(image)
            return image
        finally:
            self._release_vae_decoder()

    def _active_unet(self):
        unet = getattr(self, "_unet", None)
        return self.unet if unet is None else unet

    def _ensure_unet_forwards(self, shape: tuple, batch: int, dtype) -> None:
        unet = self._active_unet()
        if unet is None:
            raise RuntimeError("SDXL UNet is not loaded")
        key = (
            self._unet_generation,
            id(unet),
            self._lora_signature,
            tuple(shape),
            int(batch),
            dtype,
        )
        if self._forward_key == key and self._unet_forward is not None:
            return
        # Do not wrap the full UNet forward in mx.compile. Measured 17-82 s/step
        # and a Metal OOM on this 16 GB M1, versus 1.1-1.5 s/step for the plain
        # graph. Only the elementwise Euler update is compiled.
        forward_with_cache = getattr(unet, "forward_with_cache", None)

        def _forward(x, t, ctx, pooled, tids):
            if forward_with_cache is not None:
                return forward_with_cache(x, t, ctx, pooled, tids)
            return unet(x, t, ctx, pooled, tids), x

        def _deep(x, t, ctx, pooled, tids, deep):
            return unet(x, t, ctx, pooled, tids, cache=deep)

        self._unet_forward = _forward
        self._deep_forward = _deep
        self._forward_key = key

    def denoise(
        self,
        latents: mx.array,
        context: mx.array,
        pooled: mx.array,
        time_ids: mx.array,
        scheduler,
        steps: int,
        guidance: float,
        progress_cb=None,
    ) -> mx.array:
        """Run the sampler loop with CFG and optional DeepCache reuse."""
        steps = max(1, int(steps))
        guidance = float(guidance)
        use_cfg = guidance > 1.0
        batch = 2 if use_cfg else 1
        scheduler.set_timesteps(steps)
        timesteps = scheduler.timesteps
        self._ensure_unet_forwards(tuple(latents.shape), batch, latents.dtype)
        cache = DeepCache(getattr(self, "cache_interval", 1) or 1)
        cache.reset()
        prediction_type = getattr(
            getattr(scheduler, "config", None), "prediction_type", "epsilon"
        )
        prediction_code = _PREDICTION_TYPE_CODES.get(prediction_type, 0)
        is_euler = type(scheduler) is EulerDiscreteScheduler
        final_step = len(timesteps) - 1
        for i, t_i in enumerate(timesteps):
            scaled = scheduler.scale_model_input(latents, t_i)
            if use_cfg:
                model_input = mx.concatenate([scaled, scaled], axis=0)
            else:
                model_input = scaled
            if i == final_step:
                cache.force_final()
            if cache.should_reuse() and cache.deep is not None:
                noise = self._deep_forward(
                    model_input, t_i, context, pooled, time_ids, cache.deep
                )
            else:
                noise, deep = self._unet_forward(
                    model_input, t_i, context, pooled, time_ids
                )
                cache.deep = deep
            if use_cfg:
                cond = noise[0:1]
                uncond = noise[1:2]
                noise = uncond + guidance * (cond - uncond)
            if is_euler:
                index = scheduler._step_index
                latents = _compiled_euler_step(
                    noise,
                    latents,
                    scheduler.sigmas[index],
                    scheduler.sigmas[index + 1],
                    prediction_code,
                )
                scheduler._step_index = index + 1
                mx.eval(latents)
            else:
                latents = scheduler.step(noise, t_i, latents)
                mx.eval(latents)
            if progress_cb is not None:
                progress_cb(i)
        return latents

    def _prepare_conditioning(
        self, prompt: str, negative_prompt: str, clip_skip: int, loras
    ) -> tuple:
        """Load resident TEs first, then apply LoRAs, then encode.

        Loading a text encoder after a UNet-only LoRA used to invalidate every
        hook, silently dropping the adapter on the first cold generation.
        """
        self.ensure_text_encoders_loaded()
        self.ensure_loras(loras)
        return self.encode_prompt(prompt, negative_prompt, clip_skip)

    def generate(self, req: dict) -> dict:
        """Validate, denoise, decode, and write raw RGB bytes to ``dest``."""
        req = _validate_request(req)
        width = req["width"]
        height = req["height"]
        steps = req["steps"]
        seed = req["seed"]
        guidance = req["guidance"]
        clip_skip = req["clip_skip"]
        loras = req["loras"]
        cache_interval = req["cache_interval"]
        tile_vae = req["tile_vae"]
        fast_vae = req["fast_vae"]
        image_id = req["image_id"]
        dest = Path(req["dest"])

        from sdxl_layout import sdxl_layout_report

        ok, reason = sdxl_layout_report(self.model_dir)
        if not ok:
            raise ValueError(reason)

        scheduler = make_scheduler(req["sampler"])
        scheduler._seed = seed
        self.cache_interval = cache_interval

        self.load_unet()
        gen_started = time.monotonic()
        try:
            _write_ipc_json(
                {
                    "phase": "preparing",
                    "detail": "Preparing prompt conditioning & latents...",
                }
            )
            if loras:
                _write_ipc_json(
                    {
                        "phase": "loading_model",
                        "detail": f"Applying {len(loras)} LoRA adapter(s)...",
                    }
                )
            context, pooled = self._prepare_conditioning(
                req["prompt"], req["negative_prompt"], clip_skip, loras
            )
        except BaseException:
            gc.collect()
            try:
                mx.clear_cache()
            except Exception:
                pass
            raise

        batch = 2 if guidance > 1.0 else 1
        if batch == 1:
            context = context[0:1]
            pooled = pooled[0:1]

        latent_h = height // 8
        latent_w = width // 8
        scheduler.set_timesteps(steps)
        # Substepped and custom schedulers expand beyond the requested count.
        total_steps = max(1, len(scheduler.timesteps))
        latents = mx.random.normal(
            (1, latent_h, latent_w, 4), key=mx.random.key(seed)
        )
        latents = (latents * scheduler.init_noise_sigma).astype(mx.float16)
        time_ids = mx.broadcast_to(
            mx.array(
                [[float(height), float(width), 0.0, 0.0, float(height), float(width)]]
            ),
            (batch, 6),
        ).astype(pooled.dtype)
        mx.eval(latents, time_ids)

        prev_wired = None
        effective_wired = 0
        try:
            try:
                info = mx.device_info()
                cap = (
                    info.get("max_recommended_working_set_size")
                    or info.get("recommended_max_working_set_size")
                    or 0
                )
                requested_wired = req.get("wired_limit_bytes")
                if requested_wired is None:
                    limit = cap if cap > 0 else 0
                else:
                    limit = max(0, int(requested_wired))
                    if cap > 0 and limit > 0:
                        limit = min(limit, cap)
                effective_wired = limit
                prev_wired = mx.set_wired_limit(limit) if limit > 0 else None
            except Exception:
                prev_wired = None

            _write_ipc_json(
                {
                    "phase": "compiling",
                    "detail": "Compiling Metal scheduler & encoding prompt...",
                }
            )

            def _progress(index):
                _write_ipc_json(
                    {
                        "phase": "generating",
                        "detail": (
                            f"Denoising step {index + 1}/{total_steps}..."
                        ),
                        "progress": {
                            "step": index + 1,
                            "steps": total_steps,
                        },
                    }
                )

            latents = self.denoise(
                latents,
                context,
                pooled,
                time_ids,
                scheduler,
                steps,
                guidance,
                progress_cb=_progress,
            )
            mx.eval(latents)

            _write_ipc_json(
                {
                    "phase": "saving",
                    "detail": "Decoding VAE latents & saving image...",
                }
            )
            image = self.decode(
                latents.astype(mx.float32),
                fast_vae=fast_vae,
                tile_vae=tile_vae,
            )
            mx.eval(image)
            arr = np.asarray(image, dtype=np.float32)[0]
            if arr.min() < 0.0:
                arr = (arr + 1.0) / 2.0
            img = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            if img.shape[0] != height or img.shape[1] != width:
                raise RuntimeError(
                    f"SDXL returned {img.shape[1]}x{img.shape[0]}, "
                    f"expected {width}x{height}"
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                dest.write_bytes(img.tobytes())
            except Exception:
                try:
                    dest.unlink()
                except OSError:
                    pass
                raise
            generation_time = round(time.monotonic() - gen_started, 2)
        except BaseException:
            gc.collect()
            try:
                mx.clear_cache()
            except Exception:
                pass
            raise
        finally:
            if prev_wired is not None:
                try:
                    mx.set_wired_limit(prev_wired)
                except Exception:
                    pass

        try:
            del image
            del arr
            del img
            del latents
        except (NameError, UnboundLocalError):
            pass
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

        return {
            "id": image_id,
            "generation_time": generation_time,
            "load_time": round(float(getattr(self, "_load_time", 0.0)), 2),
            "sampler": req["sampler"],
            "scheduler": type(scheduler).__name__,
            "guidance": guidance,
            "width": width,
            "height": height,
            "quantization": (
                None
                if self.quantize_unet is None
                else int(self.quantize_unet)
            ),
            "loras": loras,
            "fast_vae": bool(fast_vae),
            "cache_interval": cache_interval,
            "tile_vae": bool(tile_vae),
            "wired_limit_bytes": effective_wired,
            "cache_final_step": cache_interval > 1,
            "clip_skip": clip_skip,
        }


_engine = None
_engine_key = None
_generation_lock = threading.Lock()


def get_pipeline(
    model_dir: str | Path | None = None,
    quantize_unet: int = 4,
) -> MLXSDXLEngine:
    """Return the resident engine, creating it when the key changes."""
    global _engine, _engine_key
    target_dir = Path(model_dir or DEFAULT_MODEL_DIR).resolve()
    key = (target_dir, quantize_unet)
    if _engine is None or _engine_key != key:
        if _engine is not None:
            # Drop the old resident weights before allocating the replacement.
            _engine = None
            gc.collect()
            try:
                mx.clear_cache()
            except Exception:
                pass
        _engine = MLXSDXLEngine(target_dir, quantize_unet)
        _engine_key = key
    return _engine


def _reset_engine() -> None:
    global _engine, _engine_key
    _engine = None
    _engine_key = None


def generate(req: dict) -> dict:
    """Daemon entry point: validate, select the engine, and delegate."""
    normalized = _validate_request(req)
    with _generation_lock:
        engine = get_pipeline(
            model_dir=normalized.get("model_dir"),
            quantize_unet=normalized.get("quantize_unet", 4),
        )
        return engine.generate(normalized)


def _handle_signal(signum, frame):
    try:
        _reset_engine()
        mx.clear_cache()
    except Exception:
        pass
    sys.exit(0)


def _arm_parent_watchdog_if_requested() -> None:
    """Exit with the parent instead of lingering as a ~10 GB orphan.

    Only armed when the spawner passes --parent-pid (the backend always
    does); hand-launched engines see no flag and behave exactly as before.
    """
    try:
        from engine_watchdog import arm_orphan_exit, parent_pid_from_argv
    except Exception:
        return
    expected = parent_pid_from_argv(sys.argv[1:])
    if expected is not None:
        arm_orphan_exit(expected)


def _serve():
    """Daemon mode: one JSON request per stdin line, JSON responses on stdout.
    Model stays resident across requests."""
    import signal

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    _arm_parent_watchdog_if_requested()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = generate(req)
            _write_ipc_json(res)
        except Exception as e:
            import traceback

            traceback.print_exc(file=sys.stderr)
            _write_ipc_json({"error": f"{type(e).__name__}: {e}"})
        finally:
            gc.collect()
            try:
                mx.clear_cache()
            except Exception:
                pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        _serve()
    else:
        req = json.loads(sys.argv[1])
        try:
            _write_ipc_json(generate(req))
        except Exception as e:
            import traceback

            traceback.print_exc(file=sys.stderr)
            _write_ipc_json({"error": f"{type(e).__name__}: {e}"})
            sys.exit(1)
