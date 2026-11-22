import gc
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.flux2.model.flux2_vae.decoder.decoder import Flux2Decoder
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping


def _shards(path: Path):
    import json

    for index_name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        index = path / index_name
        if index.is_file():
            weight_map = json.loads(index.read_text("utf-8")).get("weight_map") or {}
            return sorted(set(path / name for name in weight_map.values()))
    diffusers_single = path / "diffusion_pytorch_model.safetensors"
    if diffusers_single.is_file():
        return [diffusers_single]
    return sorted(path.glob("*.safetensors"))


from moss_paths import find_shared_moss_vae_dir, has_moss_vae_weights


def _resolve_vae_dir(root: Path) -> Path:
    direct = root / "vae"
    if has_moss_vae_weights(direct):
        return direct
    if has_moss_vae_weights(root) and (root / "config.json").is_file():
        return root
    shared = find_shared_moss_vae_dir(exclude_dir=root)
    if shared is not None:
        return shared
    return direct if direct.is_dir() else root


def load_moss_vae(root: Path) -> Flux2VAE:
    vae_dir = _resolve_vae_dir(root)
    vae = Flux2VAE()
    vae.decoder = Flux2Decoder(block_out_channels=(96, 192, 384, 384))

    raw = {}
    for shard in _shards(vae_dir):
        raw.update(mx.load(str(shard)))

    quant_paths = {key[:-7] for key in raw if key.endswith(".scales")}
    if quant_paths:
        nn.quantize(
            vae,
            bits=4,
            group_size=64,
            class_predicate=lambda path, module: path in quant_paths,
        )
    expected = {k: v for k, v in tree_flatten(vae.parameters())}
    mapped = {}
    for key, val in raw.items():
        target_key = key
        if target_key not in expected and ".mid_block.attentions.0.to_out.0." in target_key:
            alt = target_key.replace(".mid_block.attentions.0.to_out.0.", ".mid_block.attentions.0.to_out.")
            if alt in expected:
                target_key = alt
        if target_key not in expected:
            continue
        target_shape = tuple(expected[target_key].shape)
        if (
            val.ndim == 4
            and tuple(val.shape) != target_shape
            and (val.shape[0], val.shape[2], val.shape[3], val.shape[1]) == target_shape
        ):
            val = mx.transpose(val, (0, 2, 3, 1))
        mapped[target_key] = val
    del raw

    vae.update(tree_unflatten(list(mapped.items())), strict=False)
    missing_keys = sorted(set(expected.keys()) - set(mapped.keys()))
    if missing_keys:
        raise RuntimeError(
            f"MOSS VAE weights incomplete ({len(missing_keys)} missing keys, e.g. {missing_keys[:5]})"
        )
    mx.eval(vae.parameters())
    del mapped
    gc.collect()
    mx.clear_cache()
    return vae


def release_vae(vae=None):
    if vae is not None:
        del vae
    gc.collect()
    mx.clear_cache()
