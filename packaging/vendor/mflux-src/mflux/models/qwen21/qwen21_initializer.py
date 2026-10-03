from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition


class Qwen21Initializer:
    @staticmethod
    def init(
        model,
        quantize: int | None,
        model_path: str | None,
        model_config: ModelConfig,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        Qwen21Initializer._init_config(model, model_config)
        weights = Qwen21Initializer._load_weights(path)
        Qwen21Initializer._init_tokenizers(model, path)
        Qwen21Initializer._init_models(model)
        Qwen21Initializer._apply_weights(model, weights, quantize)

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=Qwen21WeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Qwen21WeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = Qwen21VAE()
        model.transformer = Qwen21Transformer()
        model.text_encoder = Qwen21TextEncoder()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        # 16GB build: accept native MLX pre-quantized repos (e.g.
        # mlx-community/Qwen-Image-2.1-MLX-4bit) whose group-quantized layers are stored
        # as flat {weight, scales, biases} triplets. The plain-bf16 modules built in
        # _init_models cannot hold those tensors and update(strict=False) would skip them
        # silently, so rebuild any such layer as QuantizedLinear/QuantizedEmbedding first.
        quantized_layers = 0
        for name in ("vae", "transformer", "text_encoder"):
            tree = weights.components.get(name)
            if tree:
                quantized_layers += Qwen21Initializer._rebuild_quantized_layers(getattr(model, name), tree)
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Qwen21WeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )
        if quantized_layers and model.bits is None:
            model.bits = 4  # pre-quantized repo; report worst-case on-disk level like the CLI does

    @staticmethod
    def _rebuild_quantized_layers(module, tree) -> int:
        """Rebuild folded MLX quantized layers before weights are applied.

        A native save stores each group-quantized layer as packed ``weight`` plus
        ``scales``/``biases``; plain bf16 modules can't hold them. Recursively walk the
        mapped tree and swap any child found with such a triplet for a
        QuantizedLinear/QuantizedEmbedding sized from the stored shapes. Both dict
        attributes (``attn.to_q``) and list elements (``attn.to_out.0``,
        ``modulation.1``) end up as triplet containers, so both are handled. Returns
        the number of layers rebuilt so callers can report a pre-quantized repo loaded.
        """
        from mlx import nn as _nn

        count = 0

        def make_replacement(child, sub):
            outer_dims, group_size, bits = Qwen21Initializer._infer_quantization(child, sub)
            inner_dims = sub["scales"].shape[1] * group_size
            if isinstance(child, (_nn.Embedding, _nn.QuantizedEmbedding)):
                return _nn.QuantizedEmbedding(
                    num_embeddings=outer_dims, dims=inner_dims, group_size=group_size, bits=bits
                )
            return _nn.QuantizedLinear(
                inner_dims, outer_dims, bias="bias" in sub, group_size=group_size, bits=bits
            )

        def rebuild(module, tree) -> None:
            nonlocal count
            if isinstance(tree, list):
                # Resolve the REAL child container so mutations stick:
                #  - plain python lists (transformer_blocks, attn.to_out)
                #  - nn.Sequential keeps children in ``.layers``
                #  - anything else falls back to whatever ``iter(module)`` yields
                if isinstance(module, list):
                    children = module
                elif hasattr(module, "layers") and isinstance(module.layers, list):
                    children = module.layers
                else:
                    children = list(module) if hasattr(module, "__iter__") else []
                for idx, sub in enumerate(tree):
                    if idx >= len(children):
                        continue
                    if (
                        isinstance(sub, dict)
                        and {"weight", "scales", "biases"} <= set(sub)
                        and not isinstance(children[idx], (_nn.QuantizedLinear, _nn.QuantizedEmbedding))
                    ):
                        child = children[idx]
                        if not isinstance(child, (_nn.Linear, _nn.Embedding, _nn.QuantizedLinear, _nn.QuantizedEmbedding)):
                            continue
                        children[idx] = make_replacement(child, sub)
                        count += 1
                    elif isinstance(sub, (dict, list)):
                        rebuild(children[idx], sub)
                return
            if not isinstance(tree, dict):
                return
            for key, sub in tree.items():
                if not isinstance(sub, (dict, list)):
                    continue
                child = getattr(module, key, None)
                if child is None and isinstance(module, dict):
                    child = module.get(key)
                if child is None:
                    continue
                if (
                    isinstance(sub, dict)
                    and "scales" in sub
                    and "biases" in sub
                    and "weight" in sub
                    and not isinstance(child, (_nn.QuantizedLinear, _nn.QuantizedEmbedding))
                ):
                    if isinstance(module, dict):
                        module[key] = make_replacement(child, sub)
                    else:
                        setattr(module, key, make_replacement(child, sub))
                    count += 1
                else:
                    rebuild(child, sub)

        rebuild(module, tree)
        return count

    @staticmethod
    def _infer_quantization(child, sub) -> tuple[int, int, int]:
        """Recover (outer_dims, group_size, bits) from a stored quantized layer.

        Quantized linears and embeddings both store ``scales`` as (outer_dims,
        num_groups) and packed ``weight`` as (outer_dims, inner_dims * bits // 32). The
        existing module is the authority on the inner dimension.
        """
        scales = sub["scales"]
        outer_dims, num_groups = scales.shape[0], scales.shape[1]
        inner_dims = getattr(child, "in_features", None) or getattr(child, "dims", None)
        if not isinstance(inner_dims, int) or inner_dims <= 0:
            weight = getattr(child, "weight", None)
            if weight is not None and getattr(weight, "ndim", 0) == 2:
                inner_dims = weight.shape[-1]
        if not isinstance(inner_dims, int) or inner_dims <= 0:
            inner_dims = num_groups * 64
        return outer_dims, inner_dims // num_groups, (sub["weight"].shape[-1] * 32) // inner_dims
