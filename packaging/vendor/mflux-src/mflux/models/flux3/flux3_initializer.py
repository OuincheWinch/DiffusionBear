import mlx.core as mx

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config import ModelConfig
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux3.model.flux3_transformer.flux3_transformer import Flux3Transformer
from mflux.models.flux3.weights.flux3_weight_definition import (
    Flux3KleinBaseWeightDefinition,
    Flux3KleinWeightDefinition,
)
from mflux.utils.exceptions import ModelConfigError


class Flux3Initializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        compute_precision: mx.Dtype | None = None,
    ) -> None:
        precision = ComputePrecision(compute_precision)
        path = model_path if model_path else model_config.model_name
        Flux3Initializer._init_config(model, model_config)
        definition = Flux3Initializer.weight_definition(model_config)
        weights = Flux3Initializer._load_weights(path, definition)

        # Check checkpoint kind: auto-switch model_config if checkpoint structure differs from expected flow_map
        flow_map = model_config.transformer_overrides.get("flow_map", True)
        has_embedder = Flux3Initializer._has_target_embedder(weights)
        if has_embedder != flow_map:
            model_config = (
                ModelConfig.flux3_image_klein()
                if has_embedder
                else ModelConfig.flux3_image_klein_base()
            )
            model.model_config = model_config
            definition = Flux3Initializer.weight_definition(model_config)

        Flux3Initializer._check_checkpoint_kind(weights, model_config, path)
        Flux3Initializer._init_tokenizers(model, path)
        Flux3Initializer._init_models(model)
        Flux3Initializer._apply_weights(model, weights, quantize, definition)
        model.compute_precision = precision
        if compute_precision is not None:
            # Last, so it casts the final parameters, whatever quantization produced.
            model.transformer.apply_compute_precision(precision)

    @staticmethod
    def _has_target_embedder(weights: LoadedWeights) -> bool:
        t_weights = weights.transformer or {}
        # In diffusers format: target_timestep_embedder.*
        # In BFL native format: ref_time_in.*
        return any(
            k.startswith("target_timestep_embedder")
            or k.startswith("ref_time_in")
            or "target_timestep_embedder" in k
            or "ref_time_in" in k
            for k in t_weights
        )

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model._noise_override = None  # test hook: golden-parity scripts replace the seeded noise with a fixed tensor

    @staticmethod
    def weight_definition(model_config: ModelConfig):
        flow_map = model_config.transformer_overrides.get("flow_map", True)
        return Flux3KleinWeightDefinition if flow_map else Flux3KleinBaseWeightDefinition

    @staticmethod
    def _load_weights(model_path: str, definition) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=definition,
            model_path=model_path,
        )

    @staticmethod
    def _check_checkpoint_kind(weights: LoadedWeights, model_config: ModelConfig, model_path: str) -> None:
        # Only the distilled checkpoint has target_timestep_embedder (or ref_time_in in BFL native format).
        # A mismatch would otherwise load with strict=False and leave that embedder random (Base read as distilled)
        # or drop it (the reverse).
        flow_map = model_config.transformer_overrides.get("flow_map", True)
        has_embedder = Flux3Initializer._has_target_embedder(weights)
        if has_embedder == flow_map:
            return
        kind, fix = (
            ("Distilled", ModelConfig.flux3_image_klein())
            if has_embedder
            else ("Base", ModelConfig.flux3_image_klein_base())
        )
        raise ModelConfigError(
            f"The checkpoint at {model_path} {'has' if has_embedder else 'has no'} target_timestep_embedder / ref_time_in, "
            f"so it is a FLUX 3 Image Klein {kind} checkpoint, but it was loaded as "
            f"{'Base' if has_embedder else 'Distilled'}. Pass --base-model {fix.aliases[0]}."
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Flux3KleinWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.vae = Flux2VAE()
        model.transformer = Flux3Transformer(**model.model_config.transformer_overrides)
        model.text_encoder = Qwen3TextEncoder(**model.model_config.text_encoder_overrides)

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None, definition) -> None:
        t_weights = weights.components.get("transformer")
        if t_weights is not None:
            from mlx.utils import tree_flatten
            flat_items = tree_flatten(t_weights)
            flat_keys = [k for k, _ in flat_items]
            has_bfl_keys = any(
                k.startswith("time_in")
                or k.startswith("ref_time_in")
                or k.startswith("txt_in")
                or k.startswith("emb_in")
                or k.startswith("content_mode_blocks")
                or k.startswith("single_blocks")
                or "content_mode_blocks" in k
                or "single_blocks" in k
                for k in flat_keys
            )
            if has_bfl_keys:
                from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
                flow_map = model.model_config.transformer_overrides.get("flow_map", True)
                mapping = definition._transformer_mapping_getter()(flow_map)
                raw_dict = dict(flat_items)
                mapped = WeightMapper.apply_mapping(hf_weights=raw_dict, mapping=mapping)
                weights.components["transformer"] = mapped

        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=definition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )