from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.iris.latent_creator.iris_latent_creator import IrisLatentCreator
from mflux.models.iris.variants import Iris
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs when --model is omitted; the parser keys the --steps default (4) off it.
DEFAULT_MODEL = "iris-3b"

# LoRAs stay separate layers unless --bake-lora (see Iris.DEFAULT_BAKE_LORA)
DEFAULT_BAKE_LORA = Iris.DEFAULT_BAKE_LORA

# The other entries this CLI serves: the undistilled base model (100 steps, CFG 3.0).
FAMILY_MODELS = tuple(key for key in AVAILABLE_MODELS if key.startswith("iris-") and key != DEFAULT_MODEL)

IGNORED_OPTIONS = {
    "--scheduler": "Iris always runs the sampler its checkpoints were trained for (4-step flow for Turbo, FlowDPM-Solver++ for base).",
    "--vae-tiling": "Iris generates directly in pixel space; there is no VAE to tile.",
    "--vae-tile-size": "Iris generates directly in pixel space; there is no VAE to tile.",
}

CONDITIONAL_OPTIONS = {
    "--steps": {
        "condition": "the base model (iris-3b-base)",
        "reason": "Turbo runs its fixed 4-step distilled schedule.",
    },
    "--guidance": {
        "condition": "the base model (iris-3b-base)",
        "reason": "Turbo has guidance distilled into its weights and no CFG branch.",
    },
    "--negative-prompt": {
        "condition": "the base model (iris-3b-base)",
        "reason": "Turbo runs without CFG, so the negative prompt is never encoded.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Iris-3B.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_lora_arguments(bake_by_default=DEFAULT_BAKE_LORA)
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_output_arguments()
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    model_config = ConfigResolution.resolve_restricted(
        args.model,
        DEFAULT_MODEL,
        model_path=args.model_path,
        extra_keys=FAMILY_MODELS,
        base_model=args.base_model,
    )

    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)
    if not model_config.supports_guidance:
        CommandLineParser.warn_ignored_options({name: option["reason"] for name, option in CONDITIONAL_OPTIONS.items()})

    width, height = DimensionResolver.resolve(width=args.width, height=args.height, reference_image_path=None)
    # Check the size before the model loads, so that a bad size gives a usage error and not a traceback.
    if width % IrisLatentCreator.PATCH_SIZE or height % IrisLatentCreator.PATCH_SIZE:
        parser.error(f"--width and --height must be multiples of {IrisLatentCreator.PATCH_SIZE}, got {width}x{height}.")

    model = Iris(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        **lora_init_kwargs_from_args(args),
    )
    # The parser fills --steps with the base model's 100; a few-step LoRA runs its own count unless --steps or a
    # metadata sidecar set one
    steps = args.steps
    if model.few_step is not None and getattr(args, "steps_is_default", False):
        steps = None
    if model.few_step is not None:
        CommandLineParser.warn_ignored_options({
            "--guidance": "a few-step LoRA has its guidance distilled in and runs without CFG.",
            "--negative-prompt": "a few-step LoRA runs without CFG, so the negative prompt is never encoded.",
        })  # fmt: off

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=IrisLatentCreator,
    )

    try:
        for seed in args.seed:
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance=args.guidance,
                negative_prompt=args.negative_prompt,
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
