from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.lens.variants.txt2img.lens_image import LensImage
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs when --model is omitted. The parser needs it too, to key the
# --steps default off the right model instead of falling back to FLUX.1-dev's 25.
DEFAULT_MODEL = "lens-turbo"

# Single source of truth for options this CLI accepts but cannot honour: the runtime
# warning and the mflux-capabilities dump both read it.
IGNORED_OPTIONS = {
    "--guidance": "Lens Turbo is a 4-step distillation with CFG internalized; guidance is never applied.",
    "--negative-prompt": "CFG is disabled on Lens Turbo, so the negative prompt is never encoded.",
    "--scheduler": "Lens Turbo runs the shifted sigma schedule its distillation was trained on.",
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Microsoft Lens (Turbo).")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_output_arguments()
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)

    # --model accepts only lens aliases; anything else errors instead of being
    # silently run as Lens Turbo.
    model = LensImage(
        model_config=ConfigResolution.resolve_restricted(args.model, "lens-turbo", model_path=args.model_path),
        quantize=args.quantize,
        model_path=args.model_path,
    )

    # Lens rides FLUX.2 latents, so the flux2 unpacker is the one that turns its packed
    # (1, seq, 128) tensor back into something the VAE can decode for stepwise output.
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Flux2LatentCreator,
    )

    try:
        width, height = DimensionResolver.resolve(width=args.width, height=args.height)
        for seed in args.seed:
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=width,
                height=height,
                num_inference_steps=args.steps,
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
