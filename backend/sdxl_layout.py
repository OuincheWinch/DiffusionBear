"""Does this directory hold an SDXL model the engine can actually load?

THE FAILURE THIS EXISTS TO PREVENT
`sdxl_engine.py` calls `StableDiffusionXLPipeline.from_diffusers(...)`, which needs
a diffusers directory: `model_index.json` to resolve the pipeline, and a populated
`unet/` to build the UNet. A missing or short `unet/` surfaces as

    FileNotFoundError: No .safetensors files in .../unet

after a forty-line traceback, having told the user nothing about the cause. That
was reported from a second machine, where the download had been interrupted and
left a half-written 9.5 GB shard.

WHY A SEPARATE MODULE
It is needed in three places that cannot import each other cleanly: the download
worker, the adopt/register route (both in routers/downloads.py) and the
generation path (generator.py, which routers/downloads.py already imports).
Putting it here keeps the layering acyclic. routers/downloads.py re-exports it so
existing imports and tests keep working.

WHY NOT SIMILAR TO _has_weights()
That check searches recursively for any .safetensors anywhere, which is correct for
the mflux engines -- they load a single file. It is far too weak for SDXL, and
satisfied by a single-file checkpoint, which is how almost every SDXL model is
published on Hugging Face.
"""

from pathlib import Path

# A diffusers SDXL directory needs model_index.json to resolve the pipeline, and a
# unet/ holding the actual weights.
SDXL_REQUIRED = ("model_index.json",)
SDXL_WEIGHT_DIR = "unet"


def _incomplete_files(directory: Path) -> list[Path]:
    """Partially written downloads.

    huggingface_hub writes into `.cache/huggingface/download/` and only moves a
    file into place once it is whole, so an interrupted download leaves an
    `.incomplete` or `.part` behind and the final name never appears. Distinguishing
    this from "wrong format" is the whole point: the first is resumable, the second
    is not.
    """
    root = directory / ".cache" / "huggingface" / "download"
    if not root.is_dir():
        return []
    try:
        return [
            p for p in root.rglob("*")
            if p.is_file() and p.name.endswith((".incomplete", ".part"))
        ]
    except OSError:
        return []


def sdxl_layout_report(directory: Path) -> tuple[bool, str]:
    """Is `directory` a diffusers SDXL model this engine can load?

    Returns (ok, reason). The reason names the missing part, in terms the owner can
    act on, because the failure it replaces was a raw traceback.
    """
    directory = Path(directory)

    if not directory.is_dir():
        return False, f"{directory.name} does not exist, so the model was never installed."

    # Interrupted before the layout check: say so, because it is resumable and the
    # other messages in here would send the owner down the wrong path.
    partial = _incomplete_files(directory)
    if partial:
        megabytes = sum(p.stat().st_size for p in partial) // (1 << 20)
        return False, (
            f"{directory.name} has an unfinished download ({megabytes} MB still partial). "
            f"The app was closed or the download was interrupted partway through. "
            f"Re-download the model from the Models tab; nothing else will fix this."
        )

    for required in SDXL_REQUIRED:
        if not (directory / required).is_file():
            single = any(directory.glob("*.safetensors"))
            if single:
                return False, (
                    f"{directory.name} is a single-file checkpoint, but SDXL needs the "
                    f"diffusers folder layout ({SDXL_REQUIRED[0]} plus {SDXL_WEIGHT_DIR}/). "
                    f"Convert it first, or pick a repo published in diffusers format."
                )
            return False, (
                f"{directory.name} has no {required}, so it is not a diffusers directory. "
                f"SDXL needs {SDXL_REQUIRED[0]} plus {SDXL_WEIGHT_DIR}/."
            )

    weight_dir = directory / SDXL_WEIGHT_DIR
    if not weight_dir.is_dir():
        return False, (
            f"{directory.name} has no {SDXL_WEIGHT_DIR}/ directory, so it is not a "
            f"diffusers SDXL model."
        )

    if not any(weight_dir.glob("*.safetensors")):
        has_bin = any(weight_dir.glob("*.bin"))
        detail = (
            f"{SDXL_WEIGHT_DIR}/ holds .bin files, but this engine reads safetensors"
            if has_bin
            else f"{SDXL_WEIGHT_DIR}/ is empty"
        )
        return False, (
            f"{directory.name}/{SDXL_WEIGHT_DIR}/ has no .safetensors files -- {detail}. "
            f"The download is incomplete; re-download the model from the Models tab."
        )

    return True, ""


def is_sdxl_engine(model_id: str, models: dict | None = None) -> bool:
    """SDXL runs through venv-sdxl/mlx_diffuser and needs the layout above."""
    if models is None:
        import generator

        models = generator.MODELS
    info = models.get(model_id) or {}
    return info.get("ecosystem") == "SDXL" or info.get("lora_format") == "SDXL"