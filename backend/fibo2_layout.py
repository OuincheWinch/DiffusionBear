"""Does this directory hold a FIBO 2.0 Edit Turbo model the engine can load?

A FIBO 2.0 4-bit package requires three components:
1. transformer/ : FIBO 2.0 DiT Transformer (config.json + 4-bit safetensors)
2. text_encoder/: Qwen3-VL-4B Instruct Text Encoder (config.json + 4-bit safetensors)
3. vae/         : FLUX.2 AutoencoderKL VAE Decoder (config.json + fp16 safetensors)
"""

from pathlib import Path


FIBO2_REQUIRED_DIRS = ("transformer", "text_encoder", "vae")


def _incomplete_files(directory: Path) -> list[Path]:
    """Partially written downloads."""
    root = directory / ".cache" / "huggingface" / "download"
    if not root.is_dir():
        return []
    try:
        return [
            p for p in root.rglob("*")
            if p.is_file() and (p.name.endswith(".incomplete") or p.name.endswith(".part"))
        ]
    except OSError:
        return []


def is_fibo2_engine(model_id: str | None) -> bool:
    """True if model_id belongs to the FIBO 2.0 engine family."""
    mid = str(model_id or "").lower()
    return "fibo2" in mid or "fibo" in mid


def fibo2_layout_report(directory: Path | None) -> tuple[bool, str | None]:
    """Inspect directory and report whether it can be loaded by fibo_engine."""
    if directory is None:
        return False, "no directory provided"
    p = Path(directory)
    if not p.is_dir():
        return False, f"{p} is not a directory"

    incompletes = _incomplete_files(p)
    if incompletes:
        sample = incompletes[0].name
        return False, f"download incomplete ({len(incompletes)} partial file(s) remain, e.g. {sample})"

    for req_dir in FIBO2_REQUIRED_DIRS:
        sub = p / req_dir
        if not sub.is_dir():
            return False, f"missing required directory: {req_dir}/"
        cfg = sub / "config.json"
        if not cfg.is_file():
            return False, f"{req_dir}/ is missing config.json"
        has_safetensors = any(sub.glob("*.safetensors"))
        if not has_safetensors:
            return False, f"{req_dir}/ contains no .safetensors weights"

    return True, None
