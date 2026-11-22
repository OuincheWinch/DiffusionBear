"""Filesystem-only helpers to locate shared MOSS/Kromoss assets without importing MLX."""

from __future__ import annotations

from pathlib import Path

TE_DIRNAME = "qwen3_5-te-q8g64"
MOSS_SIBLINGS = ("moss-image-sft", "kromoss", "moss-image-base")


def has_complete_moss_te(te_dir: Path | None = None) -> bool:
    """Return True when the shared MLX Qwen 3.5 text encoder directory is complete on disk."""
    candidates = [te_dir] if te_dir is not None else []
    if te_dir is None:
        try:
            import app_settings
            candidates.append(Path(app_settings.ASSET_DIR) / "models" / TE_DIRNAME)
        except Exception:
            pass
        candidates.append(Path(__file__).resolve().parent / "data" / "models" / TE_DIRNAME)
    for cand in candidates:
        if cand is None or not cand.is_dir():
            continue
        cfg = cand / "config.json"
        weights = cand / "model.safetensors"
        if not (cfg.is_file() and weights.is_file()):
            continue
        try:
            if weights.stat().st_size <= 1024:
                continue
        except OSError:
            continue
        if any(p.is_file() and p.name.endswith((".incomplete", ".part")) for p in cand.rglob("*")):
            continue
        return True
    return False


def has_moss_vae_weights(vae_dir: Path) -> bool:
    if not vae_dir.is_dir():
        return False
    if any(p.is_file() and p.name.endswith((".incomplete", ".part")) for p in vae_dir.rglob("*")):
        return False
    return (
        (vae_dir / "diffusion_pytorch_model.safetensors").is_file()
        or (vae_dir / "0.safetensors").is_file()
    )


def find_shared_moss_vae_dir(
    exclude_dir: Path | None = None,
    models_dir: Path | None = None,
) -> Path | None:
    """Return an already-installed sibling MOSS VAE directory on disk, or None."""
    roots: list[Path] = []
    if models_dir is not None:
        roots.append(Path(models_dir))
    if exclude_dir is not None and exclude_dir.parent not in roots:
        roots.append(exclude_dir.parent)
    if models_dir is None:
        try:
            import app_settings
            asset_models = Path(app_settings.ASSET_DIR) / "models"
            if asset_models not in roots:
                roots.append(asset_models)
        except Exception:
            pass
    seen: set[Path] = set()
    excl_resolved = exclude_dir.resolve() if exclude_dir is not None else None
    for models_root in roots:
        for sib in MOSS_SIBLINGS:
            cand = models_root / sib / "vae"
            if not cand.is_dir():
                continue
            try:
                resolved = cand.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if excl_resolved is not None and (models_root / sib).resolve() == excl_resolved:
                continue
            if has_moss_vae_weights(cand):
                return cand
    return None


def find_shared_moss_tokenizer_dir(
    exclude_dir: Path | None = None,
    models_dir: Path | None = None,
    *,
    include_repo_data: bool = True,
) -> Path | None:
    """Return an existing MOSS tokenizer directory on disk outside exclude_dir, or None."""
    roots: list[Path] = []
    if models_dir is not None:
        roots.append(Path(models_dir))
    if exclude_dir is not None and exclude_dir.parent not in roots:
        roots.append(exclude_dir.parent)
    if models_dir is None:
        try:
            import app_settings
            asset_models = Path(app_settings.ASSET_DIR) / "models"
            if asset_models not in roots:
                roots.append(asset_models)
        except Exception:
            pass

    candidates: list[Path] = []
    for models_root in roots:
        candidates.append(models_root / "moss-tokenizer")
        for sib in MOSS_SIBLINGS:
            candidates.append(models_root / sib / "tokenizer")
    if include_repo_data:
        candidates.append(Path(__file__).resolve().parent / "data" / "models" / "moss-tokenizer")

    excl_resolved = exclude_dir.resolve() if exclude_dir is not None else None
    for cand in candidates:
        if not cand.is_dir():
            continue
        if not ((cand / "chat_template.jinja").is_file() and (cand / "tokenizer.json").is_file()):
            continue
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if excl_resolved is not None and (resolved == excl_resolved or resolved.is_relative_to(excl_resolved)):
            continue
        return cand
    return None
