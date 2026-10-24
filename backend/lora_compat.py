"""Is this LoRA built for the model you are about to generate with?

The failure this exists to prevent arrives at the END of a generation:

    44 of the 144 layers named by Logo.Redmond__Logo_Lora__FLUX.2.Klein... could not
    be applied (single_transformer_blocks.23.attn.to_out, ... and 39 more); each
    reason is printed above. The adapter and this model do not match.

That is a true statement, arrived at far too late, phrased in terms nobody can act on.
It also says "144" -- the total the adapter plus the model happened to add up to -- as
though that number meant something. It does not. What means something is WHICH layers
could not be applied, grouped by block family, said before the generation starts.

So this answers the same question the loader answers, earlier, and specifically.

HOW IT ANSWERS IT WITHOUT LOADING ANYTHING
------------------------------------------
mflux resolves an adapter by string-matching its keys against a per-model list of
patterns, each naming a module path in the transformer. Whether a target resolves
depends on two facts, both of which are readable without weights:

  1. which patterns the model's mapping class provides, and
  2. which module paths actually exist in the model.

Rather than restate either, this borrows mflux's own `_build_pattern_mappings` and
`_match_pattern`. A hand-rolled comparison is exactly the trap recorded in AGENTS.md --
a key checked against a vocabulary that can agree with itself while both are wrong.
Here the vocabulary is the same object the loader will use, so a verdict cannot drift
from the behaviour it predicts.

Module paths come from the weight index in the local Hugging Face cache, so no network
call is needed: a check that required the internet would be useless exactly when it
matters. The model's weights are never loaded, so this is cheap enough to run on every
LoRA toggle rather than at generation time.

Deliberately conservative: "unknown" is never reported as compatible. A model whose
weights are absent cannot be verified, and quietly passing it is how a mismatch
reaches the user as a ninety-second failure.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

# Trailing tensor names in a quantised mflux weight index that are NOT module paths:
# a quantised Linear stores the weight plus its scales and biases beside it.
_TENSOR_SUFFIXES = ("weight", "scales", "biases", "bias", "scale")

_BLOCK_FAMILY = re.compile(r"(?:^|\.)([a-z_]+_blocks)(?:\.|$)")

def _load_flux2_mapping():
    from mflux.models.flux2.weights.flux2_lora_mapping import Flux2LoRAMapping
    return Flux2LoRAMapping


def _load_flux_mapping():
    from mflux.models.flux.weights.flux_lora_mapping import FluxLoRAMapping
    return FluxLoRAMapping


def _load_z_image_mapping():
    from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping
    return ZImageLoRAMapping


def _load_krea2_mapping():
    from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping
    return Krea2LoRAMapping


def _load_qwen_mapping():
    from mflux.models.qwen.weights.qwen_lora_mapping import QwenLoRAMapping
    return QwenLoRAMapping


# model_id prefix -> a callable that imports and returns mflux's LoRA mapping for it.
#
# Kept explicit rather than derived from the model class, because the mapping is chosen
# inside mflux's initializer, not exposed on the class. Prefix matching is deliberate:
# every FLUX.2 Klein size shares one mapping, which is the point -- a LoRA valid for the
# 9B is valid for the 4B and vice versa, and a difference in coverage is then a real
# finding rather than an artefact of which size was typed.
#
# Each entry is a one-line function with a PLAIN import rather than a string module path
# handed to a dynamic importer. Two reasons, both learned:
#
#   * Eagerly importing all four costs 4.1 s and 2,064 modules and pulls in mlx.core,
#     for a check that runs only when someone picks a LoRA. That is why this is lazy.
#   * The obvious lazy spelling hands a string module path to Python's dynamic import
#     builtin, and test_repo_security fails on that -- correctly. The guard exists to
#     stop code being loaded from a computed string; it cannot tell a hardcoded
#     constant from an injected one, so an allowance for it would have to be taken on
#     trust. Four trivial functions cost nothing and leave the guard strict for
#     everything else. (The guard is a plain regex over file text, so even naming the
#     builtin in a comment trips it -- hence the wording here.)
_MAPPING_CLASSES: dict[str, Callable[[], Any]] = {
    "flux2-klein": _load_flux2_mapping,
    "flux1": _load_flux_mapping,
    "z-image": _load_z_image_mapping,
    "krea2": _load_krea2_mapping,
    "qwen-image": _load_qwen_mapping,
}


# Custom-registry family -> the _MAPPING_CLASSES key whose loader it uses.
# sdxl is absent on purpose: the SDXL engine is not mflux, so no mapping.
_CUSTOM_FAMILY_MAPPINGS = {
    "flux2": "flux2-klein",
    "krea": "krea2",
    "zit": "z-image",
    "qwen": "qwen-image",
}


def mapping_for(model_id: str):
    """mflux's LoRA mapping class for a registry model id, or None if not mflux.

    A custom registry entry is consulted first: its explicit family decides,
    via the same loaders the prefix table names. Anything else falls through
    to the built-in prefix match; unknown ids still return None.
    """
    import custom_models

    for entry in custom_models.read_custom_models():
        if entry.get("id") == model_id:
            key = _CUSTOM_FAMILY_MAPPINGS.get(entry.get("family"))
            if key is None:
                return None
            try:
                return _MAPPING_CLASSES[key]()
            except (ImportError, AttributeError):
                return None
    for prefix, load in _MAPPING_CLASSES.items():
        if model_id.startswith(prefix):
            try:
                return load()
            except (ImportError, AttributeError):
                return None
    return None


@lru_cache(maxsize=64)
def lora_keys(path: str | Path) -> tuple[str, ...]:
    """Adapter tensor names, from the safetensors header only.

    Header-only matters: this answers from a few KB what would otherwise open a
    165 MB file just to look at its key list. Cached on the resolved path so two
    entries pointing at one file cost one read. Returns a tuple for the same
    unhashable-mutation reason as `model_module_paths`.
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return tuple(f.keys())


@lru_cache(maxsize=256)
def model_module_paths(model_id: str) -> frozenset[str]:
    """Every module path the model's transformer actually has.

    Read from the weight index rather than by instantiating the model, which is the
    whole point: no weights, no memory, no minutes.

    Cached because the LoRA picker polls this per selection, and re-reading the index
    every time buys nothing. A model gaining or losing weights mid-session would not be
    seen until restart -- which is the acceptable direction for a mistake to go stale,
    since it is a cache MISS on the new path that is cheap and a cache HIT that is wrong
    only until restart.

    Returns a frozenset so a caller cannot mutate the cached value.
    """
    index = _find_weight_index(model_id)
    if index is None:
        return frozenset()
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return frozenset()

    paths: set[str] = set()
    for name in (data.get("weight_map") or {}):
        parts = name.split(".")
        if parts and parts[-1] in _TENSOR_SUFFIXES:
            parts = parts[:-1]
        if parts:
            paths.add(".".join(parts))
    return frozenset(paths)


def _find_weight_index(model_id: str) -> Path | None:
    import generator

    model = generator.get_model_info(model_id)
    if model is None:
        return None

    # An explicitly configured local path is the weights actually in use and must win
    # over the published repo.
    local = model.get("local_path")
    candidates: list[Path] = []
    if local:
        base = Path(str(local)).expanduser()
        candidates += [
            base / "transformer" / "model.safetensors.index.json",
            base / "model.safetensors.index.json",
        ]
    repo = model.get("repo")
    if isinstance(repo, str) and not repo.startswith("local:"):
        candidates += list(
            (generator.ASSET_DIR / "models").glob(f"{model_id}/transformer/*.index.json")
        )
        try:
            from huggingface_hub import snapshot_download

            snapshot = Path(snapshot_download(
                repo, allow_patterns=["transformer/*.index.json"], local_files_only=True))
            candidates += sorted(snapshot.glob("transformer/*.index.json"))
        except Exception:
            pass
    else:
        candidates += list(
            (generator.ASSET_DIR / "models").glob(f"{model_id}/transformer/*.index.json")
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _resolve_with(mapping_class, model_paths: set[str],
                  keys: tuple[str, ...]) -> tuple[set[str], set[str]]:
    """The loader's own matching against an explicit module list.

    Split out so a sibling model's fit can be probed without re-resolving
    which mapping belongs to it: same matcher, different module list, no
    second implementation to drift.
    """
    from mflux.models.common.lora.mapping.lora_loader import LoRALoader

    pattern_mappings = LoRALoader._build_pattern_mappings(mapping_class.get_mapping())

    hit: set[str] = set()
    for key in keys:
        for mapping in pattern_mappings:
            block = LoRALoader._match_pattern(key, mapping.source_pattern)
            if block is None:
                continue
            target = mapping.target_path
            if block is not None and "{block}" in target:
                target = target.format(block=block)
            hit.add(target)

    return hit & model_paths, hit - model_paths


def resolve_targets(model_id: str, keys: tuple[str, ...]) -> tuple[set[str], set[str]]:
    """Split the targets a LoRA resolves to into (found, missing) against the model.

    This is mflux's own matching, run against the model's real module list, so it
    predicts `_apply_single_lora`'s `failed_targets` exactly rather than approximately.
    """
    mapping_class = mapping_for(model_id)
    if mapping_class is None:
        return set(), set()

    return _resolve_with(mapping_class, model_module_paths(model_id), keys)


def _mapping_prefix(model_id: str) -> str | None:
    """The registry family a model id belongs to, without importing anything.

    Deliberately prefix-based rather than class-based: deciding "same family"
    must not import every mapping module (one of those imports costs seconds
    and pulls in MLX), and a prefix that matches nothing simply yields no
    siblings rather than a wrong answer.
    """
    for prefix in _MAPPING_CLASSES:
        if model_id.startswith(prefix):
            return prefix
    return None


def fitting_siblings(model_id: str, keys: tuple[str, ...]) -> list[str]:
    """Registry models this adapter DOES fit, when it does not fit `model_id`.

    Only same-family models that support LoRAs and whose weights are on disk
    are probed; anything unverifiable is skipped, never claimed. An empty
    list means "fits nowhere we can check", not "fits nowhere".
    """
    import generator

    prefix = _mapping_prefix(model_id)
    if prefix is None:
        return []
    mapping_class = mapping_for(model_id)
    if mapping_class is None:
        return []

    fits = []
    for mid, model in generator.MODELS.items():
        if mid == model_id or not mid.startswith(prefix):
            continue
        if not model.get("supports_loras", True):
            continue
        paths = model_module_paths(mid)
        if not paths:
            continue
        found, missing = _resolve_with(mapping_class, paths, keys)
        if found and not missing:
            fits.append(mid)
    return fits


def group_by_family(paths: set[str]) -> dict[str, int]:
    """Collapse module paths to {"block_family": count}, for a message a person reads."""
    counts: dict[str, int] = {}
    for path in paths:
        match = _BLOCK_FAMILY.search(path)
        key = match.group(1) if match else "other"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def check(path: str | Path, model_id: str) -> dict[str, Any]:
    """Decide whether `path` can be applied to `model_id`, and say why not.

    Returns a dict rather than raising: "incompatible" is an answer the UI needs in
    order to explain itself, not an exception to be caught.
    """
    import generator

    model = generator.get_model_info(model_id)
    if model is None:
        return {
            "compatible": None, "unknown": True, "model_id": model_id,
            "reason": "unknown_model",
            "message": f"Unknown model {model_id}.",
        }
    if mapping_for(model_id) is None:
        return {
            "compatible": None, "unknown": True, "model_id": model_id,
            "reason": "unsupported",
            "message": "LoRA compatibility is only checked for mflux models.",
        }
    if not model.get("supports_loras", True):
        return {
            "compatible": None, "unknown": True, "model_id": model_id,
            "reason": "no_lora_support",
            "message": f"{model_id} does not support LoRAs.",
        }

    model_paths = model_module_paths(model_id)
    if not model_paths:
        return {
            "compatible": None, "unknown": True, "model_id": model_id,
            "reason": "weights_missing",
            "message": "This model's weights are not downloaded, so compatibility "
                       "could not be checked.",
        }

    try:
        keys = lora_keys(path)
    except Exception:
        return {
            "compatible": None, "unknown": True, "model_id": model_id,
            "reason": "unreadable",
            "message": "This LoRA file could not be read.",
        }
    if not keys:
        return {
            "compatible": None, "unknown": True, "model_id": model_id,
            "reason": "unreadable",
            "message": "This LoRA file contains no tensors.",
        }

    found, missing = resolve_targets(model_id, keys)
    if not found and not missing:
        return {
            "compatible": False, "unknown": False, "model_id": model_id,
            "reason": "no_recognised_keys",
            "message": f"No layer in this LoRA is recognised by {model_id}'s layout. "
                       "It was saved in a format this model does not read.",
        }

    missing_families = group_by_family(missing)
    if missing:
        total = len(found) + len(missing)
        label = model.get("label") or model_id
        fits = fitting_siblings(model_id, keys)
        if fits:
            fit_labels = [(generator.get_model_info(mid) or {}).get("label") or mid for mid in fits]
            return {
                "compatible": False, "unknown": False, "model_id": model_id,
                "reason": "size_mismatch",
                "message": f"This LoRA fits {', '.join(fit_labels)}, not {label}. "
                           "Switch model or remove the LoRA.",
                "resolved_layers": len(found), "total_layers": total,
                "missing_families": missing_families,
                "fits_models": fits,
            }
        return {
            "compatible": False, "unknown": False, "model_id": model_id,
            "reason": "architecture_mismatch",
            "message": f"This LoRA does not match {label}. "
                       "Remove it or switch models.",
            "resolved_layers": len(found), "total_layers": total,
            "missing_families": missing_families,
            "fits_models": [],
        }

    return {
        "compatible": True, "unknown": False, "model_id": model_id,
        "reason": "ok", "message": "",
        "resolved_layers": len(found), "total_layers": len(found),
        "missing_families": {}, "fits_models": [],
    }