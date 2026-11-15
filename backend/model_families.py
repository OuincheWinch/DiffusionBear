"""Canonical model-family identification.

This module is the single source of truth for what counts as a model family in
DiffusionBear, how every known spelling maps onto it, and which families can load
LoRAs. It imports nothing from the project and performs no I/O, so every backend
module and the frontend can depend on it without cycles.

The vocabulary split it replaces was real: the LoRA registry said ``flux2`` for every
FLUX adapter, the custom-model registry said ``krea`` and ``zit``, and the download
allow-list omitted ``moss``. A FLUX.1 LoRA was refused from Civitai and, had it been
stored, would have borrowed FLUX.2's block names and parameter count. Families are
now distinct and a family never borrows another family's metrics or mapping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

CANONICAL_FAMILIES: tuple[str, ...] = (
    "sdxl",
    "flux1",
    "flux2",
    "flux3",
    "krea2",
    "z-image",
    "qwen",
    "moss",
    "fibo2",
)

LORA_FAMILIES: frozenset[str] = frozenset(
    {"sdxl", "flux1", "flux2", "krea2", "z-image", "qwen", "moss"}
)


@dataclass(frozen=True)
class Family:
    id: str
    label: str
    aliases: tuple[str, ...]
    lora_supported: bool
    lora_mapping_key: str | None
    metrics: str


@dataclass(frozen=True)
class Resolved:
    family: str
    subtype: str | None = None
    variant: str | None = None
    source: str = ""
    reason: str | None = None
    unsupported: bool = False


class UnsupportedArchitecture(ValueError):
    """A recognised architecture for which this app has no engine or loader."""


class UnknownArchitecture(ValueError):
    """Metadata that names no architecture this app recognises."""


FAMILIES: dict[str, Family] = {
    "sdxl": Family("sdxl", "SDXL", ("sdxl", "stable diffusion xl"), True, None, "3.5B-class"),
    "flux1": Family("flux1", "FLUX.1", ("flux.1", "flux1", "flux dev", "flux-dev", "flux schnell", "flux-schnell", "flux kontext", "flux"), True, "flux1", "12B"),
    "flux2": Family("flux2", "FLUX.2", ("flux.2", "flux2", "klein"), True, "flux2-klein", "4B/9B"),
    "flux3": Family("flux3", "FLUX.3", ("flux.3", "flux3", "flux-3", "flux_3"), False, None, "unknown"),
    "krea2": Family("krea2", "Krea 2", ("krea",), True, "krea2", "unknown"),
    "z-image": Family("z-image", "Z-Image", ("z-image", "zimage", "z_image", "zit"), True, "z-image", "unknown"),
    "qwen": Family("qwen", "Qwen-Image", ("qwen",), True, "qwen-image", "unknown"),
    "moss": Family("moss", "MOSS", ("moss",), True, "moss", "5B"),
    "fibo2": Family("fibo2", "FIBO 2.0", ("fibo 2", "fibo2", "fibo-2", "fibo"), False, None, "unknown"),
}

# Ordered: the first rule whose needle is a substring wins. Specific spellings must
# precede generic ones. `Flux.1 Krea` is FLUX.1, not Krea 2; `Flux.2` must beat `Flux`.
_RULES: tuple[tuple[tuple[str, ...], str, str | None, bool], ...] = (
    (("flux.3", "flux3", "flux-3", "flux_3"), "flux3", None, False),
    (("flux.2", "flux2", "klein"), "flux2", None, False),
    (("flux.1", "flux1", "flux dev", "flux-dev", "flux schnell", "flux-schnell", "flux kontext", "flux"), "flux1", None, False),
    (("illustrious",), "sdxl", "illustrious", False),
    (("pony",), "sdxl", "pony", False),
    (("sdxl", "stable diffusion xl", "sd xl", "noobai"), "sdxl", None, False),
    (("krea",), "krea2", None, False),
    (("z-image", "zimage", "z_image", "zit"), "z-image", None, False),
    (("qwen",), "qwen", None, False),
    (("moss",), "moss", None, False),
    (("fibo 2", "fibo2", "fibo-2", "fibo"), "fibo2", None, False),
    (("sd 1.5", "sd1.5", "sd 1.4", "sd14", "sd15", "sd 2.0", "sd 2.1", "sd20", "sd21", "stable diffusion 1.5"), "sd15", None, True),
    (("sd3", "stable diffusion 3"), "sd3", None, True),
    (("hunyuan",), "hunyuan", None, True),
    (("auraflow",), "auraflow", None, True),
    (("pixart",), "pixart", None, True),
    (("cascade",), "cascade", None, True),
    (("wan",), "wan", None, True),
    (("ltx",), "ltx", None, True),
    (("hidream",), "hidream", None, True),
    (("kolors",), "kolors", None, True),
)

_VARIANT_RE = re.compile(r"(?:^|[^a-z0-9])(4b|9b)(?:[^a-z0-9]|$)")

_CUSTOM_TO_CANONICAL = {
    "sdxl": "sdxl",
    "flux1": "flux1",
    "flux2": "flux2",
    "flux3": "flux3",
    "krea": "krea2",
    "zit": "z-image",
    "qwen": "qwen",
    "moss": "moss",
    "fibo2": "fibo2",
}
_CANONICAL_TO_CUSTOM = {value: key for key, value in _CUSTOM_TO_CANONICAL.items()}

_KEY_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("to_qkv_mlp_proj", "time_guidance_embed", "double_stream_modulation", "single_stream_modulation", "ff.linear_in", "ff.linear_out"), "flux2"),
    (("proj_mlp", "time_text_embed", "ff.linear1", "ff.linear2"), "flux1"),
    (("txtfusion", "tmlp", "tproj"), "krea2"),
    (("noise_refiner", "context_refiner", "adaln_modulation", "feed_forward.w1"), "z-image"),
    (("img_mlp",), "qwen"),
    (("diffusion_model.layers.",), "moss"),
    (("lora_te1_", "lora_te2_", "down_blocks", "up_blocks", "input_blocks", "middle_block", "output_blocks"), "sdxl"),
)


def _lower(value: object) -> str:
    return str(value or "").strip().lower()


def _matches(needle: str, haystack: str) -> bool:
    if needle == "flux":
        # `mflux` is a packaging suffix, not a FLUX model. Bare `flux` must be a token.
        return re.search(r"(?<![a-z])flux(?![a-z])", haystack) is not None
    return needle in haystack


def _variant_for(family: str, haystack: str) -> str | None:
    if family == "flux2":
        match = _VARIANT_RE.search(haystack)
        return match.group(1) if match else None
    if family == "qwen":
        for variant in ("edit", "turbo", "base"):
            if variant in haystack:
                return variant
    return None


def _resolve_text(value: str | None, source: str, extra: str = "") -> Resolved | None:
    haystack = f"{_lower(value)} {_lower(extra)}".strip()
    if not haystack:
        return None
    for needles, family, subtype, unsupported in _RULES:
        if any(_matches(needle, haystack) for needle in needles):
            reason = f"unsupported architecture: {family}" if unsupported else None
            return Resolved(
                family=family,
                subtype=subtype,
                variant=_variant_for(family, haystack),
                source=source,
                reason=reason,
                unsupported=unsupported,
            )
    return None


def resolve_civitai_base(value: str | None) -> Resolved | None:
    return _resolve_text(value, "civitai")


def resolve_hf_repo(repo_id: str, tags: list[str] | None = None) -> Resolved | None:
    # A structured `base_model:` tag is stronger evidence than the repository
    # name: LoRAs are routinely named after their base model, and a stale or
    # mirrored repo name must not override the publisher's declaration.
    for tag in tags or []:
        text = str(tag).strip()
        if not text.lower().startswith("base_model:"):
            continue
        value = text.split(":", 1)[1].strip()
        resolved = _resolve_text(value, "huggingface")
        if resolved is not None:
            return resolved
    return _resolve_text(repo_id, "huggingface", " ".join(str(tag) for tag in (tags or [])))


def canonical(value: str | None) -> str | None:
    resolved = resolve_civitai_base(value)
    if resolved is None or resolved.unsupported:
        return None
    return resolved.family


def custom_family_to_canonical(family: str | None) -> str | None:
    return _CUSTOM_TO_CANONICAL.get(_lower(family)) if family else None


def canonical_to_custom_family(family: str | None) -> str | None:
    return _CANONICAL_TO_CUSTOM.get(_lower(family)) if family else None


def is_lora_supported(family: str | None) -> bool:
    return _lower(family) in LORA_FAMILIES


def lora_mapping_key(family: str | None) -> str | None:
    record = FAMILIES.get(_lower(family))
    return record.lora_mapping_key if record else None


def require_known(value: str | None) -> Resolved:
    resolved = resolve_civitai_base(value)
    if resolved is None:
        raise UnknownArchitecture(
            f"Model architecture '{value}' was not recognised. Pick the base model manually."
        )
    if resolved.unsupported:
        raise UnsupportedArchitecture(
            f"Model architecture '{value}' is not supported by DiffusionBear."
        )
    return resolved


def require_lora_family(value: str | None) -> Resolved:
    resolved = require_known(value)
    if not is_lora_supported(resolved.family):
        raise UnsupportedArchitecture(
            f"No engine in DiffusionBear can load {FAMILIES[resolved.family].label} LoRAs."
        )
    return resolved


def _subtype_from_title(meta: Mapping[str, Any]) -> str | None:
    title = _lower(meta.get("modelspec.title"))
    if "illustrious" in title:
        return "illustrious"
    if "pony" in title:
        return "pony"
    return None


def _family_from_keys(keys: list[str]) -> str | None:
    haystack = " ".join(str(key).lower() for key in keys)
    matches: list[str] = []
    for needles, family in _KEY_RULES:
        if any(needle in haystack for needle in needles):
            matches.append(family)
    if len(matches) == 1:
        return matches[0]
    # `single_transformer_blocks.*.norm.linear` is FLUX.1-only; FLUX.2 uses
    # `single_stream_modulation` and `to_qkv_mlp_proj`, which are checked above.
    if "flux1" not in matches and "single_transformer_blocks" in haystack and "norm.linear" in haystack:
        return "flux1"
    if len(matches) > 1:
        return None
    return None


def resolve_safetensors(
    header: Mapping[str, Any], filename: str = "", *, allow_name_fallback: bool = True
) -> Resolved | None:
    if not isinstance(header, Mapping):
        return None
    meta = header.get("__metadata__")
    meta = meta if isinstance(meta, Mapping) else {}
    for field in ("modelspec.architecture", "ss_base_model_version"):
        resolved = _resolve_text(meta.get(field), "safetensors")
        if resolved is not None:
            if resolved.unsupported:
                return resolved
            if resolved.family == "sdxl" and resolved.subtype is None:
                resolved = Resolved(
                    family="sdxl",
                    subtype=_subtype_from_title(meta),
                    variant=resolved.variant,
                    source="safetensors",
                )
            return resolved
    # MOSS adapters reuse generic `lora_unet_`-style keys, so the filename is the only
    # conclusive signal for that family. This is not a name-over-layout rule: the
    # layout is ambiguous exactly here.
    named = _resolve_text(filename, "safetensors")
    if named is not None and named.family == "moss" and not named.unsupported:
        return named
    keys = [str(key) for key in header.keys() if key != "__metadata__"]
    family = _family_from_keys(keys)
    if family is not None:
        subtype = _subtype_from_title(meta) if family == "sdxl" else None
        variant = _variant_for(family, _lower(meta.get("modelspec.title")) + " " + _lower(filename))
        return Resolved(family=family, subtype=subtype, variant=variant, source="safetensors")
    if allow_name_fallback:
        return _resolve_text(filename, "safetensors")
    return None


def resolve_model(model_id: str, meta: Mapping[str, Any] | None = None) -> Resolved | None:
    meta = meta if isinstance(meta, Mapping) else {}
    custom = custom_family_to_canonical(meta.get("family"))
    if custom:
        return Resolved(family=custom, source="model")
    for field in ("lora_format", "ecosystem"):
        resolved = _resolve_text(meta.get(field), "model")
        if resolved is not None:
            return resolved
    engine = _lower(meta.get("engine"))
    if engine in ("sdxl", "moss", "qwen"):
        return Resolved(family=engine, source="model")
    return _resolve_text(model_id, "model")


if __name__ == "__main__":
    import sys

    print(resolve_model(sys.argv[1] if len(sys.argv) > 1 else ""))
