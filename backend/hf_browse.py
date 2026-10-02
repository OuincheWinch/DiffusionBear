"""Hugging Face model discovery for the Models tab.

There was no way to search Hugging Face from the app at all: `HfApi` was only ever
imported for `model_info` on a single, already-known repo (hf_service.fetch_hf_metadata,
downloads._run_model_download). Searching therefore needs its own module.

Two things here are load-bearing and easy to get wrong:

* **Quantisation is only ever encoded in the repo id.** There is no `quantization` field
  anywhere in `GET /api/models` -- the registry infers it implicitly (`flux2-klein-4b` ->
  `quantize=4`) and `mlx-community` encodes it as a name suffix (`-4bit`, `-8bit`,
  `-3bit`, `-bf16`). So filtering on quantisation necessarily means parsing names.
* **`mlx-community` ships a text encoder mislabelled as quantised.** Its
  `krea-2-turbo-mflux-q4` index declares `quantization_level: 4` but has 399 keys, zero
  `.scales`, and 7672MB of shards -- it is bf16. Do not infer bit-depth from that metadata
  field; infer it from the repo id.
"""

from __future__ import annotations

import re
from typing import Any

from huggingface_hub import HfApi

import hf_service

DEFAULT_AUTHOR = "mlx-community"
MAX_LIMIT = 100

# Orgs worth offering in the picker. mlx-community is where MLX-ready weights actually
# live, but the families Fred cares about are split across orgs -- there is no Illustrious or
# Pony repo in mlx-community at all, so an org-locked browser can never show them.
ORG_CHOICES = [
    {"key": "mlx-community", "label": "mlx-community"},
    {"key": "mflux-community", "label": "mflux-community (mflux layout)"},
    {"key": "all", "label": "All of Hugging Face"},
    {"key": "cagliostrolab", "label": "cagliostrolab"},
    {"key": "ostris", "label": "ostris (Pony)"},
    {"key": "black-forest-labs", "label": "black-forest-labs"},
    {"key": "krea", "label": "krea"},
    {"key": "Qwen", "label": "Qwen"},
]

# HF pipeline tags that correspond to something this app can actually generate. Anything
# else -- notably text-to-speech like Ming-omni-tts, which shares a name fragment with the
# Ming image models and was being returned as a "generator" -- is excluded by default.
_IMAGE_PIPELINES = {
    "text-to-image",
    "image-to-image",
    "image-to-text",
    "text-to-video",
    "image-to-video",
    "image-inpainting",
    "image-editing",
    "unconditional-image-generation",
    "fill-mask",
    "inpainting",
}

# Ordered longest-first so "flux.2" wins over "flux", and "illustrious" over a bare "ill".
_ARCHITECTURES: list[tuple[str, str]] = [
    ("flux.2", "FLUX.2"),
    ("flux2", "FLUX.2"),
    ("flux.1", "FLUX.1"),
    ("flux1", "FLUX.1"),
    ("flux-dev", "FLUX.1"),
    ("flux-schnell", "FLUX.1"),
    ("krea", "Krea"),
    ("z-image", "Z-Image"),
    ("zimage", "Z-Image"),
    ("qwen-image", "Qwen-Image"),
    ("qwenimage", "Qwen-Image"),
    ("ming-image", "Ming"),
    ("ming", "Ming"),
    ("illustrious", "Illustrious"),
    ("pony", "Pony"),
    ("sdxl", "SDXL"),
    ("xl", "SDXL"),
    ("hunyuan", "Hunyuan"),
    ("auraflow", "AuraFlow"),
    ("pixart", "PixArt"),
    ("cascade", "Cascade"),
    ("sd3", "Stable Diffusion 3"),
    ("stable-diffusion-3", "Stable Diffusion 3"),
    ("sd15", "SD 1.5"),
    ("sd-15", "SD 1.5"),
    ("sd-v1-5", "SD 1.5"),
    ("ltx", "LTX"),
    ("wan", "Wan"),
    ("deepseek", "DeepSeek"),
    ("gemma", "Gemma"),
    ("qwen", "Qwen"),
]

# Upscalers and aux models are listed alongside generators but are not engines. Tagging
# them lets the UI show them without pretending they can be picked as a Generate model.
_UPSCALERS: list[tuple[str, str]] = [
    ("restormer", "Restormer"),
    ("realesrgan", "Real-ESRGAN"),
    ("esrgan", "ESRGAN"),
    ("swinir", "SwinIR"),
    ("dedsr", "Real-ESRGAN"),
    ("upscal", "Upscaler"),
    ("spandrel", "Upscaler"),
]

_LORA_HINTS = ("lora", "lycoris", "locon", "dora")

# Some community mirrors carry no usable metadata at all: mlx-community/Ming-omni-tts-
# 16.8B-A3B-4bit has pipeline_tag=None and tags of just ["safetensors","bailingmm",
# "custom_code","4-bit","region:us"] -- the upstream repo says text-to-speech, the mirror
# says nothing. Name fragments are the last discriminator we have. Deliberately narrow:
# only audio/speech terms, so Ming-Image and Ming-omni-tts can be told apart.
_NON_IMAGE_NAME_HINTS = ("tts", "speech", "asr", "audio", "voice", "whisper")

# Ming-Image renders with a native alpha channel, which is worth surfacing: it is the one
# model family here that can emit transparency the others cannot.
_ALPHA_FAMILIES = {"Ming"}

# Literal schemes first, then "<n>bit" forms. Two separate lists on purpose: mixing
# literal labels with "<n>bit" captures in one table invites exactly the KeyError an
# earlier draft of this file had, because only the latter produces a usable group.
_QUANT_LITERALS: list[tuple[str, str]] = [
    (r"(?:^|[-_])mxfp4", "MXFP4"),
    (r"(?:^|[-_])mfp4", "MFP4"),
    (r"(?:^|[-_])xfp4", "XFP4"),
    (r"(?:^|[-_])nf4", "NF4"),
    (r"(?:^|[-_])bf16", "BF16"),
    (r"(?:^|[-_])fp16", "FP16"),
    (r"(?:^|[-_])fp32", "FP32"),
    (r"(?:^|[-_])float16", "FP16"),
    (r"(?:^|[-_])float32", "FP32"),
    (r"(?:^|[-_])bnb", "BNB"),
]

_QUANT_BIT_PATTERNS: list[str] = [
    r"(?:^|[-_])([2-8])[-_]?bit",
    r"(?:^|[-_])q([2-8])(?![a-z0-9])",
    r"(?:^|[-_])int([2-8])(?![a-z0-9])",
    r"(?:^|[-_])(1[2-6])[-_]?bit",
]

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def guess_architecture(repo_id: str, tags: list[str] | None = None) -> tuple[str, str]:
    """Return (key, human label) for a repo id. Key is lowercase, used for filtering."""
    haystack = f"{repo_id} {' '.join(tags or [])}".lower()
    for needle, label in _ARCHITECTURES:
        if needle in haystack:
            return label.lower().replace(" ", "-").replace(".", ""), label
    return "other", "Other"


def guess_quantization(repo_id: str) -> str | None:
    """Return a display label for the quantisation encoded in the repo id, if any."""
    name = repo_id.split("/")[-1].lower()
    for needle, label in _QUANT_LITERALS:
        if re.search(needle, name):
            return label
    for pattern in _QUANT_BIT_PATTERNS:
        match = re.search(pattern, name)
        if match:
            return f"{match.group(1)}bit"
    return None


def guess_kind(repo_id: str, tags: list[str] | None = None, architecture_label: str = "") -> str:
    """diffusion | lora | upscaler. Kept separate from architecture on purpose."""
    haystack = f"{repo_id} {' '.join(tags or [])}".lower()
    for needle, _ in _UPSCALERS:
        if needle in haystack:
            return "upscaler"
    if any(hint in haystack for hint in _LORA_HINTS):
        return "lora"
    return "diffusion"


def supports_alpha(repo_id: str, architecture_label: str = "") -> bool:
    return architecture_label in _ALPHA_FAMILIES


def safe_install_name(repo_id: str) -> str:
    """Directory name for a downloaded repo, always inside ASSET_DIR."""
    return _SAFE_NAME.sub("-", repo_id.split("/")[-1]).strip("-") or "model"


_QUANT_SUFFIX = re.compile(
    r"(?:[-_.]?(?:m?fp4|nf4|int[2-8]|q[2-8]|[2-8]bit|1[2-6]bit|bf16|fp16|fp32|float16|float32|bnb))+$"
)


def _stem(repo_id: str) -> str:
    """Repo name with any quantisation suffix stripped, for fuzzy installed matching.

    Exact repo-id equality is the only sound test for "this exact artefact is installed",
    and it is what `installed` reports. The stem is used for a weaker, separately-labelled
    signal: `mlx-community/FLUX.2-Klein-4B-4bit` is a *different* artefact from the
    `black-forest-labs/FLUX.2-klein-4B` the registry actually fetches, so calling it
    installed would be a lie -- but "you already have this model, from another repo" is
    genuinely useful, so it gets its own field.
    """
    name = repo_id.split("/")[-1].lower()
    return _QUANT_SUFFIX.sub("", name).strip("-_.")


def _installed_repo_ids() -> tuple[dict[str, str], dict[str, str]]:
    """Return (repo_id -> model_id, stem -> model_id) for what the app already has.

    Deliberately built from the registry rather than the HF cache: the app only treats a
    model as usable when `GET /api/models` reports installed, and that is driven by
    generator.MODELS. A repo merely sitting in the hub cache is not something the app can
    generate with.
    """
    import generator

    by_repo: dict[str, str] = {}
    by_stem: dict[str, str] = {}
    for model_id, minfo in generator.MODELS.items():
        try:
            repo = generator.model_download_repo(model_id, minfo)
        except Exception:
            continue
        if not repo or str(repo).startswith("local:"):
            continue
        repo = str(repo)
        by_repo[repo] = model_id
        by_stem.setdefault(_stem(repo), model_id)
    return by_repo, by_stem


def _matches(item: dict, architecture: str | None, quantization: str | None, kind: str | None) -> bool:
    if architecture and item.get("architecture") != architecture:
        return False
    if quantization and (item.get("quantization") or "").lower() != quantization.lower():
        return False
    if kind and item.get("kind") != kind:
        return False
    return True


_VALID_SORTS = {"downloads", "likes", "lastModified"}


def _list_models(api, *, author, search, limit, sort):
    """Call HfApi.list_models with only the kwargs this hub version actually accepts.

    The keyword set has churned across releases -- huggingface_hub 1.28.0 has no
    `direction` at all, and the repo venv and the packaged runtime venv can be different
    versions. Passing a stale kwarg raises TypeError, which would otherwise surface as a
    blank browser with a 502 and no clue why.
    """
    import inspect

    supported = set(inspect.signature(api.list_models).parameters)
    wanted = {
        "author": author or None,
        "search": search,
        "limit": limit,
        "sort": sort if sort in _VALID_SORTS else "downloads",
        "direction": -1,          # only on hub versions that still have it
        "full": False,
        "cardData": False,
    }
    kwargs = {k: v for k, v in wanted.items() if k in supported and v is not None}
    return api.list_models(**kwargs)


def _looks_like_image_model(entry, tags: list[str], repo_id: str = "") -> bool | None:
    """True/False when we can tell, None when the repo carries no usable signal.

    `pipeline_tag` is often empty on community repos, so tags are the fallback -- a repo
    tagged text-to-speech or automatic-speech-recognition is not a generator no matter what
    its name suggests. Ming-omni-tts is the case that matters: it shares the "Ming"
    fragment with Ming-Image and was being offered as a downloadable image model.
    """
    pipeline = getattr(entry, "pipeline_tag", None)
    if pipeline:
        return pipeline in _IMAGE_PIPELINES
    haystack = {t.lower() for t in tags}
    if haystack & {"text-to-speech", "automatic-speech-recognition", "text-to-audio", "voice-cloning"}:
        return False
    if haystack & _IMAGE_PIPELINES:
        return True
    name = repo_id.split("/")[-1].lower()
    if any(hint in name for hint in _NON_IMAGE_NAME_HINTS):
        return False
    return None


def search_models(
    search: str | None = None,
    author: str | None = None,
    architecture: str | None = None,
    quantization: str | None = None,
    kind: str | None = None,
    limit: int = 40,
    sort: str = "downloads",
    models_only: bool = True,
) -> dict[str, Any]:
    """Search one Hugging Face organisation and normalise the results for the UI.

    Filtering is applied client-visible-side after normalisation rather than pushed into
    the HF query, because "4bit" and "FLUX.2" are only expressible in a repo *name*. That
    means over-fetching a little and returning facets computed from what actually matched,
    so the dropdowns can only ever offer options that exist.
    """
    author = (author or "").strip() or DEFAULT_AUTHOR
    # author="" means "every org", which is how the Illustrious/Pony families are reachable
    # at all. DEFAULT_ORGS would just re-hide them.
    author = "" if author in {"*", "all"} else author
    limit = max(1, min(int(limit or 40), MAX_LIMIT))
    query = (search or "").strip() or None

    api = HfApi(token=hf_service.get_hf_token())
    raw: list[Any] = []
    try:
        raw = list(_list_models(api, author=author, search=query, limit=min(limit * 3, 200), sort=sort))
    except Exception as exc:
        raise RuntimeError(f"Hugging Face search failed: {exc}") from exc

    installed, installed_by_stem = _installed_repo_ids()
    items: list[dict[str, Any]] = []
    for entry in raw:
        repo_id = str(getattr(entry, "id", "") or "")
        if not repo_id:
            continue
        tags = [str(t) for t in (getattr(entry, "tags", None) or [])]
        arch_key, arch_label = guess_architecture(repo_id, tags)
        quant = guess_quantization(repo_id)
        kind = guess_kind(repo_id, tags, arch_label)
        image_like = _looks_like_image_model(entry, tags, repo_id)
        if kind == "diffusion" and image_like is False:
            # e.g. Ming-omni-tts shares the "Ming" name fragment but is a voice model.
            kind = "other"
        item = {
            "id": repo_id,
            "repo_id": repo_id,
            "label": repo_id.split("/")[-1],
            "architecture": arch_key,
            "architecture_label": arch_label,
            "quantization": quant,
            "kind": kind,
            "supports_alpha": supports_alpha(repo_id, arch_label),
            "downloads": int(getattr(entry, "downloads", 0) or 0),
            "likes": int(getattr(entry, "likes", 0) or 0),
            "pipeline_tag": getattr(entry, "pipeline_tag", None),
            "is_image_model": image_like,
            "tags": tags[:12],
            "updated_at": str(getattr(entry, "lastModified", "") or ""),
            "install_name": safe_install_name(repo_id),
            "installed": repo_id in installed,
            "installed_as": installed.get(repo_id),
            # Weaker, distinct signal: same model, different repo/quantisation.
            "related_installed_as": None if repo_id in installed else installed_by_stem.get(_stem(repo_id)),
        }
        if kind == "other" and models_only:
            continue
        if _matches(item, architecture, quantization, kind):
            items.append(item)
        if len(items) >= limit:
            break

    # Facets are computed from the full fetched set, not the filtered page, so switching
    # a filter off cannot leave the user with an empty dropdown.
    facet_source = []
    for entry in raw:
        repo_id = str(getattr(entry, "id", "") or "")
        if not repo_id:
            continue
        tags = [str(t) for t in (getattr(entry, "tags", None) or [])]
        _, arch_label = guess_architecture(repo_id, tags)
        facet_source.append(
            {
                "architecture": arch_label.lower().replace(" ", "-").replace(".", ""),
                "architecture_label": arch_label,
                "quantization": guess_quantization(repo_id),
                "kind": guess_kind(repo_id, tags, arch_label),
                "is_image_model": _looks_like_image_model(entry, tags, repo_id),
            }
        )
    if models_only:
        facet_source = [e for e in facet_source if e["is_image_model"] is not False]

    def facet(field: str) -> list[dict[str, Any]]:
        counts: dict[tuple[str, str], int] = {}
        for entry in facet_source:
            value = entry.get(field)
            if not value:
                continue
            label = entry.get(f"{field}_label") if f"{field}_label" in entry else value
            counts[(value, label)] = counts.get((value, label), 0) + 1
        return [
            {"key": key, "label": label, "count": count}
            for (key, label), count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0]))
        ]

    return {
        "query": query or "",
        "author": author,
        "orgs": ORG_CHOICES,
        "fetched": len(raw),
        "items": items,
        "facets": {
            "architecture": facet("architecture"),
            "quantization": facet("quantization"),
            "kind": facet("kind"),
        },
    }