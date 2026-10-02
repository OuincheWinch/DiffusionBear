"""Civitai model discovery, for the Models tab's Civitai source.

Complements hf_browse.py rather than duplicating it: the two sources return the same
normalised shape, so the browser renders them through one code path and one set of filters.

Civitai is structurally the *better* source for two of the fields this UI filters on. It
publishes a structured `baseModel` ("SDXL 1.0", "Pony", "Illustrious XL", "Flux.1"), and each
file carries `metadata.format`/`size`/`fp`. Hugging Face has neither, so hf_browse has to
infer both from the repo name. Where Civitai states a value, use it; fall back to the name
heuristics only when it does not.

Downloading is NOT here. Civitai's endpoint (/api/download/models/{versionId}) answers with a
302 to a pre-signed URL on someone else's storage, and that redirect plus its auth rules are
handled by civitai_service.open_public_https_stream, which already strips the Authorization
header when the redirect leaves civitai.com.
"""

from __future__ import annotations

from typing import Any

import civitai_service
import hf_browse

API_BASE = civitai_service.CIVITAI_API_BASE
MAX_LIMIT = 100

# Civitai's own vocabulary for what a file is. Anything that is not Checkpoint/Model is not
# something this app can run as a Generate engine -- LORA and TextualInversion belong to the
# LoRA registry, which is a different pipeline entirely.
_CHECKPOINT_TYPES = {"Checkpoint", "Model", "TextualInversion", "VAE", "MotionModule", "Controlnet"}

# Civitai sorts by these names in its query string.
_VALID_SORTS = {
    "Most Downloaded": "Most Downloaded",
    "Highest Rated": "Highest Rated",
    "Newest": "Newest",
    "Most Liked": "Most Liked",
    "Most Images": "Most Images",
}

# Civitai baseModel -> the same canonical family labels hf_browse emits, so a filter chosen
# for one source means the same thing in the other. Keys are matched lowercased.
_BASE_MODEL_MAP: list[tuple[tuple[str, ...], str]] = [
    (("illustrious",), "illustrious"),
    (("pony",), "pony"),
    (("sdxl 1.0", "sdxl", "xl"), "sdxl"),
    (("sd 1.5", "sd1.5", "stable diffusion 1.5"), "sd15"),
    (("flux.1", "flux1", "flux dev", "flux schnell"), "flux1"),
    (("flux.2", "flux2"), "flux2"),
    (("krea",), "krea"),
    (("z-image", "zimage"), "z-image"),
    (("qwen-image", "qwenimage"), "qwen-image"),
    (("hunyuan",), "hunyuan"),
    (("auraflow",), "auraflow"),
    (("pixart",), "pixart"),
    (("sd3", "stable diffusion 3"), "sd3"),
]


def _architecture_from(base_model: str | None, name: str) -> tuple[str, str]:
    """Prefer Civitai's structured baseModel; fall back to parsing the name."""
    if base_model:
        haystack = base_model.strip().lower()
        for needles, key in _BASE_MODEL_MAP:
            if any(n in haystack for n in needles):
                return key, base_model.strip()
    return hf_browse.guess_architecture(name, [])


def _quantization_from_file(file_meta: dict) -> str | None:
    """Civitai states precision per file, so no name parsing is needed here."""
    if not isinstance(file_meta, dict):
        return None
    fp = str(file_meta.get("fp") or "").strip().lower()
    fmt = str(file_meta.get("format") or "").strip().lower()
    if fp in {"fp8", "fp16", "bf16"}:
        return fp.upper()
    if fmt == "safetensor" and not fp:
        # A single-file checkpoint has no quantisation; FP16 is what the size implies.
        return None
    return fp.upper() if fp else None


def _pick_file(version: dict) -> dict | None:
    """Prefer the primary SafeTensor, else the largest SafeTensor on the version."""
    files = [f for f in (version.get("files") or []) if isinstance(f, dict)]
    safetensors = [f for f in files if str((f.get("metadata") or {}).get("format", "")).lower() == "safetensor"]
    if not safetensors:
        safetensors = [f for f in files if str(f.get("name", "")).lower().endswith(".safetensors")]
    if not safetensors:
        return None
    for candidate in safetensors:
        if candidate.get("primary"):
            return candidate
    return max(safetensors, key=lambda f: float(f.get("sizeKB") or 0))


def search_models(
    query: str | None = None,
    types: str | None = None,
    sort: str = "Most Downloaded",
    limit: int = 40,
    token: str | None = None,
) -> dict[str, Any]:
    """Search Civitai and normalise to the same shape hf_browse returns."""
    import requests

    limit = max(1, min(int(limit or 40), MAX_LIMIT))
    params: dict[str, Any] = {
        "limit": limit,
        "sort": _VALID_SORTS.get(sort, "Most Downloaded"),
    }
    if query:
        params["query"] = query.strip()
    if types:
        params["types"] = types

    headers = {"Accept": "application/json"}
    auth = civitai_service._valid_token(token) or civitai_service.get_civitai_api_key()
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    try:
        response = requests.get(f"{API_BASE}/models", params=params, headers=headers, timeout=20)
    except Exception as exc:
        raise RuntimeError(f"Civitai search failed: {exc}") from exc
    if response.status_code == 401:
        raise RuntimeError("Civitai rejected the API token (401). Add one in Parameters → Secrets.")
    if response.status_code >= 400:
        raise RuntimeError(f"Civitai search failed: HTTP {response.status_code}")
    payload = response.json()

    items: list[dict[str, Any]] = []
    for entry in payload.get("items") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        versions = entry.get("modelVersions") or []
        version = versions[0] if versions and isinstance(versions[0], dict) else None
        if not version:
            continue
        file = _pick_file(version)
        if not file:
            continue

        base_model = version.get("baseModel")
        arch_key, arch_label = _architecture_from(base_model, name)
        quant = _quantization_from_file(file.get("metadata") or {})
        entry_type = str(entry.get("type") or "")
        kind = "lora" if entry_type == "LORA" else ("upscaler" if entry_type == "Upscaler" else "diffusion")
        usable, usable_label = hf_browse.usable_as(f"civitai/{name}", kind)
        if not usable and entry_type in _CHECKPOINT_TYPES and arch_key in {"sdxl", "sd15"}:
            # Being precise here matters more than being generic. "No engine in this app
            # can run this architecture" reads like SDXL is unsupported, which is false --
            # the app runs four SDXL checkpoints. The actual obstacle is the FILE SHAPE:
            # Civitai serves one .safetensors, and the SDXL engine loads a diffusers
            # directory. Converting is not implemented.
            usable_label = (
                "Single-file checkpoint. The SDXL engine loads a diffusers directory, "
                "so this needs converting before it can be used."
            )

        items.append(
            {
                "id": f"civitai/{entry.get('id')}",
                "source": "civitai",
                "civitai_model_id": entry.get("id"),
                "civitai_version_id": version.get("id"),
                "repo_id": f"civitai/{entry.get('id')}/{version.get('id')}",
                "label": str(file.get("name") or name),
                "architecture": arch_key,
                "architecture_label": arch_label,
                "quantization": quant,
                "kind": kind,
                "civitai_type": entry_type,
                "file_type": str((file.get("metadata") or {}).get("format") or ""),
                "size_bytes": int(float(file.get("sizeKB") or 0) * 1024),
                "supports_alpha": False,
                "downloads": int(entry.get("stats", {}).get("downloadCount") or 0),
                "likes": int(entry.get("stats", {}).get("likeCount") or 0),
                "rating": float(entry.get("stats", {}).get("rating") or 0),
                "tags": [str(t) for t in (entry.get("tags") or [])][:12],
                "updated_at": str(version.get("updatedAt") or entry.get("createdAt") or ""),
                "creator": str((entry.get("creator") or {}).get("username") or ""),
                "nsfw": bool(entry.get("nsfw")),
                "is_lora": entry_type == "LORA",
                "usable_as": usable,
                "usable_as_label": usable_label,
                "installed": False,
                "installed_as": None,
                "related_installed_as": None,
            }
        )

    return {
        "source": "civitai",
        "query": (query or "").strip(),
        "fetched": len(payload.get("items") or []),
        "items": items,
        "facets": {
            "architecture": [],
            "quantization": [],
            "kind": [{"key": k, "label": k, "count": sum(1 for i in items if i["kind"] == k)} for k in ("diffusion", "lora", "upscaler")],
        },
        "authenticated": bool(auth),
    }