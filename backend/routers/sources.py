"""HTTP surface for the unified sources.

One browse route over three sources, plus the Draw Things import actions. The
existing `/api/hf/*` and `/api/civitai/*` routes stay exactly as they are -- this
is additive, and Task 7 moves bodies behind `Source` without changing any path.

Three rules the endpoints below enforce, each because the alternative is a wrong
result that loads silently:

- **A Draw Things `filename` is a basename, never a path.** The folder is
  read-only and the output lands in the asset directory, so a traversal in
  either would be a real write outside where it belongs.
- **A conversion never overwrites an existing output.** It refuses with 409.
- **`skipped_tensors` is always in the response.** FLUX norm2 modulation has no
  module in the target model; those pairs are skipped and counted, and the count
  travels back so the UI can show it rather than swallow it.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import generator
import lora_convert
import sources
from sources import diffusionbee, drawthings

router = APIRouter(tags=["sources"])

# The Draw Things folder, resolved through `drawthings.model_dir()` rather than a
# baked constant so tests can point it elsewhere. `Path.home()` is monkeypatched
# process-wide by an existing test (`test_model_download_path` replaces
# `hf_service.Path.home` with a temp dir and never restores it), so anything that
# reaches for the real user folder mid-suite finds a temp dir instead. Routing every
# read through one overridable function keeps that leak out of these endpoints.
def _dt_dir() -> Path:
    return drawthings.model_dir()


def _db_dirs() -> list[Path]:
    return diffusionbee.model_dirs()


class ConvertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str = Field(min_length=1, max_length=255)
    base_model: str = Field(pattern="^(sdxl|flux)$")


class CivitaiMatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    query: str | None = Field(default=None, max_length=200)
    version_url: str | None = Field(default=None, max_length=500)


def _resolve_ckpt(filename: str):
    """Resolve a bare filename to a `.ckpt` inside the Draw Things folder.

    Rejects anything that is not a plain basename, then proves containment by
    resolving -- belt and braces, because a basename check alone still has to
    survive a symlink.
    """
    if not filename or "/" in filename or "\\" in filename or "\x00" in filename:
        raise HTTPException(400, "filename must be a plain name inside the Draw Things folder")
    if filename != Path(filename).name:
        raise HTTPException(400, "filename must be a plain name inside the Draw Things folder")
    root = _dt_dir().resolve()
    candidate = (root / filename).resolve()
    if root not in candidate.parents:
        raise HTTPException(400, "filename must be a plain name inside the Draw Things folder")
    if not candidate.is_file():
        raise HTTPException(404, f"{filename} is not in the Draw Things folder")
    return candidate


def _resolve_tdict(filename: str) -> Path:
    """Resolve a bare filename to a .tdict inside the DiffusionBee folders.

    Rejects anything that is not a plain basename, then proves containment by
    resolving.
    """
    if not filename or "/" in filename or "\\" in filename or "\x00" in filename:
        raise HTTPException(
            400, "filename must be a plain name inside the DiffusionBee folders"
        )
    if filename != Path(filename).name:
        raise HTTPException(
            400, "filename must be a plain name inside the DiffusionBee folders"
        )

    for root in _db_dirs():
        resolved_root = root.resolve()
        candidate = (resolved_root / filename).resolve()
        if resolved_root not in candidate.parents:
            raise HTTPException(
                400, "filename must be a plain name inside the DiffusionBee folders"
            )
        if candidate.is_file():
            return candidate

    raise HTTPException(404, f"{filename} is not in the DiffusionBee folders")


@router.get("/api/sources/{name}/items")
def list_source_items(name: str, limit: int = 40, q: str | None = None):
    """Search one source through the shared `Item` shape.

    Never fetches a licence per item: a cached `license` on the item is
    serialised, a missing one stays null. A slow or unreachable source must not
    be able to stall the models page.
    """
    source_cls = sources.SOURCES.get(name)
    if source_cls is None:
        raise HTTPException(404, f"unknown source {name!r}")
    try:
        items = source_cls().search(q, limit=max(1, min(limit, 100)))
    except Exception as exc:  # a dead source degrades to empty, never to a 500
        return {"source": name, "items": [], "error": str(exc)[:200]}
    return {"source": name, "items": [item.to_dict() for item in items]}


@router.get("/api/sources/drawthings/files")
def list_drawthings_files(kind: str | None = None):
    """List the user's Draw Things `.ckpt` files, read-only."""
    files = drawthings.list_ckpt_files(_dt_dir())
    if kind:
        files = [f for f in files if f.kind == kind]
    return {"items": [f.to_dict() for f in files]}


@router.post("/api/sources/drawthings/convert")
def convert_drawthings(req: ConvertRequest):
    """Convert one Draw Things LoRA to safetensors and report what happened."""
    source = _resolve_ckpt(req.filename)
    if source.suffix.lower() != ".ckpt":
        raise HTTPException(400, "not a Draw Things .ckpt file")

    detected = drawthings.detect_base_model(source)
    if detected and detected != req.base_model:
        raise HTTPException(
            400,
            f"{req.filename} is a {detected} LoRA but {req.base_model} was requested")

    out_dir = generator.ASSET_DIR / "lora_files"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (source.stem + ".safetensors")
    if out_path.exists():
        raise HTTPException(409, f"{out_path.name} already exists; refusing to overwrite")

    try:
        summary = lora_convert.convert(
            str(source), out_path, req.base_model, allow_root=generator.ASSET_DIR)
    except lora_convert.ConversionError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"conversion failed: {type(exc).__name__}") from exc

    # `convert` already threads the report through; re-read it so the skipped
    # count is guaranteed present even if `build_pairs` gained a new skip path.
    summary["output_path"] = str(out_path)
    summary["base_model"] = req.base_model
    summary["skipped_tensors"] = int(summary.get("skipped_tensors", 0))
    return summary


@router.get("/api/sources/diffusionbee/files")
def list_diffusionbee_files(kind: str | None = None):
    """List the user's DiffusionBee .tdict files, strictly read-only."""
    files = diffusionbee.list_tdict_files(_db_dirs())
    if kind:
        files = [f for f in files if f.kind == kind]
    return {"items": [f.to_dict() for f in files]}


@router.post("/api/sources/diffusionbee/convert")
def convert_diffusionbee(req: ConvertRequest):
    """Convert one DiffusionBee LoRA to safetensors and report what happened."""
    source = _resolve_tdict(req.filename)
    if source.suffix.lower() != ".tdict":
        raise HTTPException(400, "not a DiffusionBee .tdict file")

    keys = diffusionbee.read_tdict_keys(source)
    kind = diffusionbee.detect_kind_keys(keys)
    if kind != "lora":
        raise HTTPException(
            400,
            f"{req.filename} is a {kind or 'unknown'} (not a LoRA); full models cannot be converted in v1",
        )

    detected = diffusionbee.detect_base_model_keys(keys)
    if detected and detected != req.base_model:
        raise HTTPException(
            400,
            f"{req.filename} is a {detected} LoRA but {req.base_model} was requested",
        )

    out_dir = generator.ASSET_DIR / "lora_files"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (source.stem + ".safetensors")
    if out_path.exists():
        raise HTTPException(409, f"{out_path.name} already exists; refusing to overwrite")

    try:
        summary = diffusionbee.convert_tdict_lora(
            source, out_path, req.base_model, allow_root=generator.ASSET_DIR
        )
    except lora_convert.ConversionError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"conversion failed: {type(exc).__name__}") from exc

    summary["output_path"] = str(out_path)
    summary["base_model"] = req.base_model
    summary["skipped_tensors"] = int(summary.get("skipped_tensors", 0))
    return summary


@router.post("/api/sources/diffusionbee/civitai-match")
def match_civitai_diffusionbee(req: CivitaiMatchRequest):
    """Find Civitai candidates for a converted DiffusionBee LoRA. Attaches nothing."""
    return match_civitai(req)


@router.post("/api/sources/drawthings/civitai-match")
def match_civitai(req: CivitaiMatchRequest):
    """Find Civitai candidates for a converted LoRA. Attaches nothing.

    Returns a candidate list for the user to choose from. Never auto-attaches a
    licence: a wrong match puts the wrong terms on the user's LoRA, which is the
    one error here that must never happen silently.
    """
    import civitai_service

    if req.version_url:
        return {"candidates": _candidates_from_url(req.version_url, civitai_service),
                "attached": False}
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(400, "provide either a query or a version_url")
    return {"candidates": _candidates_from_query(query, civitai_service),
            "attached": False}


def _candidate(model_id, version, payload) -> dict:
    import licensing
    info = licensing.fetch_civitai_license_from_payload(payload)
    return {
        "model_id": str(model_id),
        "version_id": str(version.get("id")) if version.get("id") else None,
        "version_name": version.get("name"),
        "source_url": info.source_url,
        "license": info.to_dict(),
    }


def _candidates_from_query(query: str, civitai_service) -> list[dict]:
    # civitai_service has no search(); civitai_browse does, and it is the same
    # helper the existing /api/civitai/models route uses. Reuse it rather than
    # reimplementing a second query path.
    import civitai_browse
    try:
        data = civitai_browse.search_models(query, limit=10)
    except Exception:
        return []
    entries = data.get("items", data) if isinstance(data, dict) else data
    out = []
    for entry in entries or []:
        model_id = entry.get("civitai_model_id") or entry.get("id")
        version_id = entry.get("civitai_version_id")
        payload = civitai_service.fetch_by_model_id(
            model_id, civitai_service.get_civitai_api_key()) or {"id": model_id}
        out.append(_candidate(model_id, {"id": version_id,
                                         "name": entry.get("label")}, payload))
    return out


def _candidates_from_url(url: str, civitai_service) -> list[dict]:
    import re
    import licensing
    match = re.search(r"modelVersionId=(\d+)", url)
    if not match:
        return []
    version_id = int(match.group(1))
    payload = licensing._safe(lambda: licensing._get_json(
        f"{licensing.CIVITAI_API_BASE}/model-versions/{version_id}",
        civitai_service.get_civitai_api_key()))
    if not payload:
        return []
    info = licensing.fetch_civitai_license_from_payload(payload)
    model_id = payload.get("modelId")
    if model_id is None:
        return [{"model_id": None, "version_id": str(version_id),
                 "version_name": payload.get("name"),
                 "source_url": info.source_url, "license": info.to_dict()}]
    # Permissions live on the model, not the version.
    model = licensing._safe(lambda: licensing._get_json(
        f"{licensing.CIVITAI_API_BASE}/models/{model_id}",
        civitai_service.get_civitai_api_key())) or {}
    return [_candidate(model_id, {"id": version_id, "name": payload.get("name")}, model)]
