"""Storage manifest endpoints.

Never destructive. There is deliberately no delete verb in this router: the
manifest's job is to make 28 GB of weights legible and to name what is
unaccounted for, and a report that can also destroy things is a report people stop
opening.

The one write verb is adopt, which points a registry model at a directory already
sitting in the store. It copies nothing, moves nothing and deletes nothing; it
records a path, and unregistering the model undoes it.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import app_settings
import generator
import storage
from .downloads import _has_weights

router = APIRouter(tags=["storage"])


@router.get("/api/storage")
def get_storage(force: bool = False):
    """Everything on disk that the app owns, grouped by category.

    Results are cached for a minute because the UI polls this; pass force=true to
    rescan on demand. Nothing is hashed, so a scan of 28 GB takes milliseconds and
    never contends with generation.
    """
    return storage.get_report(force=force)


class AdoptModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=512)
    model_id: str = Field(min_length=1, max_length=120)


@router.post("/api/storage/adopt")
def adopt_model(req: AdoptModelRequest):
    """Point a registry model at a directory already sitting in the model store.

    This is the same model_paths override as the downloader's register route, so an
    adopted model then reports as installed, is counted as referenced instead of
    unrecognised, and appears in the model dropdown alongside everything else. It is
    reversible: unregistering clears the entry and the directory returns to the
    unrecognised list.

    The directory must be inside the model store and contain weights. Nothing is
    copied, moved or deleted -- adoption is a pointer, so it costs no disk.
    """
    model_id = req.model_id.strip()
    minfo = generator.MODELS.get(model_id)
    if minfo is None:
        raise HTTPException(404, "unknown model")

    models_root = (app_settings.ASSET_DIR / "models").resolve()
    candidate = (models_root / req.name).resolve()
    try:
        candidate.relative_to(models_root)
    except ValueError:
        raise HTTPException(400, "path must stay inside the model store")
    if not candidate.is_dir():
        raise HTTPException(404, f"{req.name} is not in the model store")
    if not _has_weights(candidate):
        raise HTTPException(400, f"{candidate.name} contains no weights")

    updated = app_settings.update_settings({"model_paths": {model_id: str(candidate)}})
    return {
        "status": "adopted",
        "model_id": model_id,
        "name": candidate.name,
        "local_path": str(candidate),
        "model_paths": updated.get("model_paths") or {},
    }
