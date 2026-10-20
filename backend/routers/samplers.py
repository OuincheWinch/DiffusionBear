"""API routes for the custom-sampler store: list, install from URL or local
file, remove."""
from __future__ import annotations

import fastapi
import pydantic

import samplers_store

router = fastapi.APIRouter()


class InstallBody(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    url: str = pydantic.Field(min_length=8, max_length=2048)
    name: str | None = pydantic.Field(default=None, max_length=40)


class InstallLocalBody(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    path: str = pydantic.Field(min_length=1, max_length=2048)
    name: str | None = pydantic.Field(default=None, max_length=40)


@router.get("/api/samplers")
def list_samplers():
    return {"samplers": samplers_store.list_samplers()}


@router.post("/api/samplers/install")
def install_sampler(body: InstallBody):
    try:
        entry = samplers_store.install_from_url(body.url, body.name)
    except PermissionError as exc:
        raise fastapi.HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise fastapi.HTTPException(
            status_code=502, detail=f"download failed: {exc}") from exc
    return entry


@router.post("/api/samplers/install-local")
def install_local_sampler(body: InstallLocalBody):
    try:
        entry = samplers_store.install_from_path(body.path, body.name)
    except PermissionError as exc:
        raise fastapi.HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise fastapi.HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise fastapi.HTTPException(status_code=500, detail=str(exc)) from exc
    return entry


@router.delete("/api/samplers/{name}")
def remove_sampler(name: str):
    try:
        samplers_store.remove(name)
    except PermissionError as exc:
        raise fastapi.HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise fastapi.HTTPException(status_code=404, detail=str(exc)) from exc
    return {"removed": name}


@router.post("/api/samplers/verify")
def verify_samplers():
    return {"restored": samplers_store.self_heal()}
