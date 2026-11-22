"""API key management routes."""

from __future__ import annotations

import os
import socket

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

import app_settings
import security_keys

router = APIRouter(tags=["security"])


class ApiPortRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    port: int | None = Field(default=None, ge=1024, le=65535)


class ApiModeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool


class CreateKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    label: str = Field(min_length=1, max_length=security_keys.MAX_LABEL_LENGTH)

    @field_validator("label")
    @classmethod
    def _trim_label(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("label is required")
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("label must not contain control characters")
        return value


def _active_port(request: Request) -> int | None:
    try:
        return request.url.port
    except Exception:
        return None


def _port_status(request: Request) -> dict:
    configured = int(app_settings.get_setting("api_port") or 8001)
    active = _active_port(request)
    return {
        "active": active,
        "configured": configured,
        "next_base_url": f"http://127.0.0.1:{configured}",
        "restart_required": active is not None and int(active) != configured,
        "api_mode": security_keys.api_mode_enabled(),
        "bind_host": os.environ.get("DIFFUSIONBEAR_BIND_HOST", "127.0.0.1"),
    }


@router.get("/api/system/port")
def get_api_port(request: Request):
    return _port_status(request)


@router.put("/api/system/port")
def set_api_port(req: ApiPortRequest, request: Request):
    if not security_keys.api_mode_enabled():
        raise HTTPException(409, "enable API mode before choosing a port")
    port = int(req.port or 8001)
    active = _active_port(request)
    if active is not None and port != int(active):
        host = os.environ.get("DIFFUSIONBEAR_BIND_HOST", "127.0.0.1")
        probe_host = "0.0.0.0" if host in ("0.0.0.0", "::") else host.strip("[]")
        family = socket.AF_INET6 if ":" in probe_host else socket.AF_INET
        try:
            probe = socket.socket(family, socket.SOCK_STREAM)
            try:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((probe_host, port))
            finally:
                probe.close()
        except OSError as exc:
            raise HTTPException(409, f"port {port} is not available") from exc
    try:
        app_settings.update_settings({"api_port": port})
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return _port_status(request)


@router.put("/api/security/mode")
def update_api_mode(req: ApiModeRequest):
    try:
        return security_keys.set_api_mode(req.enabled)
    except security_keys.NoUsableKeyError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/api/security/keys")
def list_api_keys(request: Request):
    ring = security_keys.load_key_ring()
    current = getattr(request.state, "api_key_identity", None)
    return security_keys.redacted_status(ring, current)


@router.post("/api/security/keys", status_code=201)
def create_api_key(req: CreateKeyRequest):
    try:
        full_key, record = security_keys.create_key(req.label)
    except security_keys.DuplicateLabelError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"key": full_key, "record": record.public_dict()}


@router.delete("/api/security/keys/{key_id}", status_code=204)
def revoke_api_key(key_id: str):
    try:
        security_keys.revoke_key(key_id)
    except security_keys.KeyNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except security_keys.LastKeyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return Response(status_code=204)
