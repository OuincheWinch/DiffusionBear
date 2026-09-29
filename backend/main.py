from contextlib import asynccontextmanager
import hmac
import ipaddress
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

import generator
from state import _discover_local_loras, _init_gallery_index
from routers import jobs, gallery, loras, tokens, uploads, downloads, settings as settings_router
import app_version


import threading


_LOCAL_API_TOKEN = (
    os.environ.get("MLX_DIFFUSION_API_TOKEN")
    or os.environ.get("MLX_API_TOKEN")
    or os.environ.get("LOCAL_API_TOKEN", "")
)
_DEV_FALLBACK = any(
    os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
    for name in ("MLX_DIFFUSION_API_DEV_FALLBACK", "MLX_DIFFUSION_DEV_FALLBACK", "MLX_ALLOW_DEV_NO_TOKEN")
)
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_ALLOWED_ORIGINS = {"http://localhost:5174", "http://127.0.0.1:5174"}
_MAX_API_BODY_BYTES = 8 * 1024 * 1024


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    value = host.strip().lower().split("%", 1)[0]
    if value in ("localhost", "testclient"):
        return True
    try:
        address = ipaddress.ip_address(value)
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        return address.is_loopback
    except ValueError:
        return False


def _requires_api_token(method: str, path: str) -> bool:
    if method not in _SAFE_METHODS:
        return True
    normalized = path.rstrip("/") or "/"
    if normalized == "/api/version" or normalized.startswith("/api/uploads/") or (
        normalized.startswith("/api/images/") and normalized.endswith("/file")
    ):
        return False
    return normalized.startswith("/api/")


def _request_token(request: Request) -> bytes | None:
    header_token = request.headers.get("X-MLX-API-Token") or request.headers.get("X-API-Token")
    if header_token:
        return header_token.encode("utf-8")
    authorization = request.headers.get("Authorization", "")
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() == "bearer" and value:
        return value.encode("utf-8")
    return None


class LocalHostMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.hostname not in {"localhost", "127.0.0.1", "::1", "testserver", "testclient"}:
            return JSONResponse({"detail": "host is not allowed"}, status_code=400)
        return await call_next(request)


class RequestSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") and not (
            path.startswith("/api/uploads") or path.startswith("/api/loras/upload")
        ):
            raw_length = request.headers.get("content-length")
            if raw_length:
                try:
                    if int(raw_length) > _MAX_API_BODY_BYTES:
                        return JSONResponse({"detail": "request body is too large"}, status_code=413)
                except ValueError:
                    return JSONResponse({"detail": "invalid content length"}, status_code=400)
        return await call_next(request)


class LocalAPIMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path == "/api" or path.startswith("/api/"):
            if not _is_loopback(request.client.host if request.client else None):
                return JSONResponse({"detail": "local API access is restricted to loopback clients"}, status_code=403)
            origin = request.headers.get("origin")
            if origin and origin not in _ALLOWED_ORIGINS:
                return JSONResponse({"detail": "request origin is not allowed"}, status_code=403)
            if _requires_api_token(request.method, path) and _LOCAL_API_TOKEN and not _DEV_FALLBACK:
                supplied = _request_token(request)
                if supplied is None or not hmac.compare_digest(supplied, _LOCAL_API_TOKEN.encode("utf-8")):
                    return JSONResponse(
                        {"detail": "local API token required in X-MLX-API-Token, X-API-Token, or Authorization: Bearer"},
                        status_code=401,
                        headers={"WWW-Authenticate": "Bearer"},
                    )
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    generator.cleanup_orphan_artifacts()
    if not generator._cleanup_stale_sdxl_daemon():
        raise RuntimeError("could not verify the previous SDXL daemon")
    if not generator._cleanup_stale_qwen_daemon():
        raise RuntimeError("could not verify the previous Qwen daemon")
    _init_gallery_index()
    threading.Thread(target=_discover_local_loras, daemon=True).start()
    yield
    try:
        generator._kill_sdxl_daemon()
        generator._kill_qwen_process()
        generator._drop_mflux_pipeline()
    except Exception:
        pass


app = FastAPI(title="MLX-Diffusion", version=app_version.APP_VERSION, lifespan=lifespan)

app.add_middleware(LocalHostMiddleware)
app.add_middleware(RequestSizeMiddleware)
app.add_middleware(LocalAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["Content-Type", "Authorization", "X-MLX-API-Token", "X-API-Token"],
)

app.include_router(jobs.router)
app.include_router(gallery.router)
app.include_router(loras.router)
app.include_router(tokens.router)
app.include_router(uploads.router)
app.include_router(downloads.router)
app.include_router(settings_router.router)


@app.get("/api/version")
def api_version():
    return {
        "name": app_version.APP_NAME,
        "version": app_version.APP_VERSION,
        "version_label": app_version.APP_VERSION_LABEL,
        "author": app_version.APP_AUTHOR,
        "website": app_version.APP_WEBSITE,
        "repo": app_version.APP_REPO,
        "ai_credits": app_version.AI_CREDITS,
    }
