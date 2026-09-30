from contextlib import asynccontextmanager
import hmac
import ipaddress
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

import generator
from state import _discover_local_loras, _init_gallery_index
from routers import jobs, gallery, loras, tokens, uploads, downloads, settings as settings_router, storage as storage_router
import app_version


import threading


# Renamed MLX_DIFFUSION_* -> DIFFUSIONBEAR_* with the app. The legacy names are kept
# as fallbacks so an existing launchd unit or export in a shell still works.
_LOCAL_API_TOKEN = (
    os.environ.get("DIFFUSIONBEAR_API_TOKEN")
    or os.environ.get("MLX_DIFFUSION_API_TOKEN")
    or os.environ.get("MLX_API_TOKEN")
    or os.environ.get("LOCAL_API_TOKEN", "")
)
_DEV_FALLBACK = any(
    os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
    for name in (
        "DIFFUSIONBEAR_API_DEV_FALLBACK",
        "DIFFUSIONBEAR_DEV_FALLBACK",
        "MLX_DIFFUSION_API_DEV_FALLBACK",
        "MLX_DIFFUSION_DEV_FALLBACK",
        "MLX_ALLOW_DEV_NO_TOKEN",
    )
)
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# The two Vite dev ports, plus the app's own port. The standalone build serves the
# SPA from this same origin, so its fetch() calls are same-origin and need no CORS
# grant at all -- these entries only matter if a browser context ever presents that
# Origin explicitly. Both are loopback-only, which LocalHostMiddleware already enforces.
_ALLOWED_ORIGINS = {
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
}
# Built SPA, served by the backend in production so the standalone app is one
# process and one port instead of uvicorn + a Vite dev server. Absent in dev, where
# Vite serves the SPA on :5174 and proxies /api here -- the routes below simply
# do not register, so dev behaviour is unchanged.
_SPA_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
# Paths that must keep returning their own response instead of index.html, so an
# unknown API route stays a JSON 404 rather than silently becoming the SPA shell.
_SPA_RESERVED = ("api", "docs", "redoc", "openapi.json")
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


app = FastAPI(title="DiffusionBear", version=app_version.APP_VERSION, lifespan=lifespan)

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
app.include_router(storage_router.router)


# GET and HEAD both, deliberately: this is the launcher's readiness probe target, and
# FastAPI registers only GET for a plain @app.get, so a HEAD request would 405 and
# the app would time out even though the backend was serving normally.
@app.api_route("/api/version", methods=["GET", "HEAD"])
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


if _SPA_DIR.is_dir():
    class _ImmutableStatic(StaticFiles):
        """Cache hashed asset filenames forever; they change name when content changes."""

        def file_response(self, *args, **kwargs):
            response = super().file_response(*args, **kwargs)
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            return response

    app.mount(
        "/assets",
        _ImmutableStatic(directory=_SPA_DIR / "assets" if (_SPA_DIR / "assets").is_dir() else _SPA_DIR),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        """Serve the built SPA, falling back to index.html for client-side routes."""
        head = full_path.split("/", 1)[0]
        if head in _SPA_RESERVED:
            raise HTTPException(status_code=404, detail="not found")
        if full_path:
            candidate = (_SPA_DIR / full_path).resolve()
            # resolve() collapses ../ so a crafted path cannot escape dist/
            if candidate.is_file() and candidate.is_relative_to(_SPA_DIR.resolve()):
                return FileResponse(candidate)
        # The shell must never be cached. Without an explicit header WKWebView
        # heuristically caches it, so after a rebuild the window kept rendering the
        # PREVIOUS build's JavaScript against the new API -- which showed as "Load
        # failed" and an empty gallery while every endpoint was returning 200.
        response = FileResponse(_SPA_DIR / "index.html")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        return response

else:  # no build present -- dev mode, Vite owns the UI

    @app.get("/")
    def dev_hint():
        return JSONResponse(
            {
                "detail": "No production build found. Run `npm run build` in frontend/, "
                "or use the Vite dev server on http://localhost:5174.",
                "spa_dir": str(_SPA_DIR),
            },
            status_code=503,
        )
