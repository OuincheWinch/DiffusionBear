from contextlib import asynccontextmanager
import hmac
import ipaddress
import os
from urllib.parse import urlsplit

# Hugging Face's Xet storage backend stalls indefinitely on large LFS files: a fresh
# flux2-klein-4b install hung with one 2 GB shard at 575 MB and zero bytes of progress,
# holding its blob lock, with no error and no timeout. The classic HTTP path is slower
# per request but resumable and reliable. This must be set before huggingface_hub is
# imported anywhere, because it reads the variable at import time to choose a backend.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Hugging Face Hub otherwise posts anonymous usage telemetry. This app does not
# send analytics of any kind; set it before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.concurrency import run_in_threadpool

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

import torch_guard
torch_guard.install()

import generator
from state import _discover_local_loras, _init_gallery_index
from routers import jobs, gallery, loras, models, tokens, uploads, downloads, settings as settings_router, storage as storage_router, sources as sources_router, samplers as samplers_router, security as security_router
import app_settings
import app_version
import samplers_store
import security_pro

import threading


_LOCAL_API_TOKEN = (
    os.environ.get("DIFFUSIONBEAR_API_TOKEN")
    or os.environ.get("MLX_API_TOKEN")
    or os.environ.get("LOCAL_API_TOKEN", "")
)
_DEV_FALLBACK = any(
    os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
    for name in (
        "DIFFUSIONBEAR_API_DEV_FALLBACK",
        "DIFFUSIONBEAR_DEV_FALLBACK",
        "MLX_ALLOW_DEV_NO_TOKEN",
    )
)
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# The two Vite dev ports, plus the app's own port. The standalone build serves the
# SPA from this same origin, so its fetch() calls are same-origin and need no CORS
# grant at all -- these entries only matter if a browser context ever presents that
# Origin explicitly. Both are loopback-only, which LocalHostMiddleware already enforces.
_ALLOWED_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
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
    if value in ("localhost", "testserver", "testclient"):
        return True
    try:
        address = ipaddress.ip_address(value)
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        return address.is_loopback
    except ValueError:
        return False


def _lan_browser_enabled() -> bool:
    """LAN browser access requires API mode and a strong configured key."""
    return security_pro.has_api_mode() and security_pro.has_strong_configured_key()


def _configured_allowed_hosts() -> set[str]:
    raw = os.environ.get("DIFFUSIONBEAR_ALLOWED_HOSTS", "")
    return {
        value.strip().lower().strip("[]")
        for value in raw.replace(",", " ").split()
        if value.strip()
    }


def _host_is_allowed(hostname: str | None) -> bool:
    """Allow loopback, or a LAN IP/hostname only when the API key is strong."""
    value = (hostname or "").strip().lower().strip("[]")
    if not value:
        return False
    if _is_loopback(value):
        return True
    if not _lan_browser_enabled():
        return False
    if value in _configured_allowed_hosts():
        return True
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _origin_is_allowed(origin: str, request: Request) -> bool:
    if origin in _ALLOWED_ORIGINS:
        return True
    parsed = urlsplit(origin)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    if not _host_is_allowed(parsed.hostname):
        return False
    return parsed.netloc.lower() == request.headers.get("host", "").strip().lower()


def _requires_api_token(method: str, path: str) -> bool:
    if method == "OPTIONS":
        return False
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
    _PROXY_HEADERS = ("forwarded", "x-forwarded-for", "x-real-ip", "x-forwarded-host")

    async def dispatch(self, request: Request, call_next):
        api_mode = await run_in_threadpool(security_pro.has_api_mode)
        if not api_mode and any(
            name in request.headers for name in self._PROXY_HEADERS
        ):
            return JSONResponse(
                {"detail": "proxy headers are not trusted while API mode is disabled"},
                status_code=403,
            )
        allowed = await run_in_threadpool(_host_is_allowed, request.url.hostname)
        if not allowed:
            return JSONResponse({"detail": "host is not allowed"}, status_code=400)
        client_host = request.client.host if request.client else None
        if not _is_loopback(client_host) and not await run_in_threadpool(_lan_browser_enabled):
            return JSONResponse({"detail": "access is restricted to loopback clients"}, status_code=403)
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
            elif request.method in ("POST", "PUT", "PATCH"):
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > _MAX_API_BODY_BYTES:
                        return JSONResponse({"detail": "request body is too large"}, status_code=413)
                request._body = bytes(body)
                async def receive():
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                request._receive = receive
        return await call_next(request)


class LocalAPIMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path == "/api" or path.startswith("/api/"):
            if not _is_loopback(request.client.host if request.client else None) and not await run_in_threadpool(_lan_browser_enabled):
                return JSONResponse({"detail": "local API access is restricted to loopback clients"}, status_code=403)
            origin = request.headers.get("origin")
            if origin and not _origin_is_allowed(origin, request):
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


def _release_bootstrap_worker():
    try:
        import release_bootstrap

        release_bootstrap.bootstrap_release_assets()
    except Exception as exc:
        print(f"[bootstrap] release asset bootstrap failed: {exc}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    restored = samplers_store.self_heal()
    if restored:
        print(f"[samplers] self-heal: {', '.join(restored)}", flush=True)
    generator.cleanup_orphan_artifacts()
    if not generator._cleanup_stale_sdxl_daemon():
        raise RuntimeError("could not verify the previous SDXL daemon")
    if not generator._cleanup_stale_qwen_daemon():
        raise RuntimeError("could not verify the previous Qwen daemon")
    if not generator._cleanup_stale_moss_daemon():
        raise RuntimeError("could not verify the previous MOSS daemon")
    if not generator._cleanup_stale_fibo_daemon():
        raise RuntimeError("could not verify the previous FIBO daemon")
    from routers.downloads import recover_model_install_staging
    recover_model_install_staging()
    _init_gallery_index()
    threading.Thread(target=_discover_local_loras, daemon=True).start()
    threading.Thread(target=_release_bootstrap_worker, daemon=True).start()
    try:
        import model_integrity
        model_integrity.start_first_run_verification_if_needed()
    except Exception as exc:
        print(f"[integrity] startup verification skipped: {exc}", flush=True)
    parent_pid = os.environ.get("DIFFUSIONBEAR_PARENT_PID", "").strip()
    if parent_pid.isdigit():
        import engine_watchdog
        engine_watchdog.arm_orphan_exit(int(parent_pid))
    yield
    try:
        generator._kill_sdxl_daemon()
        generator._kill_qwen_process()
        generator._kill_moss_process()
        generator._kill_fibo_daemon()
        generator._drop_mflux_pipeline()
    except Exception:
        pass


_DOCS_ENABLED = os.environ.get("DIFFUSIONBEAR_ENABLE_DOCS") == "1"
app = FastAPI(
    title="DiffusionBear",
    version=app_version.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if _DOCS_ENABLED else None,
    redoc_url="/redoc" if _DOCS_ENABLED else None,
    openapi_url="/openapi.json" if _DOCS_ENABLED else None,
)

app.add_middleware(LocalHostMiddleware)
app.add_middleware(RequestSizeMiddleware)
app.add_middleware(LocalAPIMiddleware)
# Rejects a burst before FastAPI parses a generation body or queues a job.
app.add_middleware(security_pro.GenerationRateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-MLX-API-Token", "X-API-Token"],
)

# Every API router is wrapped by the enterprise dependency. /api/version is
# registered directly on the app below, so it stays public by construction.
_api_auth = [Depends(security_pro.require_api_key)]
app.include_router(jobs.router, dependencies=_api_auth)
app.include_router(gallery.router, dependencies=_api_auth)
app.include_router(loras.router, dependencies=_api_auth)
app.include_router(models.router, dependencies=_api_auth)
app.include_router(tokens.router, dependencies=_api_auth)
app.include_router(uploads.router, dependencies=_api_auth)
app.include_router(downloads.router, dependencies=_api_auth)
app.include_router(settings_router.router, dependencies=_api_auth)
app.include_router(storage_router.router, dependencies=_api_auth)
app.include_router(sources_router.router, dependencies=_api_auth)
app.include_router(samplers_router.router, dependencies=_api_auth)
app.include_router(security_router.router, dependencies=_api_auth)


# GET and HEAD both, deliberately: this is the launcher's readiness probe target, and
# FastAPI registers only GET for a plain @app.get, so a HEAD request would 405 and
# the app would time out even though the backend was serving normally.
@app.api_route("/api/version", methods=["GET", "HEAD"])
def api_version(request: Request):
    return {
        "name": app_version.APP_NAME,
        "version": app_version.APP_VERSION,
        "version_label": app_version.APP_VERSION_LABEL,
        "api_mode": security_pro.has_api_mode(),
        "port": request.url.port,
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


def run(host: str | None = None, port: int | None = None, **uvicorn_kwargs) -> None:
    """Start uvicorn with the enterprise bind fail-safe applied.

    ``--host 0.0.0.0`` is honoured only when a strong API key is configured.
    Otherwise the effective host is forced to loopback and a CRITICAL warning
    is logged before uvicorn is started.
    """
    requested = host or os.environ.get("DIFFUSIONBEAR_BIND_HOST", "127.0.0.1")
    effective = security_pro.resolve_bind_host(requested)
    if effective != requested:
        print(
            f"[security] bind host {requested!r} refused; using {effective!r}",
            flush=True,
        )
    if port is not None:
        effective_port = int(port)
    elif os.environ.get("DIFFUSIONBEAR_PORT", "").strip():
        effective_port = int(os.environ["DIFFUSIONBEAR_PORT"])
    elif security_pro.has_api_mode():
        effective_port = int(app_settings.get_setting("api_port") or 8001)
    else:
        effective_port = 8001
    import uvicorn

    uvicorn.run(
        "main:app",
        host=effective,
        port=effective_port,
        proxy_headers=False,
        forwarded_allow_ips="",
        **uvicorn_kwargs,
    )


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="DiffusionBear PRO API server")
    parser.add_argument(
        "--host",
        default=os.environ.get("DIFFUSIONBEAR_BIND_HOST", "127.0.0.1"),
        help="Bind host. 0.0.0.0/:: requires DIFFUSIONBEAR_API_KEY.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DIFFUSIONBEAR_PORT", "8001")),
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")
    args = parser.parse_args()
    run(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    _cli()
