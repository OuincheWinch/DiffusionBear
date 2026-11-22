"""PRO/Enterprise security gateway for DiffusionBear.

The local build trusts loopback. An enterprise build may bind a LAN or public
interface, so three independent controls are required:

* **Authentication** -- a strong API key from ``DIFFUSIONBEAR_API_KEY`` or
  ``<DATA_DIR>/security.json``. Every ``/api/*`` route except the public
  ``/api/version`` health check requires it.
* **Bind fail-safe** -- an open bind (``0.0.0.0``, ``::``, ``*``) is refused
  unless a strong key is configured; the effective host falls back to
  ``127.0.0.1`` instead of exposing an unauthenticated GPU.
* **Generation flood guard** -- a bounded per-client rate window plus an early
  queue-depth check rejects bursts with ``429`` before a request reaches the
  worker queue. This is intentionally in-process: one backend owns one GPU.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import os
import re
import threading
import time
from collections import deque

import security_keys
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("diffusionbear.security")

_MIN_KEY_LENGTH = 16
_MAX_SECURITY_FILE_BYTES = 64 * 1024
_OPEN_BINDS = {
    "0.0.0.0",
    "0.0.0.0/0",
    "::",
    "[::]",
    "::/0",
    "[::]/0",
    "*",
}
_bearer = HTTPBearer(auto_error=False)
# Query credentials are limited to GET media delivery, where <img> cannot send a
# header. They are never accepted on a state-changing or JSON endpoint.
_MEDIA_GET_RE = re.compile(r"^/api/(?:uploads/[^/]+|images/[^/]+/file)$")


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer environment override, falling back safely."""
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _security_file_path() -> Path:
    """Return the enterprise config path inside the app's data directory."""
    import app_settings

    return Path(app_settings.DATA_DIR) / "security.json"


def _read_security_file() -> str | None:
    """Read ``api_key``/``token`` from ``security.json``; failures fail closed."""
    path = _security_file_path()
    try:
        if not path.is_file():
            return None
        if path.stat().st_size > _MAX_SECURITY_FILE_BYTES:
            logger.critical("[security] security.json is too large; ignoring it")
            return None
        if path.stat().st_mode & 0o077:
            logger.warning(
                "[security] security.json is readable by group/others; chmod 600 it"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        value = data.get("api_key") or data.get("token") or ""
        value = str(value).strip()
        return value or None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.critical("[security] cannot read security.json: %s", exc)
        return None


def configured_api_key() -> str | None:
    """Return the configured API key, environment first, then security.json."""
    env_key = os.environ.get("DIFFUSIONBEAR_API_KEY", "").strip()
    if env_key:
        return env_key
    return _read_security_file()


def has_api_mode() -> bool:
    return security_keys.api_mode_enabled()


def has_strong_configured_key() -> bool:
    """True when the environment, legacy or managed ring has a strong key."""
    return security_keys.has_strong_key(security_keys.load_key_ring())


def is_strong_api_key(value: str | None) -> bool:
    """True when a key is long enough, non-blank, and not one repeated character."""
    if not isinstance(value, str):
        return False
    key = value.strip()
    if len(key) < _MIN_KEY_LENGTH:
        return False
    if any(char.isspace() for char in key):
        return False
    return len(set(key)) > 1


def is_open_bind(host: str | None) -> bool:
    """True for wildcard bind addresses that expose every interface.

    Accept every spelling Uvicorn/socket accept: expanded/compressed IPv6,
    bracketed IPv6, a port suffix, and the unspecified address as parsed by
    ``ipaddress``. A literal set missed ``::0`` and
    ``0:0:0:0:0:0:0:0`` and let an open bind bypass the key fail-safe.
    """
    value = (host or "").strip().lower()
    if not value:
        return False
    if value == "*":
        return True
    if value in {"0.0.0.0/0", "::/0"}:
        return True
    if value.startswith("[") and "]" in value:
        value = value[1:value.index("]")]
    elif value.count(":") == 1:
        head, _, port = value.rpartition(":")
        if port.isdigit():
            value = head
    try:
        return ipaddress.ip_address(value).is_unspecified
    except ValueError:
        return False


def resolve_bind_host(requested_host: str | None, api_key: str | None = None) -> str:
    """Return a safe bind host, forcing loopback when an open bind has no key.

    A caller that asks for ``0.0.0.0`` without a strong key gets
    ``127.0.0.1`` and a ``CRITICAL`` log line. The fallback is deliberate: a
    misconfigured enterprise start must not expose an unauthenticated API.
    """
    host = (requested_host or "127.0.0.1").strip() or "127.0.0.1"
    if not is_open_bind(host):
        return host
    if api_key is not None:
        strong = bool(api_key and is_strong_api_key(api_key))
    else:
        strong = security_keys.api_mode_enabled() and has_strong_configured_key()
    if strong:
        logger.warning(
            "[security] API bound to %s with enterprise authentication enabled",
            host,
        )
        return host
    logger.critical(
        "[security] refusing to bind %s without a strong DIFFUSIONBEAR_API_KEY "
        "or security.json api_key (>= %d chars); falling back to 127.0.0.1",
        host,
        _MIN_KEY_LENGTH,
    )
    return "127.0.0.1"


def _is_loopback_host(host: str | None) -> bool:
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


_REMOTE_PROXY_HEADERS = (
    "forwarded",
    "x-forwarded-for",
    "x-real-ip",
    "x-forwarded-host",
    "x-forwarded-proto",
    "cf-connecting-ip",
    "true-client-ip",
    "tailscale-user-login",
    "tailscale-user-name",
    "tailscale-headers-info",
)


def _header_hostname(host_header: str | None) -> str | None:
    if not host_header:
        return None
    raw = host_header.strip()
    if not raw:
        return None
    if raw.startswith("["):
        end = raw.find("]")
        return raw[1:end] if end != -1 else raw
    if ":" in raw:
        return raw.rsplit(":", 1)[0]
    return raw


def is_direct_local_request(request: Request) -> bool:
    """True only for unproxied loopback requests originating on the host Mac."""
    client_host = request.client.host if request.client else None
    if not _is_loopback_host(client_host):
        return False
    if any(name in request.headers for name in _REMOTE_PROXY_HEADERS):
        return False
    req_host = _header_hostname(request.headers.get("host")) or request.url.hostname
    if req_host and not _is_loopback_host(req_host):
        return False
    return True


def require_local_loopback(request: Request, detail: str = "Action is restricted to local sessions on the host Mac.") -> None:
    """Reject remote, LAN, or proxied (e.g. Tailscale) requests with 403."""
    if not is_direct_local_request(request):
        raise HTTPException(status_code=403, detail=detail)


def _supplied_key(
    request: Request, credentials: HTTPAuthorizationCredentials | None
) -> str | None:
    """Extract a key from Bearer auth or one of the accepted API-key headers."""
    if credentials is not None and credentials.scheme.lower() == "bearer":
        return credentials.credentials
    for header in ("X-API-Key", "X-MLX-API-Token", "X-API-Token"):
        value = request.headers.get(header)
        if value:
            return value
    return None


def _media_query_key(request: Request) -> str | None:
    """Return ``?token=``/``?api_key=`` only for GET media endpoints.

    Browser <img> tags cannot attach Authorization. This narrow exception keeps
    the gallery working without a reverse proxy; every other route requires a
    header. Query values can appear in access logs, so callers should prefer the
    header for machine-to-machine traffic and keep media URLs out of referrers.
    """
    if request.method != "GET":
        return None
    path = request.url.path.rstrip("/")
    if not _MEDIA_GET_RE.fullmatch(path):
        return None
    return request.query_params.get("token") or request.query_params.get("api_key")


def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """FastAPI dependency enforcing the enterprise API key.

    When no key is configured, loopback clients keep the local build's
    behaviour and non-loopback clients get ``401``. When a key is configured,
    every caller -- loopback included -- must present it, because a PRO
    deployment may be driven by a local agent as well as a remote one.
    """
    if not security_keys.api_mode_enabled():
        client_host = request.client.host if request.client else None
        if _is_loopback_host(client_host):
            return
        raise HTTPException(
            status_code=401,
            detail="API mode is disabled; non-loopback access is refused",
            headers={"WWW-Authenticate": "Bearer"},
        )
    ring = security_keys.load_key_ring()
    if not (ring.environment_key or ring.legacy_key or ring.managed):
        client_host = request.client.host if request.client else None
        if _is_loopback_host(client_host):
            return
        raise HTTPException(
            status_code=401,
            detail="API key required for non-loopback access",
            headers={"WWW-Authenticate": "Bearer"},
        )
    supplied = _supplied_key(request, credentials)
    if supplied is None:
        supplied = _media_query_key(request)
    identity = security_keys.authenticate(supplied or "", ring)
    if identity is None:
        raise HTTPException(
            status_code=401,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    request.state.api_key_identity = identity


@dataclass(frozen=True)
class RateLimitPolicy:
    """In-process limits for the single-GPU generation endpoint."""

    per_minute: int = 30
    burst: int = 4
    max_queued_jobs: int = 64
    window_seconds: float = 60.0
    burst_window_seconds: float = 1.0


class GenerationRateLimiter:
    """Bounded sliding-window limiter keyed by API key hash or client IP."""

    def __init__(self, policy: RateLimitPolicy | None = None) -> None:
        self.policy = policy or RateLimitPolicy()
        self._events: dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, identity: str, now: float | None = None) -> tuple[bool, int]:
        """Record one request and return ``(allowed, retry_after_seconds)``."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            events = self._events.setdefault(identity, deque())
            cutoff = moment - self.policy.window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.policy.per_minute:
                retry = max(1, int(self.policy.window_seconds - (moment - events[0])))
                return False, retry
            recent_cutoff = moment - self.policy.burst_window_seconds
            recent = sum(1 for value in events if value > recent_cutoff)
            if recent >= self.policy.burst:
                return False, max(1, int(self.policy.burst_window_seconds))
            events.append(moment)
            if len(self._events) > 4096:
                # Bound the identity map itself: a spoofed-source flood must not
                # turn the limiter into a memory leak. Drop the emptiest/oldest
                # buckets first.
                ordered = sorted(
                    self._events.items(),
                    key=lambda item: item[1][-1] if item[1] else 0.0,
                )
                for key, _ in ordered[: len(self._events) - 4096]:
                    self._events.pop(key, None)
            return True, 0


_GENERATE_LIMITER = GenerationRateLimiter(
    RateLimitPolicy(
        per_minute=_env_int("DIFFUSIONBEAR_GENERATE_RATE_PER_MINUTE", 30),
        burst=_env_int("DIFFUSIONBEAR_GENERATE_BURST", 4),
        max_queued_jobs=_env_int("DIFFUSIONBEAR_GENERATE_MAX_QUEUED_JOBS", 64),
    )
)


def _request_identity(request: Request) -> str:
    """Identify a client for rate limiting.

    A valid key gets its own bucket. An invalid or absent key is bucketed by
    source IP, so rotating fake API keys cannot buy an unlimited request rate.
    """
    if not security_keys.api_mode_enabled():
        host = request.client.host if request.client else "unknown"
        return f"ip:{host}"
    key = _supplied_key(request, None)
    identity = security_keys.authenticate(key or "")
    if identity is not None:
        if identity.kind == "environment":
            return "key:env"
        if identity.kind == "legacy":
            return "key:legacy"
        return f"key:{identity.id}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


def _active_generation_jobs() -> int:
    """Count queued and running generation jobs without importing at module load."""
    try:
        import state

        with state._JOBS_LOCK:
            return sum(
                1
                for job in state.JOBS.values()
                if job.get("status") in ("generating", "queued")
            )
    except Exception:
        return 0


def enforce_generation_limits(request: Request) -> None:
    """Raise ``429`` for a burst or a full queue before the body is processed."""
    allowed, retry_after = _GENERATE_LIMITER.check(_request_identity(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="generation rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )
    if _active_generation_jobs() >= _GENERATE_LIMITER.policy.max_queued_jobs:
        raise HTTPException(
            status_code=429,
            detail="generation queue is full",
            headers={"Retry-After": "30"},
        )


class GenerationRateLimitMiddleware(BaseHTTPMiddleware):
    """Reject generation floods before the request body reaches a route."""

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path.rstrip("/") == "/api/generate":
            try:
                enforce_generation_limits(request)
            except HTTPException as exc:
                return JSONResponse(
                    {"detail": exc.detail},
                    status_code=exc.status_code,
                    headers=exc.headers,
                )
        return await call_next(request)
