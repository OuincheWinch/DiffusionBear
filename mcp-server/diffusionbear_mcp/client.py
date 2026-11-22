"""HTTP client for the DiffusionBear REST API (loopback, LAN or Tailscale).

Deliberately free of any `mcp` import so it is unit-testable with httpx's
MockTransport. Spec: docs/agentic-api-reference.md (published at
https://www.diffusionbear.com/agentic-api.html).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8001"


class ApiError(RuntimeError):
    def __init__(self, status: int, body: str, route: str):
        self.status = status
        self.body = body
        self.route = route
        super().__init__(f"DiffusionBear API {route} -> HTTP {status}: {body[:500]}")


class DiffusionBearClient:
    """Thin wrapper over the REST API. Auth: Bearer key on every /api/*
    route except the public GET /api/version health check."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("DIFFUSIONBEAR_API_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("DIFFUSIONBEAR_API_KEY", "")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def _headers(self, auth: bool) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if auth:
            if not self.api_key:
                raise ApiError(
                    401,
                    "DIFFUSIONBEAR_API_KEY is not set. Export it (or set it in the app's "
                    "Parameters > API Keys view) — every /api/* route except /api/version "
                    "requires the Bearer key.",
                    "auth",
                )
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _request(self, method: str, route: str, *, auth: bool = True, json_body: Any = None) -> Any:
        resp = self._client.request(method, route, headers=self._headers(auth), json=json_body)
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, resp.text, route)
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    # ---- public health check -------------------------------------------------
    def version(self) -> Any:
        return self._request("GET", "/api/version", auth=False)

    # ---- catalog -------------------------------------------------------------
    def models(self) -> Any:
        return self._request("GET", "/api/models")

    def loras(self) -> Any:
        return self._request("GET", "/api/loras")

    # ---- generation ----------------------------------------------------------
    def generate(self, payload: dict[str, Any]) -> Any:
        """POST /api/generate. Unknown fields are rejected with 422, so callers
        must only include known fields (see build_generate_payload). Never send
        `base_model` here — it is a LoRA-registry field, not a generation field."""
        return self._request("POST", "/api/generate", json_body=payload)

    def job(self, job_id: str) -> Any:
        return self._request("GET", f"/api/jobs/{job_id}")

    def jobs(self) -> Any:
        return self._request("GET", "/api/jobs")

    def cancel_job(self, job_id: str) -> Any:
        return self._request("POST", f"/api/jobs/{job_id}/cancel")

    # ---- outputs -------------------------------------------------------------
    def image_metadata(self, image_id: str) -> Any:
        return self._request("GET", f"/api/images/{image_id}")

    def image_bytes(self, image_id: str) -> bytes:
        resp = self._client.get(f"/api/images/{image_id}/file", headers=self._headers(auth=True))
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, resp.text, f"/api/images/{image_id}/file")
        return resp.content

    # ---- workflows -----------------------------------------------------------
    def wait_for_job(
        self,
        job_id: str,
        timeout_seconds: float = 900.0,
        poll_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Any:
        """Poll GET /api/jobs/{id} until done (the API has no SSE/WebSocket).
        Raises ApiError on error/cancelled and TimeoutError past the deadline."""
        deadline = time.monotonic() + timeout_seconds
        while True:
            job = self.job(job_id)
            status = job.get("status")
            if status == "done":
                return job
            if status in ("error", "cancelled"):
                raise ApiError(0, f"job {job_id} {status}: {job.get('error')}", f"/api/jobs/{job_id}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"job {job_id} still {status} after {timeout_seconds}s")
            sleep(poll_seconds)

    def save_image(self, image_id: str, dest_path: str | Path, with_metadata: bool = False) -> dict[str, Any]:
        """Write the original bytes to dest_path; optionally the JSON sidecar
        next to it (same name, .json)."""
        dest = Path(dest_path).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = self.image_bytes(image_id)
        dest.write_bytes(data)
        out: dict[str, Any] = {"image_id": image_id, "path": str(dest), "bytes": len(data)}
        if with_metadata:
            meta = self.image_metadata(image_id)
            sidecar = dest.with_suffix(".json")
            sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            out["metadata_path"] = str(sidecar)
            out["seed"] = meta.get("seed") if isinstance(meta, dict) else None
        return out

    def generate_and_save(
        self,
        payload: dict[str, Any],
        dest_path: str | Path,
        timeout_seconds: float = 900.0,
        poll_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        """Queue a job, wait for it, then save every image (+ sidecar).
        A batch writes name.png, name-2.png, ..."""
        job_id = self.generate(payload)["job_id"]
        job = self.wait_for_job(job_id, timeout_seconds, poll_seconds, sleep)
        dest = Path(dest_path).expanduser()
        saved = []
        for index, result in enumerate(job.get("results") or [], start=1):
            target = dest if index == 1 else dest.with_name(f"{dest.stem}-{index}{dest.suffix}")
            saved.append(self.save_image(result["id"], target, with_metadata=True))
        return {"job_id": job_id, "images": saved}


# Fields accepted by POST /api/generate (strict schema — unknown fields -> 422).
GENERATE_FIELDS = {
    "model", "prompt", "negative_prompt", "width", "height", "steps", "guidance",
    "seed", "quantization", "loras", "batch", "output_format", "sampler",
    "cache_interval", "reference_images", "reference_strength", "stealth",
    "fast_vae", "max_pixels", "use_teacache", "use_kv_cache", "clip_skip",
}


def build_generate_payload(**fields: Any) -> dict[str, Any]:
    """Keep only known generate fields and drop unset (None) ones so the server
    applies its own defaults. Raises if a caller tries to pass base_model."""
    if "base_model" in fields:
        raise ValueError("`base_model` is not a /api/generate field (422). Use `model`.")
    payload: dict[str, Any] = {}
    for key, value in fields.items():
        if key not in GENERATE_FIELDS:
            raise ValueError(f"unknown /api/generate field: {key!r}")
        if value is not None:
            payload[key] = value
    return payload
