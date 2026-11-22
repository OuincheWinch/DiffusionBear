#!/usr/bin/env python3
"""Minimal DiffusionBear REST client — standard library only (Python 3.9+).

Generate one image, follow progress, save the PNG *and* its JSON sidecar:

    export DIFFUSIONBEAR_URL=http://127.0.0.1:8001     # or http://<mac-ip>:8001, https://mac.tailnet.ts.net
    export DIFFUSIONBEAR_API_KEY=dbk_...                # needed when API mode is ON
    python3 docs/examples/diffusionbear_client.py "a cozy cabin in a snowy forest, golden hour" \
        --model flux2-klein-4b --steps 4 --size 768x1024 --seed 42 --out cabin.png

Ctrl-C cancels the running job on the server. Import it from your own code:

    from diffusionbear_client import DiffusionBear
    db = DiffusionBear()
    png, meta = db.create_image("a red apple", out="apple.png")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TERMINAL = {"done", "error", "cancelled"}


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str, retry_after: float | None = None):
        super().__init__(f"HTTP {status}: {detail}")
        self.status, self.detail, self.retry_after = status, detail, retry_after


class DiffusionBear:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 30):
        self.base_url = (base_url or os.environ.get("DIFFUSIONBEAR_URL", "http://127.0.0.1:8001")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("DIFFUSIONBEAR_API_KEY", "")
        self.timeout = timeout

    # -- transport -------------------------------------------------------------
    def _request(self, method: str, path: str, body: dict | None = None, raw: bool = False):
        headers = {"Accept": "application/json"}
        if self.api_key:  # every /api/* route except /api/version when API mode is ON
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except (ValueError, AttributeError):
                pass
            retry = exc.headers.get("Retry-After")
            raise ApiError(exc.code, str(detail), float(retry) if retry else None) from None
        if raw:
            return payload
        return json.loads(payload) if payload else None

    # -- API -------------------------------------------------------------------
    def version(self) -> dict:
        return self._request("GET", "/api/version")

    def installed_models(self) -> list[dict]:
        return [m for m in self._request("GET", "/api/models") if m.get("installed")]

    def submit(self, **fields) -> str:
        """POST /api/generate. Retries once on 429, honouring Retry-After."""
        for attempt in range(2):
            try:
                return self._request("POST", "/api/generate", fields)["job_id"]
            except ApiError as exc:
                if exc.status != 429 or attempt:
                    raise
                time.sleep(exc.retry_after or 5)
        raise AssertionError("unreachable")

    def job(self, job_id: str) -> dict:
        return self._request("GET", f"/api/jobs/{job_id}")

    def cancel(self, job_id: str) -> dict:
        return self._request("POST", f"/api/jobs/{job_id}/cancel")

    def wait(self, job_id: str, poll: float = 2.0, timeout: float = 900, on_progress=None) -> dict:
        """Poll until done/error/cancelled (the API has no SSE/WebSocket)."""
        deadline = time.monotonic() + timeout
        while True:
            job = self.job(job_id)
            if on_progress:
                on_progress(job)
            if job["status"] in TERMINAL:
                if job["status"] != "done":
                    raise RuntimeError(f"job {job['status']}: {job.get('error')}")
                return job
            if time.monotonic() > deadline:
                raise TimeoutError(f"job {job_id} still {job['status']} after {timeout}s")
            time.sleep(poll)

    def save(self, image_id: str, out: str | Path) -> tuple[Path, Path]:
        """Write the original image bytes and its JSON sidecar next to each other."""
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self._request("GET", f"/api/images/{image_id}/file", raw=True))
        meta = self._request("GET", f"/api/images/{image_id}")
        sidecar = out.with_suffix(".json")
        sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return out, sidecar

    def create_image(self, prompt: str, out: str | Path = "out.png", on_progress=None, **fields):
        """Submit, wait, save. Ctrl-C cancels the job server-side."""
        job_id = self.submit(prompt=prompt, **fields)
        try:
            job = self.wait(job_id, on_progress=on_progress)
        except KeyboardInterrupt:
            self.cancel(job_id)
            raise
        return self.save(job["results"][0]["id"], out)


def _print_progress(job: dict) -> None:
    p = job.get("progress") or {}
    if job["status"] == "generating" and p.get("step"):
        eta = p.get("eta_seconds")
        print(f"\r  step {p['step']}/{p.get('steps', '?')}  eta {eta if eta is not None else '?'}s   ",
              end="", file=sys.stderr, flush=True)
    elif job["status"] in TERMINAL:
        print(f"\r  {job['status']}{' ' * 30}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompt")
    ap.add_argument("--model", help="runnable id from GET /api/models (default: first installed)")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--size", help="WIDTHxHEIGHT, e.g. 768x1024")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default="out.png")
    args = ap.parse_args()

    db = DiffusionBear()
    print(f"DiffusionBear {db.version()['version']} at {db.base_url}", file=sys.stderr)
    model = args.model or next((m["id"] for m in db.installed_models()), None)
    if not model:
        sys.exit("No installed model: download one in the Models tab first.")
    fields = {"model": model}
    if args.steps:
        fields["steps"] = args.steps
    if args.size:
        fields["width"], fields["height"] = (int(v) for v in args.size.lower().split("x"))
    if args.seed is not None:
        fields["seed"] = args.seed
    try:
        png, sidecar = db.create_image(args.prompt, out=args.out, on_progress=_print_progress, **fields)
    except ApiError as exc:
        hint = {401: " (missing/wrong key: set DIFFUSIONBEAR_API_KEY)",
                422: " (unknown or invalid field — e.g. `base_model`)"}.get(exc.status, "")
        sys.exit(f"API error {exc}{hint}")
    except KeyboardInterrupt:
        sys.exit("cancelled")
    meta = json.loads(sidecar.read_text())
    print(f"saved {png} + {sidecar.name}  seed={meta.get('seed')}  {meta.get('generation_time')}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
