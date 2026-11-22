"""MCP server for DiffusionBear (Model Context Protocol, stdio transport).

Tools exposed to agents:
  server_info        -> GET /api/version (public health check)
  list_models        -> GET /api/models (installed engines, defaults, capabilities)
  list_loras         -> GET /api/loras (LoRA registry; `base_model` lives here)
  generate_image     -> POST /api/generate -> {"job_id": ...}
  job_status         -> GET /api/jobs/{job_id}
  wait_for_image     -> poll until done/error/cancelled -> image ids + filenames
  list_jobs          -> GET /api/jobs
  cancel_job         -> POST /api/jobs/{job_id}/cancel
  image_metadata     -> GET /api/images/{image_id}
  save_image         -> GET /api/images/{image_id}/file -> writes bytes (+ optional JSON sidecar)
  create_image       -> one call: generate + wait + save image and sidecar

Configuration (env):
  DIFFUSIONBEAR_API_URL  default http://127.0.0.1:8001. A LAN or Tailscale URL
                         also works when the backend runs in API mode (key required).
  DIFFUSIONBEAR_API_KEY  Bearer key (dbk_...) required by every /api/* route
                         except /api/version

Run:  diffusionbear-mcp            (stdio, for Claude Desktop / Cursor / Cline)
      python -m diffusionbear_mcp.server
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .client import DiffusionBearClient, build_generate_payload

mcp = FastMCP(
    "diffusionbear-local",
    instructions=(
        "Local AI image generation on Apple Silicon via DiffusionBear. "
        "Call list_models(installed_only=True) first and pick a model id. "
        "For one image, create_image(prompt, dest_path, model=...) does everything. "
        "For finer control: generate_image() -> wait_for_image(job_id) -> save_image(image_id, dest_path). "
        "Everything runs on the user's Mac — no cloud, no per-image cost."
    ),
)

_client: DiffusionBearClient | None = None


def client() -> DiffusionBearClient:
    global _client
    if _client is None:
        _client = DiffusionBearClient()
    return _client


@mcp.tool()
def server_info() -> Any:
    """App/backend identity and version (public health check, no key needed)."""
    return client().version()


@mcp.tool()
def list_models(installed_only: bool = False) -> Any:
    """Runnable model engines with defaults and capabilities. Filter to
    installed=True when you want to generate right away."""
    models = client().models()
    if installed_only and isinstance(models, list):
        models = [m for m in models if m.get("installed")]
    return {"models": models}


@mcp.tool()
def list_loras() -> Any:
    """LoRA registry. Entries carry `base_model` (the family they attach to);
    match it against the chosen model's family before using a LoRA."""
    return {"loras": client().loras()}


@mcp.tool()
def generate_image(
    prompt: str,
    model: str | None = None,
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    negative_prompt: str | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    sampler: str | None = None,
    batch: int | None = None,
    output_format: str | None = None,
    loras: list[dict[str, Any]] | None = None,
    reference_images: list[str] | None = None,
    reference_strength: float | None = None,
    stealth: bool | None = None,
) -> Any:
    """Queue one generation job on the local DiffusionBear app.

    Returns {"job_id": "..."} (HTTP 429 if the queue is full). Omitted fields
    fall back to the model's defaults. `model` is the runnable id from
    list_models() — never `base_model`.
    """
    payload = build_generate_payload(
        prompt=prompt,
        model=model,
        steps=steps,
        width=width,
        height=height,
        negative_prompt=negative_prompt,
        guidance=guidance,
        seed=seed,
        sampler=sampler,
        batch=batch,
        output_format=output_format,
        loras=loras,
        reference_images=reference_images,
        reference_strength=reference_strength,
        stealth=stealth,
    )
    return client().generate(payload)


@mcp.tool()
def job_status(job_id: str) -> Any:
    """Poll a job: status is queued | generating | done | error | cancelled.
    On done, results[] carries image ids (`id`) and filenames (`file`)."""
    return client().job(job_id)


@mcp.tool()
def wait_for_image(job_id: str, timeout_seconds: float = 900.0, poll_seconds: float = 2.0) -> Any:
    """Block until the job reaches a terminal state, then return its results.

    Returns the job dict (status done + results with image ids) or raises with
    the job's error message on error/cancelled/timeout.
    """
    return client().wait_for_job(job_id, timeout_seconds, poll_seconds)


@mcp.tool()
def list_jobs() -> Any:
    """Active and recent generation jobs."""
    return {"jobs": client().jobs()}


@mcp.tool()
def cancel_job(job_id: str) -> Any:
    """Cancel a queued or running job."""
    return client().cancel_job(job_id)


@mcp.tool()
def image_metadata(image_id: str) -> Any:
    """JSON sidecar of a generated image (prompt, model, seed, Civitai PNG tEXt...)."""
    return client().image_metadata(image_id)


@mcp.tool()
def save_image(image_id: str, dest_path: str, with_metadata: bool = False) -> Any:
    """Download the original image bytes (PNG/JPEG) to dest_path on this machine.

    with_metadata=True also writes the JSON sidecar (prompt, seed, model...) as
    dest_path with a .json suffix. Parent directories are created.
    """
    return client().save_image(image_id, dest_path, with_metadata=with_metadata)


@mcp.tool()
def create_image(
    prompt: str,
    dest_path: str,
    model: str | None = None,
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    negative_prompt: str | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    batch: int | None = None,
    output_format: str | None = None,
    loras: list[dict[str, Any]] | None = None,
    timeout_seconds: float = 900.0,
) -> Any:
    """One-shot: queue a generation, wait for it and save the image(s) with
    their JSON sidecars. Returns {"job_id", "images": [{"image_id", "path",
    "metadata_path", "seed", "bytes"}]}. Batches write name-2.png, name-3.png...
    """
    payload = build_generate_payload(
        prompt=prompt,
        model=model,
        steps=steps,
        width=width,
        height=height,
        negative_prompt=negative_prompt,
        guidance=guidance,
        seed=seed,
        batch=batch,
        output_format=output_format,
        loras=loras,
    )
    return client().generate_and_save(payload, dest_path, timeout_seconds=timeout_seconds)


def main() -> None:
    # stdio transport is the default for desktop agents (Claude Desktop, Cursor, ...)
    mcp.run()


if __name__ == "__main__":
    main()
