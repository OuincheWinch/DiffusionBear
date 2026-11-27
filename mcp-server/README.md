# diffusionbear-mcp — MCP server for DiffusionBear

An **MCP (Model Context Protocol) server** that exposes DiffusionBear's REST API
(`http://127.0.0.1:8001` by default, or a LAN/Tailscale backend) as MCP **tools**, so agents —
Claude Desktop, Cursor, Cline, Claude Code, or any MCP client — can generate
images **locally on an Apple Silicon Mac** with zero cloud keys and zero
per-image billing.

This package implements the MCP protocol with the official Python SDK
(`mcp.server.fastmcp.FastMCP`, stdio transport) and is a thin, typed wrapper
over the REST API documented at
[diffusionbear.com/agentic-api.html](https://www.diffusionbear.com/agentic-api.html)
(raw spec: [docs/agentic-api-reference.md](../docs/agentic-api-reference.md)).
Ready-made client configs and prompts: [docs/examples/mcp-clients.md](../docs/examples/mcp-clients.md).

## Install

```bash
# from a clone of this repo
pip install ./mcp-server

# or directly from GitHub with uv
uvx --from git+https://github.com/OuincheWinch/DiffusionBear.git#subdirectory=mcp-server diffusionbear-mcp
```

Requires Python ≥ 3.10. Dependencies: `mcp>=1.8,<2` and `httpx`.

## Configure

| Env var | Default | Meaning |
|---|---|---|
| `DIFFUSIONBEAR_API_URL` | `http://127.0.0.1:8001` | Backend URL: loopback, LAN (`http://192.168.1.20:8001`) or Tailscale Serve (`https://mac.tailnet.ts.net`) |
| `DIFFUSIONBEAR_API_KEY` | — | Bearer `dbk_...` key; required by every `/api/*` route except `/api/version` |

Prerequisites: DiffusionBear v0.5.0+ running (or `backend` served on 8001), API
mode ON and a key created in **Parameters → API** (shown once).

## Client config

Claude Code:

```bash
claude mcp add diffusionbear -e DIFFUSIONBEAR_API_KEY=dbk_... -- diffusionbear-mcp
```

Claude Desktop / Cursor / Cline (use the absolute path from `which diffusionbear-mcp`
if the client does not inherit your shell `PATH`):

```json
{
  "mcpServers": {
    "diffusionbear-local": {
      "command": "diffusionbear-mcp",
      "env": {
        "DIFFUSIONBEAR_API_URL": "http://127.0.0.1:8001",
        "DIFFUSIONBEAR_API_KEY": "dbk_..."
      }
    }
  }
}
```

## Tools

| Tool | Maps to | Purpose |
|---|---|---|
| `server_info()` | `GET /api/version` | Health check (public) |
| `list_models(installed_only=False)` | `GET /api/models` | Runnable engines, defaults, capabilities |
| `list_loras()` | `GET /api/loras` | LoRA registry (`base_model` field = family) |
| `generate_image(prompt, model?, steps?, width?, height?, …)` | `POST /api/generate` | Queue a job → `{"job_id"}` (429 if queue full) |
| `job_status(job_id)` | `GET /api/jobs/{id}` | queued / generating / done / error / cancelled |
| `wait_for_image(job_id, timeout_seconds=900)` | polls `GET /api/jobs/{id}` | Block until done, return results (image ids + files) |
| `list_jobs()` | `GET /api/jobs` | Recent jobs |
| `cancel_job(job_id)` | `POST /api/jobs/{id}/cancel` | Cancel one job |
| `image_metadata(image_id)` | `GET /api/images/{id}` | JSON sidecar (prompt, seed, Civitai tEXt) |
| `save_image(image_id, dest_path, with_metadata=False)` | `GET /api/images/{id}/file` (+ `GET /api/images/{id}`) | Write original PNG/JPEG bytes to disk, optionally the JSON sidecar next to it |
| `create_image(prompt, dest_path, model?, …)` | generate + poll + save | One call: queue, wait, write image(s) and sidecar(s); returns paths and seed |

Shortest agent flow: `list_models(installed_only=True)` →
`create_image(prompt=…, dest_path="~/Desktop/bear.png", model=…)`.

Step-by-step flow (progress, cancellation, batches): `generate_image(prompt=…, model=…)` →
`job_status(job_id)` / `wait_for_image(job_id)` → `save_image(results[0].id, "out.png", with_metadata=True)`.

Example `create_image` result:

```json
{
  "job_id": "790c480080da4b5aa7046bbb6f33cc97",
  "images": [
    {
      "image_id": "7af0522216164857b535d3ab8bd6998a",
      "path": "/Users/me/Desktop/bear.png",
      "bytes": 171173,
      "metadata_path": "/Users/me/Desktop/bear.json",
      "seed": 7
    }
  ]
}
```

Naming rule: send `model` (runnable id from `list_models`) to `generate_image`.
`base_model` is a LoRA-registry field and is rejected (422) by `/api/generate`.

## Security note

The API is a local developer tool, not a hosted service. The app binds
`127.0.0.1` unless API mode is ON with a strong key (16+ chars); wildcard binds
fall back to loopback otherwise. Any valid key has full rights (generation,
gallery, settings, key management), so give each agent its own key and revoke it
when done. Remote access goes through the LAN or Tailscale with the key — never
publish the port on the Internet.

## Tests

```bash
# client unit tests (httpx MockTransport, no server needed)
cd mcp-server && python -m unittest discover -s tests -v

# end-to-end over stdio against the stub API (no GPU needed)…
python mcp-server/tests/stub_api.py &      # 127.0.0.1:8123, key dbk_stub
DIFFUSIONBEAR_API_URL=http://127.0.0.1:8123 DIFFUSIONBEAR_API_KEY=dbk_stub python mcp-server/e2e_mcp_test.py

# …or against a real backend with an installed model (13 checks, real images)
DIFFUSIONBEAR_API_KEY=dbk_... python mcp-server/e2e_mcp_test.py
```

License: MIT (same as the app).
