# Drop-in instructions for coding agents

Paste this block into `AGENTS.md`, `CLAUDE.md`, a Cursor rule or a system prompt
so an agent can use a DiffusionBear backend through plain HTTP.

```markdown
## Image generation: DiffusionBear (local, Apple Silicon)

- Base URL: $DIFFUSIONBEAR_URL (default http://127.0.0.1:8001). Key: $DIFFUSIONBEAR_API_KEY.
- Send `Authorization: Bearer $DIFFUSIONBEAR_API_KEY` on every /api/* call except GET /api/version.
  Never print the key, never put it in a URL for JSON calls, never commit it.
- Workflow:
  1. GET /api/version (health).
  2. GET /api/models -> keep entries with "installed": true; use their "id" as `model`.
  3. POST /api/generate {"model", "prompt", "width", "height", "steps"?, "seed"?} -> {"job_id"}.
     Unknown fields are rejected with 422. Never send `base_model`.
  4. Poll GET /api/jobs/{job_id} every 2 s until status is done | error | cancelled.
     There is no SSE/WebSocket. Model loading can take longer than inference.
  5. On done, for each results[i].id: GET /api/images/{id}/file (bytes) and
     GET /api/images/{id} (JSON sidecar). Keep both side by side.
- One generation runs at a time: queue sequentially or use "batch"; don't fan out requests.
- 401 = key missing/wrong/revoked. 429 = wait Retry-After seconds. Job "error" = read the
  message (model not installed, size too large…) and don't retry the same request blindly.
- To stop: POST /api/jobs/{job_id}/cancel.
- Ready-made clients: docs/examples/diffusionbear_client.py (Python stdlib),
  docs/examples/generate.mjs (Node 18+), docs/examples/quickstart.sh (curl).
```

An MCP-capable agent can use the [`diffusionbear-mcp`](mcp-clients.md) tools
instead; `create_image` covers steps 2–5 in a single call.
