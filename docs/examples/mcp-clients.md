# Plugging DiffusionBear into an MCP agent

The `diffusionbear-mcp` server (in [`mcp-server/`](../../mcp-server/)) speaks MCP over
stdio. Install it once, then point your agent at it.

```bash
pip install ./mcp-server            # from a clone of this repo
which diffusionbear-mcp             # absolute path, handy for GUI clients
```

Create the key in **Parameters → API → Create key** (shown once), turn API mode
ON, and keep the key out of shared config files when you can.

## Claude Code

```bash
claude mcp add diffusionbear \
  -e DIFFUSIONBEAR_API_URL=http://127.0.0.1:8001 \
  -e DIFFUSIONBEAR_API_KEY=dbk_... \
  -- diffusionbear-mcp
```

## Claude Desktop / Cursor / Cline / Windsurf

`claude_desktop_config.json`, `~/.cursor/mcp.json`, … all use the same shape:

```json
{
  "mcpServers": {
    "diffusionbear": {
      "command": "/absolute/path/to/diffusionbear-mcp",
      "env": {
        "DIFFUSIONBEAR_API_URL": "http://127.0.0.1:8001",
        "DIFFUSIONBEAR_API_KEY": "dbk_..."
      }
    }
  }
}
```

## Generating on another Mac (LAN or Tailscale)

The MCP server runs next to the agent; only `DIFFUSIONBEAR_API_URL` changes:

| Where the backend runs | `DIFFUSIONBEAR_API_URL` |
|---|---|
| Same Mac | `http://127.0.0.1:8001` |
| Mac on the LAN, backend started with `--host 0.0.0.0` and API mode ON | `http://192.168.1.20:8001` |
| Mac behind `tailscale serve --bg 8001` | `https://my-mac.tailnet-name.ts.net` |

The key is required in every remote case. Tailscale encrypts the link but does not
replace the key. Never publish the port on the Internet.

## What to ask the agent

> Generate a 768×1024 poster of a bear astronaut with DiffusionBear and save it to
> ~/Desktop/bear.png.

The agent typically calls:

```text
list_models(installed_only=True)               -> picks "flux2-klein-4b"
create_image(prompt="poster of a bear astronaut, retro print",
             dest_path="~/Desktop/bear.png",
             model="flux2-klein-4b", width=768, height=1024, seed=42)
-> {"job_id": "…", "images": [{"path": "/Users/me/Desktop/bear.png",
                               "metadata_path": "/Users/me/Desktop/bear.json",
                               "seed": 42, "bytes": 512345}]}
```

For long or batched work it can use the lower-level tools:
`generate_image` → `job_status` / `wait_for_image` → `save_image(…, with_metadata=True)`,
and `cancel_job` if the user changes their mind.

## Smoke test

```bash
DIFFUSIONBEAR_API_URL=http://127.0.0.1:8001 DIFFUSIONBEAR_API_KEY=dbk_... \
  python mcp-server/e2e_mcp_test.py      # spawns the server over stdio, 13 checks
```
