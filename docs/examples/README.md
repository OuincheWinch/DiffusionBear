# DiffusionBear API examples

Working, copy-paste examples for the REST API described in
[`../agentic-api-reference.md`](../agentic-api-reference.md). Each one: checks
health, picks an installed model, queues a job, follows progress by polling,
then saves the image **and** its JSON sidecar.

| File | For | Needs |
|---|---|---|
| [`quickstart.sh`](quickstart.sh) | shell scripts, CI, a first try | `curl`, `python3` |
| [`diffusionbear_client.py`](diffusionbear_client.py) | Python apps and agents (CLI + importable class, Ctrl-C cancels) | Python 3.9+, no packages |
| [`generate.mjs`](generate.mjs) | Node / TypeScript tooling | Node 18+ |
| [`mcp-clients.md`](mcp-clients.md) | Claude Code, Claude Desktop, Cursor, Cline… | `pip install ./mcp-server` |
| [`agent-instructions.md`](agent-instructions.md) | `AGENTS.md` / system prompt block for HTTP agents | — |

## Setup

```bash
export DIFFUSIONBEAR_URL=http://127.0.0.1:8001   # quickstart.sh reads DB_URL / DB_KEY
export DIFFUSIONBEAR_API_KEY=dbk_...             # Parameters → API → Create key
```

On the same Mac with API mode OFF, no key is needed. From another device (LAN,
phone, Tailscale), API mode must be ON and the key is required. See
[Remote access](../agentic-api-reference.md#remote-access-lan-tailscale-phone).

## Try it

```bash
bash docs/examples/quickstart.sh "a lighthouse at dusk, oil painting"
python3 docs/examples/diffusionbear_client.py "a cozy cabin in a snowy forest" \
    --model flux2-klein-4b --size 768x1024 --seed 42 --out cabin.png
node docs/examples/generate.mjs "an origami fox, studio light" fox.png
```

Typical output of the Python client:

```text
DiffusionBear 0.5.1 at http://192.168.1.20:8001
  step 4/4  eta 0.0s
  done
saved cabin.png + cabin.json  seed=42  7.58s
```
