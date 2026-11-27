# DiffusionBear Agentic API Reference

Version: DiffusionBear 0.5.0
Audience: local LLM agents, coding assistants, automation scripts
Transport: HTTP/JSON over loopback, LAN, or a TLS-terminated PRO endpoint
Default base URL: `http://127.0.0.1:8001`

DiffusionBear exposes its local FastAPI backend with no cloud dependency. Once the
chosen model weights are installed, every request and every generated image stays on
the machine. There is no telemetry; uploads (reference images, LoRAs) only go to the
backend you point at. The local build does not
require a key on loopback; a PRO/Enterprise deployment requires `Authorization:
Bearer <TOKEN>` on every `/api/*` route except the public health check.

> Want working code first? See [`docs/examples/`](examples/README.md): curl, Python
> (stdlib), Node and MCP examples, all tested against a real backend.

> The port is normally `8001`. If the app started on another port, use the port shown
> in the app header. `GET /api/version` confirms the backend is alive.

## Authentication & Rate Limiting (v0.5.0+ PRO)

The local build still accepts loopback requests without a key. A PRO deployment is
started with `DIFFUSIONBEAR_API_KEY` (16+ characters) or a `security.json` file, and
then **every `/api/*` route except `/api/version` requires**:

```http
Authorization: Bearer <TOKEN>
```

`X-API-Key: <TOKEN>` is accepted as an alias (legacy `X-MLX-API-Token` and
`X-API-Token` too). When several are sent, `Authorization` wins: a wrong Bearer is
refused even if `X-API-Key` is right. Basic auth is not supported. `/api/version` is intentionally public
for readiness probes; it is the only unauthenticated API route.

When a key is configured, the key applies to loopback too. A missing or wrong token
returns:

```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer
```

`POST /api/generate` is additionally protected by an in-process flood guard:

| Budget | Default | Tuning variable |
|---|---|---|
| Requests per identity per 60 s | 30 | `DIFFUSIONBEAR_GENERATE_RATE_PER_MINUTE` |
| Burst per identity per 1 s | 4 | `DIFFUSIONBEAR_GENERATE_BURST` |
| Queued + running jobs | 64 | `DIFFUSIONBEAR_GENERATE_MAX_QUEUED_JOBS` |

A burst or a full queue returns `429 Too Many Requests` with a `Retry-After` header.
Clients must wait at least that long before retrying. See
`docs/enterprise-security.md` for the full deployment and key-rotation guide.

### Media query-token exception

Browser `<img>` tags cannot send an `Authorization` header. The gateway accepts
`?token=<TOKEN>` or `?api_key=<TOKEN>` **only** on `GET` media routes:

```text
/api/images/{image_id}/file
/api/uploads/{filename}
```

Query credentials are rejected on every JSON or mutating route. Prefer the
Authorization header for agents; query strings can appear in access logs.

API mode is the master switch: OFF keeps loopback-only behaviour and ignores
stored keys; ON enables key authentication, LAN access and rate limiting. The
header shows API when enabled, and CLIENT when the UI targets a remote backend.

Managed keys can be created and revoked in **Parameters → API**. The environment
key remains valid and read-only; a new key is shown only once and stored as a
SHA-256 hash. Use `GET/POST/DELETE /api/security/keys` for automation.

LAN browser access is supported when a strong key is configured and the request
is same-origin; non-IP hostnames require `DIFFUSIONBEAR_ALLOWED_HOSTS`.

Architectural constraints: TLS is not built in and requires a reverse proxy; the
rate limiter is in-process and assumes one worker; `?token=` media URLs must be
protected by TLS to prevent log, history, and referrer leakage.

## Agent contract at a glance

1. `GET /api/models` and choose a model whose `installed` field is `true`.
2. `GET /api/loras` and, if needed, select entries whose `base_model` matches the
   chosen model family.
3. `POST /api/generate` with a JSON body. The response contains a `job_id`.
4. Poll `GET /api/jobs/{job_id}` until `status` is `done`, `error`, or `cancelled`.
5. Read each `results[i].file`, then download it with
   `GET /api/images/{id}/file` or read it from the local data directory.

One image batch is processed at a time. Submit independent work sequentially, or let
`batch` create several images from one prompt. Do not fire many concurrent generation
requests and expect parallel inference.

## Important naming rule: `model` vs `base_model`

`POST /api/generate` accepts **`model`**, the runnable model id
(`flux1-schnell`, `flux2-klein-4b`, `sdxl`, `krea2-turbo`, ...). It does **not** accept
a field named `base_model`; unknown fields are rejected with `422`.

`base_model` is a LoRA-registry field returned by `GET /api/loras`. It tells an agent
which family an adapter belongs to. Filter LoRAs by it, then send the adapter's
`path` in the generation payload.

## Core endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/version` | App and backend identity |
| `GET` | `/api/models` | Runnable models, defaults, capabilities, install state |
| `GET` | `/api/loras` | LoRA registry, including canonical `base_model` |
| `GET` | `/api/loras/{name}/compatibility?model={model_id}` | Preflight adapter compatibility |
| `POST` | `/api/generate/steps/preflight` | Check a step count and obtain an extended-steps ack token |
| `POST` | `/api/generate` | Queue one generation job |
| `GET` | `/api/jobs/{job_id}` | Poll job status and results |
| `GET` | `/api/jobs` | List active and recent jobs |
| `POST` | `/api/jobs/{job_id}/cancel` | Cancel a queued or running job |
| `POST` | `/api/jobs/cancel-all` | Cancel every active job |
| `GET` | `/api/images/{image_id}` | Image metadata sidecar as JSON |
| `GET` | `/api/images/{image_id}/file` | Original image bytes (PNG/JPEG) |

## `POST /api/generate/steps/preflight`

Ask before submitting a step count above the machine-safe ceiling:

```json
{ "model": "juggernaut-xi", "steps": 500, "width": 512, "height": 768, "batch": 1 }
```

The response contains `safe_max_steps`, `hard_max_steps`, `estimated_seconds`,
and, when the count is above the safe ceiling, `requires_ack: true` and an
`ack_token`. Submit the generation with `allow_extended_steps: true` and that
token in `steps_ack`. The token expires after five minutes and is bound to the
model, steps, width, height and batch. Above `hard_max_steps` the request is
refused even with a token. Distilled and Qwen models keep their trained ceilings.

## `GET /api/version`

Useful as a health check before submitting work.

```json
{
  "name": "DiffusionBear",
  "version": "0.5.0",
  "version_label": "v0.5.0",
  "author": "Ouinche",
  "website": "https://www.DiffusionBear.com",
  "repo": "https://github.com/OuincheWinch/DiffusionBear"
}
```

## `GET /api/models`

Returns a JSON array. Each entry is a runnable engine or a known model that can be
installed. Key fields an agent needs:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Send this value as `model` in `/api/generate` |
| `label` | string | Human-readable name |
| `engine` | string | Backend engine (`mflux`, `sdxl`, `qwen`, `moss`, `fibo2`) |
| `ecosystem` | string | Display ecosystem (`FLUX.1`, `FLUX.2`, `SDXL`, ...) |
| `lora_format` | string or null | Adapter format the engine accepts |
| `supports_loras` | boolean | Whether `loras` may be sent for this model |
| `supports_guidance` | boolean | Whether `guidance` has an effect |
| `supports_negative` | boolean | Whether `negative_prompt` has an effect |
| `installed` | boolean | `false` means generation will fail until downloaded |
| `default_steps` | integer | Suggested step count |
| `default_width`, `default_height` | integer | Suggested size |
| `default_guidance` | number | Suggested CFG/guidance |
| `samplers` | string[] | Accepted sampler ids |
| `max_pixels` | integer | Per-image pixel ceiling |
| `family` | string or null | Present on custom model entries |

### Built-in models and canonical families

The family is not always a separate field on built-in entries. Map it from the model
id or `ecosystem`:

| Model id | Canonical family | LoRA support |
|---|---|---|
| `flux1-schnell` | `flux1` | yes |
| `flux2-klein-4b`, `flux2-klein-9b` | `flux2` | yes |
| `flux3-klein` | `flux3` | no |
| `krea2-turbo` | `krea2` | yes |
| `z-image-turbo` | `z-image` | yes |
| `qwen-image-2.1`, `qwen-image-2.1-turbo` | `qwen` | yes |
| `qwen-image-edit` | `qwen` | no |
| `moss-image-sft`, `moss-image-base` | `moss` | yes |
| `fibo2-turbo` | `fibo2` | no |
| `juggernaut-xl-lightning`, `juggernaut-xi`, `realvis-xl-v5`, `realvis-xl-v5-lightning`, `Pony` | `sdxl` | yes |

Pony and Illustrious adapters are still family `sdxl`. A registry entry may also carry
`family_subtype: "pony"` or `"illustrious"` for display; that subtype does not change
the engine or the LoRA mapping.

Example, trimmed:

```json
[
  {
    "id": "flux1-schnell",
    "label": "FLUX.1-schnell",
    "engine": "mflux",
    "ecosystem": "FLUX.1",
    "lora_format": "FLUX.1",
    "supports_loras": true,
    "supports_negative": false,
    "default_steps": 4,
    "default_width": 512,
    "default_height": 768,
    "installed": true,
    "samplers": ["flow_match_euler_discrete", "euler"],
    "max_pixels": 4194304
  }
]
```

Agent rule: never assume a model is installed. Check `installed === true`; otherwise
stop and report which model must be downloaded.

## `GET /api/loras`

Returns the LoRA registry array. Each entry:

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Registry name, used by the compatibility endpoint |
| `path` | string | Server-side absolute path; send this in `loras[].path` |
| `scale` | number | Stored default scale |
| `triggers` | string[] | Activation phrases |
| `base_model` | string | Canonical family: `sdxl`, `flux1`, `flux2`, `krea2`, `z-image`, `qwen`, `moss` |
| `family_subtype` | string or null | `pony` / `illustrious` display hint for SDXL |
| `sha256` | string or null | Adapter digest |
| `civitai_version_id` | integer or null | Civitai version id when known |
| `civitai_model_id` | integer or null | Civitai model id when known |

Example:

```json
[
  {
    "name": "fir-needles-flux-v1",
    "path": "/Users/you/DiffusionBear/data/lora_files/fir-needles-flux-v1.safetensors",
    "scale": 1.0,
    "triggers": ["fir needles", "fir needle style", "made out of fir needles"],
    "base_model": "flux1",
    "family_subtype": null,
    "sha256": "19914243A8372C98A1BA542AF09DD461115EFE1D342BBC87089ED84278820803"
  }
]
```

The returned `path` is authoritative. Do not construct paths yourself and do not send
a `name` or `base_model` inside the generation `loras` array: the API accepts only
`path` and `scale`.

### Optional compatibility preflight

```http
GET /api/loras/{name}/compatibility?model={model_id}
```

Returns a verdict before generation starts. `compatible` can be `true`, `false`, or
`null` when the answer cannot be verified (for example, weights are not installed).
Treat `null` as unknown, not as compatible.

## `POST /api/generate`

Queues a generation job. The request is strict: unknown fields are rejected.

### Minimal request

```json
{
  "model": "flux1-schnell",
  "prompt": "a small red apple on a white table",
  "width": 512,
  "height": 768,
  "steps": 4
}
```

### Full agent request

```json
{
  "model": "flux1-schnell",
  "prompt": "a small red apple on a white table, soft window light",
  "negative_prompt": "",
  "width": 512,
  "height": 768,
  "steps": 4,
  "guidance": 1.0,
  "seed": 1794308128,
  "quantization": 4,
  "sampler": "flow_match_euler_discrete",
  "batch": 1,
  "output_format": "png",
  "loras": [
    {
      "path": "/Users/you/DiffusionBear/data/lora_files/fir-needles-flux-v1.safetensors",
      "scale": 1.0
    }
  ]
}
```

### Request fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | string | app default | Runnable id from `/api/models` |
| `prompt` | string | `""` | Required by the engine; non-empty after trim |
| `negative_prompt` | string | `""` | Ignored by guidance-distilled models that do not support it |
| `width`, `height` | integer | model default | 128..8192; engine may enforce stricter pixel caps |
| `steps` | integer | model default | 1..10000 schema; `GET /api/models` returns `steps_policy` with the machine-safe and hard ceilings. Above `safe_max_steps`, send `allow_extended_steps: true` plus a fresh `steps_ack`. Distilled and Qwen models keep their trained ceiling. |
| `guidance` | number | model default | 0..30; effect depends on `supports_guidance` |
| `seed` | integer | random | 0..2^63-1; repeat for reproducibility |
| `quantization` | integer | `4` | Allowed values `2`, `3`, `4`, `8`; SDXL accepts only 4 or 8 |
| `loras` | array | `[]` | Max 16; each item is exactly `{ "path": string, "scale": number }` |
| `batch` | integer | `1` | 1..16 schema; the server may lower it on low-memory machines |
| `output_format` | string | `"png"` | `png`, `jpeg`, or `jpg` |
| `sampler` | string or null | model default | Must be one of the model's `samplers` |
| `cache_interval` | integer | `1` | 1..10; engine-specific step cache |
| `reference_images` | string[] | `[]` | Max 10; server-side paths/ids for supported edit engines |
| `reference_strength` | number | model default | >0..1 for engines that use strength |
| `stealth` | boolean | `false` | Strip metadata from the saved image |
| `fast_vae` | boolean | `true` | Used only when the model supports it |
| `max_pixels` | integer or null | model cap | Per-request pixel ceiling |
| `use_teacache` | boolean | `true` | Step-skipping where supported |
| `use_kv_cache` | boolean | `true` | Reference-image KV cache where supported |
| `clip_skip` | integer | `2` | 1..12; SDXL text-encoder layer |

Do not send `base_model` to `/api/generate`. It is not a generation field and will
fail validation.

### Response

```json
{
  "job_id": "40478339923948c88c10e82338505fda"
}
```

HTTP status is usually `200`. Authentication failures return `401` on a keyed PRO
deployment. Schema errors (unknown field, out-of-range value) return `422` and an
empty prompt returns `400`; a generation burst or a full queue returns `429` with
`Retry-After`. Everything checked by the engine — including **a model that is not
installed** — is accepted with `200` and reported later through the job's `error`
field, e.g. `"FLUX.1-schnell is not installed."`. Check `installed` first.

## Polling `GET /api/jobs/{job_id}`

Polling is the only progress channel: there is no SSE or WebSocket endpoint.
Poll every 1–2 s while `generating`; `progress.step`/`progress.steps` and
`progress.eta_seconds` drive a progress bar.

Status values:

- `queued`: accepted, waiting for the worker
- `generating`: inference is running
- `done`: all requested images are saved
- `error`: generation failed; see `error`
- `cancelled`: cancelled by the user or another agent

Example (trimmed):

```json
{
  "id": "40478339923948c88c10e82338505fda",
  "status": "done",
  "created_at": 1794308211.09,
  "finished_at": 1794308356.24,
  "phase": "done",
  "progress": {
    "step": 4,
    "steps": 4,
    "elapsed": 15.9,
    "eta_seconds": 0,
    "image_index": 0,
    "batch": 1,
    "saved_id": "feb87d2de34a48668a5f5681007c82a5",
    "saved_index": 1
  },
  "results": [
    {
      "id": "feb87d2de34a48668a5f5681007c82a5",
      "file": "feb87d2de34a48668a5f5681007c82a5.png",
      "prompt": "a small red apple on a white table",
      "negative_prompt": "",
      "width": 512,
      "height": 768,
      "steps": 4,
      "guidance": 1.0,
      "seed": 1794308128,
      "quantization": 4,
      "sampler": "flow_match_euler_discrete",
      "loras": [],
      "generation_time": 145.15,
      "load_time": 2.19,
      "created_at": 1794308211.09,
      "model": "mflux-community/flux-1-schnell-mflux-q4",
      "format": "png"
    }
  ],
  "error": null
}
```

While generating, `last_saved` can identify an image that is already on disk:

```json
{
  "status": "generating",
  "last_saved": {
    "id": "feb87d2de34a48668a5f5681007c82a5",
    "index": 1,
    "batch": 3
  }
}
```

`partial_results` may be present when a batch fails after some images were saved.

## Getting the output image

Each result has two identifiers:

- `results[i].id`: image id
- `results[i].file`: filename under the app's generated-image directory

Recommended agent path:

```http
GET /api/images/{id}/file
```

This returns the original PNG or JPEG bytes. No base64 conversion is done by the API;
if an LLM tool needs base64, read the HTTP body and encode it client-side.

Metadata is available at:

```http
GET /api/images/{id}
```

The JSON sidecar includes the final prompt, seed, size, sampler, LoRA paths and
generation timing. The file also lives on disk under the app's data directory, in
`generated/<file>`. Agents running on the same machine may read that path directly;
agents in another process should prefer the HTTP route.

## `GET /api/jobs`

Returns active jobs first, then recently finished jobs. Use it for queue visibility or
crash recovery. Common fields: `id`, `status`, `error`, `created_at`, `finished_at`,
`progress`, `request`.

## Cancellation

```http
POST /api/jobs/{job_id}/cancel
POST /api/jobs/cancel-all
```

Cancellation is cooperative. A running engine stops at its next safe step and may
still leave `partial_results`. Do not delete files while a job is still generating.

| Call | Response |
|---|---|
| cancel a `queued` or `generating` job | `200 {"status": "cancelled", "job_id": "…"}` |
| cancel a finished job | `400 {"detail": "job already done"}` (or `error` / `cancelled`) |
| unknown id | `404 {"detail": "unknown job"}` |
| `cancel-all` | `200 {"cancelled": ["…"]}` |

## Python agent example

A complete, dependency-free client with progress, 429 handling, Ctrl-C
cancellation and sidecar saving lives in
[`examples/diffusionbear_client.py`](examples/diffusionbear_client.py). The core
loop with `requests`:

```python
import os
import time
import requests

BASE = os.environ.get("DIFFUSIONBEAR_URL", "http://127.0.0.1:8001")
API_KEY = os.environ.get("DIFFUSIONBEAR_API_KEY", "")
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

models = requests.get(f"{BASE}/api/models", headers=HEADERS, timeout=10).json()
installed = [m for m in models if m.get("installed")]
if not installed:
    raise SystemExit("No installed model: download one in the Models tab first.")
model = next((m for m in installed if m["id"] == "flux2-klein-4b"), installed[0])

payload = {
    "model": model["id"],
    "prompt": "a small red apple on a white table",
    "width": 512,
    "height": 768,
    "steps": model["default_steps"],
    "seed": 42,
}
if model.get("supports_loras"):
    family = {"FLUX.1": "flux1", "FLUX.2": "flux2"}.get(model.get("ecosystem"))
    loras = requests.get(f"{BASE}/api/loras", headers=HEADERS, timeout=10).json()
    # Pony/Illustrious adapters are base_model == "sdxl".
    usable = [l for l in loras if family and l.get("base_model") == family]
    if usable:
        payload["loras"] = [{"path": usable[0]["path"], "scale": 1.0}]

resp = requests.post(f"{BASE}/api/generate", json=payload, headers=HEADERS, timeout=30)
resp.raise_for_status()  # 401 key, 422 schema, 429 -> wait resp.headers["Retry-After"]
job_id = resp.json()["job_id"]

while True:
    status = requests.get(f"{BASE}/api/jobs/{job_id}", headers=HEADERS, timeout=10).json()
    if status["status"] in {"done", "error", "cancelled"}:
        break
    time.sleep(2)

if status["status"] != "done":
    raise RuntimeError(status.get("error") or status["status"])

for result in status["results"]:
    image_id = result["id"]
    image = requests.get(f"{BASE}/api/images/{image_id}/file", headers=HEADERS, timeout=60)
    sidecar = requests.get(f"{BASE}/api/images/{image_id}", headers=HEADERS, timeout=10)
    with open(f"{image_id}.png", "wb") as fh:
        fh.write(image.content)
    with open(f"{image_id}.json", "w") as fh:
        fh.write(sidecar.text)
    print("saved", image_id, "seed", result["seed"])
```

## curl example (local loopback or authenticated remote)

```bash
export DB_URL="http://127.0.0.1:8001"        # or https://mac.tailnet.ts.net
export DB_KEY="dbk_your_managed_or_env_key"  # omit Authorization header when API mode is OFF on localhost

curl -sS "$DB_URL/api/generate" \
  -H "Authorization: Bearer $DB_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "flux2-klein-4b",
    "prompt": "a small red apple on a white table",
    "width": 512,
    "height": 768,
    "steps": 4
  }'
```

Then poll (`X-API-Key: $DB_KEY` also works):

```bash
curl -sS -H "Authorization: Bearer $DB_KEY" "$DB_URL/api/jobs/<job_id>"
curl -sS -H "Authorization: Bearer $DB_KEY" "$DB_URL/api/images/<image_id>/file" -o image.png
curl -sS -H "Authorization: Bearer $DB_KEY" "$DB_URL/api/images/<image_id>" -o image.json
```

The end-to-end script [`examples/quickstart.sh`](examples/quickstart.sh) chains
these calls (model pick, polling, image + sidecar download).

## Agent behavior rules

1. **Health first.** Call `GET /api/version` before a long workflow.
2. **Select only installed models.** `installed` must be true.
3. **Filter LoRAs by `base_model`.** Send only `path` and `scale`; never construct a
   LoRA path or cross families.
4. **Use `model`, not `base_model`, at generation time.**
5. **Poll with backoff.** Two to five seconds is reasonable; a cold model load can take
   longer than the inference itself.
6. **Respect the queue.** One active generation at a time is the practical limit.
   Batch related images with `batch` rather than many concurrent HTTP requests.
7. **Treat `error` as authoritative.** Do not retry an unchanged request blindly; check
   the error, model install state, LoRA compatibility, and pixel limits first.
8. **Keep seeds.** Store the seed from the result when reproducibility matters.
9. **No base64 endpoint.** Download the bytes from `/api/images/{id}/file` and encode
   them in the client if a downstream model requires base64.
10. **Respect the deployment mode.** Local loopback needs no key. A PRO deployment
    requires `Authorization: Bearer <TOKEN>` on every `/api/*` route except
    `/api/version`; never expose an unauthenticated backend to a network interface.
    For media URLs that cannot carry a header, use the documented `?token=` exception.

## Common failure modes

| Symptom | Likely cause |
|---|---|
| `422` with an unknown field | You sent `base_model` or another field not in the schema |
| `200` then job `error` "… is not installed" | Selected model weights are absent; check `installed` before submitting |
| `400` LoRA error | Adapter path is wrong, family mismatch, or model has `supports_loras: false` |
| `401` | Missing, malformed, wrong or revoked API key; send `Authorization: Bearer <TOKEN>` |
| `403` "Disallowed CORS origin" / `400` on preflight | Browser app served from an origin other than the backend or the Vite dev ports; serve the UI from the backend itself |
| `403` "local-only" | `DELETE /api/images/{id}` and `/api/import-path` refuse remote clients by design |
| `429` | Generation burst or full queue; wait for the `Retry-After` value before retrying |
| Job `error` with Metal/OOM text | Image size or reference set exceeds the machine budget |
| `results` missing | Job is still running; check `status` and `progress` |
| `null` compatibility | Weights missing; the answer is unknown, not necessarily incompatible |

## Remote access (LAN, Tailscale, phone)

| Setup | How | Key |
|---|---|---|
| Same Mac, API mode OFF | `http://127.0.0.1:8001` | not needed |
| Same Mac, API mode ON | `http://127.0.0.1:8001` | required (loopback included) |
| LAN | start the backend with `python backend/main.py --host 0.0.0.0 --port 8001`, API mode ON, then `http://<mac-ip>:8001` | required |
| Tailscale | `tailscale serve --bg 8001` on the Mac, then `https://<mac>.<tailnet>.ts.net` | required |

- The packaged `.app` always binds `127.0.0.1`; for a phone or another computer use
  Tailscale Serve (HTTPS) or a CLI backend on `0.0.0.0`.
- Asking for `0.0.0.0` without a strong key (16+ chars) is refused at startup and the
  backend falls back to `127.0.0.1` (`[security] bind host '0.0.0.0' refused`).
- The web UI is served by the backend itself, so a phone can open
  `http://<mac-ip>:8001/`, paste the key in **Parameters → API → Backend connection**,
  press **Test**, then **Save connection**. Generation and the gallery then work as on the Mac.
- CORS only allows same-origin requests and the Vite dev server (`localhost:5173/5174`).
  A UI served from another origin cannot call a remote backend from the browser.
- TLS is not built in: plain `http://` on a LAN sends the key in clear text. Prefer
  Tailscale Serve or a reverse proxy with HTTPS, especially because `<img>` URLs carry
  `?token=`.
- Every valid key is an administrator key: it can change settings, read the workspace
  and Hugging Face token status, change the port and create/revoke keys. Give each
  client its own key and revoke it when the device is lost.
- Never forward the port on your router or expose it through a public tunnel.
- Generated images are written to the app's data directory. `results[i].file` is the
  filename inside `generated/`; `results[i].id` is its API identity.
