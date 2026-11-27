# DiffusionBear Enterprise Security Guide

Version: 0.5.0
Audience: agency operators, enterprise platform teams, remote-agent integrators
Scope: the native FastAPI security gateway in `backend/security_pro.py`

DiffusionBear is a local-first image-generation studio. The default installer binds
`127.0.0.1` and trusts loopback clients. A PRO/Enterprise deployment may expose the
API to a LAN, a private VPN, or a public reverse proxy. This guide covers the three
controls that make that safe:

1. API-key authentication.
2. A bind-address fail-safe that refuses to expose an unauthenticated GPU.
3. A generation rate limiter and queue guard sized for one Apple Silicon device.

> **Security warning.** An API key sent over plain HTTP on a LAN is visible to
> anyone who can capture traffic. Put a TLS-terminating reverse proxy in front of
> DiffusionBear for any non-loopback deployment. Keep the backend on a private
> interface or a private network whenever possible.

---

## 1. Authentication

### 1.1 Key requirements

The API key must be:

- at least **16 characters** long;
- free of whitespace;
- not a single repeated character (`xxxxxxxxxxxxxxxx` is rejected).

Use a generated value, not a passphrase:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Example:

```text
4g7Yt2mN8qR1sV6xZ0aB3cD5eF9hJ2kL
```

### 1.2 Environment variable

The simplest deployment is an environment variable:

```bash
export DIFFUSIONBEAR_API_KEY='4g7Yt2mN8qR1sV6xZ0aB3cD5eF9hJ2kL'
```

The process must see the variable before it starts. Restart the backend after
changing it. An environment variable wins over `security.json`.

### 1.3 `security.json`

For managed deployments, store the key in the app's data directory:

```text
<DATA_DIR>/security.json
```

On a source checkout that is normally:

```text
backend/data/security.json
```

On the packaged macOS app the data directory is the resolved
`DIFFUSIONBEAR_DATA_DIR` (see `AGENTS.md`; the live gallery and settings may live in
the shared data tree).

File content:

```json
{
  "api_key": "4g7Yt2mN8qR1sV6xZ0aB3cD5eF9hJ2kL"
}
```

Only the `api_key` field is read; a `token` field is ignored (the bind guard then
treats the file as having no key).

Lock the file down:

```bash
chmod 600 backend/data/security.json
```

The backend logs a warning when the file is readable by group or others. A
`security.json` larger than 64 KiB, unreadable, or malformed is treated as **no
key**; the bind guard then refuses an open interface.

`security.json` is read per request, so rotating the file key takes effect without a
restart. Environment-variable changes require a restart.

### 1.4 Public health check

`GET /api/version` is intentionally public. It is the only `/api/*` route that does
not require authentication, so load balancers and readiness probes can use it:

```bash
curl -s http://127.0.0.1:8001/api/version
```

Every other `/api/*` route requires one of:

```http
Authorization: Bearer <TOKEN>
```

```http
X-API-Key: <TOKEN>
```

Legacy `X-MLX-API-Token` and `X-API-Token` headers are also accepted.

When **no key is configured**, loopback clients keep the local-build behaviour and
work without authentication; non-loopback clients receive `401`. When a key **is**
configured, every caller, including loopback, must present it.

### 1.5 `401 Unauthorized`

Missing, malformed, or wrong credentials return:

```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer
Content-Type: application/json

{"detail":"invalid or missing API key"}
```

The check uses `hmac.compare_digest`; it is not a string `==`.

### 1.5 API mode and client mode

**Parameters → API → API mode** is the master switch.

- **OFF (default):** API keys are stored but inactive. Only loopback clients
  are accepted; open LAN bindings fall back to `127.0.0.1`. Existing local
  behaviour continues.
- **ON:** key authentication, LAN/Tailscale access, managed keys and rate
  limiting become active.

When API mode is on, the header shows **API** next to the version. If the UI is
pointed at a non-local backend, it also shows **CLIENT**.

The header reflects `/api/version.api_mode`; it is not a cosmetic local toggle.

### 1.6 Managing keys in Parameters → API

Field-by-field guide: [`parameters-api-guide.md`](parameters-api-guide.md).

v0.5.0 adds **Parameters → API**. It supports multiple named managed keys for
agents, devices and browser profiles.

- The environment key is read-only and always remains valid.
- A legacy `security.json` `"api_key"` is read-only and listed separately.
- New keys are shown once, then stored only as SHA-256 in `security.json`.
- Revoke is immediate and per key. The final usable credential cannot be revoked.
- The environment key cannot be deleted from the UI.
- “Use in this browser” writes the selected key to
  `localStorage["DIFFUSIONBEAR_API_KEY"]`.

`security.json` is written atomically with mode `0600`. The full managed key is
never returned by `GET /api/security/keys`.

---

## 2. Network binding and the fail-safe

### 2.1 Safe default

The default bind is loopback:

```text
127.0.0.1:8001
```

The packaged macOS app and `run.sh` keep that default. Do not change it unless you
are deliberately deploying the API to another machine.

### 2.2 Starting PRO mode

Use the built-in CLI runner so the bind guard is applied:

```bash
cd backend
DIFFUSIONBEAR_API_KEY='4g7Yt2mN8qR1sV6xZ0aB3cD5eF9hJ2kL' \
  ../venv/bin/python main.py --host 0.0.0.0 --port 8001
```

The equivalent environment form:

```bash
export DIFFUSIONBEAR_API_KEY='4g7Yt2mN8qR1sV6xZ0aB3cD5eF9hJ2kL'
export DIFFUSIONBEAR_BIND_HOST='0.0.0.0'
export DIFFUSIONBEAR_PORT='8001'
cd backend && ../venv/bin/python main.py
```

Do **not** start `uvicorn --host 0.0.0.0` directly and expect the bind guard to run.
The guard lives in `main.run()`. Authentication still protects the API if uvicorn is
launched another way, but the explicit bind fail-safe only runs through the runner.

### 2.3 Open-interface detection

These requested hosts are treated as open interfaces:

- `0.0.0.0`
- `0.0.0.0/0`
- `0.0.0.0:<port>`
- `::`
- `[::]`
- `::/0`
- `[::]/0`
- `*`

A normal LAN address such as `192.168.1.20` or `10.0.0.5` is not wildcard-open, but
it is still reachable from the network. Protect it with the API key and TLS.

### 2.4 What the fail-safe does

If an open bind is requested and:

- no key is configured, or
- the key is shorter than 16 characters, or
- the key is whitespace or one repeated character,

the effective host is forced to `127.0.0.1` and a `CRITICAL` log line is emitted:

```text
[security] refusing to bind 0.0.0.0 without a strong DIFFUSIONBEAR_API_KEY
or security.json api_key (>= 16 chars); falling back to 127.0.0.1
```

The process continues on loopback instead of exposing an unauthenticated GPU. A
strong key produces a `WARNING` that the API is exposed with authentication enabled.

### 2.5 LAN browser access

A browser on another machine can open the backend-served UI directly:

```text
http://<LAN-IP>:8001/
```

This requires a strong configured key. Without one, host/client checks still
refuse the LAN request before authentication runs.

Hostnames are accepted only when they are IP literals or explicitly listed:

```bash
export DIFFUSIONBEAR_ALLOWED_HOSTS="macbook.local,diffusion.local"
```

Run the server with `--host 0.0.0.0` as in §2.2. Then in the browser origin:

```javascript
localStorage.setItem("DIFFUSIONBEAR_API_KEY", "PASTE_THE_KEY");
location.reload();
```

Only same-origin browser requests are accepted; a page on another host or port
gets `403`. DNS-rebinding `Host` values are rejected unless explicitly allowed.
Requests without an `Origin` header (agents, curl) still require the Bearer key.

### 2.6 Remote backend from the UI

**Parameters → API → Backend connection** can point the frontend at a remote
backend. Empty URL means the local origin.

- Backend URL: `https://<mac>.<tailnet>.ts.net` via Tailscale Serve, or the Mac's
  Tailscale IP.
- API key: the same managed or environment key.
- **Test** checks `/api/version` and an authenticated `/api/models`.
- **Save** stores `DIFFUSIONBEAR_API_BASE` and `DIFFUSIONBEAR_API_KEY` in local
  storage, then reloads.
- **Reset to local** removes both overrides.

Tailscale is transport only. The API key remains mandatory. Remote CORS must
allow the local UI origin if the UI is not served by the remote backend itself.

### 2.7 Reverse proxy and TLS

For production, bind the backend to a private interface and terminate TLS at a
reverse proxy:

```nginx
server {
    listen 443 ssl;
    server_name diffusion.example.com;

    ssl_certificate     /etc/ssl/certs/diffusion.crt;
    ssl_certificate_key /etc/ssl/private/diffusion.key;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600s;
        # Authentication stays on the API. The backend does not trust a
        # loopback peer as an authenticated user behind a proxy.
        proxy_set_header Authorization $http_authorization;
    }
}
```

Keep the backend on loopback behind the proxy when possible. Do not publish port
8001 itself to the internet. The proxy must forward the API key; a loopback
connection alone is not authentication. `uvicorn` is started with
`proxy_headers=False`, so source-IP forwarding headers are not trusted.

---

## 3. Rate limiting and queue limits

### 3.1 Why the gateway limits generation

DiffusionBear runs one generation at a time on one Apple Silicon unified-memory
device. A remote script that fires hundreds of `/api/generate` requests cannot make
the GPU faster; it only fills the queue and the host's memory. The gateway rejects
the flood before FastAPI parses the body or the request reaches the worker queue.

### 3.2 Defaults

| Budget | Default | Environment variable |
|---|---|---|
| Requests per identity per 60 s | 30 | `DIFFUSIONBEAR_GENERATE_RATE_PER_MINUTE` |
| Requests per identity per 1 s (burst) | 4 | `DIFFUSIONBEAR_GENERATE_BURST` |
| Queued + running generation jobs | 64 | `DIFFUSIONBEAR_GENERATE_MAX_QUEUED_JOBS` |

An identity is the SHA-256 hash of a **valid** API key. A request with no key, an
invalid key, or a rotating fake key is bucketed by source IP instead, so rotating
junk keys does not create unlimited buckets.

The limiter is in-process and thread-safe. It assumes one backend worker, which is
the supported deployment for one GPU. Multiple uvicorn workers would each have an
independent budget.

### 3.3 Tuning

For a single trusted operator:

```bash
export DIFFUSIONBEAR_GENERATE_RATE_PER_MINUTE=60
export DIFFUSIONBEAR_GENERATE_BURST=8
export DIFFUSIONBEAR_GENERATE_MAX_QUEUED_JOBS=128
```

For a shared agency gateway:

```bash
export DIFFUSIONBEAR_GENERATE_RATE_PER_MINUTE=20
export DIFFUSIONBEAR_GENERATE_BURST=2
export DIFFUSIONBEAR_GENERATE_MAX_QUEUED_JOBS=32
```

Lower the queue before lowering the rate if memory pressure is the problem. Every
queued image still costs disk and holds a job record. The per-image hardware pixel
ceiling and step policy still apply after the request is accepted.

### 3.4 `429 Too Many Requests`

A burst or a full queue returns:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 1
Content-Type: application/json

{"detail":"generation rate limit exceeded"}
```

A full queue returns `Retry-After: 30`:

```json
{"detail":"generation queue is full"}
```

Clients must honor `Retry-After`; do not retry immediately. The limiter counts the
rejected request itself in the window, so a tight retry loop stays rejected.

Only `POST /api/generate` is rate-limited. Reads such as `/api/models`,
`/api/jobs`, and image delivery are not burst-limited by this guard.

### 3.5 Client-side concurrency

Even under the budget, do not run parallel generations against one device. The
practical pattern is:

1. submit one job;
2. poll `/api/jobs/{job_id}` with a 2–5 s backoff;
3. download the results;
4. submit the next job.

Use `batch` for multiple images from one prompt instead of many concurrent HTTP
requests.

---

## 4. Client integration

### 4.1 `curl`

Health check, no key:

```bash
curl -s http://127.0.0.1:8001/api/version
```

Authenticated model list:

```bash
export DIFFUSIONBEAR_API_KEY='4g7Yt2mN8qR1sV6xZ0aB3cD5eF9hJ2kL'

curl -s http://127.0.0.1:8001/api/models \
  -H "Authorization: Bearer $DIFFUSIONBEAR_API_KEY"
```

`X-API-Key` is equivalent:

```bash
curl -s http://127.0.0.1:8001/api/models \
  -H "X-API-Key: $DIFFUSIONBEAR_API_KEY"
```

Submit a generation:

```bash
curl -s http://127.0.0.1:8001/api/generate \
  -H "Authorization: Bearer $DIFFUSIONBEAR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "flux2-klein-4b",
    "prompt": "a clean product photo of a ceramic cup on a white table",
    "width": 512,
    "height": 768,
    "steps": 4,
    "seed": 42
  }'
```

Poll:

```bash
curl -s http://127.0.0.1:8001/api/jobs/<job_id> \
  -H "Authorization: Bearer $DIFFUSIONBEAR_API_KEY"
```

Download an image with a header:

```bash
curl -s http://127.0.0.1:8001/api/images/<image_id>/file \
  -H "Authorization: Bearer $DIFFUSIONBEAR_API_KEY" \
  -o image.png
```

### 4.2 Python `requests`

```python
import os
import time

import requests

BASE = "http://127.0.0.1:8001"
TOKEN = os.environ["DIFFUSIONBEAR_API_KEY"]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

health = requests.get(f"{BASE}/api/version", timeout=10)
health.raise_for_status()

models = requests.get(f"{BASE}/api/models", headers=HEADERS, timeout=30)
models.raise_for_status()

payload = {
    "model": "flux2-klein-4b",
    "prompt": "a clean product photo of a ceramic cup on a white table",
    "width": 512,
    "height": 768,
    "steps": 4,
    "seed": 42,
}

job = requests.post(
    f"{BASE}/api/generate",
    headers=HEADERS,
    json=payload,
    timeout=30,
)
job.raise_for_status()
job_id = job.json()["job_id"]

while True:
    status = requests.get(
        f"{BASE}/api/jobs/{job_id}", headers=HEADERS, timeout=30
    )
    status.raise_for_status()
    body = status.json()
    if body["status"] in {"done", "error", "cancelled"}:
        break
    time.sleep(3)

if body["status"] != "done":
    raise RuntimeError(body.get("error") or body["status"])

for result in body.get("results", []):
    image_id = result["id"]
    image = requests.get(
        f"{BASE}/api/images/{image_id}/file",
        headers=HEADERS,
        timeout=120,
    )
    image.raise_for_status()
    with open(f"{image_id}.png", "wb") as handle:
        handle.write(image.content)
```

### 4.3 Six-line retry policy for `429`

```python
import time

def post_with_retry(url, **kwargs):
    for _ in range(5):
        response = requests.post(url, **kwargs)
        if response.status_code != 429:
            return response
        time.sleep(int(response.headers.get("Retry-After", "5")))
    return response
```

### 4.4 Media query-token exception

Browser `<img>` tags cannot attach an `Authorization` header. For that reason only,
the gateway accepts a query key on **GET media routes**:

```text
GET /api/images/{image_id}/file?token=<TOKEN>
GET /api/images/{image_id}/file?thumb=true&token=<TOKEN>
GET /api/uploads/{filename}?token=<TOKEN>
```

`?api_key=<TOKEN>` is an alias.

The exception is deliberately narrow:

- only `GET`;
- only `/api/images/{id}/file` and `/api/uploads/{filename}`;
- the value is checked with the same constant-time comparison as a Bearer token;
- query keys are rejected on every JSON or mutating route.

Query strings can appear in reverse-proxy access logs, browser history, and
`Referer` headers. Prefer the Authorization header for machine-to-machine traffic,
serve media over TLS, and do not paste signed image URLs into third-party pages.

### 4.5 Browser frontend

The React frontend reads the key at runtime from local storage:

```javascript
// Browser console, once per browser profile.
localStorage.setItem("DIFFUSIONBEAR_API_KEY", "4g7Yt2mN8qR1sV6xZ0aB3cD5eF9hJ2kL");
location.reload();
```

To remove it:

```javascript
localStorage.removeItem("DIFFUSIONBEAR_API_KEY");
location.reload();
```

Production builds deliberately ignore `VITE_API_TOKEN`: a token baked into
`/assets/*.js` is public, even when the API itself requires a key. Set the key at
runtime through the app's connection settings or `localStorage`. `api.js` attaches
it to normal `fetch()` calls with `Authorization: Bearer`, and appends `?token=`
to gallery image URLs automatically.

Do not put a production key in a public web build, a screenshot, or a shared
browser profile. For a multi-user frontend, terminate auth at a reverse proxy and
issue per-user credentials.

---

## 5. Operations checklist

- [ ] Generate a random key of at least 32 characters.
- [ ] Store it in an environment variable or `security.json` with `chmod 600`.
- [ ] Start with `python main.py --host ...`, not bare `uvicorn`.
- [ ] Confirm the startup log says authentication is enabled.
- [ ] Put TLS in front of every non-loopback deployment.
- [ ] Keep `/api/version` public only for health checks.
- [ ] Tune `DIFFUSIONBEAR_GENERATE_*` for your expected client count.
- [ ] Teach clients to honor `Retry-After`.
- [ ] Rotate the key by rewriting `security.json` or restarting with a new env var.
- [ ] Never expose port 8001 directly to the public internet.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `401 invalid or missing API key` | No or wrong `Authorization: Bearer` | Send the exact key; check header capitalization is irrelevant but the value is exact |
| `401` on loopback after setting a key | A configured key applies to every caller, including loopback | Add the header to local scripts and the frontend |
| Server still on `127.0.0.1` after `--host 0.0.0.0` | Missing or weak key | Set a 16+ character key and restart |
| `429 generation rate limit exceeded` | Per-key/per-IP budget exhausted | Sleep for `Retry-After`, reduce concurrency |
| `429 generation queue is full` | 64 queued/running jobs by default | Wait, lower batch, or raise `DIFFUSIONBEAR_GENERATE_MAX_QUEUED_JOBS` deliberately |
| Images broken in the browser | Key configured but image URL has no token | Use the bundled frontend, or append `?token=` to media URLs |
| Key works with curl but not browser fetch | CORS/origin or the runtime key is not set | Use the app's own origin and set the key in connection settings; `VITE_API_TOKEN` is dev-only |

---

## 7. Security model and limits

- The gateway protects API access; it does not sandbox the local operating system.
- The rate limiter is in-memory and per process. It resets when the backend restarts.
- The bind fail-safe is a startup guard, not a firewall. Use host firewall rules and
  a reverse proxy for defense in depth.
- A query token is exposed wherever URLs are logged. Treat media URLs as secrets.
- `security.json` and environment variables are only as safe as the host account.
  Run the backend under a dedicated, least-privilege user.
- TLS, key rotation, audit logging, and network segmentation remain the operator's
  responsibility.
