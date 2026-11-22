# Parameters → API: Field-by-Field Guide

Version: 0.5.0
Audience: operators configuring local, LAN, Tailscale or remote API access

This page explains every control in **Parameters → API**. It covers the master
API switch, backend connection, and the three key sources: environment, legacy
and managed keys.

> Tailscale is transport only. It never replaces the API key. Every non-loopback
> deployment should use a strong key, and TLS is required outside a private
> Tailscale network.

---

## 1. Header badges

| Badge | Meaning |
|---|---|
| **API** | API mode is ON. Key authentication and external-access rules are active. |
| **CLIENT** | The UI is pointed at a non-local backend. The local UI is acting as a client of a remote API. |

Both can appear together: API ON on the remote backend, and CLIENT on the app
pointing to it.

---

## 2. API mode — the large ON/OFF switch

**Location:** top of Parameters → API, large toggle.

**OFF (default):**
- No external connections are accepted.
- Local loopback continues to work without a key.
- Stored keys are preserved but inactive.
- Opening the backend on `0.0.0.0` falls back to `127.0.0.1`.
- This is the normal local desktop behaviour.

**ON:**
- Key authentication is enforced on every `/api/*` route except `/api/version`.
- LAN, Tailscale and remote clients can connect, subject to key and origin checks.
- Managed keys can be revoked; revoking a key takes effect immediately.
- The header shows **API**.

**To enable:** a strong key must already exist. If none exists, the backend
returns `409` and the switch stays OFF. Create a managed key first, or set
`DIFFUSIONBEAR_API_KEY`.

**To disable:** turn the switch OFF. Keys remain stored; external access stops.

---

## 2a. Backend port

**Location:** API tab, below the API mode switch.

- Default: `8001`.
- Allowed: `1024`–`65535`.
- The port can only be changed while API mode is ON.
- Saving validates the range and probes the port on the configured bind host.
- A saved port does not move the running process. The app/backend must be
  relaunched. The panel shows the active port, configured port, and the next URL.
- In the packaged app, the launcher reads `api_port` from `settings.json` and
  uses it strictly when API mode is enabled. If that port is busy, launch fails
  clearly instead of silently climbing to another port.
- CLI: explicit `--port` wins, then `DIFFUSIONBEAR_PORT`, then `api_port`.

If browser clients are used, update their backend URL after the relaunch.

## 3. Backend connection

These fields tell the UI which backend to call.

### Backend URL

- Empty = the local backend that served the UI.
- Local URL example: `http://127.0.0.1:8001`.
- Remote LAN example: `http://192.168.1.20:8001`.
- Tailscale Serve example: the HTTPS URL printed by `tailscale serve status`.
- Tailscale IP example: `http://100.x.y.z:8001`.

An empty field is the safe default. A non-local URL activates **CLIENT** mode
after a successful test.

### API key

The key used for the target backend.

- Paste the environment key or a managed key.
- The field is masked.
- It is stored in this browser/app profile under
  `localStorage["DIFFUSIONBEAR_API_KEY"]`.
- It is not written to `security.json` by this field.

### Test

Runs two calls against the entered backend:

1. `GET /api/version` — public reachability.
2. `GET /api/models` with `Authorization: Bearer <key>` — authentication.

A remote URL is not saved until this succeeds. A `401` means the key is wrong or
missing. A network error means the URL, CORS, firewall or Tailscale route is wrong.

### Save connection

- Local URL: stores the key and removes the remote URL override.
- Remote URL: requires a key and a successful Test, then stores both and reloads.
- The header then shows **CLIENT** if the URL is non-local.

### Reset to local

Removes `DIFFUSIONBEAR_API_BASE` and `DIFFUSIONBEAR_API_KEY` from this browser/app
profile and reloads. It does not change the server's keys or API mode.

---

## 4. API keys

### Environment key

The `DIFFUSIONBEAR_API_KEY` value of the running backend process.

- **Read-only in the UI.** The UI cannot rotate or delete it.
- Always valid while the process has that environment variable.
- It remains valid even if every managed key is revoked.
- The card shows only a masked prefix and last 4 characters.
- To change it: stop the backend, change the environment variable, restart.

This is the key used by most deployments started from a terminal.

### Legacy key

The old plaintext `"api_key"` field inside `security.json`.

```json
{
  "api_key": "old-shared-key"
}
```

- Accepted for backward compatibility.
- Read-only in the UI.
- Preserved during every new key write.
- It exists only if an older setup or manual edit created it.
- New deployments should use a managed key or the environment key instead.
- To remove it, edit `security.json` with the backend stopped and keep at least
  one other usable key.

It is called “legacy” because it is a single plaintext shared key, not a named
managed key.

### Managed keys

Named keys created from this page. Recommended for agents, phones, tablets and
separate browser profiles.

Each row shows:

- **Label** — human name, for example `iPhone`, `Fred laptop`, `render agent`.
- **Masked key** — prefix and last 4 characters only.
- **Created** — creation timestamp.
- **Last used** — last authenticated request, flushed at most once per minute.
- **This browser** — this key is the one currently stored in this browser.
- **Revoke** — deletes that key immediately.

### Create key

1. Enter a unique label.
2. Click **Create key**.
3. The full key is shown once.
4. Copy it or click **Use in this browser**.
5. Close the dialog. The server keeps only the SHA-256 hash.

If this is the first usable key, the key is applied to this browser
automatically before the dialog can close. That prevents locking yourself out.

The full key cannot be recovered later. If lost, revoke it and create another.

### Revoke

- Revokes only the selected managed key.
- Other keys and the environment key remain valid.
- The final usable key cannot be revoked: the backend returns `409`.
- Revoking the key used by this browser requires another key to be available.
  The UI warns first.

---

## 5. Common workflows

### Local desktop only

1. API mode OFF.
2. Backend URL empty.
3. No key needed on loopback.
4. Keys may exist but are inactive.

### Local backend with API mode ON

1. Create a managed key, or set `DIFFUSIONBEAR_API_KEY`.
2. Enable API mode.
3. Keep the key in this browser with **Use in this browser**.
4. External clients must send `Authorization: Bearer <key>`.

### Tailscale from iPhone/iPad (or any single web browser)

`DiffusionBear.app` always launches its local backend on `127.0.0.1:8001` (`main.swift`). To generate remotely from another Mac, PC, Windows laptop, Linux desktop, iPhone, or iPad **using only one web browser** (no client install):

1. **On the host Mac (one-time setup):**
   - Open **Parameters → API → Managed keys**, create a key per browser/device (for instance `Chrome Laptop` or `Safari iPad`), copy it once, and click **Use in this browser** so the Mac browser keeps access.
   - Flip **API mode** to **ON**. The top bar shows the **API** badge.
   - Expose local loopback safely over HTTPS via Tailscale Serve:
     ```bash
     tailscale serve --bg --https=443 http://127.0.0.1:8001
     tailscale serve status
     ```
     *(Prefer direct plain-HTTP LAN access without Tailscale? Run from CLI instead of the `.app`: `cd backend && DIFFUSIONBEAR_API_KEY="dbk_or_16char_secret" ../venv/bin/python main.py --host 0.0.0.0 --port 8001`. Add `DIFFUSIONBEAR_ALLOWED_HOSTS` if connecting via a hostname instead of an IP literal.)*
2. **On the remote browser:**
   - Open the HTTPS URL (`https://<mac-name>.<tailnet>.ts.net`) or direct LAN IP (`http://192.168.1.20:8001`).
   - Open **Parameters → API → Backend connection**.
   - **Leave `Backend URL` EMPTY** (`Empty = local`). When the browser already loaded the page from the remote Mac, empty means *"send API calls back to the server serving this tab"* — keeping everything same-origin.
   - Paste the device's API key into **API key**, click **Test** (verifies `GET /api/version` + `GET /api/models`), then click **Save connection**.
   - Start generating, browsing, or editing images from that single tab.

#### Browser-specific notes (Chrome, Firefox, Opera, Safari)

- **Google Chrome (Desktop / Android):**
  - Prefer **`https://<mac>.ts.net`** via Tailscale Serve so HTTPS avoids Chrome's **Mixed Content** and private-network blocks.
  - On plain HTTP (`http://192.168.x.x:8001`), accept Chrome's **Local network access** prompt when connecting on the same Wi-Fi/LAN.
  - The API key is stored per Chrome profile in `localStorage["DIFFUSIONBEAR_API_KEY"]`. Separate profiles stay isolated; Incognito forgets the key when the last window closes.
- **Mozilla Firefox (Desktop / Android):**
  - Open **`https://<mac>.ts.net`** or an explicit numeric IP (**`192.168.x.x`** or **`100.x.y.z`**). If Firefox **DNS over HTTPS (DoH)** is enabled, Bonjour `.local` hostnames often fail to resolve.
  - Ensure **Delete cookies and site data when Firefox is closed** is not purging your DiffusionBear URL (or add an exception), otherwise you will need to paste the key again next launch.
- **Opera (Desktop / Mobile):**
  - **Turn OFF Opera's built-in VPN** before opening a LAN (`192.168.x.x`) or Tailscale (`100.x.y.z` / `.ts.net`) URL: Opera VPN routes browser traffic through Opera's external proxy, making your local Mac unreachable.
  - Follow the same Chromium HTTPS / Mixed Content rules as Chrome above.
- **Apple Safari (macOS / iPhone / iPad):**
  - Pair the Tailscale app on your iPhone/iPad with **`https://<mac>.ts.net`** (or `http://<LAN-IP>:8001` on trusted home Wi-Fi), open **Parameters → API → Backend connection**, and leave **Backend URL EMPTY**. Every `/api/*` request and `/api/images/{id}/file?token=...` thumbnail remains strictly same-origin, so Safari's **Prevent Cross-Site Tracking (ITP)** never strips or blocks gallery images.
  - Paste your `dbk_...` key into **API key**, tap **Test**, then **Save connection**.
  - On iPhone or iPad, tap **Share → Add to Home Screen** to run DiffusionBear as a full-screen standalone web app (note: iOS Home Screen web apps use their own separate `localStorage`, so paste and save the key once inside the Home Screen app if you add it after configuring Safari).

### How not to lock yourself out (and how to recover via `security.json`)

When **API mode** is **ON**, *every* client—including another browser on the same Mac—must send a valid key. Because `localStorage` is isolated per browser profile and per app (`DiffusionBear.app`, Safari, Chrome, Brave, and iOS Home Screen web apps do **not** share keys), enabling API mode in one window will prompt for a key in any other browser.

- **Before enabling API mode:** create a managed key (`dbk_...`), click **Use in this browser**, and save a copy of the secret in your password manager (it is shown once at creation and stored on disk only as a SHA-256 hash).
- **If you are locked out on your own Mac:** open `security.json` and change `"enabled": true` to `"enabled": false`. Your existing `api_keys` stay intact; API mode simply turns OFF so `127.0.0.1` loopback works immediately without a key.
  - Packaged `DiffusionBear.app`: `~/Library/Application Support/DiffusionBear/data/security.json`
  - Source checkout: `backend/data/security.json`

```bash
# Quick Terminal unlock for DiffusionBear.app (keeps your saved keys)
python3 -c 'import json, pathlib; p = pathlib.Path.home() / "Library/Application Support/DiffusionBear/data/security.json"; d = json.loads(p.read_text()); d["enabled"] = False; p.write_text(json.dumps(d, indent=2))'
```

### Local app UI, remote backend (`CLIENT` mode)

1. Open your **local** DiffusionBear app, go to **Parameters → API → Backend connection**.
2. Enter the remote backend URL (`https://<mac>.ts.net` or `http://192.168.1.20:8001`).
3. Paste its API key.
4. Click **Test**; it must succeed (`200 OK` on `/api/version` and `/api/models`).
5. Click **Save connection**.
6. The header shows **CLIENT** and routes API calls and gallery image requests to the remote Mac.
*(Note: Cross-origin CORS on the remote Mac allows Vite dev ports `:5173/:5174` and same-origin requests; if another origin is blocked by browser CORS, open the remote URL directly in your browser tab and leave `Backend URL` empty instead.)*

### Quick AI Agent / `curl` example

```bash
export DB_URL="https://mac.tailnet.ts.net"   # or http://127.0.0.1:8001
export DB_KEY="dbk_your_managed_or_env_key"

# 1. Queue one generation
curl -sS -X POST "$DB_URL/api/generate" \
  -H "Authorization: Bearer $DB_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"flux2-klein-4b","prompt":"wooden cabin at dusk","width":512,"height":768,"steps":4}'

# 2. Poll job status (X-API-Key alias also works)
curl -sS -H "X-API-Key: $DB_KEY" "$DB_URL/api/jobs/<job_id>"

# 3. Download generated PNG bytes
curl -sS -H "Authorization: Bearer $DB_KEY" "$DB_URL/api/images/<image_id>/file" -o output.png
```

The local UI still needs a small local process to render; it does not load local
models in client mode.

---

## 6. Troubleshooting

| Symptom | Meaning | Fix |
|---|---|---|
| API mode switch returns 409 | No strong usable key exists | Create a managed key or set `DIFFUSIONBEAR_API_KEY` |
| `401 invalid or missing API key` | Target backend has API mode ON and the key is missing/wrong | Paste the correct key, or put it in the Authorization header |
| Test fails with network error | URL unreachable, CORS, firewall or Tailscale route | Check URL, Tailscale status, reverse proxy, CORS origin |
| Test returns 401 | Key rejected | Check key source; environment key and managed keys are distinct |
| CLIENT badge missing after save | Backend URL is empty or loopback | Enter a non-local URL and pass Test |
| Environment key card says “Not set” | Backend has no `DIFFUSIONBEAR_API_KEY` | Restart backend with the variable |
| Legacy key card says “Not set” | `security.json` has no `api_key` field | Normal for new installs |
| Full key cannot be shown again | By design: only a hash is stored | Revoke and create a new key |

---

## 7. Security rules

- Never put a full key in a URL except the narrow media `?token=` exception.
- Prefer `Authorization: Bearer` for agents.
- Use TLS or a private Tailscale network; plain HTTP on a LAN is sniffable.
- `security.json` is written atomically with mode `0600`.
- Managed keys are stored as SHA-256 only.
- API mode OFF never means “keys deleted”; it means “keys inactive”.
- The environment key is immutable from the UI. Rotation requires a restart.
