#!/usr/bin/env bash
# DiffusionBear REST quickstart — curl + python3 (for JSON parsing) only.
#
#   export DB_URL=http://127.0.0.1:8001      # LAN: http://<mac-ip>:8001 · Tailscale: https://mac.tailnet.ts.net
#   export DB_KEY=dbk_...                    # leave empty on localhost when API mode is OFF
#   bash docs/examples/quickstart.sh "a lighthouse at dusk, oil painting"
set -euo pipefail

DB_URL="${DB_URL:-http://127.0.0.1:8001}"
PROMPT="${1:-a small red apple on a white table}"
OUT="${OUT:-out.png}"
AUTH=()
[ -n "${DB_KEY:-}" ] && AUTH=(-H "Authorization: Bearer $DB_KEY")
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

# 1. Health check (public, no key needed)
curl -fsS "$DB_URL/api/version" | json '"DiffusionBear", d["version"]'

# 2. Pick an installed model (or force one with MODEL=flux2-klein-4b)
MODEL="${MODEL:-$(curl -fsS ${AUTH[@]+"${AUTH[@]}"} "$DB_URL/api/models" \
  | json 'next((m["id"] for m in d if m.get("installed")), "")')}"
[ -n "$MODEL" ] || { echo "No installed model — download one in the Models tab." >&2; exit 1; }
echo "model: $MODEL"

# 3. Queue the job. Send `model`, never `base_model` (422).
BODY=$(PROMPT="$PROMPT" MODEL="$MODEL" python3 -c 'import json,os; print(json.dumps({
  "model": os.environ["MODEL"], "prompt": os.environ["PROMPT"],
  "width": 512, "height": 768, "seed": 42}))')
JOB=$(curl -fsS ${AUTH[@]+"${AUTH[@]}"} -H 'Content-Type: application/json' -d "$BODY" \
  "$DB_URL/api/generate" | json 'd["job_id"]')
echo "job: $JOB   (cancel: curl -X POST ${DB_URL}/api/jobs/$JOB/cancel -H 'Authorization: Bearer …')"

# 4. Poll — the API has no SSE/WebSocket; 1–2 s is plenty.
while :; do
  STATE=$(curl -fsS ${AUTH[@]+"${AUTH[@]}"} "$DB_URL/api/jobs/$JOB")
  STATUS=$(echo "$STATE" | json 'd["status"]')
  echo "$STATE" | json '"  %s step %s/%s" % (d["status"], (d.get("progress") or {}).get("step","-"), (d.get("progress") or {}).get("steps","-"))'
  case "$STATUS" in done) break ;; error|cancelled) echo "$STATE" | json 'd.get("error")' >&2; exit 1 ;; esac
  sleep 2
done

# 5. Download the image and its JSON sidecar (prompt, seed, timings…)
IMG=$(echo "$STATE" | json 'd["results"][0]["id"]')
curl -fsS ${AUTH[@]+"${AUTH[@]}"} "$DB_URL/api/images/$IMG/file" -o "$OUT"
case "$(basename "$OUT")" in *.*) SIDECAR="${OUT%.*}.json" ;; *) SIDECAR="$OUT.json" ;; esac
curl -fsS ${AUTH[@]+"${AUTH[@]}"} "$DB_URL/api/images/$IMG" -o "$SIDECAR"
echo "saved $OUT and $SIDECAR"
