#!/usr/bin/env bash
# Exercises shell/PortProbe.swift against real sockets.
#
# Both halves of the decision matter and only one is testable with fakes:
#   * a rung held by something ELSE must be stepped over
#   * a rung already serving OUR backend must be REUSED, or launching the app twice
#     leaves two engines resident with two copies of the model in memory
#
# The second needs a genuine /api/version answering with our name, so the script
# starts the real backend on 9025 and occupies 8001 with a fake 404.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d)"
cleanup() { [ -n "${FAKE_PID:-}" ] && kill "$FAKE_PID" 2>/dev/null || true
           [ -n "${BE_PID:-}" ] && kill "$BE_PID" 2>/dev/null || true
           rm -rf "$WORK"; }
trap cleanup EXIT

echo "==> occupying 8001 with a server that answers 404"
python3 "$HERE/fake_listener.py" 8001:notfound > "$WORK/fake.log" 2>&1 &
FAKE_PID=$!

echo "==> starting the real backend on 9025"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-$REPO/venv/bin/python}"
STORE="${DIFFUSIONBEAR_ASSET_DIR:-$(dirname "$(dirname "$REPO")")}"
# cwd must be backend/: uvicorn imports `main:app` relative to where it runs.
( cd "$REPO/backend" && env DIFFUSIONBEAR_ASSET_DIR="$STORE" DIFFUSIONBEAR_DATA_DIR="$STORE" \
  "$PY" -u -m uvicorn main:app --host 127.0.0.1 --port 9025 --no-access-log \
  > "$WORK/backend.log" 2>&1 ) &
BE_PID=$!

for _ in $(seq 1 40); do
  sleep 1
  curl -sf --max-time 2 http://127.0.0.1:9025/api/version > /dev/null && break
done
curl -sf --max-time 2 http://127.0.0.1:9025/api/version > /dev/null || {
  echo "backend did not come up on 9025" >&2; tail -20 "$WORK/backend.log" >&2; exit 1; }

echo "==> compiling and running the probe test against shell/PortProbe.swift"
mkdir -p "$WORK/src"
cp "$HERE/port_probe_test.swift" "$WORK/src/main.swift"
swiftc -O -target arm64-apple-macos15.0 \
  -o "$WORK/probe" "$REPO/packaging/shell/PortProbe.swift" "$WORK/src/main.swift"
"$WORK/probe"
