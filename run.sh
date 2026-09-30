#!/usr/bin/env bash
set -euo pipefail

SILENT=0
for arg in "$@"; do
    case "$arg" in
        -s|--silent) SILENT=1 ;;
        *) printf 'Usage: %s [-s|--silent]\n' "$0" >&2; exit 2 ;;
    esac
done

say() { printf '%s\n' "$*"; }
say_n() { printf '%s' "$*"; }

# Resolve the project root from this script's own location, so run.sh works
# from any current directory and any install path (resolves symlinks too).
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
PROJECT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"

# Private DEV convenience: reuse the sibling working copy's model/SDXL/LoRA store
# so DEV never re-downloads weights. Never published to the public repo.
if [ -z "${DIFFUSIONBEAR_ASSET_DIR:-}" ] && [ -z "${MLX_DIFFUSION_ASSET_DIR:-}" ]; then
    SHARED_ASSET_DIR="$PROJECT_DIR/../MLX-DIFFUSION OpenCode/backend/data"
    if [ -d "$SHARED_ASSET_DIR/models" ]; then
        export DIFFUSIONBEAR_ASSET_DIR="$SHARED_ASSET_DIR"
    fi
fi

# Locate a usable Python venv (prefer repo venv, then venv-sdxl isn't one for the API).
# NOTE: use venv/bin/python (not the venv/bin/uvicorn console script) — console-script
# shebangs can point at a stale no-space venv path after a folder rename, silently
# booting the whole app under the WRONG (old) python/mflux. Always launch via `-m`.
VENV_PY=""
for cand in "$PROJECT_DIR/venv/bin/python" "$PROJECT_DIR/backend/venv/bin/python"; do
    if [ -x "$cand" ]; then
        VENV_PY="$cand"
        break
    fi
done
if [ -z "$VENV_PY" ]; then
    echo "❌ No venv found (looked for $PROJECT_DIR/venv/bin/python). Run setup first."
    exit 1
fi

APP_VERSION="$("$VENV_PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); import app_version; print(app_version.APP_VERSION_LABEL)' "$BACKEND_DIR" 2>/dev/null || true)"
if [ -z "$APP_VERSION" ]; then
    APP_VERSION="unknown"
fi

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    say "⚠️  Frontend deps not installed — run: cd \"$FRONTEND_DIR\" && npm install"
fi

say "================================================="
say "        🚀 Starting MLX-DIFFUSION Studio — $APP_VERSION"
say "================================================="
say "  project: $PROJECT_DIR"
say "  backend: $BACKEND_DIR"
say "  frontend: $FRONTEND_DIR"
say "  python : $VENV_PY"

BACKEND_PID=""
FRONTEND_PID=""
EVENT_PID=""
EVENT_LOG=""
API_TOKEN_VALUE="${DIFFUSIONBEAR_API_TOKEN:-${MLX_DIFFUSION_API_TOKEN:-${MLX_API_TOKEN:-${LOCAL_API_TOKEN:-}}}}"
if [ -n "$API_TOKEN_VALUE" ] && [ -z "${VITE_API_TOKEN:-}" ]; then
    export VITE_API_TOKEN="$API_TOKEN_VALUE"
fi
OWNED_PIDS=()

collect_process_tree() {
    local pid="$1"
    local child
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        collect_process_tree "$child"
    done
    OWNED_PIDS+=("$pid")
}

cleanup() {
    local status=$?
    trap - SIGINT SIGTERM EXIT
    if [ "$SILENT" -eq 1 ]; then
        exec 2>/dev/null
    fi
    OWNED_PIDS=()
    for pid in "$BACKEND_PID" "$FRONTEND_PID" "$EVENT_PID"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            collect_process_tree "$pid"
        fi
    done
    for pid in "${OWNED_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for _ in {1..20}; do
        local alive=0
        for pid in "${OWNED_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive=1
                break
            fi
        done
        [ "$alive" -eq 0 ] && break
        sleep 0.25
    done
    for pid in "${OWNED_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    wait "$BACKEND_PID" 2>/dev/null || true
    wait "$FRONTEND_PID" 2>/dev/null || true
    wait "$EVENT_PID" 2>/dev/null || true
    if [ -n "$EVENT_LOG" ]; then
        rm -f "$EVENT_LOG" 2>/dev/null || true
    fi
    say ""
    say "Processes stopped."
    exit "$status"
}

if command -v lsof >/dev/null 2>&1; then
    if lsof -nP -iTCP:8001 -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Port 8001 is already in use. Stop that listener and retry."
        exit 1
    fi
    if lsof -nP -iTCP:5174 -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Port 5174 is already in use. Stop that listener and retry."
        exit 1
    fi
fi

trap cleanup SIGINT SIGTERM EXIT

# 2. Launch Backend
if [ "$SILENT" -eq 1 ]; then
    EVENT_LOG="$(mktemp "${TMPDIR:-/tmp}/diffusionbear-events.XXXXXX")"
    : > "$EVENT_LOG"
    tail -F "$EVENT_LOG" &
    EVENT_PID=$!
    export DIFFUSIONBEAR_EVENT_LOG="$EVENT_LOG"
    say "📦 Starting FastAPI backend on http://127.0.0.1:8001..."
    cd "$BACKEND_DIR"
    caffeinate -s "$VENV_PY" -m uvicorn main:app --reload --port 8001 --no-access-log --log-level info &
else
    unset DIFFUSIONBEAR_EVENT_LOG || true
    say "📦 Starting FastAPI backend on http://127.0.0.1:8001..."
    cd "$BACKEND_DIR"
    caffeinate -s "$VENV_PY" -m uvicorn main:app --reload --port 8001 &
fi
BACKEND_PID=$!

# Wait for backend to be ready
say_n "⏳ Waiting for backend to initialize..."
BACKEND_READY=0
for _ in {1..30}; do
    READY=1
    if [ -n "$API_TOKEN_VALUE" ]; then
        curl -s -H "X-MLX-API-Token: $API_TOKEN_VALUE" http://127.0.0.1:8001/api/models >/dev/null 2>&1 || READY=0
    else
        curl -s http://127.0.0.1:8001/api/models >/dev/null 2>&1 || READY=0
    fi
    if [ "$READY" -eq 1 ]; then
        say " Ready!"
        BACKEND_READY=1
        break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        break
    fi
    sleep 0.5
    say_n "."
done
if [ "$BACKEND_READY" -ne 1 ]; then
    say ""
    say "Backend failed to become ready."
    exit 1
fi

# 3. Launch Frontend
say "💻 Starting Vite frontend on http://localhost:5174..."
cd "$FRONTEND_DIR"
npm run dev >/dev/null 2>&1 &
FRONTEND_PID=$!

sleep 1.5

say ""
say "================================================="
say "  🎨 MLX-DIFFUSION is LIVE!                      "
say "  👉 Web UI:        http://localhost:5174         "
say "  👉 API Docs:      http://localhost:8001/docs    "
say "  (Press Ctrl+C to stop all services)            "
say "================================================="
say ""

# Automatically open UI in default browser on macOS
if command -v open >/dev/null 2>&1; then
    open http://localhost:5174 2>/dev/null || true
fi

# Keep script running and wait for background processes
wait "$BACKEND_PID" "$FRONTEND_PID"