#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"

# Private DEV convenience: reuse the sibling working copy's model store.
if [[ -z "${DIFFUSIONBEAR_ASSET_DIR:-}" ]]; then
    SHARED_ASSET_DIR="$PROJECT_DIR/../MLX-DIFFUSION OpenCode/backend/data"
    if [[ -d "$SHARED_ASSET_DIR/models" ]]; then
        export DIFFUSIONBEAR_ASSET_DIR="$SHARED_ASSET_DIR"
    fi
fi

VENV_PY=""
for candidate in "$PROJECT_DIR/venv/bin/python" "$PROJECT_DIR/backend/venv/bin/python"; do
    if [[ -x "$candidate" ]]; then
        VENV_PY="$candidate"
        break
    fi
done
if [[ -z "$VENV_PY" ]]; then
    print -u2 "No backend venv found. Run setup first."
    exit 1
fi
cd "$PROJECT_DIR/backend"
exec caffeinate -s "$VENV_PY" -m uvicorn main:app --reload --port 8001
