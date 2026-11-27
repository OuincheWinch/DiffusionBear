#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"

# Honour the folders chosen in Settings > Workspace (the same pointer files
# the macOS app reads), unless the environment already sets them.
WORKSPACE_POINTERS="$HOME/Library/Application Support/DiffusionBear"
for pair in DIFFUSIONBEAR_DATA_DIR:data_path DIFFUSIONBEAR_ASSET_DIR:asset_path; do
    var="${pair%%:*}"
    file="$WORKSPACE_POINTERS/${pair#*:}"
    if [ -z "$(printenv "$var" || true)" ] && [ -f "$file" ]; then
        dir="$(head -n 1 "$file" | tr -d '\r')"
        if [[ "$dir" == /* ]] && [ -d "$dir" ]; then
            export "$var=$dir"
        fi
    fi
done

# Reuse a sibling checkout's model store when present.
if [[ -z "${DIFFUSIONBEAR_ASSET_DIR:-}" ]]; then
    SHARED_ASSET_DIR="$PROJECT_DIR/../DiffusionBear/backend/data"
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
