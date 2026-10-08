#!/usr/bin/env bash
# Build the relocatable, torch-free Python runtime that ships inside
# DiffusionBear.app.
#
# Why python-build-standalone (pbs) and not a copy of venv/:
#   venv/pyvenv.cfg says `home = /opt/homebrew/opt/python@3.10/bin`, so the
#   existing venv needs Homebrew installed and is not relocatable. 38 of its
#   console scripts also carry shebangs containing the space in
#   "DiffusionBear", which macOS truncates at the first space -- the
#   documented cause of the app silently booting under a stale interpreter.
#   pbs has no absolute paths (resolves its stdlib from its own executable), so
#   the whole tree can live inside an .app at a space-free path.
#
# torch is NOT optional. mflux's mflux/models/common/weights/loading/weight_loader.py
# imports it at module level, and that module sits on the inference path because
# checkpoints are read through it. Measured: importing main:app with torch blocked
# succeeds (12 routes), but a real 1-step generation raises ImportError. An earlier
# note here claimed torch was 511 MB of dead weight on the strength of the import
# test alone; that was wrong and generation is the test that counts.
#
# NEVER set PYTHONHOME here or when spawning the engine: it leaks into every
# grandchild process and kills interpreters that are not this one. PYTHONPATH is
# safe because unrelated interpreters ignore foreign-ABI extensions.
set -euo pipefail

HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-$HERE/build}"
PBS_TGZ="${PBS_TGZ:-/tmp/pbs.tar.gz}"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20260929/cpython-3.10.21+20260929-aarch64-apple-darwin-install_only.tar.gz"

PYVER="3.10.21"

say() { printf '\n\033[1m==>\033[0m %s\n' "$*"; }

mkdir -p "$STAGE"

# ---------------------------------------------------------------- pbs
if [ ! -x "$STAGE/python/bin/python3" ]; then
  say "fetching python-build-standalone $PYVER (aarch64-apple-darwin)"
  if [ ! -f "$PBS_TGZ" ]; then
    curl -sL --max-time 600 -o "$PBS_TGZ" "$PBS_URL"
  fi
  say "extracting interpreter"
  tar xzf "$PBS_TGZ" -C "$STAGE"
  test -x "$STAGE/python/bin/python3" || { echo "interpreter missing after extract" >&2; exit 1; }
fi
PBS_PY="$STAGE/python/bin/python3"
say "interpreter: $("$PBS_PY" -V)  ($PBS_PY)"

# ---------------------------------------------------------------- venv
if [ ! -x "$STAGE/venv/bin/python" ]; then
  say "creating venv from pbs"
  "$PBS_PY" -m venv "$STAGE/venv"
fi
VPY="$STAGE/venv/bin/python"

# ---------------------------------------------------------------- packages
say "upgrading pip"
"$VPY" -m pip install --quiet --upgrade pip setuptools wheel

# Backend requirements, minus mflux (installed from the vendored tree so the
# Qwen-Image 2.1 port is present) and minus torch (absent from requirements.txt
# already; it only ever arrived as an mflux dependency).
say "installing backend runtime dependencies"
"$VPY" -m pip install --quiet \
  "fastapi>=0.115.0" "uvicorn[standard]>=0.30.0" "pydantic>=2.0.0" \
  "python-multipart>=0.0.12" "pillow>=10.0.0" "pillow-heif>=1.7.0" \
  "safetensors>=0.4.0" "numpy>=1.26.0" "huggingface-hub>=0.25.0" \
  "mlx==0.32.1" "mlx-lm==0.31.3" "mlx-taef==0.8.1" "mlx-teacache==0.11.0" \
  "torch>=2.13.0,<3.0" \
  "diffusers>=0.40.0" "omegaconf>=2.3.0" "accelerate>=1.0.0"

say "installing vendored mflux (Qwen-Image 2.1 port)"
# pip installs from a directory in place, which drops build/ and mflux.egg-info/ INTO
# that directory. That silently mutates the vendored fork: it doubles the file count
# (816 -> 1617) and changes its digest, so backend/test_repo_security.py fails, and the
# next bundle ships the artefacts too. Build from a throwaway copy and clean up after.
_VENDOR_COPY="$(mktemp -d)/mflux-src"
cp -R "$HERE/vendor/mflux-src" "$_VENDOR_COPY"
trap 'rm -rf "$(dirname "$_VENDOR_COPY")"' EXIT
"$VPY" -m pip install --quiet "$_VENDOR_COPY"
rm -rf "$HERE/vendor/mflux-src/build" "$HERE/vendor/mflux-src"/*.egg-info

# ---------------------------------------------------------------- verify
# diffusers/omegaconf/accelerate are for the single-file -> diffusers checkpoint
# conversion in backend/sdxl_convert.py. Without them the shipped app reports
# "conversion unavailable" even though a source checkout can convert: this venv is built
# from the list above, not from whatever happens to be installed in the repo venv.
say "verifying: mlx + mflux + the qwen21 port must import, and torch must be present"
"$VPY" - <<'PY'
import importlib, importlib.util, sys
if importlib.util.find_spec("torch") is None:
    sys.exit("FAIL: torch is missing; mflux's weight loader needs it at inference time")
import torch
import mlx.core as mx
for _dep, _why in (
    ("diffusers", "single-file -> diffusers checkpoint conversion"),
    ("omegaconf", "diffusers from_single_file"),
    ("accelerate", "diffusers from_single_file"),
):
    if importlib.util.find_spec(_dep) is None:
        sys.exit(f"FAIL: {_dep} is missing; {_why} would be unavailable in the shipped app")
print(f"  torch     {torch.__version__}")
print("  diffusers present (checkpoint conversion enabled)")
print(f"  mlx       device={mx.default_device()}")
import mflux
print("  mflux     ok")
mod = importlib.import_module("mflux.models.qwen21.variants.txt2img.qwen_image_21")
print("  qwen21    port importable:", mod.__name__)
from fastapi import FastAPI
print("  fastapi   ok")
PY

# The pbs interpreter must not need PYTHONHOME. Prove relocation by running it
# from a copy at a different, space-free path.
say "verifying relocatability: running from a copy at a different path"
COPY="$STAGE/relocate-test"
rm -rf "$COPY"; mkdir -p "$COPY"
cp -R "$STAGE/python" "$COPY/python"
cp -R "$STAGE/venv" "$COPY/venv"
# venvs embed their absolute home; rewrite it to the copy so the interpreter
# resolves the relocated stdlib.
printf 'home = %s\ninclude-system-site-packages = false\nversion = %s\n' \
  "$COPY/python/bin" "$PYVER" > "$COPY/venv/pyvenv.cfg"
( cd / && "$COPY/venv/bin/python" -c "import mlx.core, mflux, fastapi; print('  relocated import OK from', '$COPY')" )
rm -rf "$COPY"

say "runtime build complete"
du -sh "$STAGE/venv" | sed 's/^/  /'
"$VPY" -m pip list --format=freeze | wc -l | sed 's/^/  packages: /'
