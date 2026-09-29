#!/usr/bin/env bash
# Build the SDXL engine runtime that ships alongside the main runtime.
#
# This is the optional second engine (Juggernaut XL Lightning) and it does not need
# torch at inference time: mlx_diffuser contains zero `import torch` statements (the
# 22 "torch" hits in its source are all `pytorch_compatible=True` kwargs and
# comments), and sdxl_engine.py imports it 0 times. It DOES need transformers, but
# only for the CLIP tokenizer -- see the install step below for why that is not the
# same thing as needing torch.
#
# Consequence: adding a NEW SDXL checkpoint to the app needs the dev tree's
# venv-sdxl (which has torch/diffusers/transformers/accelerate) to convert it.
# That is deliberate, and it is the only capability left out of the bundle.
set -euo pipefail

HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-$HERE/build}"
PBS_TGZ="/tmp/pbs314.tar.gz"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20260929/cpython-3.14.7+20260929-aarch64-apple-darwin-install_only.tar.gz"
MLX_DIFFUSER_REF="a26b42aee4e31999dbb4429226b66d896d49e1d8"
PYVER="3.14.7"

say() { printf '\n\033[1m==>\033[0m %s\n' "$*"; }
mkdir -p "$STAGE"

SDXLDIR="$STAGE/sdxl"
mkdir -p "$SDXLDIR"

if [ ! -x "$SDXLDIR/python/bin/python3" ]; then
  say "fetching python-build-standalone $PYVER"
  [ -f "$PBS_TGZ" ] || curl -sL --max-time 600 -o "$PBS_TGZ" "$PBS_URL"
  say "extracting interpreter into sdxl/"
  tar xzf "$PBS_TGZ" -C "$SDXLDIR"
  test -x "$SDXLDIR/python/bin/python3" || { echo "interpreter missing" >&2; exit 1; }
fi
PBS_PY="$SDXLDIR/python/bin/python3"
say "interpreter: $("$PBS_PY" -V)"

if [ ! -x "$SDXLDIR/venv/bin/python" ]; then
  say "creating venv"
  "$PBS_PY" -m venv "$SDXLDIR/venv"
fi
VPY="$SDXLDIR/venv/bin/python"

say "upgrading pip"
"$VPY" -m pip install --quiet --upgrade pip setuptools wheel

say "installing mlx_diffuser from git @ $MLX_DIFFUSER_REF"
"$VPY" -m pip install --quiet \
  "git+https://github.com/AmirHossein-razlighi/mlx_diffuser.git@$MLX_DIFFUSER_REF"

say "installing the rest of the SDXL runtime"
# transformers IS required even though nothing imports torch. mlx_diffuser's SDXL
# pipeline needs transformers' CLIPTokenizer, and an SDXL run without it fails with
# "The SDXL pipeline needs \`transformers\` for tokenization". It is safe here because
# only the tokenizer is used: transformers 5.x prints "PyTorch was not found. Models
# won't be available and only tokenizers, configuration and file/data utilities can be
# used" and works fine in that mode, while the UNet runs in MLX. Verified end to end:
# a 4-step juggernaut-xl-lightning render from the assembled bundle.
#
# torch/diffusers/accelerate are still excluded -- they exist in the dev tree's
# venv-sdxl only to convert new checkpoints to diffusers format, which the bundle
# does not do. Adding a NEW SDXL checkpoint needs the dev tree for the conversion.
"$VPY" -m pip install --quiet \
  "mlx==0.32.1" "pillow>=10.0.0" "pillow-heif>=1.7.0" \
  "safetensors>=0.4.0" "numpy>=1.26.0" "transformers>=4.40.0"

say "verifying: torch must be ABSENT here, and the SDXL tokenizer path must work"
"$VPY" - <<'PY'
import importlib.util, sys
if importlib.util.find_spec("torch") is not None:
    sys.exit("FAIL: torch is installed; the SDXL runtime is meant to be torch-free")
import mlx.core as mx
import mlx_diffuser
print(f"  mlx           device={mx.default_device()}")
print("  mlx_diffuser  ok")
# The thing that actually broke SDXL: the tokenizer import must resolve without torch.
from transformers import CLIPTokenizer
print("  transformers  CLIPTokenizer importable (tokenizer-only, no torch)")
PY

say "verifying relocatability"
COPY="$SDXLDIR/relocate-test"
rm -rf "$COPY"; mkdir -p "$COPY"
cp -R "$SDXLDIR/python" "$COPY/python"
cp -R "$SDXLDIR/venv" "$COPY/venv"
printf 'home = %s\ninclude-system-site-packages = false\nversion = %s\n' \
  "$COPY/python/bin" "$PYVER" > "$COPY/venv/pyvenv.cfg"
( cd / && "$COPY/venv/bin/python" -c "import mlx_diffuser, mlx.core; print('  relocated import OK')" )
rm -rf "$COPY"

say "sdxl runtime build complete"
du -sh "$SDXLDIR/venv" | sed 's/^/  /'
