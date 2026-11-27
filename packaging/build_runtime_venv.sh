#!/usr/bin/env bash
# Build the relocatable, torch-free Python runtime that ships inside
# DiffusionBear.app.
#
# Why python-build-standalone (pbs) and not a copy of venv/:
#   venv/pyvenv.cfg says `home = /opt/homebrew/opt/python@3.13/bin`, so the
#   existing venv needs Homebrew installed and is not relocatable. 38 of its
#   console scripts also carry shebangs containing the space in
#   "DiffusionBear", which macOS truncates at the first space -- the
#   documented cause of the app silently booting under a stale interpreter.
#   pbs has no absolute paths (resolves its stdlib from its own executable), so
#   the whole tree can live inside an .app at a space-free path.
#
# PyTorch, diffusers, and accelerate are uninstalled below (and verified absent at
# the end of this script): every built-in generation engine runs directly on Apple
# MLX, safetensors, NumPy, transformers, and tokenizers without torch on the path.
# Raw single-file Civitai SDXL checkpoints still require external diffusers/torch if
# converted out-of-process via backend/sdxl_convert.py.
#
# NEVER set PYTHONHOME here or when spawning the engine: it leaks into every
# grandchild process and kills interpreters that are not this one. PYTHONPATH is
# safe because unrelated interpreters ignore foreign-ABI extensions.
set -euo pipefail

HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-$HERE/build}"
# Cache lives under the build stage, never /tmp: a predictable path plus
# skip-if-present is how a pre-planted archive wins. PBS_TGZ stays overridable
# for vendored builds -- any file found there is hash-checked below, so an
# override cannot smuggle an unverified interpreter in.
PBS_TGZ="${PBS_TGZ:-$STAGE/.cache/pbs-3.13.tar.gz}"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261110/cpython-3.13.18+20261110-aarch64-apple-darwin-install_only.tar.gz"
PBS_SHA256="1afd3983237c2a10277345f235be730e9cb49a123e0574ed252dbf3a841a24e4"
# shellcheck disable=SC1091
source "$HERE/pbs_fetch.sh"

PYVER="3.13.18"

say() { printf '\n\033[1m==>\033[0m %s\n' "$*"; }

mkdir -p "$STAGE"

# ---------------------------------------------------------------- pbs
# A stage reused across a PYVER bump keeps the old interpreter and its venv.
if [ -x "$STAGE/python/bin/python3" ] &&
   [ "$("$STAGE/python/bin/python3" -c 'import platform; print(platform.python_version())')" != "$PYVER" ]; then
  say "stage interpreter is not $PYVER: discarding stale python + venv"
  rm -rf "$STAGE/python" "$STAGE/venv"
fi
if [ ! -x "$STAGE/python/bin/python3" ]; then
  say "fetching python-build-standalone $PYVER (aarch64-apple-darwin)"
  fetch_verified_tarball "$PBS_URL" "$PBS_SHA256" "$PBS_TGZ"
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
say "upgrading pip and purging legacy PyTorch/diffusers bloat if reusing stage"
rm -rf "$STAGE/sdxl"
"$VPY" -m pip install --quiet --upgrade pip setuptools wheel
"$VPY" -m pip uninstall --quiet -y \
  torch torchvision torchaudio diffusers accelerate omegaconf sympy networkx mpmath fsspec \
  2>/dev/null || true

# Backend requirements: backend/requirements.txt is the single source of pinned
# versions (dev venv, this runtime and CI). mflux comes from the vendored tree
# below; mlx-vlm from requirements-nodeps.txt. 100% PyTorch/diffusers-free.
say "installing backend runtime dependencies"
"$VPY" -m pip install --quiet -r "$HERE/../backend/requirements.txt"

# mlx-vlm: --no-deps to avoid dragging in unused training/eval/audio bloat
# (opencv-python GUI, mlx-audio, datasets, scipy, ...). Its runtime imports are
# already covered by requirements.txt.
"$VPY" -m pip install --quiet --no-deps -r "$HERE/../backend/requirements-nodeps.txt"

say "installing vendored mflux (Qwen-Image 2.1 port)"
# pip installs from a directory in place, which drops build/ and mflux.egg-info/ INTO
# that directory. That silently mutates the vendored fork: it doubles the file count
# (816 -> 1617) and changes its digest, so backend/test_repo_security.py fails, and the
# next bundle ships the artefacts too. Build from a throwaway copy and clean up after.
_VENDOR_COPY="$(mktemp -d)/mflux-src"
cp -R "$HERE/vendor/mflux-src" "$_VENDOR_COPY"
trap 'rm -rf "$(dirname "$_VENDOR_COPY")"' EXIT
# pip skips a same-version install, and this venv is REUSED between builds
# (dist stage survives). The fork kept its version at 0.20.0 while gaining
# whole model families (flux3, moss), so a reused venv silently kept the OLD
# package and every FLUX.3 generation died with "No module named
# 'mflux.models.flux3'". Uninstall first so the vendored copy always lands.
"$VPY" -m pip uninstall --quiet -y mflux 2>/dev/null || true
# --no-cache-dir: pip serves a CASHED wheel keyed by name+version, and the fork
# stays at 0.20.0 while gaining model families (flux3, moss) — without this the
# fresh venv kept the months-old cached wheel and FLUX.3 died with
# "No module named 'mflux.models.flux3'" even straight after rm -rf dist.
# --no-deps: mflux's declared deps include GUI/plotting wheels (opencv-python,
# matplotlib); its real runtime deps are pinned in backend/requirements.txt.
"$VPY" -m pip install --quiet --no-cache-dir --no-deps "$_VENDOR_COPY"
rm -rf "$HERE/vendor/mflux-src/build" "$HERE/vendor/mflux-src"/*.egg-info

# ---------------------------------------------------------------- prune
# mflux declares `opencv-python`, so pip pulls the GUI wheel on top of the
# headless one installed above. Both write the same cv2/ tree; the GUI wheel
# wins and drags SDL2/X11 dylibs into a Mac app that never opens a window.
# Remove it and restore the headless wheel. cv2 keeps working: mflux only calls
# seamlessClone and mlx-vlm's Qwen3-VL path only uses the image codecs.
#
# mlx-vlm is installed --no-deps by design (see above). miniaudio/sounddevice
# are audio extras of that package, never imported by the image path; they also
# arrive in a reused stage from older builds, so remove them explicitly.
say "pruning duplicate GUI OpenCV and unused audio/plotting extras"
"$VPY" -m pip uninstall --quiet -y \
  opencv-python sounddevice miniaudio \
  matplotlib fonttools kiwisolver cycler contourpy pyparsing \
  2>/dev/null || true
"$VPY" -m pip install --quiet --force-reinstall --no-deps "opencv-python-headless>=4.10.0,<5.0"

# ---------------------------------------------------------------- verify
say "verifying: mlx + mflux + the qwen21 port must import without torch"
"$VPY" - "$HERE/vendor/mflux-src/pyproject.toml" <<'PY'
import importlib, importlib.util, re, sys
import mlx.core as mx
# Model families each backend path imports at generation time. A reused venv
# with a stale mflux passed this check when it only listed flux3/z_image and
# still killed Anima runs: shipped 0.5.1 bundled mflux 0.20.0 (no anima, no
# fibo2) -> "No module named 'mflux.models.anima...'" at generation time.
for _mod, _why in (
    ("mflux.models.anima.variants.txt2img.anima", "Anima Turbo 2B generation"),
    ("mflux.models.dreamlite.variants.txt2img.dreamlite", "DreamLite Mobile generation"),
    ("mflux.models.flux", "FLUX.1 generation"),
    ("mflux.models.flux2", "FLUX.2-klein generation"),
    ("mflux.models.flux3", "FLUX.3-klein generation"),
    ("mflux.models.krea2", "Krea 2 Turbo generation"),
    ("mflux.models.qwen", "Qwen-Image-Edit generation"),
    ("mflux.models.qwen21", "Qwen-Image 2.1 generation"),
    ("mflux.models.z_image", "Z-Image Turbo generation"),
    ("mlx_vlm", "FIBO 2.0 Edit Turbo generation"),
    ("cv2", "OpenCV headless for inpainting fill and controlnet"),
):
    if importlib.util.find_spec(_mod) is None:
        sys.exit(f"FAIL: {_mod} is missing; {_why} would be unavailable in the shipped app")
# The installed mflux must be exactly the vendored version. A reused stage venv
# keeps whatever mflux was installed last build; without this check that is
# invisible here and only surfaces as a missing module at generation time.
_vendor_version = re.search(r'^version = "([^"]+)"', open(sys.argv[1], encoding="utf-8").read(), re.M).group(1)
from importlib.metadata import version as _dist_version
_installed = _dist_version("mflux")
if _installed != _vendor_version:
    sys.exit(f"FAIL: installed mflux {_installed} != vendored {_vendor_version}; the runtime venv is stale -- rebuild it from vendor/mflux-src")
from importlib.metadata import PackageNotFoundError, version
def _has(name):
    try:
        version(name)
        return True
    except PackageNotFoundError:
        return False
if _has("opencv-python"):
    sys.exit("FAIL: the GUI opencv-python wheel is installed; it shadows the headless cv2 tree")
if not _has("opencv-python-headless"):
    sys.exit("FAIL: opencv-python-headless is missing; inpainting fill would be unavailable")
for _pkg in ("sounddevice", "miniaudio"):
    if _has(_pkg):
        sys.exit(f"FAIL: unused audio package {_pkg} is still in the runtime")
if _has("torch") or "torch" in sys.modules:
    sys.exit("FAIL: torch is still installed or imported in the runtime")
print(f"  mflux     {importlib.import_module('mflux').__version__ if hasattr(importlib.import_module('mflux'), '__version__') else 'vendored'} (flux3 present)")
print("  mlx_vlm   ok (FIBO 2.0 Edit Turbo supported)")
print("  cv2       ok (OpenCV headless present)")
print(f"  mlx       device={mx.default_device()}")
import mflux
assert "torch" not in sys.modules, "FAIL: importing mflux imported torch"
print("  mflux     ok (torch-free)")
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
