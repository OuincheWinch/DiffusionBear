#!/usr/bin/env bash
# Assemble MLX-Diffusion.app from the built runtimes, the production SPA and the
# Swift shell, then ad-hoc sign it. No Apple Developer ID required.
#
# Layout is dictated by the backend's own path expectations (see Paths in
# shell/main.swift): Resources/{backend,venv,venv-sdxl,python,frontend/dist}.
#
# Signing, in the order codesign requires: nested dylibs and the executables
# first, then the outer bundle. On arm64 the signing itself is what repairs the
# arm64e link-edit that install_name_tool leaves behind -- an unsigned modified
# dylib will not load.
#
# LSMinimumSystemVersion is 15.0 because MLX >= 0.29.2 (0.32.1 here) calls a Metal
# API that does not exist on Sonoma. Setting it lets macOS refuse cleanly instead
# of crashing at import.
set -euo pipefail

HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -P "$HERE/.." && pwd)"
STAGE="${1:-$HERE/build}"
DIST="${2:-$HERE/dist}"
APP_NAME="DiffusionBear"
APP="$DIST/$APP_NAME.app"
BUNDLE_ID="com.ouinchewinch.diffusionbear"
VERSION="$(cat "$REPO/backend/app_version.py" 2>/dev/null | sed -n 's/^APP_VERSION *= *"\([^"]*\)".*/\1/p' | head -1)"
VERSION="${VERSION:-0.0.0}"

say() { printf '\n\033[1m==>\033[0m %s\n' "$*"; }

[ -d "$STAGE/venv" ]      || { echo "missing $STAGE/venv -- run build_runtime_venv.sh" >&2; exit 1; }
[ -d "$STAGE/sdxl/venv" ] || { echo "missing $STAGE/sdxl/venv -- run build_sdxl_venv.sh" >&2; exit 1; }
[ -f "$REPO/frontend/dist/index.html" ] || {
  echo "missing frontend/dist -- run: (cd frontend && VITE_API_BASE= npx vite build)" >&2; exit 1; }

say "cleaning $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# ---------------------------------------------------------------- payload
say "copying backend"
# Only code. The whole data/ tree is excluded, not just data/models: it also holds
# lora_files (425 MB of .safetensors) and -- critically -- hf_token.txt and
# civitai_token.txt. Those are real credentials and must never be copied into a
# distributable bundle. Everything there lives in the external store anyway, which
# MLX_DIFFUSION_ASSET_DIR / MLX_DIFFUSION_DATA_DIR point at.
rsync -a --quiet \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
  --exclude 'data' \
  "$REPO/backend/" "$APP/Contents/Resources/backend/"
mkdir -p "$APP/Contents/Resources/backend/data"

say "copying runtimes (this is the bulk of the app)"
cp -R "$STAGE/venv"        "$APP/Contents/Resources/venv"
cp -R "$STAGE/python"      "$APP/Contents/Resources/python"
cp -R "$STAGE/sdxl/venv"   "$APP/Contents/Resources/venv-sdxl"
cp -R "$STAGE/sdxl/python" "$APP/Contents/Resources/python-sdxl"

# pbs resolves its stdlib from the executable, but a venv's pyvenv.cfg records an
# absolute `home`. Repoint both at the final install location so the copied
# interpreters find their stdlib after the bundle moves.
say "repointing pyvenv.cfg at the final bundle path"
for pair in "venv:python" "venv-sdxl:python-sdxl"; do
  v="${pair%%:*}"; p="${pair##*:}"
  cat > "$APP/Contents/Resources/$v/pyvenv.cfg" <<CFG
home = $APP/Contents/Resources/$p/bin
include-system-site-packages = false
version = $("$APP/Contents/Resources/$v/bin/python" -V 2>&1 | awk '{print $2}')
CFG
done
# pbs resolves its stdlib from the executable, but a venv created from it points at
# the interpreter with an ABSOLUTE symlink:
#     venv/bin/python3 -> /Volumes/.../packaging/build/python/bin/python3
# Left alone, the bundle would run on the build machine and fail on every other Mac,
# because the target does not exist there. Rewrite those as relative symlinks that
# resolve inside Resources, and replace them with real files where a symlink would
# be ambiguous.
say "repointing interpreter symlinks into the bundle"
relink() {  # $1=venv bin dir  $2=target dir relative to it (e.g. ../../python/bin)
  local bindir="$1" target="$2"
  # Symlinks, never copies. A pbs interpreter locates its stdlib through the
  # symlink chain; replacing venv/bin/python3 with a *copy* of the base binary
  # makes sys._base_executable point into the venv, and the venv has no stdlib of
  # its own -- it fails with "Could not find platform independent libraries" and
  # "No module named 'encodings'". That is exactly what happened to venv-sdxl.
  for base in python3 python; do
    [ -x "$bindir/$target/$base" ] || continue
    rm -f "$bindir/$base"
    ln -s "$target/$base" "$bindir/$base"
  done
  # python3.10 / python3.14 should point at the primary name, not at the base dir,
  # so there is exactly one place that names the base interpreter.
  for ver in $(ls "$bindir" 2>/dev/null | grep -E '^python3\.[0-9]+$' || true); do
    rm -f "$bindir/$ver"
    ln -s python3 "$bindir/$ver"
  done
  printf '    %s/{python,python3,python3.x} -> %s\n' "$(basename "$bindir")" "$target"
}
relink "$APP/Contents/Resources/venv/bin"        "../../python/bin"
relink "$APP/Contents/Resources/venv-sdxl/bin"   "../../python-sdxl/bin"

say "verifying no absolute symlinks remain in the runtimes"
LEAKED=0
while IFS= read -r -d '' l; do
  case "$(readlink "$l")" in /*) echo "    LEAK: $l -> $(readlink "$l")"; LEAKED=1;; esac
done < <(find "$APP/Contents/Resources/venv" "$APP/Contents/Resources/venv-sdxl" -type l -print0)
[ "$LEAKED" -eq 0 ] || { echo "absolute symlinks would break the bundle" >&2; exit 1; }
echo "    none"

say "verifying every interpreter in the bundle can find its stdlib"
# The failure this catches is silent at build time and only appears as an SDXL
# engine that "died" at generation time, so prove it here for both runtimes.
for pair in "venv:3.10" "venv-sdxl:3.14"; do
  v="${pair%%:*}"; want="${pair##*:}"
  p="$APP/Contents/Resources/$v/bin/python"
  [ -e "$p" ] || { echo "    MISSING: $p" >&2; exit 1; }
  got="$(cd / && "$p" -c 'import sys, os; print("%d.%d|%s" % (sys.version_info.major, sys.version_info.minor, os.path.realpath(os.__file__)))' 2>&1 | tail -1)" || true
  case "$got" in
    "$want|"*"/python$want/os.py")
      echo "    $v OK -> ${got#*|}";;
    *)
      echo "    $v FAILED: $got" >&2
      echo "bundle runtime is broken; refusing to ship" >&2; exit 1;;
  esac
done

# venv launchers embed the absolute interpreter path too.
for v in venv venv-sdxl; do
  real="$APP/Contents/Resources/$v/bin/python"
  for f in "$APP/Contents/Resources/$v/bin/"*; do
    [ -f "$f" ] || continue
    head -1 "$f" 2>/dev/null | grep -q '^#!' || continue
    # Rewrite only the shebang line, in place, byte for byte otherwise.
    python3 - "$f" "$real" <<'PY'
import sys
path, real = sys.argv[1], sys.argv[2]
with open(path, "rb") as fh:
    data = fh.read()
nl = data.find(b"\n")
if nl < 0 or not data.startswith(b"#!"):
    raise SystemExit(0)
data = b"#!" + real.encode() + data[nl:]
with open(path, "wb") as fh:
    fh.write(data)
PY
  done
done

say "copying the production SPA"
mkdir -p "$APP/Contents/Resources/frontend"
rsync -a --quiet "$REPO/frontend/dist/" "$APP/Contents/Resources/frontend/dist/"

# ---------------------------------------------------------------- shell
say "compiling the Swift shell"
swiftc -O -wmo \
  -target arm64-apple-macos15.0 \
  -framework AppKit -framework WebKit \
  -o "$APP/Contents/MacOS/$APP_NAME" \
  "$HERE/shell/main.swift"

# ---------------------------------------------------------------- plist
say "writing Info.plist"
# No XML DOCTYPE: it is a legacy prolog that macOS does not need, and its
# apple.com DTD URL trips test_outbound_hosts_are_allowlisted for no benefit.
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
  <key>CFBundleName</key>                  <string>$APP_NAME</string>
  <key>CFBundleDisplayName</key>           <string>$APP_NAME</string>
  <key>CFBundleExecutable</key>            <string>$APP_NAME</string>
  <key>CFBundleIdentifier</key>            <string>$BUNDLE_ID</string>
  <key>CFBundlePackageType</key>           <string>APPL</string>
  <key>CFBundleShortVersionString</key>    <string>$VERSION</string>
  <key>CFBundleVersion</key>               <string>$VERSION</string>
  <key>LSMinimumSystemVersion</key>        <string>15.0</string>
  <key>NSHighResolutionCapable</key>       <true/>
  <key>LSApplicationCategoryType</key>     <string>public.app-category.graphics-design</string>
  <!-- The backend is a local loopback server; no outbound network entitlement is
       claimed, and no camera/microphone/disk-access prompt is needed. -->
  <key>NSAppTransportSecurity</key>
  <dict>
    <key>NSAllowsLocalNetworking</key><true/>
  </dict>
</dict>
</plist>
PLIST

# ---------------------------------------------------------------- sign
# No Developer ID: ad-hoc only. A locally built bundle is not quarantined, so
# Gatekeeper does not intervene; moving it to another Mac would need a right-click
# Open (or a real Developer ID + notarisation, which is out of scope here).
#
# Signed bottom-up and WITHOUT --deep on purpose. --deep re-signs nested code by
# walking the bundle, which chases the python/python3/python3.10 symlink web and
# leaves the verify step reporting "file modified". Signing each real Mach-O file
# once, in dependency order, then the outer bundle, is both correct and verifiable.
say "ad-hoc signing native libraries (dylibs)"
signed=0
while IFS= read -r -d '' lib; do
  codesign --force --sign - --timestamp=none "$lib" 2>/dev/null && signed=$((signed+1)) || true
done < <(find "$APP/Contents/Resources" -name '*.dylib' -print0)
echo "    $signed dylib(s)"

say "ad-hoc signing Python extension modules"
signed=0
for v in venv venv-sdxl python python-sdxl; do
  d="$APP/Contents/Resources/$v"
  [ -d "$d" ] || continue
  while IFS= read -r -d '' so; do
    codesign --force --sign - --timestamp=none "$so" 2>/dev/null && signed=$((signed+1)) || true
  done < <(find "$d" -name '*.so' -print0 2>/dev/null)
done
echo "    $signed extension module(s)"

say "ad-hoc signing the interpreters (real files only, never symlinks)"
signed=0
for p in "$APP/Contents/Resources/python/bin" "$APP/Contents/Resources/python-sdxl/bin"; do
  [ -d "$p" ] || continue
  for exe in "$p"/*; do
    [ -L "$exe" ] && continue      # a symlink is not signable on its own
    [ -f "$exe" ] || continue
    file "$exe" 2>/dev/null | grep -q 'Mach-O' || continue
    codesign --force --sign - --timestamp=none "$exe" 2>/dev/null && signed=$((signed+1)) || true
  done
done
echo "    $signed interpreter(s)"

say "ad-hoc signing the shell and the bundle"
codesign --force --sign - --timestamp=none "$APP/Contents/MacOS/$APP_NAME"
codesign --force --sign - --timestamp=none "$APP"
echo "    verifying"
if ! codesign --verify --deep --strict "$APP" 2>&1 | sed 's/^/    /'; then
  echo "signature verification failed" >&2
  exit 1
fi
echo "    signature OK (ad-hoc, identifier: $BUNDLE_ID)"

# ---------------------------------------------------------------- audit
# A bundle is a thing you hand to someone. Anything that looks like a credential
# or a model weight must not be inside it. This is the check that would have caught
# hf_token.txt being copied in, and it runs on every build rather than relying on
# whoever edits the rsync line to remember.
say "auditing the bundle for credentials and stray weights"
FAIL=0
# The runtimes legitimately contain cacert.pem (a public CA root bundle) and
# thousands of .so files, so every check below skips them. What matters is that
# the app's OWN payload carries no secrets and no weights.
is_runtime() { case "$1" in */venv/*|*/venv-sdxl/*|*/python/*|*/python-sdxl/*) return 0;; *) return 1;; esac; }

# 1. credential-shaped filenames in the app's own payload
while IFS= read -r -d '' f; do
  is_runtime "$f" && continue
  echo "    CREDENTIAL: ${f#$APP/}"
  FAIL=1
done < <(find "$APP" -path "$APP/Contents/Resources/venv" -prune -o \
                   -path "$APP/Contents/Resources/venv-sdxl" -prune -o \
                   -path "$APP/Contents/Resources/python" -prune -o \
                   -path "$APP/Contents/Resources/python-sdxl" -prune -o \
                   \( -name '*token*.txt' -o -name '*.pem' -o -name '*.key' \
                   -o -name 'id_rsa*' -o -name '.env' -o -name '*.p12' \) -print0 2>/dev/null)
# 2. model weights anywhere in the payload
while IFS= read -r -d '' f; do
  echo "    WEIGHT IN BUNDLE: ${f#$APP/}"
  FAIL=1
done < <(find "$APP/Contents/Resources" -name '*.safetensors' -print0 2>/dev/null)
# 3. any unexpectedly large file outside the runtimes
while IFS= read -r -d '' f; do
  is_runtime "$f" && continue
  sz=$(stat -f%z "$f")
  [ "$sz" -gt 52428800 ] || continue
  echo "    LARGE FILE OUTSIDE RUNTIMES: ${f#$APP/} ($((sz / 1048576)) MB)"
  FAIL=1
done < <(find "$APP" -type f -print0)
if [ "$FAIL" -ne 0 ]; then
  echo "bundle audit FAILED -- do not ship this" >&2
  exit 1
fi
echo "    clean: no credentials, no model weights, no stray large files"

# ---------------------------------------------------------------- report
say "bundle ready"
echo
du -sh "$APP" | sed 's/^/  total      /'
du -sh "$APP/Contents/Resources/venv"        | sed 's/^/  venv       /'
du -sh "$APP/Contents/Resources/venv-sdxl"   | sed 's/^/  venv-sdxl  /'
du -sh "$APP/Contents/Resources/python"      | sed 's/^/  python     /'
du -sh "$APP/Contents/Resources/python-sdxl" | sed 's/^/  python-sdxl /'
du -sh "$APP/Contents/Resources/backend"     | sed 's/^/  backend    /'
du -sh "$APP/Contents/Resources/frontend"    | sed 's/^/  frontend   /'
du -sh "$APP/Contents/MacOS/$APP_NAME"      | sed 's/^/  shell      /'
echo
echo "  models stay external: ~/Library/Application Support/MLX-Diffusion/data"
echo "  install with: ditto $APP /Applications/"
