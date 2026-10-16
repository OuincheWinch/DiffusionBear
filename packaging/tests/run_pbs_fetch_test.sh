#!/usr/bin/env bash
# Verifies packaging/pbs_fetch.sh: the interpreter archive is never extracted
# unverified, and a pre-planted file is replaced rather than used.
#
# The flaw it pins (C6): both venv build scripts fetched python-build-standalone
# to a predictable /tmp path and extracted it with no hash check -- worse, they
# SKIPPED the download when the file already existed, so a pre-planted archive
# won outright and landed inside the signed app bundle.
#
# Every case runs against file:// URLs under a mktemp dir, so no network and no
# real interpreter bytes are involved. The work dir deliberately contains a space:
# build stages live under paths like "MLX-DIFFUSION OpenCode", and the shasum
# check-file format has to survive that.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(mktemp -d)"
SRC="$BASE/src"
# a space in the work path: build stages live under directories like
# "MLX-DIFFUSION OpenCode", and the shasum check-file format has to survive it
WORK="$BASE/with space"
mkdir -p "$WORK" "$SRC"
cleanup() { rm -rf "$BASE"; }
trap cleanup EXIT

bash -n "$HERE/../pbs_fetch.sh"
# shellcheck disable=SC1091
source "$HERE/../pbs_fetch.sh"

mkdir -p "$SRC/files" && printf 'payload' > "$SRC/files/f.txt"
tar czf "$SRC/good.tar.gz" -C "$SRC/files" f.txt
printf 'tampered' > "$SRC/bad.tar.gz"
GOOD_SHA="$(shasum -a 256 "$SRC/good.tar.gz" | awk '{print $1}')"
CACHE="$WORK/cache/a.tar.gz"
pass=0; fail=0

check() { # name, expected-exit, actual-exit, [extra-assertion...]
  local name="$1" want="$2"; shift 2
  if [ "$1" = "$want" ]; then pass=$((pass + 1)); printf '  ok   %s\n' "$name";
  else fail=$((fail + 1)); printf '  FAIL %s (want exit %s, got %s)\n' "$name" "$want" "$1"; fi
}

echo "==> fresh download lands verified"
fetch_verified_tarball "file://$SRC/good.tar.gz" "$GOOD_SHA" "$CACHE"
check "fresh download exits 0" 0 "$?"
cmp -s "$SRC/good.tar.gz" "$CACHE" && check "bytes identical" 0 0 || check "bytes identical" 0 1

echo "==> pre-planted file is replaced, never used"
printf 'planted' > "$CACHE"
fetch_verified_tarball "file://$SRC/good.tar.gz" "$GOOD_SHA" "$CACHE"
check "plant succeeds" 0 "$?"
cmp -s "$SRC/good.tar.gz" "$CACHE" && check "plant replaced" 0 0 || check "plant replaced" 0 1

echo "==> tampered upstream is refused"
rm -f "$CACHE"
if fetch_verified_tarball "file://$SRC/bad.tar.gz" "$GOOD_SHA" "$CACHE"; then rc=0; else rc=1; fi
check "tamper exits non-zero" 1 "$rc"
[ ! -e "$CACHE" ] && check "nothing left at dest" 0 0 || check "nothing left at dest" 0 1

echo "==> corrupt cache plus dead upstream fails closed"
printf 'stale-garbage' > "$CACHE"
if fetch_verified_tarball "file://$SRC/no-such-file.tar.gz" "$GOOD_SHA" "$CACHE"; then rc=0; else rc=1; fi
check "dead upstream exits non-zero" 1 "$rc"

echo "==> $pass passed, $fail failed"
[ "$fail" = 0 ]
