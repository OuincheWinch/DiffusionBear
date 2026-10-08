#!/usr/bin/env bash
# Export the filtered public source tree and push it.
#
# WHY THIS IS A SCRIPT AND NOT A SHELL HISTORY
# The public repo is a filtered copy of the private one, and the exclusion list has
# only ever lived in someone's terminal. That is how 0.3.4 nearly published
# data/lora_files (425 MB) and data/hf_token.txt -- a real credential -- into a
# public repository. One forgotten line, or one typo, and it ships. The list
# belongs in version control where it is reviewable.
#
# WHAT IS EXCLUDED, AND WHY
#   docs/ENGINEERING.md        internal measurements and incident history
#   docs/HANDOFF.md            names this machine's paths; operational only
#   design-system/...MASTER.md design source of truth, not needed by readers
#   backend/civitai_browse.py.bak  dead backup of a deleted module
#   backend/scripts/repeatability_test.py  private benchmarking harness
#   musehandoff.MD            review handoff for another model; not project docs
#   exofreehandoff.md         next-model handoff; not project docs
#
# WHAT IS DELIBERATELY KEPT
#   backend/test_*.py          public CI runs them; they are the project's proof
#   packaging/vendor/mflux-src vendored upstream, needed to audit a path claim
#   packaging/shell/main.swift the launcher is the interesting part of the build
#
# USAGE
#   ./export_public.sh              build the tree and print the diff stat
#   ./export_public.sh --push       also commit, force-push upstream, print the SHA
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BRANCH="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"

WORK="${TMPDIR:-/tmp}/diffusionbear-public-export"
EXCLUDES=(
  "docs/ENGINEERING.md"
  "docs/HANDOFF.md"
  "musehandoff.MD"
  "exofreehandoff.md"
  "design-system/diffusionbear/MASTER.md"
  "backend/civitai_browse.py.bak"
  "backend/scripts/repeatability_test.py"
)

say() { printf '\033[1m==> %s\033[0m\n' "$1"; }

# What counts as "a local path", DERIVED at run time rather than written down.
#
# Hardcoding the build machine's volume and username here meant this file could not
# be published without republishing them -- the exact leak that shipped in 0.3.4
# and that had to be scrubbed from main.swift and a test canary. So the defaults
# are computed: the volume the repo lives on, and the current home directory. The
# volume, not the repo path, because naming only the repo missed the sibling store
# path that also shipped.
#
# Override both to audit a different machine:
#   REPO_LEAK_PATH=/Volumes/Other HOME_LEAK_PATH=/Users/someone ./export_public.sh
_repo_volume="$(cd "$REPO/../.." 2>/dev/null && pwd -P || echo "")"
REPO_LEAK_PATH="${REPO_LEAK_PATH:-$_repo_volume}"
HOME_LEAK_PATH="${HOME_LEAK_PATH:-$HOME}"

say "cleaning $WORK"
rm -rf "$WORK"
mkdir -p "$WORK"

say "exporting $BRANCH"
# git archive, not cp: it carries committed content only, so an untracked
# credential or scratch file cannot leak in even if it sits in the worktree.
git -C "$REPO" archive "$BRANCH" | tar -x -C "$WORK"

say "applying the exclusion list"
for rel in "${EXCLUDES[@]}"; do
  if [ -e "$WORK/$rel" ]; then
    rm -f "$WORK/$rel"
    echo "    excluded $rel"
  else
    echo "    NOTE: $rel does not exist (renamed or removed?)"
  fi
done
rmdir "$WORK/design-system/diffusionbear" "$WORK/design-system" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Audit the exported tree before it can be published. Cheap, and the failure it
# prevents is unrecoverable once a credential reaches a public repo's history.
# ---------------------------------------------------------------------------
say "auditing the export"
fail=0
if [ -z "$REPO_LEAK_PATH" ] || [ -z "$HOME_LEAK_PATH" ]; then
  echo "  could not derive the local-path patterns; refusing to audit" >&2
  exit 1
fi

# This machine's own paths, by name. Deliberately NOT a bare '/Volumes/' or
# '/Users/' search: the launcher legitimately does hasPrefix("/Volumes/"), build_app.sh
# documents the /Volumes/... case in prose, an upstream mflux docstring shows
# /Volumes/flux2-klein-9b-experiments, and a test fixture uses /Users/alice. A gate
# that flags those trains you to ignore it. What actually leaked was THIS machine's
# identifiers, so those are what is forbidden.
# No self-exclusion is needed any more: the patterns are derived at run time, so
# this script contains no literal path to trip over. An earlier version spelled them
# out and had to exclude itself, which is how the exception became a hole.
leaks=$(grep -rlI --exclude-dir=.git \
        -e "$REPO_LEAK_PATH" -e "$HOME_LEAK_PATH" "$WORK" 2>/dev/null || true)
if [ -n "$leaks" ]; then
  echo "    these files name a local path and must not be published:" >&2
  echo "$leaks" | sed "s|$WORK/|      |" >&2
  fail=1
fi

# Credentials, by CONTENT signature rather than by filename. Grepping for
# "hf_token" flags every module that legitimately constructs DATA_DIR/hf_token.txt
# -- which is most of hf_service.py, and a gate that cries wolf gets ignored. A
# Hugging Face token is hf_ followed by 30+ alphanumerics; zero tracked files
# contain that today, which is what makes it safe to fail on.
creds=$(grep -rlIE --exclude-dir=.git -e 'hf_[A-Za-z0-9]{30,}' -e '-----BEGIN [A-Z ]*PRIVATE KEY-----' \
        "$WORK" 2>/dev/null || true)
if [ -n "$creds" ]; then
  echo "    POSSIBLE CREDENTIALS in:" >&2
  echo "$creds" | sed "s|$WORK/|      |" >&2
  fail=1
fi

# Weights. git archive only carries tracked files, so the 15 GB under
# backend/data cannot appear here -- but the tripwire is worth keeping, because the
# day someone git-adds a model the gate should notice rather than notice nothing.
# packaging/vendor/mflux-src legitimately contains one 580 KB positional-embedding
# table that mflux imports; anything larger is a mistake.
weights=$(find "$WORK" \( -name '*.ckpt' -o -name '*.gguf' -o -name '*.npz' \) -type f 2>/dev/null || true)
if [ -n "$weights" ]; then
  echo "    checkpoint files present:" >&2
  echo "$weights" | sed "s|$WORK/|      |" >&2
  fail=1
fi
big=$(find "$WORK" -type f -size +10M -not -path '*/vendor/mflux-src/*' 2>/dev/null || true)
if [ -n "$big" ]; then
  echo "    file(s) over 10 MB outside vendored upstream:" >&2
  echo "$big" | sed "s|$WORK/|      |" >&2
  fail=1
fi

if [ -d "$WORK/backend/data" ]; then
  echo "    backend/data is present; the store must never be exported" >&2
  fail=1
fi

if [ "$fail" != "0" ]; then
  echo "EXPORT REFUSED -- fix the above before publishing" >&2
  exit 1
fi
echo "    clean: no local paths, no credential signatures, no weights, no store"

say "result"
echo "    files: $(find "$WORK" -type f | wc -l | tr -d ' ')"
echo "    tree:  $WORK"

if [ "${1:-}" != "--push" ]; then
  echo
  echo "Review it, then:  $0 --push"
  exit 0
fi

cd "$WORK"
git init -q .

# A fresh `git init` has no remotes, and `upstream` is configured only in the
# private repo's local config -- so this push used to die here with
# "'upstream' does not appear to be a git repository". Take the URL from that
# config instead of spelling it out a second time, and refuse loudly if it is
# missing rather than pushing somewhere unintended.
UPSTREAM_URL="${UPSTREAM_URL:-$(git -C "$REPO" remote get-url upstream 2>/dev/null || true)}"
if [ -z "$UPSTREAM_URL" ]; then
  echo "    no 'upstream' remote in $REPO; set UPSTREAM_URL" >&2
  exit 1
fi
git remote add upstream "$UPSTREAM_URL"

git add -A
if git diff --cached --quiet; then
  echo "    nothing to publish"
  exit 0
fi
git -c user.name="${GIT_NAME:-Ouinche}" -c user.email="${GIT_EMAIL:-ouinche@ouinche.com}" \
    commit -q -m "${COMMIT_MESSAGE:-Update public source}"

echo
echo "About to force-push $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo HEAD) to upstream/main."
echo "The public history has different hashes than the private one by design, so this"
echo "rewrites the public branch. Re-run with no argument to inspect first."
if [ "${CONFIRM_PUSH:-0}" != "1" ]; then
  echo
  echo "Set CONFIRM_PUSH=1 to actually push."
  exit 0
fi
# A fresh `git init` keeps every object loose, and shipping ~1100 of them in one
# request fails with "RPC failed; HTTP 400" however small the resulting pack is.
# Pack first, and give that single request a bigger buffer.
git repack -a -d -q
git -c http.postBuffer=524288000 push --force upstream HEAD:main
echo "pushed $(git rev-parse --short HEAD)"