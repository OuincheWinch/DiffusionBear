#!/usr/bin/env bash
# Shared fetch of the python-build-standalone interpreter archive, used by
# build_runtime_venv.sh and build_sdxl_venv.sh.
#
# Contract: fetch_verified_tarball <url> <sha256> <dest_tgz>
#   * dest_tgz is the caller's cache path, under the build STAGE -- never /tmp.
#   * A cached file whose digest matches is reused (re-verified every run, so a
#     cache corrupted between builds is caught, not trusted).
#   * Otherwise the archive is downloaded to a fresh mktemp dir, verified, and
#     moved into place. A pre-planted or corrupted file fails verification and
#     is replaced -- never extracted.
#   * Any mismatch fails closed (non-zero return, nothing extracted). Callers
#     run under `set -e`, so a failure aborts the build before `tar` runs.
#
# The pinned digests live in the callers next to their URLs. Trust basis,
# stated plainly: they were fetched over HTTPS from github.com (an allowed
# host) on 2026-10-14. The pin protects against subsequent tampering and
# pre-planting -- it cannot protect against a compromised origin at pin time.
# Re-pin deliberately, never by copying a digest from an unverified mirror.
#
# This file defines functions only and executes nothing on source, so tests
# can source it without side effects.
fetch_verified_tarball() {
  local url="$1" want="$2" dest="$3"
  if [ -z "$url" ] || [ -z "$want" ] || [ -z "$dest" ]; then
    echo "fetch_verified_tarball: url, sha256 and dest path are all required" >&2
    return 1
  fi
  if [ -f "$dest" ] && printf '%s  %s\n' "$want" "$dest" | shasum -a 256 -c - >/dev/null 2>&1; then
    echo "reusing verified $(basename "$dest")"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  local work
  work="$(mktemp -d)" || return 1
  local tmp="$work/dl.tar.gz"
  if ! curl -sL --max-time 600 -o "$tmp" "$url"; then
    echo "download failed: $url" >&2
    rm -rf "$work"
    return 1
  fi
  if ! printf '%s  %s\n' "$want" "$tmp" | shasum -a 256 -c - >/dev/null 2>&1; then
    echo "SHA-256 mismatch for $url -- refusing a tampered or corrupted archive" >&2
    rm -rf "$work"
    return 1
  fi
  mv "$tmp" "$dest"
  rm -rf "$work"
}
