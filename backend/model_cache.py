"""On-disk cache for artifacts derived from model weights.

Quantising a multi-gigabyte checkpoint, converting a LoRA or rebuilding a text
encoder are all deterministic functions of an input file set plus engine and
option versions. This module persists those derived files so the next run can
skip the work.

A hit is only served when a JSON manifest names this exact key and the artifact
still hashes to the SHA-256 recorded in that manifest. A wrong hit is worse than
a miss: engines would load corrupt weights and fail far away from the cause.

Hashing a multi-gigabyte artifact on every hit dominates load time, so a
successful check records the artifact's size, mtime, ctime and inode in a stamp next
to the manifest. Later hits skip the hash while that identity is unchanged; any
rewrite, replacement or truncation changes it and forces a full re-hash.

The cache is an optimisation, never a correctness dependency. A disabled cache,
a missing entry, a corrupt manifest or a failed write degrades to a miss and
never raises into generation. ``DIFFUSIONBEAR_DISABLE_MODEL_CACHE=1`` disables
all use.
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

DEFAULT_MAX_BYTES = 20 * 1024 * 1024 * 1024
_DISABLE_ENV = "DIFFUSIONBEAR_DISABLE_MODEL_CACHE"
_KEY_RE = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACT = "artifact.bin"
_MANIFEST = "manifest.json"
_VERIFIED = "verified.json"
_CHUNK = 1024 * 1024
_SCHEMA = 1


def _default_root():
    import app_settings

    return Path(app_settings.DATA_DIR) / ".cache" / "derived-artifacts"


def _canonical(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("cache key dict keys must be strings")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item) for item in value]
        return sorted(items, key=_sort_key)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported cache key value: {type(value).__name__}")


def _sort_key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _remove_path(path):
    try:
        if path.is_symlink() or not path.is_dir():
            path.unlink()
        else:
            shutil.rmtree(path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _file_identity(path):
    st = os.stat(path)
    return {
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        # ctime cannot be set from userland: any write, even one followed by
        # os.utime() restoring mtime, moves it and invalidates the stamp.
        "ctime_ns": st.st_ctime_ns,
        "ino": st.st_ino,
        "dev": st.st_dev,
    }


def _read_stamp(entry):
    try:
        stamp = json.loads((entry / _VERIFIED).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return stamp if isinstance(stamp, dict) else None


def _stamp_sha(entry):
    stamp = _read_stamp(entry)
    return stamp.get("sha256") if stamp else None


def _stamp_matches(entry, artifact, sha256):
    """True when ``artifact`` is the exact file that last hashed to ``sha256``."""
    stamp = _read_stamp(entry)
    if not stamp or not sha256 or stamp.get("sha256") != sha256:
        return False
    try:
        identity = _file_identity(artifact)
    except OSError:
        return False
    return all(stamp.get(k) == v for k, v in identity.items())


def _write_stamp(entry, artifact, sha256):
    if not sha256:
        return
    tmp = None
    try:
        stamp = dict(_file_identity(artifact), sha256=sha256)
        fd, tmp = tempfile.mkstemp(prefix=".verified.", dir=str(entry))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(stamp, handle, sort_keys=True)
        os.replace(tmp, entry / _VERIFIED)
        tmp = None
    except Exception:
        pass
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _sha256_file(path):
    digest = hashlib.sha256()
    buffer = bytearray(_CHUNK)
    with open(path, "rb") as handle:
        while True:
            count = handle.readinto(buffer)
            if not count:
                break
            digest.update(memoryview(buffer)[:count])
    return digest.hexdigest()


class DerivedArtifactCache:
    def __init__(self, root=None, max_bytes=DEFAULT_MAX_BYTES):
        self.root = Path(root).expanduser() if root is not None else _default_root()
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    @property
    def disabled(self):
        return os.environ.get(_DISABLE_ENV, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def key(kind, sources, engine_version, options):
        """Return a SHA-256 digest of the derivation inputs.

        Sources are fingerprinted by resolved path, size and ``st_mtime_ns``. A
        missing source raises ``FileNotFoundError``: a key without a fingerprint
        could not be trusted to distinguish old weights from new ones.
        """
        try:
            canonical_options = _canonical(options)
        except TypeError as exc:
            raise ValueError(
                f"cache key options must be plain JSON-compatible values: {exc}"
            ) from exc

        if isinstance(sources, (str, Path)):
            sources = [sources]
        fingerprints = {}
        for source in sources:
            resolved = Path(source).expanduser().resolve()
            stat = resolved.stat()
            fingerprints[str(resolved)] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }

        payload = {
            "schema": _SCHEMA,
            "kind": str(kind),
            "engine_version": str(engine_version),
            "sources": [
                {"path": path, **fingerprints[path]}
                for path in sorted(fingerprints)
            ],
            "options": canonical_options,
        }
        blob = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _entry(self, key):
        key = str(key)
        if not _KEY_RE.fullmatch(key):
            raise ValueError("cache key must be a 64-character SHA-256 digest")
        entry = self.root / key
        root = self.root.resolve()
        try:
            resolves_to = entry.resolve()
        except OSError:
            resolves_to = entry.absolute()
        if not resolves_to.is_relative_to(root):
            raise ValueError("cache key resolves outside the cache root")
        return entry

    def get_path(self, key):
        """Return the artifact path for ``key`` without reading the cache."""
        return self._entry(key) / _ARTIFACT

    def _valid_artifact(self, key, entry, digest=None):
        """Return the artifact path only when it matches its manifest.

        ``digest`` lets a caller that just wrote the file reuse the SHA-256 it
        computed while copying instead of reading the artifact a second time.
        """
        artifact = entry / _ARTIFACT
        try:
            if not artifact.is_file():
                return None
            manifest = json.loads((entry / _MANIFEST).read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return None
            if manifest.get("schema") != _SCHEMA:
                return None
            if manifest.get("key") != str(key):
                return None
            if manifest.get("artifact") != _ARTIFACT:
                return None
            if manifest.get("size") != artifact.stat().st_size:
                return None
            if digest is not None:
                if manifest.get("sha256") != digest:
                    return None
            elif _stamp_matches(entry, artifact, manifest.get("sha256")):
                return artifact
            elif manifest.get("sha256") != _sha256_file(artifact):
                return None
            _write_stamp(entry, artifact, manifest["sha256"])
            return artifact
        except Exception:
            return None

    def get(self, key):
        if self.disabled:
            return None
        try:
            entry = self._entry(key)
        except (ValueError, OSError, RuntimeError):
            return None
        artifact = self._valid_artifact(key, entry)
        if artifact is None:
            return None
        try:
            os.utime(artifact, None)
        except OSError:
            pass
        else:
            _write_stamp(entry, artifact, _stamp_sha(entry))
        return artifact

    def put(self, key, source_path):
        """Copy ``source_path`` into the cache and return the cached path."""
        if self.disabled:
            return None
        temp_dir = None
        backup = None
        restore_failed = False
        try:
            entry = self._entry(key)
            source = Path(source_path).expanduser().resolve()
            entry.parent.mkdir(parents=True, exist_ok=True)
            temp_dir = tempfile.mkdtemp(
                prefix=f".{str(key)[:12]}.", dir=str(entry.parent)
            )
            temp_artifact = Path(temp_dir) / _ARTIFACT
            temp_manifest = Path(temp_dir) / _MANIFEST

            digest = hashlib.sha256()
            size = 0
            buffer = bytearray(_CHUNK)
            with open(source, "rb") as fin, open(temp_artifact, "wb") as fout:
                while True:
                    count = fin.readinto(buffer)
                    if not count:
                        break
                    chunk = memoryview(buffer)[:count]
                    digest.update(chunk)
                    fout.write(chunk)
                    size += count
                fout.flush()
                os.fsync(fout.fileno())

            manifest = {
                "schema": _SCHEMA,
                "key": str(key),
                "artifact": _ARTIFACT,
                "size": size,
                "sha256": digest.hexdigest(),
                "created_at": time.time(),
            }
            with open(temp_manifest, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())

            with self._lock:
                if entry.exists():
                    backup = entry.parent / (
                        f".{entry.name}.bak-{os.getpid()}-{time.time_ns()}"
                    )
                    os.replace(entry, backup)
                try:
                    os.replace(temp_dir, entry)
                except OSError:
                    if backup is not None and not entry.exists():
                        try:
                            os.replace(backup, entry)
                        except OSError:
                            restore_failed = True
                            raise
                        backup = None
                    raise
                temp_dir = None
                if backup is not None and _remove_path(backup):
                    backup = None
            self._prune(self.max_bytes, protected=entry.name)
            return self._valid_artifact(key, entry, digest=digest.hexdigest())
        except Exception:
            return None
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
            if backup is not None and not restore_failed:
                _remove_path(backup)

    def prune(self, max_bytes):
        """Delete least-recently-used entries until total bytes fit ``max_bytes``.

        ``None`` or a non-positive budget means unbounded. Entries are ordered by
        the artifact's mtime, which ``get`` refreshes on every hit.
        """
        return self._prune(max_bytes)

    def _prune(self, max_bytes, protected=None):
        if self.disabled:
            return 0
        if max_bytes is None or max_bytes <= 0:
            return 0
        try:
            with self._lock:
                entries = []
                for child in self.root.iterdir():
                    if not child.is_dir() or child.name.startswith("."):
                        continue
                    artifact = child / _ARTIFACT
                    try:
                        stat = artifact.stat()
                    except OSError:
                        continue
                    entries.append((child, stat.st_mtime, stat.st_size))

                total = sum(size for _, _, size in entries)
                if total <= max_bytes:
                    return 0
                candidates = [
                    item for item in entries if item[0].name != protected
                ]
                candidates.sort(key=lambda item: (item[1], item[0].name))
                removed = 0
                for entry, _, size in candidates:
                    if total <= max_bytes:
                        break
                    try:
                        shutil.rmtree(entry)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        continue
                    total -= size
                    removed += 1
                return removed
        except OSError:
            return 0
