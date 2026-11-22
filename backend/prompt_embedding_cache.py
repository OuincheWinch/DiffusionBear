"""Shared on-disk cache for text-encoder prompt embeddings.

Embeddings are stored as safetensors keyed by a SHA-256 digest of a canonical
JSON tuple. The prompt text is part of the hash preimage only; it is never
written to disk. Key inputs must be plain, acyclic JSON-compatible values;
cartesian sequences keep their order, while set values are sorted.

The cache is an optimisation, never a correctness dependency: a disabled cache,
an unreadable file or a failed write degrades to a miss and never raises into
generation. ``DIFFUSIONBEAR_DISABLE_PROMPT_CACHE=1`` disables all use.
"""

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

from safetensors.numpy import load_file, save_file

DEFAULT_MAX_BYTES = 512 * 1024 * 1024
_DISABLE_ENV = "DIFFUSIONBEAR_DISABLE_PROMPT_CACHE"
_KEY_RE = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA = 1
_CHUNK = 1024 * 1024


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    buffer = bytearray(_CHUNK)
    with open(path, "rb") as handle:
        while True:
            count = handle.readinto(buffer)
            if not count:
                break
            digest.update(memoryview(buffer)[:count])
    return digest.hexdigest()


def _canonical(value):
    if isinstance(value, dict):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
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


def _te_signature(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = (value,)
    return sorted(str(item) for item in value)


def _default_root():
    import app_settings

    return Path(app_settings.DATA_DIR) / ".cache" / "prompt-embeddings"


class PromptEmbeddingCache:
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
    def key(
        engine,
        model_fingerprint,
        prompt,
        negative_prompt="",
        clip_skip=2,
        te_signature=(),
        extra=None,
    ):
        """Hash plain, acyclic JSON-compatible key inputs.

        ``ValueError`` is raised for cyclic or unsupported values: silently
        collapsing distinct inputs to one digest would serve wrong embeddings.
        """
        try:
            payload = [
                1,
                _canonical(engine),
                _canonical(model_fingerprint),
                _canonical(prompt),
                _canonical(negative_prompt),
                _canonical(clip_skip),
                _te_signature(te_signature),
                _canonical(extra),
            ]
            blob = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            return hashlib.sha256(blob.encode("utf-8")).hexdigest()
        except Exception as exc:
            raise ValueError(
                "prompt cache key requires plain acyclic JSON-compatible input"
            ) from exc

    def _path(self, key):
        key = str(key)
        if not _KEY_RE.fullmatch(key):
            raise ValueError("cache key must be a 64-character SHA-256 digest")
        return self.root / f"{key}.safetensors"

    def _manifest_dir(self):
        # Sibling of the artifact root, never inside it: callers and tests treat
        # the cache root as a directory of safetensors files only.
        return self.root.parent / f".{self.root.name}.manifests"

    def _manifest_path(self, key):
        key = str(key)
        if not _KEY_RE.fullmatch(key):
            raise ValueError("cache key must be a 64-character SHA-256 digest")
        return self._manifest_dir() / f"{key}.json"

    def get(self, key):
        if self.disabled:
            return None
        try:
            path = self._path(key)
            manifest_path = self._manifest_path(key)
            if not path.is_file() or not manifest_path.is_file():
                return None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or manifest.get("schema") != _SCHEMA:
                return None
            if manifest.get("key") != str(key):
                return None
            if manifest.get("size") != path.stat().st_size:
                return None
            if manifest.get("sha256") != _sha256_file(path):
                return None
            data = load_file(str(path))
            try:
                os.utime(path, None)
            except OSError:
                pass
            return dict(data)
        except Exception:
            return None

    def put(self, key, arrays):
        if self.disabled:
            return None
        tmp_path = None
        tmp_manifest = None
        try:
            path = self._path(key)
            manifest_path = self._manifest_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".{path.stem}.",
                suffix=".tmp",
                dir=str(path.parent),
            )
            os.close(fd)
            save_file({str(name): array for name, array in arrays.items()}, tmp_path)
            with open(tmp_path, "rb") as handle:
                os.fsync(handle.fileno())
            digest = _sha256_file(tmp_path)
            size = os.path.getsize(tmp_path)
            manifest = {
                "schema": _SCHEMA,
                "key": str(key),
                "size": size,
                "sha256": digest,
                "created_at": time.time(),
            }
            mfd, tmp_manifest = tempfile.mkstemp(
                prefix=f".{path.stem}.",
                suffix=".manifest.tmp",
                dir=str(path.parent),
            )
            with os.fdopen(mfd, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            # Artifact first, manifest second. A crash between the two is a miss,
            # never a wrong hit.
            os.replace(tmp_path, path)
            tmp_path = None
            os.replace(tmp_manifest, manifest_path)
            tmp_manifest = None
            self._evict()
        except Exception:
            return None
        finally:
            for leftover in (tmp_path, tmp_manifest):
                if leftover is not None:
                    try:
                        os.unlink(leftover)
                    except OSError:
                        pass
        return None

    def _evict(self):
        if self.max_bytes is None or self.max_bytes <= 0:
            return
        with self._lock:
            try:
                files = []
                for path in self.root.glob("*.safetensors"):
                    stat = path.stat()
                    files.append((path, stat.st_mtime, stat.st_size))
                total = sum(size for _, _, size in files)
                if total <= self.max_bytes:
                    return
                files.sort(key=lambda item: (item[1], item[0].name))
                for path, _, size in files:
                    if total <= self.max_bytes:
                        break
                    try:
                        path.unlink()
                        try:
                            self._manifest_path(path.stem).unlink()
                        except OSError:
                            pass
                        total -= size
                    except OSError:
                        pass
            except Exception:
                return

    def clear(self):
        try:
            if self.root.is_dir():
                for path in self.root.iterdir():
                    if path.is_file() and (
                        path.name.endswith(".safetensors") or path.name.endswith(".tmp")
                    ):
                        try:
                            path.unlink()
                        except OSError:
                            pass
            manifest_dir = self._manifest_dir()
            if manifest_dir.is_dir():
                import shutil

                shutil.rmtree(manifest_dir, ignore_errors=True)
        except Exception:
            return
