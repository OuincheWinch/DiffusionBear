"""Hashed multi-key storage for the DiffusionBear API.

The environment key is immutable. Managed keys live in ``security.json`` as
SHA-256 hashes only; the full value is returned once at creation and never
again. A legacy plaintext ``api_key`` field is accepted for compatibility and
preserved, but is always listed as read-only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = 2
MAX_MANAGED_KEYS = 32
MAX_LABEL_LENGTH = 64
MAX_SECURITY_FILE_BYTES = 64 * 1024
_KEY_ID_RE = re.compile(r"^key_[0-9a-f]{12}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCK = threading.RLock()
_LAST_USED: dict[str, float] = {}


class DuplicateLabelError(ValueError):
    """A managed key with that label already exists."""


class LastKeyError(ValueError):
    """Revoking this key would leave no usable credential."""


class KeyNotFoundError(ValueError):
    """No managed key has that id."""


class NoUsableKeyError(ValueError):
    """API mode cannot be enabled without a usable key."""


@dataclass(frozen=True)
class ManagedKey:
    id: str
    label: str
    sha256: str
    prefix: str
    last4: str
    created_at: float
    last_used_at: float | None = None

    def public_dict(self) -> dict:
        data = asdict(self)
        data.pop("sha256", None)
        return data


@dataclass(frozen=True)
class KeyRing:
    environment_key: str | None
    legacy_key: str | None
    managed: tuple[ManagedKey, ...]


@dataclass(frozen=True)
class KeyIdentity:
    kind: str
    id: str
    label: str


def _security_file_path() -> Path:
    import app_settings

    return Path(app_settings.DATA_DIR) / "security.json"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_strong(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    key = value.strip()
    if len(key) < 16:
        return False
    if any(char.isspace() for char in key):
        return False
    return len(set(key)) > 1


def _normalize_label(label: str) -> str:
    if not isinstance(label, str):
        raise ValueError("label must be a string")
    value = label.strip()
    if not value or len(value) > MAX_LABEL_LENGTH:
        raise ValueError(f"label must be 1..{MAX_LABEL_LENGTH} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("label must not contain control characters")
    return value


def _read_raw() -> dict:
    path = _security_file_path()
    try:
        if not path.is_file():
            return {}
        if path.stat().st_size > MAX_SECURITY_FILE_BYTES:
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def _valid_record(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    if not _KEY_ID_RE.fullmatch(str(record.get("id", ""))):
        return False
    if not _SHA_RE.fullmatch(str(record.get("sha256", ""))):
        return False
    label = record.get("label")
    if not isinstance(label, str) or not 0 < len(label) <= MAX_LABEL_LENGTH:
        return False
    for key in ("prefix", "last4"):
        value = record.get(key)
        if not isinstance(value, str) or not value:
            return False
    if not isinstance(record.get("created_at"), (int, float)):
        return False
    last_used = record.get("last_used_at")
    return last_used is None or isinstance(last_used, (int, float))


def _managed_from_raw(raw: dict) -> tuple[ManagedKey, ...]:
    records = raw.get("api_keys")
    if not isinstance(records, list):
        return ()
    managed = []
    for record in records[:MAX_MANAGED_KEYS]:
        if _valid_record(record):
            managed.append(
                ManagedKey(
                    id=record["id"],
                    label=record["label"],
                    sha256=record["sha256"],
                    prefix=record["prefix"],
                    last4=record["last4"],
                    created_at=float(record["created_at"]),
                    last_used_at=(
                        float(record["last_used_at"])
                        if record.get("last_used_at") is not None
                        else None
                    ),
                )
            )
    return tuple(managed)


def load_key_ring() -> KeyRing:
    raw = _read_raw()
    legacy = raw.get("api_key")
    if not isinstance(legacy, str) or not legacy.strip():
        legacy = None
    environment = os.environ.get("DIFFUSIONBEAR_API_KEY", "").strip() or None
    return KeyRing(
        environment_key=environment,
        legacy_key=legacy,
        managed=_managed_from_raw(raw),
    )


def _is_configured(ring: KeyRing) -> bool:
    return bool(ring.environment_key or ring.legacy_key or ring.managed)


def api_mode_enabled() -> bool:
    """True only when the operator explicitly enabled the API gateway."""
    return _read_raw().get("enabled") is True


def set_api_mode(enabled: bool) -> dict:
    """Enable or disable the API gateway. Keys are preserved either way."""
    with _LOCK:
        if enabled and not has_strong_key():
            raise NoUsableKeyError(
                "configure a strong API key before enabling API mode"
            )
        raw = _read_raw()
        raw["enabled"] = bool(enabled)
        _write_raw(raw)
        return redacted_status()


def has_strong_key(ring: KeyRing | None = None) -> bool:
    ring = ring or load_key_ring()
    if ring.environment_key and _is_strong(ring.environment_key):
        return True
    if ring.legacy_key and _is_strong(ring.legacy_key):
        return True
    return bool(ring.managed)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def _write_raw(raw: dict) -> None:
    payload = dict(raw)
    payload["schema"] = SCHEMA
    payload.setdefault("api_keys", [])
    _atomic_write_text(
        _security_file_path(),
        json.dumps(payload, indent=2, allow_nan=False, sort_keys=False),
    )


def mark_used(key_id: str, when: float | None = None) -> None:
    with _LOCK:
        _LAST_USED[key_id] = time.time() if when is None else float(when)


def flush_last_used() -> None:
    with _LOCK:
        if not _LAST_USED:
            return
        raw = _read_raw()
        records = raw.get("api_keys")
        if not isinstance(records, list):
            _LAST_USED.clear()
            return
        changed = False
        for record in records:
            if not isinstance(record, dict):
                continue
            value = _LAST_USED.get(str(record.get("id", "")))
            if value is not None:
                record["last_used_at"] = value
                changed = True
        if changed:
            _write_raw(raw)
        _LAST_USED.clear()


def authenticate(supplied: str, ring: KeyRing | None = None) -> KeyIdentity | None:
    if not isinstance(supplied, str) or not supplied:
        return None
    ring = ring or load_key_ring()
    supplied_hash = _hash(supplied)

    # A weak env/legacy key must not remain a full credential just because a
    # strong managed key enabled API mode.
    if ring.environment_key and _is_strong(ring.environment_key):
        if hmac.compare_digest(supplied_hash, _hash(ring.environment_key)):
            return KeyIdentity("environment", "env", "Environment key")
    if ring.legacy_key and _is_strong(ring.legacy_key):
        if hmac.compare_digest(supplied_hash, _hash(ring.legacy_key)):
            return KeyIdentity("legacy", "legacy", "Legacy key")

    for record in ring.managed:
        if hmac.compare_digest(supplied_hash, record.sha256):
            mark_used(record.id)
            return KeyIdentity("managed", record.id, record.label)
    return None


def create_key(label: str, ring: KeyRing | None = None) -> tuple[str, ManagedKey]:
    with _LOCK:
        ring = ring or load_key_ring()
        value = _normalize_label(label)
        if len(ring.managed) >= MAX_MANAGED_KEYS:
            raise ValueError(f"maximum of {MAX_MANAGED_KEYS} managed keys")
        folded = value.casefold()
        if any(record.label.casefold() == folded for record in ring.managed):
            raise DuplicateLabelError("a key with this label already exists")

        full = f"dbk_{secrets.token_urlsafe(32)}"
        while any(hmac.compare_digest(_hash(full), record.sha256) for record in ring.managed):
            full = f"dbk_{secrets.token_urlsafe(32)}"
        record = ManagedKey(
            id=f"key_{secrets.token_hex(6)}",
            label=value,
            sha256=_hash(full),
            prefix=full[:8],
            last4=full[-4:],
            created_at=time.time(),
        )
        raw = _read_raw()
        records = raw.get("api_keys")
        if not isinstance(records, list):
            records = []
        records.append(asdict(record))
        raw["api_keys"] = records
        _write_raw(raw)
        return full, record


def revoke_key(key_id: str, ring: KeyRing | None = None) -> None:
    with _LOCK:
        ring = ring or load_key_ring()
        raw = _read_raw()
        records = raw.get("api_keys")
        if not isinstance(records, list):
            raise KeyNotFoundError("unknown key")
        target = next(
            (item for item in records if isinstance(item, dict) and item.get("id") == key_id),
            None,
        )
        if target is None:
            raise KeyNotFoundError("unknown key")
        remaining_managed = [
            item for item in records if isinstance(item, dict) and item.get("id") != key_id
        ]
        other = bool(ring.environment_key or ring.legacy_key or remaining_managed)
        if not other:
            raise LastKeyError("this is the last usable API key")
        raw["api_keys"] = remaining_managed
        _write_raw(raw)
        _LAST_USED.pop(key_id, None)


def redacted_status(
    ring: KeyRing | None = None, current: KeyIdentity | None = None
) -> dict:
    flush_last_used()
    ring = ring or load_key_ring()
    environment = None
    if ring.environment_key:
        environment = {
            "present": True,
            "prefix": ring.environment_key[:8],
            "last4": ring.environment_key[-4:],
            "strong": _is_strong(ring.environment_key),
        }
    legacy = None
    if ring.legacy_key:
        legacy = {
            "present": True,
            "prefix": ring.legacy_key[:8],
            "last4": ring.legacy_key[-4:],
            "strong": _is_strong(ring.legacy_key),
        }
    return {
        "enabled": api_mode_enabled(),
        "environment": environment or {"present": False},
        "legacy": legacy or {"present": False},
        "keys": [record.public_dict() for record in ring.managed],
        "current": (
            {"kind": current.kind, "id": current.id, "label": current.label}
            if current is not None
            else {"kind": "anonymous", "id": None, "label": None}
        ),
    }
