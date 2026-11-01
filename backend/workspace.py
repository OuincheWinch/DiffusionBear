"""Persistent pointers for the two external roots.

`data` holds settings, gallery, uploads, the LoRA registry and token files.
`asset` holds model weights and LoRA files. They can move independently.

The pointers live next to the app's default Application Support folder, outside
either root, so they stay readable after a root moves. Nothing here copies,
moves or deletes user files: switching is a pointer edit and a restart.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import app_settings


DEFAULT_ROOT = Path.home() / "Library" / "Application Support" / "DiffusionBear"
POINTERS = {"data": "data_path", "asset": "asset_path"}
LEGACY_POINTER = "store_path"


def _root(root: Path | str | None) -> Path:
    return Path(root) if root is not None else DEFAULT_ROOT


def _pointer(name: str, root: Path) -> Path | None:
    path = root / name
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute() or not candidate.is_dir():
        return None
    try:
        return candidate.resolve()
    except OSError:
        return None


def _legacy_data(root: Path) -> Path | None:
    legacy = _pointer(LEGACY_POINTER, root)
    if legacy is None:
        return None
    return legacy if legacy.name == "data" else legacy / "data"


def _write_pointer(name: str, value: Path, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    fd, tmp = tempfile.mkstemp(dir=str(root), prefix=f".{name}-", suffix=".tmp")
    try:
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(str(value))
        os.replace(tmp, root / name)
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _migrate_legacy(root: Path) -> None:
    """Turn the old single store pointer into two explicit pointers.

    Both new pointers get the old effective data directory. Only then is the old
    pointer removed. This preserves the current layout while making each root
    independently resettable.
    """
    legacy_data = _legacy_data(root)
    if legacy_data is None:
        return
    for kind, name in POINTERS.items():
        if _pointer(name, root) is None:
            _write_pointer(name, legacy_data, root)
    if all((root / name).is_file() for name in POINTERS.values()):
        try:
            (root / LEGACY_POINTER).unlink()
        except OSError:
            pass


def _inside_app_bundle(target: Path) -> bool:
    for parent in Path(__file__).resolve().parents:
        if parent.suffix.lower() != ".app":
            continue
        try:
            target.relative_to(parent)
            return True
        except ValueError:
            continue
    return False


def _validate_target(raw: str) -> Path:
    candidate = Path(str(raw).strip()).expanduser()
    if not candidate.is_absolute():
        raise ValueError("workspace path must be absolute")
    try:
        target = candidate.resolve()
    except OSError as exc:
        raise ValueError("workspace path could not be resolved") from exc
    if target == Path("/"):
        raise ValueError("the filesystem root cannot be a workspace folder")
    if _inside_app_bundle(target):
        raise ValueError("a workspace folder cannot be inside the app bundle")
    if target.exists() and not target.is_dir():
        raise ValueError("workspace path is not a folder")
    try:
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(target, 0o700)
    except OSError as exc:
        raise ValueError(f"workspace folder cannot be created: {exc}") from exc
    try:
        probe_fd, probe = tempfile.mkstemp(dir=str(target), prefix=".diffusionbear-write-")
        os.close(probe_fd)
        os.unlink(probe)
    except OSError as exc:
        raise ValueError(f"workspace folder is not writable: {exc}") from exc
    return target


def get_workspace(root: Path | str | None = None) -> dict:
    base = _root(root)
    _migrate_legacy(base)

    default_data = (base / "data").resolve()
    selected_data = _pointer(POINTERS["data"], base) or (_legacy_data(base))
    selected_asset = _pointer(POINTERS["asset"], base) or (_legacy_data(base))
    effective_data = selected_data or default_data
    effective_asset = selected_asset or default_data

    active_data = Path(app_settings.DATA_DIR).resolve()
    active_asset = Path(app_settings.ASSET_DIR).resolve()
    return {
        "active": {
            "data_dir": str(active_data),
            "asset_dir": str(active_asset),
        },
        "selected": {
            "data": str(selected_data) if selected_data else None,
            "asset": str(selected_asset) if selected_asset else None,
        },
        "defaults": {
            "data": str(default_data),
            "asset": str(default_data),
        },
        "effective": {
            "data": str(effective_data),
            "asset": str(effective_asset),
        },
        "restart_required": active_data != effective_data or active_asset != effective_asset,
    }


def set_workspace(kind: str, path: str | None, root: Path | str | None = None) -> dict:
    if kind not in POINTERS:
        raise ValueError("unknown workspace root")
    base = _root(root)
    _migrate_legacy(base)
    pointer = base / POINTERS[kind]
    if path is None or not str(path).strip():
        try:
            pointer.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ValueError(f"workspace pointer could not be cleared: {exc}") from exc
        return get_workspace(root=base)

    target = _validate_target(path)
    _write_pointer(POINTERS[kind], target, base)
    return get_workspace(root=base)
