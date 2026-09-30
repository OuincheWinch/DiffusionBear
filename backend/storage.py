"""Storage manifest: what is on disk, what belongs to what, and what is orphaned.

Read-only by design. The point of this module is to make 28 GB of model weights
legible -- which model owns which files, how big each one is, and which files
nothing references any more. It reports reclaimable space; it does not delete.
Deletion is a separate, explicit decision, and nothing here can be coerced into
one by a stray request.

Design notes that matter:
  * Sizes come from st_size. Nothing is hashed: a 28 GB integrity pass would take
    minutes and would contend with generation for I/O, and this view is about
    layout, not verification.
  * Symlinks are never followed. A symlink pointing out of the tree would otherwise
    turn a size report into an arbitrary-file-read.
  * The result is cached and computed off the request thread, so the UI can poll it
    without the scan ever blocking an API call.
  * Credential files are reported by name and size only. Their contents are never
    read, and never leave the process.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import app_settings
import generator
from app_settings import SETTINGS_FILE
from state import LORA_FILES_DIR, LORAS_FILE, SDXL_LORA_DIR, UPLOADS_DIR

CACHE_TTL_SECONDS = 60.0
MAX_FILES_PER_DIR = 20000

# Reported by name/size only. Never read.
SECRET_FILES = ("hf_token.txt", "civitai_token.txt")


def _dir_stats(root: Path, *, max_files: int = MAX_FILES_PER_DIR) -> tuple[int, int, float | None, bool]:
    """(bytes, file_count, newest_mtime, truncated). Bounded walk, no symlink follow."""
    total = 0
    count = 0
    newest: float | None = None
    truncated = False
    if not root.is_dir():
        return 0, 0, None, False
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except (OSError, PermissionError):
            continue
        for entry in entries:
            if count >= max_files:
                truncated = True
                return total, count, newest, truncated
            try:
                # follow_symlinks=False: a link is counted as a link, never walked.
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    stat = entry.stat(follow_symlinks=False)
                    total += stat.st_size
                    count += 1
                    if newest is None or stat.st_mtime > newest:
                        newest = stat.st_mtime
            except (OSError, PermissionError):
                continue
    return total, count, newest, truncated


def _dir_children(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    try:
        return sorted(e.name for e in os.scandir(root) if e.is_dir(follow_symlinks=False))
    except (OSError, PermissionError):
        return []


# Assets the app needs that are NOT registry models. Naming them here keeps them
# out of the "unrecognised" bucket -- the prompt enhancer model and the TAESD
# decoder are both loaded at runtime and neither appears in generator.MODELS.
INTERNAL_MODEL_DIRS = {
    "qwen2.5-0.5b-instruct-4bit": "prompt enhancer",
    "taesdxl": "TAESD/TAEF decoder",
}


def _referenced_model_paths() -> dict[Path, list[str]]:
    """Map resolved model dir -> the model ids that point at it.

    resolve_local_model_path() is the single source of truth, because the registry
    is not enough: krea2-turbo declares repo "local:krea2-turbo-q4" and its weights
    live at ASSET_DIR/models/krea2-turbo-q4, which appears in no model_dir field.
    Reading only model_dir reported the app's main model as reclaimable.
    """
    out: dict[Path, list[str]] = {}
    for model_id, info in generator.MODELS.items():
        resolved = None
        try:
            resolved = generator.resolve_local_model_path(model_id)
        except Exception:
            resolved = None
        if resolved is None:
            # A "local:<name>" repo means ASSET_DIR/models/<name>: no remote source,
            # weights placed on disk by hand. krea2-turbo works this way, and it was
            # still being reported as unrecognised until this was handled.
            repo = str(info.get("repo") or "")
            if repo.startswith("local:"):
                candidate = Path(generator.ASSET_DIR) / "models" / repo.split(":", 1)[1]
                if candidate.is_dir():
                    resolved = candidate
        if resolved is None:
            # No user-registered path; fall back to a registry-declared directory.
            raw = info.get("model_dir")
            if not raw:
                continue
            try:
                resolved = Path(raw)
            except (OSError, RuntimeError):
                continue
            if not resolved.is_dir():
                continue
        try:
            out.setdefault(Path(resolved).resolve(), []).append(model_id)
        except (OSError, RuntimeError):
            continue
    return out


@dataclass
class Entry:
    name: str
    path: str
    bytes: int
    files: int
    newest: float | None = None
    truncated: bool = False
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "bytes": self.bytes,
            "files": self.files,
            "newest": self.newest,
            "truncated": self.truncated,
            **self.detail,
        }


def _scan_models(models_root: Path) -> tuple[list[Entry], list[Entry]]:
    """Return (models, unrecognised).

    "unrecognised" deliberately does NOT mean "deletable". A directory nothing
    points at is usually a model the app supports but that this build's registry
    does not name, or one the user placed by hand -- so it is reported for review
    and nothing more. The only way a false positive here hurts someone is if they
    read "reclaimable" as an instruction, which is why the wording is never that.
    """
    referenced = _referenced_model_paths()
    entries: list[Entry] = []
    for name in _dir_children(models_root):
        path = models_root / name
        size, files, newest, truncated = _dir_stats(path)
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            resolved = path
        ids = referenced.get(resolved)
        if ids:
            info = generator.MODELS.get(ids[0], {})
            status, label, engine, purpose = "registered", info.get("label", ids[0]), info.get("engine"), None
        elif name in INTERNAL_MODEL_DIRS:
            status, label, engine, purpose = "internal", name, None, INTERNAL_MODEL_DIRS[name]
        else:
            status, label, engine, purpose = "unrecognised", None, None, None
        entries.append(Entry(
            name=name, path=str(path), bytes=size, files=files, newest=newest,
            truncated=truncated,
            detail={
                "status": status,
                "model_ids": ids or [],
                "label": label,
                "engine": engine,
                "purpose": purpose,
                "reclaimable": False,  # never asserted; this module does not delete
            },
        ))
    unrecognised = [e for e in entries if e.detail["status"] == "unrecognised"]
    unrecognised.sort(key=lambda e: e.bytes, reverse=True)
    return entries, unrecognised


def _scan_loras() -> dict:
    """LoRA files, and the ones no registry entry references."""
    try:
        from state import _read_loras  # local import: avoids a cycle at import time
        registry = _read_loras()
    except Exception:
        registry = []
    referenced: set[Path] = set()
    for entry in registry or []:
        raw = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(raw, str) or not raw:
            continue
        try:
            referenced.add(Path(raw).expanduser().resolve())
        except (OSError, RuntimeError):
            continue

    orphans: list[Entry] = []
    total_bytes = 0
    total_files = 0
    present: set[Path] = set()
    for root in (LORA_FILES_DIR, SDXL_LORA_DIR):
        if not root.is_dir():
            continue
        try:
            children = list(os.scandir(root))
        except (OSError, PermissionError):
            continue
        for entry in children:
            if not entry.is_file(follow_symlinks=False):
                continue
            try:
                size = entry.stat(follow_symlinks=False).st_size
            except (OSError, PermissionError):
                continue
            total_bytes += size
            total_files += 1
            path = Path(entry.path)
            try:
                resolved = path.resolve()
            except (OSError, RuntimeError):
                resolved = path
            present.add(resolved)
            if resolved not in referenced and not entry.name.startswith("."):
                orphans.append(Entry(name=entry.name, path=str(path), bytes=size, files=1))
    orphans.sort(key=lambda e: e.bytes, reverse=True)

    # Registered entries whose file is gone. The app surfaces this nowhere, so a LoRA
    # can sit in the picker as a permanent silent failure. Counted here so the storage
    # view can say so out loud.
    missing_names: list[str] = []
    for entry in registry or []:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("path")
        if not isinstance(raw, str) or not raw:
            continue
        try:
            exists = Path(raw).expanduser().is_file()
        except (OSError, RuntimeError):
            exists = False
        if not exists:
            missing_names.append(str(entry.get("name") or Path(raw).name))

    return {
        "entries": len(registry or []),
        "bytes": total_bytes,
        "files": total_files,
        "orphans": [o.to_dict() for o in orphans],
        "missing_entries": len(missing_names),
        "missing_names": sorted(missing_names)[:50],
    }


def _scan_gallery(generated_root: Path) -> dict:
    images = thumbs = 0
    image_bytes = thumb_bytes = 0
    oldest = newest = None
    if generated_root.is_dir():
        try:
            entries = list(os.scandir(generated_root))
        except (OSError, PermissionError):
            entries = []
        for entry in entries:
            if not entry.is_file(follow_symlinks=False) or not entry.name.endswith(".png"):
                continue
            try:
                stat = entry.stat(follow_symlinks=False)
            except (OSError, PermissionError):
                continue
            if entry.name.endswith("_thumb.png"):
                thumbs += 1
                thumb_bytes += stat.st_size
            else:
                images += 1
                image_bytes += stat.st_size
            if oldest is None or stat.st_mtime < oldest:
                oldest = stat.st_mtime
            if newest is None or stat.st_mtime > newest:
                newest = stat.st_mtime
    return {
        "images": images,
        "thumbnails": thumbs,
        "image_bytes": image_bytes,
        "thumbnail_bytes": thumb_bytes,
        "bytes": image_bytes + thumb_bytes,
        "oldest": oldest,
        "newest": newest,
    }


def _scan_secrets() -> list[dict]:
    """Presence and size only. Contents are never opened."""
    out = []
    for name in SECRET_FILES:
        path = app_settings.DATA_DIR / name
        try:
            size = path.stat().st_size if path.is_file() else 0
        except (OSError, PermissionError):
            size = 0
        out.append({"name": name, "present": path.is_file(), "bytes": size})
    return out


def build_report() -> dict:
    data_dir = app_settings.DATA_DIR
    asset_dir = app_settings.ASSET_DIR
    models_root = Path(asset_dir) / "models"
    generated_root = Path(data_dir) / "generated"

    models, unrecognised = _scan_models(models_root)
    loras = _scan_loras()
    gallery = _scan_gallery(generated_root)
    uploads_bytes, uploads_files, _, _ = _dir_stats(UPLOADS_DIR)
    secrets = _scan_secrets()

    unrecognised_bytes = sum(e.bytes for e in unrecognised)
    category_bytes = {
        "models": sum(e.bytes for e in models),
        "loras": loras["bytes"],
        "gallery": gallery["bytes"],
        "uploads": uploads_bytes,
    }
    return {
        "generated_at": time.time(),
        "roots": {
            "data_dir": str(data_dir),
            "asset_dir": str(asset_dir),
            "same": str(data_dir) == str(asset_dir),
        },
        "totals": {
            "bytes": sum(category_bytes.values()),
            "by_category": category_bytes,
            "unrecognised_bytes": unrecognised_bytes,
        },
        "models": [e.to_dict() for e in sorted(models, key=lambda e: e.bytes, reverse=True)],
        "unrecognised": [e.to_dict() for e in unrecognised],
        "loras": loras,
        "gallery": gallery,
        "uploads": {"bytes": uploads_bytes, "files": uploads_files},
        "config": {
            "settings_bytes": SETTINGS_FILE.stat().st_size if SETTINGS_FILE.is_file() else 0,
            "loras_registry_bytes": LORAS_FILE.stat().st_size if LORAS_FILE.is_file() else 0,
        },
        "secrets": secrets,
    }


class _Cache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: dict | None = None
        self._at = 0.0
        self._running = False

    def get(self, max_age: float = CACHE_TTL_SECONDS) -> tuple[dict | None, bool]:
        """(cached_report, still_scanning)."""
        with self._lock:
            if self._value is not None and (time.time() - self._at) <= max_age:
                return self._value, False
            if self._running:
                return (self._value if self._value else None), True
            self._running = True
        report = build_report()
        with self._lock:
            self._value = report
            self._at = time.time()
            self._running = False
        return report, False


_CACHE = _Cache()


def get_report(force: bool = False) -> dict:
    if force:
        with _CACHE._lock:  # noqa: SLF001 - same module
            _CACHE._running = False
    report, scanning = _CACHE.get(0.0 if force else CACHE_TTL_SECONDS)
    if report is None:
        report = build_report()
    report = dict(report)
    report["scanning"] = scanning
    return report
