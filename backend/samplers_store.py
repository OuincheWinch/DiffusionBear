"""User-declared custom samplers, stored under DATA_DIR/samplers.

A custom sampler is a single .py file exposing a top-level dict:

    SAMPLER = {
        "name": "my_euler",
        "label": "My Euler",
        "engines": ["mflux", "sdxl", "moss"],
        "mflux": "MyFluxScheduler",   # class in this file subclassing
                                      # mflux.models.common.schedulers.BaseScheduler
        "sdxl": "MySdxlScheduler",    # class with the sdxl_engine scheduler
                                      # protocol (set_timesteps/timesteps/
                                      # init_noise_sigma/scale_model_input/step)
        "moss": "MyMossScheduler",    # class subclassing MOSSImageScheduler
    }

Engines not listed in "engines" never see the sampler. The id exposed to the
UI and to GenerateRequest.sampler is "custom:<name>".

This module is imported by all three runtimes (repo venv for mflux, the SDXL
python 3.14 venv, the MOSS repo venv), so it must stay stdlib-only. The
protected pack (backend/sampler_pack) re-extracts into the store at startup:
pack entries are ours, cannot be deleted through the API, and are restored
byte-identically when missing or corrupted.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(os.environ.get("DIFFUSIONBEAR_DATA_DIR")
                or Path(__file__).resolve().parent / "data")
SAMPLERS_DIR = DATA_DIR / "samplers"
PACK_DIR = Path(__file__).resolve().parent / "sampler_pack"

CUSTOM_PREFIX = "custom:"
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,39}$")
MAX_SAMPLER_BYTES = 2 * 1024 * 1024
DOWNLOAD_TIMEOUT_S = 30
KNOWN_ENGINES = ("mflux", "sdxl", "moss")


def is_custom(sampler_id) -> bool:
    return str(sampler_id or "").startswith(CUSTOM_PREFIX)


def custom_name(sampler_id) -> str:
    return str(sampler_id or "")[len(CUSTOM_PREFIX):]


def custom_id(name: str) -> str:
    return f"{CUSTOM_PREFIX}{name}"


def _manifest_path(name: str) -> Path:
    return SAMPLERS_DIR / name / "manifest.json"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_module(path: Path, name: str):
    """Exec a sampler file and return its module. Raises ValueError."""
    if not path.is_file():
        raise ValueError(f"sampler file not found: {path}")
    spec = importlib.util.spec_from_file_location(f"bear_custom_sampler_{name}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load sampler module: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ValueError(f"sampler module failed to import: {exc}") from exc
    meta = getattr(module, "SAMPLER", None)
    if not isinstance(meta, dict):
        raise ValueError('sampler file must expose a SAMPLER = {...} dict')
    if str(meta.get("name") or "").strip() != name:
        raise ValueError(f"SAMPLER['name'] must be {name!r}")
    engines = meta.get("engines")
    if not isinstance(engines, list) or not engines or \
            not all(e in KNOWN_ENGINES for e in engines):
        raise ValueError(f"SAMPLER['engines'] must be a non-empty subset of {list(KNOWN_ENGINES)}")
    for engine in engines:
        cls_name = meta.get(engine)
        if not cls_name or not hasattr(module, str(cls_name)):
            raise ValueError(f"engine {engine!r} declared but class {cls_name!r} is missing")
    return module


def get_entry(name: str) -> dict | None:
    """Manifest + integrity of one installed sampler, None when absent."""
    if not NAME_RE.match(str(name or "")):
        return None
    mp = _manifest_path(name)
    if not mp.is_file():
        return None
    try:
        manifest = json.loads(mp.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    py = SAMPLERS_DIR / name / "sampler.py"
    manifest["file_present"] = py.is_file()
    manifest["intact"] = py.is_file() and manifest.get("sha256") == _file_sha256(py)
    return manifest


def list_samplers() -> list[dict]:
    if not SAMPLERS_DIR.is_dir():
        return []
    out = []
    for d in sorted(SAMPLERS_DIR.iterdir()):
        if d.is_dir():
            entry = get_entry(d.name)
            if entry is not None:
                entry["id"] = custom_id(d.name)
                out.append(entry)
    return out


def custom_ids_for_engine(engine: str) -> list[str]:
    engine = str(engine or "").lower()
    return [
        e["id"] for e in list_samplers()
        if engine in (e.get("engines") or []) and e.get("intact")
    ]


def remove(name: str) -> None:
    entry = get_entry(name)
    if entry is None:
        raise ValueError(f"unknown custom sampler: {name}")
    if entry.get("protected"):
        raise PermissionError(f"{name} is part of the protected default pack and cannot be removed")
    shutil.rmtree(SAMPLERS_DIR / name, ignore_errors=False)


def _validate_url(url: str) -> None:
    ok = (
        re.match(r"^https://[^\s]+$", str(url or ""))
        or re.match(r"^http://(localhost|127\.0\.0\.1)(:\d+)?/[^\s]*$", str(url or ""))
        # file:// installs the same trust class the download would execute:
        # locally authored sampler files declared straight from disk.
        or re.match(r"^file:///[^\s]+$", str(url or ""))
    )
    if not ok:
        raise ValueError("sampler URL must be https, http on localhost, or file://")


def install_from_url(url: str, name: str | None = None) -> dict:
    """Download a .py from url, validate, install into the store."""
    _validate_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": "DiffusionBear/1.0"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp:
        data = resp.read(MAX_SAMPLER_BYTES + 1)
    return install_from_bytes(data, name, source=url)


def install_from_path(path: str, name: str | None = None) -> dict:
    """Install a locally authored .py straight from disk (native file-picker flow).

    Same trust class as a download (the file is executed locally on use), same
    validation: only the transport differs, no HTTP involved.
    """
    p = Path(str(path or ""))
    if not p.is_file():
        raise ValueError(f"sampler file not found: {p}")
    if p.suffix != ".py":
        raise ValueError("sampler file must be a .py file")
    data = p.read_bytes()
    if len(data) > MAX_SAMPLER_BYTES:
        raise ValueError(f"sampler file exceeds the {MAX_SAMPLER_BYTES // (1 << 10)} KiB limit")
    return install_from_bytes(data, name, source=str(p))


def install_from_bytes(data: bytes, name: str | None = None, source: str = "") -> dict:
    """Validate sampler bytes and install them under the resolved name."""
    if len(data) > MAX_SAMPLER_BYTES:
        raise ValueError(f"sampler file exceeds the {MAX_SAMPLER_BYTES // (1 << 10)} KiB limit")

    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / "probe.py"
        probe.write_bytes(data)
        # First pass without a name: read SAMPLER['name'] off the file.
        module = _load_module(probe, str(name) if name else "__probe__") if name else None
        if module is not None:
            meta = module.SAMPLER
        else:
            try:
                meta = getattr(_read_meta_probe(probe), "SAMPLER", None)
            except Exception as exc:
                raise ValueError(f"cannot read sampler file: {exc}") from exc
            if not isinstance(meta, dict) or not NAME_RE.match(str(meta.get("name") or "")):
                raise ValueError("cannot determine sampler name (SAMPLER['name'] missing or invalid)")
        final_name = str(name or meta.get("name"))
        if not NAME_RE.match(final_name):
            raise ValueError(f"sampler name must match {NAME_RE.pattern}")
        # Second, real validation under the final name.
        module = _load_module(probe, final_name)
        meta = dict(module.SAMPLER)
        meta["engines"] = [str(e) for e in meta.get("engines", [])]

        target = SAMPLERS_DIR / final_name
        target.mkdir(parents=True, exist_ok=True)
        py = target / "sampler.py"
        tmp_py = target / ".sampler.py.part"
        tmp_py.write_bytes(data)
        os.replace(tmp_py, py)

        manifest = {
            "name": final_name,
            "label": str(meta.get("label") or final_name),
            "engines": meta["engines"],
            "url": source,
            "sha256": _file_sha256(py),
            "protected": False,
            "installed_at": int(time.time()),
        }
        tmp_manifest = target / ".manifest.json.part"
        tmp_manifest.write_text(json.dumps(manifest, indent=1))
        os.replace(tmp_manifest, _manifest_path(final_name))
    manifest["id"] = custom_id(final_name)
    manifest["file_present"] = True
    manifest["intact"] = True
    return manifest


def _read_meta_probe(path: Path):
    """Name discovery pass: exec and return the module without a pinned name."""
    spec = importlib.util.spec_from_file_location("bear_custom_sampler_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def self_heal() -> list[str]:
    """Re-extract the protected pack into the store. Returns action strings."""
    actions: list[str] = []
    if not PACK_DIR.is_dir():
        return actions
    for pack in sorted(PACK_DIR.iterdir()):
        pack_manifest = pack / "manifest.json"
        pack_py = pack / "sampler.py"
        if not (pack_manifest.is_file() and pack_py.is_file()):
            continue
        try:
            expected = json.loads(pack_manifest.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        name = str(expected.get("name") or "")
        if not NAME_RE.match(name):
            continue
        want_sha = expected.get("sha256") or _file_sha256(pack_py)
        current = get_entry(name)
        if current is not None and current.get("intact") and current.get("protected"):
            continue
        target = SAMPLERS_DIR / name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pack_py, target / "sampler.py")
        fixed = dict(expected)
        fixed["sha256"] = want_sha
        fixed["protected"] = True
        (_manifest_path(name)).write_text(json.dumps(fixed, indent=1))
        actions.append(f"restored {name}")
    return actions


def get_class(name: str, engine: str):
    """Loaded sampler class for engine, or None. Engines import this."""
    entry = get_entry(name)
    if entry is None or not entry.get("intact") or engine not in (entry.get("engines") or []):
        return None
    module = _load_module(SAMPLERS_DIR / name / "sampler.py", name)
    return getattr(module, str(module.SAMPLER.get(engine)), None)


_mflux_registered = False


def ensure_mflux_registered() -> int:
    """Register every intact mflux custom sampler into mflux's own registry.

    mflux resolves scheduler ids through SCHEDULER_REGISTRY first, so a
    "custom:<name>" id works natively once the class is registered under it.
    Idempotent; returns the number of live registrations.
    """
    global _mflux_registered
    if _mflux_registered:
        return 1
    try:
        from mflux.models.common.schedulers import SCHEDULER_REGISTRY, register_contrib
        from mflux.models.common.schedulers.base_scheduler import BaseScheduler
    except Exception:
        return 0
    count = 0
    for entry in list_samplers():
        if "mflux" not in (entry.get("engines") or []):
            continue
        name = entry["name"]
        try:
            module = _load_module(SAMPLERS_DIR / name / "sampler.py", name)
            cls = getattr(module, str(module.SAMPLER.get("mflux")), None)
            if cls is None or not (isinstance(cls, type) and issubclass(cls, BaseScheduler)):
                raise ValueError("mflux class must subclass BaseScheduler")
        except Exception as exc:
            print(f"[samplers] custom:{name} unusable for mflux: {exc}", flush=True)
            continue
        register_contrib(cls, scheduler_name=custom_id(name))
        count += 1
    if count:
        _mflux_registered = True
    return count
