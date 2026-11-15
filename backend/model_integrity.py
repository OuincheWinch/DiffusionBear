"""First-run-of-version SHA-256 integrity verification for built-in registry models.

Verifies only built-in (legacy/official) registry models in ``generator.MODELS``
that are currently installed on disk; user-installed custom models are never
touched. On the first launch of a new ``app_version.APP_VERSION``, a background
worker hashes installed built-in models against ``model_manifests.json`` and
exposes live progress so the UI can display a status message while hashing.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import app_settings
import app_version
import generator

MANIFEST_PATH = Path(__file__).resolve().parent / "model_manifests.json"
_HASH_CHUNK_BYTES = 4 * 1024 * 1024  # 4 MiB streaming chunks

_state_lock = threading.RLock()
_manifest_cache: dict | None = None

_live_status: dict = {
    "running": False,
    "app_version": app_version.APP_VERSION,
    "current_model": None,
    "current_label": None,
    "completed_models": 0,
    "total_models": 0,
    "message": "",
}


def _state_file() -> Path:
    return Path(app_settings.DATA_DIR) / "model_integrity.json"


def load_manifests() -> dict:
    global _manifest_cache
    with _state_lock:
        if _manifest_cache is not None:
            return _manifest_cache
        if not MANIFEST_PATH.is_file():
            _manifest_cache = {"app_version": app_version.APP_VERSION, "models": {}}
            return _manifest_cache
        try:
            data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("models"), dict):
                _manifest_cache = data
                return _manifest_cache
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            print(f"[integrity] WARN failed to load {MANIFEST_PATH}: {exc}", flush=True)
        _manifest_cache = {"app_version": app_version.APP_VERSION, "models": {}}
        return _manifest_cache


_EMPTY_STATE = {"verified_app_version": None, "last_run_at": None, "results": {}}
_state_cache_key: tuple[Path, int | None] | None = None
_state_cache_data: dict | None = None


def _read_saved_state() -> dict:
    global _state_cache_key, _state_cache_data
    path = _state_file()
    if not path.is_file():
        _state_cache_key = (path, None)
        _state_cache_data = {"verified_app_version": None, "last_run_at": None, "results": {}}
        return {**_state_cache_data, "results": {}}
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = None
    if _state_cache_data is not None and _state_cache_key == (path, mtime_ns):
        return {
            **_state_cache_data,
            "results": dict(_state_cache_data.get("results") or {}),
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            parsed = {
                "verified_app_version": data.get("verified_app_version"),
                "last_run_at": data.get("last_run_at"),
                "results": data.get("results") if isinstance(data.get("results"), dict) else {},
            }
            _state_cache_key = (path, mtime_ns)
            _state_cache_data = parsed
            return {**parsed, "results": dict(parsed["results"])}
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    return {"verified_app_version": None, "last_run_at": None, "results": {}}


def _write_saved_state(state: dict) -> None:
    global _state_cache_key, _state_cache_data
    path = _state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".model_integrity_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = None
        _state_cache_key = (path, mtime_ns)
        _state_cache_data = {
            "verified_app_version": state.get("verified_app_version"),
            "last_run_at": state.get("last_run_at"),
            "results": dict(state.get("results") or {}),
        }
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _expected_files_for_model(model_id: str, manifest_entry: dict, minfo: dict) -> tuple[dict, dict]:
    """Return (main_files, te_files) filtered for DiffusionBear's layout rules."""
    raw_files = manifest_entry.get("files") or {}
    has_model_index = "model_index.json" in raw_files
    is_sdxl = minfo.get("engine") == "sdxl" and has_model_index
    fp16_dirs: set[str] = set()
    if is_sdxl:
        fp16_dirs = {
            rel_path.split("/", 1)[0]
            for rel_path in raw_files
            if "/" in rel_path and rel_path.endswith(".fp16.safetensors")
        }
    main_files: dict[str, dict] = {}
    for rel_path, meta in raw_files.items():
        # SDXL diffusers repos sometimes also host root-level single-file checkpoints.
        if is_sdxl and "/" not in rel_path and rel_path.endswith(".safetensors"):
            continue
        # When an SDXL component ships both fp16 and fp32/sharded weights, prefer fp16.
        if is_sdxl and "/" in rel_path and rel_path.endswith(".safetensors") and not rel_path.endswith(".fp16.safetensors"):
            if rel_path.split("/", 1)[0] in fp16_dirs:
                continue
        main_files[rel_path] = meta
    te_files = dict(manifest_entry.get("te_files") or {})
    return main_files, te_files


def _resolve_builtin_local_dir(model_id: str) -> Path | None:
    """Locate local weights directory for a built-in model if installed."""
    if model_id not in generator.MODELS:
        return None
    resolved = generator.resolve_local_model_path(model_id)
    if resolved is not None and resolved.is_dir():
        return resolved
    # Fallback: an explicitly configured or default dir that failed partial validation
    # (e.g. an older MOSS install) still exists on disk and should be inspected.
    configured = app_settings.configured_model_local_path(model_id)
    if configured:
        p = Path(configured).expanduser()
        if p.is_dir():
            return generator._resolve_hf_cache_snapshot(p)
    default_dir = Path(app_settings.ASSET_DIR) / "models" / model_id
    if default_dir.is_dir() and any(default_dir.rglob("*.safetensors")):
        return default_dir
    return None


def verify_model(model_id: str, *, update_state: bool = True) -> dict | None:
    """Verify a single built-in registry model against the embedded manifest."""
    if model_id not in generator.MODELS:
        return None
    minfo = generator.MODELS[model_id]
    manifests = load_manifests()
    manifest_entry = (manifests.get("models") or {}).get(model_id)
    if not manifest_entry:
        return None

    local_dir = _resolve_builtin_local_dir(model_id)
    if local_dir is None:
        if update_state:
            with _state_lock:
                saved = _read_saved_state()
                if model_id in saved["results"]:
                    saved["results"].pop(model_id, None)
                    _write_saved_state(saved)
        return None

    main_files, te_files = _expected_files_for_model(model_id, manifest_entry, minfo)
    if model_id == "krea2-turbo" and generator.krea_text_encoder_is_quantized(local_dir):
        # Source bf16 text_encoder shards were intentionally pruned after quantizing TE.
        main_files = {k: v for k, v in main_files.items() if not k.startswith("text_encoder/")}
    if minfo.get("engine") == "qwen":
        # Qwen installs quantize text_encoder in-place to 4-bit and normalize processor/tokenizer
        # configs locally; drop non-essential conversion docs and relax size/hash on normalized files.
        te_dir_qwen = local_dir / "text_encoder"
        local_qwen_te = (
            [f for f in sorted(te_dir_qwen.glob("*.safetensors")) if f.is_file() and f.stat().st_size > 0]
            if te_dir_qwen.is_dir()
            else []
        )
        filtered_qwen: dict[str, dict] = {}
        for k, v in main_files.items():
            if k in ("QWEN21_CONVERSION.json", "mererun_model.json"):
                continue
            if local_qwen_te and k.startswith("text_encoder/"):
                continue
            if (
                k in ("model_index.json", "scheduler/scheduler_config.json")
                or k.startswith("processor/")
                or k.startswith("tokenizer/")
            ):
                cand = local_dir / k
                if cand.is_file() and cand.stat().st_size > 0:
                    filtered_qwen[k] = {"size": 0, "sha256": None}
                    continue
            filtered_qwen[k] = v
        for te_file in local_qwen_te:
            filtered_qwen[f"text_encoder/{te_file.name}"] = {"size": 0, "sha256": None}
        main_files = filtered_qwen
    if minfo.get("engine") == "sdxl":
        # SDXL installs normalize component weights to canonical diffusion_pytorch_model.safetensors /
        # model.safetensors and write normalized config/tokenizer JSONs.
        filtered_sdxl: dict[str, dict] = {}
        for k, v in main_files.items():
            comp = k.split("/", 1)[0] if "/" in k else ""
            if comp in ("unet", "vae", "text_encoder", "text_encoder_2") and (
                k.endswith(".safetensors") or k.endswith(".safetensors.index.json")
            ):
                comp_dir = local_dir / comp
                canonical_name = (
                    "diffusion_pytorch_model.safetensors"
                    if comp in ("unet", "vae")
                    else "model.safetensors"
                )
                if comp_dir.is_dir() and (comp_dir / canonical_name).is_file() and (comp_dir / canonical_name).stat().st_size > 0:
                    filtered_sdxl[f"{comp}/{canonical_name}"] = {"size": 0, "sha256": None}
                    continue
            if k.endswith((".json", ".txt")):
                cand = local_dir / k
                if cand.is_file() and cand.stat().st_size > 0:
                    filtered_sdxl[k] = {"size": 0, "sha256": None}
                    continue
                if k.endswith("/special_tokens_map.json") and (cand.parent / "tokenizer_config.json").is_file():
                    continue
            filtered_sdxl[k] = v
        main_files = filtered_sdxl
    tok_symlinked = (local_dir / "tokenizer").is_symlink() and (local_dir / "tokenizer" / "tokenizer_config.json").is_file()
    proc_symlinked = (local_dir / "processor").is_symlink() and (local_dir / "processor" / "tokenizer_config.json").is_file()
    if tok_symlinked or proc_symlinked:
        main_files = {
            k: v for k, v in main_files.items()
            if not (tok_symlinked and k.startswith("tokenizer/"))
            and not (proc_symlinked and k.startswith("processor/"))
        }

    missing_files: list[dict] = []
    mismatched_files: list[dict] = []
    valid_bytes = 0
    repair_bytes = 0
    total_bytes = 0

    def _check_group(base_dir: Path, files_map: dict, target_kind: str, repo_id: str) -> None:
        nonlocal valid_bytes, repair_bytes, total_bytes
        for rel_path, meta in files_map.items():
            expected_size = int(meta.get("size") or 0)
            expected_sha = meta.get("sha256")
            total_bytes += expected_size
            candidate = base_dir / rel_path
            if not candidate.is_file():
                missing_files.append({
                    "path": rel_path,
                    "size": expected_size,
                    "sha256": expected_sha,
                    "target": target_kind,
                    "repo": repo_id,
                })
                repair_bytes += expected_size
                continue
            try:
                actual_size = candidate.stat().st_size
            except OSError:
                missing_files.append({
                    "path": rel_path,
                    "size": expected_size,
                    "sha256": expected_sha,
                    "target": target_kind,
                    "repo": repo_id,
                })
                repair_bytes += expected_size
                continue
            if expected_size > 0 and actual_size != expected_size:
                mismatched_files.append({
                    "path": rel_path,
                    "size": expected_size,
                    "actual_size": actual_size,
                    "sha256": expected_sha,
                    "target": target_kind,
                    "repo": repo_id,
                    "reason": "size_mismatch",
                })
                repair_bytes += expected_size
                continue
            if expected_sha:
                try:
                    actual_sha = _sha256_file(candidate)
                except OSError:
                    actual_sha = None
                if actual_sha != expected_sha:
                    mismatched_files.append({
                        "path": rel_path,
                        "size": expected_size,
                        "actual_size": actual_size,
                        "sha256": expected_sha,
                        "actual_sha256": actual_sha,
                        "target": target_kind,
                        "repo": repo_id,
                        "reason": "sha256_mismatch",
                    })
                    repair_bytes += expected_size
                    continue
            valid_bytes += actual_size

    _check_group(local_dir, main_files, "main", manifest_entry.get("repo", ""))

    if minfo.get("engine") == "sdxl" and not any(k.endswith(".safetensors") for k in main_files):
        # Juggernaut-style repos distribute .bin upstream and convert locally to .safetensors.
        # Verify converted component weights exist instead of silently reporting 0-weight "ok".
        for sub in ("unet", "vae", "text_encoder", "text_encoder_2"):
            sub_dir = local_dir / sub
            if not (sub_dir.is_dir() and any(sub_dir.glob("*.safetensors"))):
                missing_files.append({
                    "path": f"{sub}/diffusion_pytorch_model.safetensors" if sub in ("unet", "vae") else f"{sub}/model.safetensors",
                    "size": 0,
                    "sha256": None,
                    "target": "main",
                    "repo": manifest_entry.get("repo", ""),
                })

    te_sibling = manifest_entry.get("te_mlx_sibling") or minfo.get("te_mlx_sibling")
    te_repo = manifest_entry.get("te_mlx_repo") or minfo.get("te_mlx_repo")
    if te_files and te_sibling and te_repo:
        te_dir = local_dir.parent / te_sibling
        _check_group(te_dir, te_files, "te", te_repo)

    if not missing_files and not mismatched_files:
        status = "ok"
    elif valid_bytes == 0 or (len(missing_files) + len(mismatched_files)) == (len(main_files) + len(te_files)):
        status = "legacy_incompatible"
    elif mismatched_files:
        status = "mismatch"
    else:
        status = "incomplete"

    result = {
        "model_id": model_id,
        "status": status,
        "checked_at": time.time(),
        "app_version": app_version.APP_VERSION,
        "local_path": str(local_dir),
        "repo": manifest_entry.get("repo"),
        "revision": manifest_entry.get("revision"),
        "missing_files": missing_files,
        "mismatched_files": mismatched_files,
        "missing_count": len(missing_files),
        "mismatched_count": len(mismatched_files),
        "valid_bytes": valid_bytes,
        "repair_bytes": repair_bytes,
        "total_bytes": total_bytes,
    }

    if update_state:
        with _state_lock:
            saved = _read_saved_state()
            saved["results"][model_id] = result
            _write_saved_state(saved)

    return result


def get_integrity_snapshot() -> dict:
    with _state_lock:
        saved = _read_saved_state()
        return {
            "running": _live_status["running"],
            "app_version": app_version.APP_VERSION,
            "verified_app_version": saved.get("verified_app_version"),
            "last_run_at": saved.get("last_run_at"),
            "current_model": _live_status["current_model"],
            "current_label": _live_status["current_label"],
            "completed_models": _live_status["completed_models"],
            "total_models": _live_status["total_models"],
            "message": _live_status["message"],
            "results": saved.get("results") or {},
        }


def get_cached_model_integrity(model_id: str) -> dict | None:
    if model_id not in generator.MODELS:
        return None
    with _state_lock:
        saved = _read_saved_state()
        return (saved.get("results") or {}).get(model_id)


def clear_model_result(model_id: str) -> None:
    """Drop cached integrity entry when a model is removed or unlinked."""
    with _state_lock:
        saved = _read_saved_state()
        if model_id in (saved.get("results") or {}):
            saved["results"].pop(model_id, None)
            _write_saved_state(saved)


def _run_verification_worker(target_model_id: str | None = None, *, mark_version_done: bool = True) -> None:
    try:
        if target_model_id:
            candidates = [target_model_id] if target_model_id in generator.MODELS else []
        else:
            candidates = [
                mid for mid in generator.MODELS
                if _resolve_builtin_local_dir(mid) is not None
            ]
        with _state_lock:
            _live_status.update({
                "running": True,
                "app_version": app_version.APP_VERSION,
                "current_model": None,
                "current_label": None,
                "completed_models": 0,
                "total_models": len(candidates),
                "message": f"Verifying {len(candidates)} installed built-in model(s) for v{app_version.APP_VERSION}...",
            })
        all_succeeded = True
        for idx, mid in enumerate(candidates):
            label = generator.MODELS[mid].get("label", mid)
            with _state_lock:
                _live_status.update({
                    "current_model": mid,
                    "current_label": label,
                    "completed_models": idx,
                    "message": f"Hashing {label} ({idx + 1}/{len(candidates)})...",
                })
            try:
                res = verify_model(mid, update_state=True)
                if res and res.get("status") != "ok":
                    print(
                        f"[integrity] {mid}: status={res['status']} "
                        f"missing={res['missing_count']} mismatched={res['mismatched_count']} "
                        f"repair_bytes={res['repair_bytes']}",
                        flush=True,
                    )
            except Exception as exc:
                all_succeeded = False
                print(f"[integrity] WARN verification failed for {mid}: {exc}", flush=True)
        with _state_lock:
            saved = _read_saved_state()
            if mark_version_done and not target_model_id and all_succeeded:
                saved["verified_app_version"] = app_version.APP_VERSION
            saved["last_run_at"] = time.time()
            _write_saved_state(saved)
    finally:
        with _state_lock:
            _live_status.update({
                "running": False,
                "current_model": None,
                "current_label": None,
                "completed_models": _live_status["total_models"],
                "message": "",
            })


def start_verification(target_model_id: str | None = None, *, mark_version_done: bool = True) -> bool:
    """Start background SHA verification if not already running."""
    with _state_lock:
        if _live_status["running"]:
            return False
        _live_status["running"] = True
        _live_status["app_version"] = app_version.APP_VERSION
        _live_status["message"] = f"Starting model verification for v{app_version.APP_VERSION}..."
    threading.Thread(
        target=_run_verification_worker,
        kwargs={"target_model_id": target_model_id, "mark_version_done": mark_version_done},
        daemon=True,
    ).start()
    return True


def start_first_run_verification_if_needed() -> bool:
    """Trigger SHA verification once per new app_version on startup."""
    saved = _read_saved_state()
    if saved.get("verified_app_version") == app_version.APP_VERSION:
        return False
    return start_verification(target_model_id=None, mark_version_done=True)
