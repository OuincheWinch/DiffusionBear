"""Persistent user-configurable settings for DiffusionBear.

Settings are stored as plain JSON in ``data/settings.json`` next to the other
runtime state (LoRA registry, downloads, …). Loads lazily, writes atomically,
and rejects unknown/invalid keys so the rest of the app can trust the shape.
"""

import json
import math
import os
import re
import tempfile
import threading
from pathlib import Path

def _abs_from_env(name: str) -> Path | None:
    configured = os.environ.get(name, "").strip()
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return path.resolve()


# Where the app's own mutable state lives: settings.json, the gallery, the LoRA
# registry, uploads, queue files and the engine pid files.
#
# Defaults to backend/data, which is what a source checkout wants. The asset dir is
# deliberately NOT reused for this: the dev workflow points it at a shared model
# store while the working copy keeps its own gallery, and collapsing the two would
# change that. But in the standalone .app the checkout *is* the bundle, so writing
# here would drop the gallery and settings inside the signed app -- growing without
# bound and invalidating the bundle's own code signature on the next launch. The
# shell therefore sets the data-dir variable alongside the asset-dir one.
#
# The variables were renamed MLX_DIFFUSION_* -> DIFFUSIONBEAR_* with the app. The old
# names still win when set, so existing launchd units, run.sh callers and any script
# a user already wrote keep working; nothing has to be updated in lockstep.
DATA_DIR = (
    _abs_from_env("DIFFUSIONBEAR_DATA_DIR")
    or _abs_from_env("MLX_DIFFUSION_DATA_DIR")
    or (Path(__file__).resolve().parent / "data")
)


def _resolve_asset_dir() -> Path:
    return (
        _abs_from_env("DIFFUSIONBEAR_ASSET_DIR")
        or _abs_from_env("MLX_DIFFUSION_ASSET_DIR")
        or DATA_DIR
    )


ASSET_DIR = _resolve_asset_dir()
SETTINGS_FILE = DATA_DIR / "settings.json"

_lock = threading.RLock()

_SAMPLERS = {
    "euler_trailing",
    "dpmpp_2m_karras",
    "euler_a_substep",
    "euler_a",
    "euler",
    "ddim",
    "linear",
    "er_sde",
    "flow_match_euler_discrete",
}

DEFAULTS = {
    # Personalization: credit embedded in Civitai/EXIF metadata. Empty -> "DiffusionBear".
    "artist_name": "",
    # Default output format for new generations ("png" | "jpeg").
    "default_output_format": "png",
    # Default stealth mode (no metadata embedded) for new generations.
    "default_stealth": False,
    # Default fast-VAE (TAEF) toggle.
    "default_fast_vae": True,
    # Optional global sampler fallback (applies to models that support it).
    "default_sampler": "",
    # Default DeepCache interval for SDXL (1 = disabled).
    "default_cache_interval": 1,
    # Interface language. "auto" follows the OS; the picker in Settings overrides it
    # with a concrete code, which is then what gets persisted.
    "language": "auto",
    # Per-model overrides: model_id -> {steps, guidance, sampler, cache_interval, fast_vae, width, height}.
    "model_defaults": {},
    # Per-model local install overrides: model_id -> directory that already holds
    # the weights on disk (an HF cache snapshot or a folder downloaded elsewhere).
    # When set, generation loads from this path instead of re-downloading, and
    # "uninstall" only clears the pointer (the user's files are never deleted).
    "model_paths": {},
    # Per-engine custom system-prompt instructions for the prompt enhancer
    # (engine key -> custom "engine guidance" text; "" = use built-in profile).
    "prompt_enhancer": {
        "flux2": "",
        "sdxl": "",
        "krea2": "",
        "z-image-turbo": "",
        "qwen": "",
    },
    # Same, but for the structured JSON output mode (independent override).
    "prompt_enhancer_json": {
        "flux2": "",
        "sdxl": "",
        "krea2": "",
        "z-image-turbo": "",
        "qwen": "",
    },
    # Engine runtime tuning (None = fall back to env var / built-in default).
    # GB wired limits for the Metal allocator; 0 = unbounded (env MLX_WIRED_LIMIT_GB,
    # MLX_KREA_WIRED_LIMIT_GB). Idle kill seconds for the mflux/SDXL watchdogs;
    # 0 = never auto-release. Applied lazily (next generation / idle rearm).
    "memory_wired_limit_gb": None,
    "memory_krea_wired_limit_gb": None,
    "krea_vae_tile_size": None,
    "idle_kill_s_mflux": None,
    "idle_kill_s_sdxl": None,
    "idle_kill_s_qwen": None,
}

# Engine keys used by prompt_enhancer.ENGINE_PROFILES.
PROMPT_ENHANCER_KEYS = {"flux2", "sdxl", "krea2", "z-image-turbo", "qwen"}
_PROMPT_ENHANCER_MAX = 8000
_MAX_SETTINGS_BYTES = 8 * 1024 * 1024
# Accepted interface languages, mirroring frontend/src/i18n/languages.js. A code
# added there and not here is silently rejected on save, and the picker's choice
# appears to do nothing -- so the two lists are asserted equal by test_i18n.py.
LANGUAGE_CODES = frozenset({"auto", "en", "fr", "de", "it", "es", "zh", "ja", "pt", "ko"})

_MAX_MODEL_SETTINGS = 64
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,119}$")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_model_id(model_id) -> bool:
    return isinstance(model_id, str) and _MODEL_ID_RE.fullmatch(model_id) and ".." not in model_id


def _valid_model_path(value) -> bool:
    if not isinstance(value, str) or not 0 < len(value) <= 4096 or "\x00" in value:
        return False
    try:
        return Path(value).expanduser().is_absolute()
    except (OSError, RuntimeError):
        return False


_MODEL_DEFAULT_KEYS = {
    "steps": lambda v: _is_int(v) and 1 <= v <= 50,
    "guidance": lambda v: _is_number(v) and 0.0 <= v <= 30.0,
    "sampler": lambda v: isinstance(v, str) and (not v or v in _SAMPLERS),
    "cache_interval": lambda v: _is_int(v) and 1 <= v <= 10,
    "fast_vae": lambda v: isinstance(v, bool),
    "width": lambda v: _is_int(v) and 256 <= v <= 4096,
    "height": lambda v: _is_int(v) and 256 <= v <= 4096,
}

_VALIDATORS = {
    "artist_name": lambda v: isinstance(v, str) and len(v) <= 200,
    "default_output_format": lambda v: v in ("png", "jpeg"),
    "default_stealth": lambda v: isinstance(v, bool),
    "default_fast_vae": lambda v: isinstance(v, bool),
    "default_sampler": lambda v: isinstance(v, str) and (not v or v in _SAMPLERS),
    "default_cache_interval": lambda v: _is_int(v) and 1 <= v <= 10,
    "language": lambda v: v in LANGUAGE_CODES,
    "model_defaults": lambda v: isinstance(v, dict) and len(v) <= _MAX_MODEL_SETTINGS,
    "model_paths": lambda v: isinstance(v, dict) and len(v) <= _MAX_MODEL_SETTINGS,
    "prompt_enhancer": lambda v: isinstance(v, dict) and len(v) <= len(PROMPT_ENHANCER_KEYS),
    "prompt_enhancer_json": lambda v: isinstance(v, dict) and len(v) <= len(PROMPT_ENHANCER_KEYS),
    "memory_wired_limit_gb": lambda v: v is None or (_is_number(v) and 0 <= v <= 128),
    "memory_krea_wired_limit_gb": lambda v: v is None or (_is_number(v) and 0 <= v <= 128),
    # 0 disables tiled VAE decoding. Bounded so a typo cannot ask for a 100k tile.
    "krea_vae_tile_size": lambda v: v is None or (_is_number(v) and 0 <= v <= 4096),
    "idle_kill_s_mflux": lambda v: v is None or (_is_int(v) and 0 <= v <= 86400),
    "idle_kill_s_sdxl": lambda v: v is None or (_is_int(v) and 0 <= v <= 86400),
    "idle_kill_s_qwen": lambda v: v is None or (_is_int(v) and 0 <= v <= 86400),
}


def _deep_merge(base: dict, updates: dict) -> dict:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _settings_signature():
    """Identity of the on-disk settings file: (mtime_ns, size, resolved path).

    The resolved path is part of the signature so that swapping settings.json
    for a symlink pointing outside DATA_DIR still busts the cache and gets the
    containment check re-run.
    """
    try:
        st = SETTINGS_FILE.stat()
    except OSError:
        return None
    try:
        resolved = str(SETTINGS_FILE.resolve())
    except OSError:
        resolved = ""
    return (st.st_mtime_ns, st.st_size, resolved)


# (signature, validated settings) for the last successful _load(). Every _load()
# caller already holds _lock, so no extra locking is needed here.
_load_cache: dict = {}


def _load() -> dict:
    signature = _settings_signature()
    cached = _load_cache.get("settings")
    if cached is not None and _load_cache.get("signature") == signature:
        return json.loads(json.dumps(cached))

    settings = json.loads(json.dumps(DEFAULTS))
    try:
        raw = (
            json.loads(SETTINGS_FILE.read_text("utf-8"))
            if signature is not None
            and SETTINGS_FILE.resolve().is_relative_to(DATA_DIR.resolve())
            and signature[1] <= _MAX_SETTINGS_BYTES
            else {}
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        print(f"[settings] failed to load {SETTINGS_FILE}: {e}", flush=True)
        return settings
    if not isinstance(raw, dict):
        return settings
    for key, validator in _VALIDATORS.items():
        if key not in raw or not validator(raw[key]):
            continue
        value = raw[key]
        if key == "model_defaults" and isinstance(value, dict):
            clean = {}
            for model_id, overrides in value.items():
                if not _valid_model_id(model_id) or not isinstance(overrides, dict):
                    continue
                valid_overrides = {k: v for k, v in overrides.items() if k in _MODEL_DEFAULT_KEYS and _MODEL_DEFAULT_KEYS[k](v)}
                if valid_overrides:
                    clean[model_id] = valid_overrides
            settings[key] = clean
        elif key == "model_paths" and isinstance(value, dict):
            settings[key] = {model_id: path for model_id, path in value.items() if _valid_model_id(model_id) and _valid_model_path(path)}
        elif key in ("prompt_enhancer", "prompt_enhancer_json") and isinstance(value, dict):
            clean_prompts = dict(settings[key])
            clean_prompts.update({
                key_name: text
                for key_name, text in value.items()
                if key_name in PROMPT_ENHANCER_KEYS and isinstance(text, str) and len(text) <= _PROMPT_ENHANCER_MAX
            })
            settings[key] = clean_prompts
        else:
            settings[key] = value
    _load_cache["signature"] = signature
    _load_cache["settings"] = settings
    return json.loads(json.dumps(settings))


def _save(settings: dict):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(SETTINGS_FILE.parent, 0o700)
    except OSError:
        pass
    fd, tmp = tempfile.mkstemp(dir=str(SETTINGS_FILE.parent), prefix=".settings-", suffix=".tmp")
    try:
        fchmod = getattr(os, "fchmod", None)
        if fchmod is not None:
            try:
                fchmod(fd, 0o600)
            except OSError:
                pass
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, SETTINGS_FILE)
        _load_cache.clear()
        try:
            dir_fd = os.open(str(SETTINGS_FILE.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def get_settings() -> dict:
    """Return a deep copy of the current settings."""
    with _lock:
        settings = _load()
        settings = _deep_merge(DEFAULTS, settings)
        for mid, overrides in list(settings.get("model_defaults", {}).items()):
            if not isinstance(overrides, dict):
                del settings["model_defaults"][mid]
        return json.loads(json.dumps(settings))


def get_setting(key: str, default=None):
    settings = get_settings()
    return settings.get(key, default)


def artist_name() -> str:
    name = (get_settings().get("artist_name") or "").strip()
    return name


def metadata_artist() -> str:
    """Artist credit embedded in image metadata/EXIF. Falls back to a neutral
    product name when the user has not personalized it."""
    return artist_name() or "DiffusionBear"


def update_settings(updates: dict) -> dict:
    """Validate and apply a partial update. Raises ValueError with the first
    validation error message on invalid input (nothing is persisted)."""
    if not isinstance(updates, dict) or not updates:
        return get_settings()

    with _lock:
        current = _load()
        errors = []

        for key, value in updates.items():
            if key not in _VALIDATORS:
                errors.append(f"unknown setting '{key}'")
                continue
            if not _VALIDATORS[key](value):
                errors.append(f"{key}: invalid value")
                continue
            if key == "model_defaults":
                for model_id, overrides in value.items():
                    if not _valid_model_id(model_id):
                        errors.append(f"model_defaults: invalid model id '{model_id}'")
                        continue
                    if not isinstance(overrides, dict):
                        errors.append(f"model_defaults.{model_id}: expected object")
                        continue
                    if not overrides:
                        # Empty overrides object -> clear saved defaults for this model.
                        current.setdefault("model_defaults", {}).pop(model_id, None)
                        continue
                    for k, v in overrides.items():
                        if k not in _MODEL_DEFAULT_KEYS:
                            errors.append(f"model_defaults.{model_id}.{k}: unknown key")
                            continue
                        if not _MODEL_DEFAULT_KEYS[k](v):
                            errors.append(f"model_defaults.{model_id}.{k}: invalid value")
                    prev = current.get("model_defaults", {}).get(model_id, {})
                    merged = dict(prev)
                    for k, v in overrides.items():
                        if k in _MODEL_DEFAULT_KEYS and _MODEL_DEFAULT_KEYS[k](v):
                            merged[k] = v
                    current.setdefault("model_defaults", {})[model_id] = merged

            if key in ("prompt_enhancer", "prompt_enhancer_json"):
                for engine_key, text in value.items():
                    if engine_key not in PROMPT_ENHANCER_KEYS:
                        errors.append(f"{key}: unknown engine '{engine_key}'")
                        continue
                    if not isinstance(text, str) or len(text) > _PROMPT_ENHANCER_MAX:
                        errors.append(f"{key}.{engine_key}: invalid value")
                        continue
                    current.setdefault(key, {})[engine_key] = text

            if key == "model_paths":
                for model_id, path in value.items():
                    if not _valid_model_id(model_id):
                        errors.append(f"model_paths: invalid model id '{model_id}'")
                        continue
                    if path is None or path == "":
                        current.setdefault("model_paths", {}).pop(model_id, None)
                        continue
                    if not _valid_model_path(path):
                        errors.append(f"model_paths.{model_id}: path must be absolute")
                        continue
                    current.setdefault("model_paths", {})[model_id] = path

        if errors:
            raise ValueError("; ".join(errors))

        for key in ("model_defaults", "model_paths"):
            if len(current.get(key, {})) > _MAX_MODEL_SETTINGS:
                errors.append(f"{key}: too many entries")

        if errors:
            raise ValueError("; ".join(errors))

        for key, value in updates.items():
            if key not in (
                "model_defaults",
                "model_paths",
                "prompt_enhancer",
                "prompt_enhancer_json",
            ):
                current[key] = value

        _save(current)
        return get_settings()


def clear_model_defaults(model_id: str) -> dict:
    if not _valid_model_id(model_id):
        raise ValueError("invalid model id")
    with _lock:
        current = _load()
        current.get("model_defaults", {}).pop(model_id, None)
        _save(current)
        return get_settings()


def prompt_enhancer_override(engine_key: str) -> str:
    """Custom system-prompt instructions for a prompt-enhancer engine, or ""."""
    return (get_settings().get("prompt_enhancer") or {}).get(engine_key, "") or ""


def prompt_enhancer_json_override(engine_key: str) -> str:
    """Custom JSON-mode system-prompt instructions for an engine, or ""."""
    return (get_settings().get("prompt_enhancer_json") or {}).get(engine_key, "") or ""


def model_local_path(model_id: str) -> str:
    """User-registered local weights directory for a model, or ""."""
    path = (get_settings().get("model_paths") or {}).get(model_id, "") or ""
    if not _valid_model_path(path):
        return ""
    resolved = Path(path).expanduser()
    return str(resolved) if resolved.is_dir() else ""


def configured_model_local_path(model_id: str) -> str:
    path = (get_settings().get("model_paths") or {}).get(model_id, "") or ""
    if not _valid_model_path(path):
        return ""
    return str(Path(path).expanduser())


# Keys a model descriptor may carry natively vs which come from settings overrides.


def apply_model_defaults(model_id: str, model: dict) -> dict:
    """Merge persisted per-model + global defaults into a model descriptor.

    Returns a copy of ``model`` where ``default_*`` fields are overridden by the
    user's saved preferences (falling back to the model's built-in values)."""
    merged = dict(model)
    per_model = (get_settings().get("model_defaults") or {}).get(model_id, {}) or {}
    merged["has_model_override"] = bool(per_model)

    if per_model.get("steps") is not None:
        merged["default_steps"] = per_model["steps"]
    if per_model.get("guidance") is not None:
        merged["default_guidance"] = per_model["guidance"]
    if per_model.get("sampler"):
        merged["default_sampler"] = per_model["sampler"]
    if per_model.get("cache_interval") is not None:
        merged["default_cache_interval"] = per_model["cache_interval"]
    if per_model.get("fast_vae") is not None:
        merged["default_fast_vae"] = per_model["fast_vae"]
    if per_model.get("width") is not None:
        merged["default_width"] = per_model["width"]
    if per_model.get("height") is not None:
        merged["default_height"] = per_model["height"]

    # Global fallbacks only apply when the model has no explicit override and the
    # feature is supported (sampler list present for SDXL, engine-scoped cache).
    samplers = model.get("samplers") or []
    default_sampler = get_settings().get("default_sampler") or ""
    if "default_sampler" not in merged and default_sampler and default_sampler in samplers:
        merged["default_sampler"] = default_sampler
    if "default_cache_interval" not in merged and model.get("engine") == "sdxl":
        merged["default_cache_interval"] = get_settings().get("default_cache_interval", 1)
    if "default_fast_vae" not in merged:
        merged["default_fast_vae"] = get_settings().get("default_fast_vae", True)
    if not model.get("supports_fast_vae", True):
        merged["default_fast_vae"] = False
    if "default_output_format" not in merged:
        merged["default_output_format"] = get_settings().get("default_output_format", "png")
    local_path = model_local_path(model_id)
    if local_path:
        merged["local_path"] = local_path
    return merged
