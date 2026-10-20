"""Custom model registry: data, not code.

Mirrors the LoRA registry pattern in state.py (_read_loras/_write_loras):
a DATA_DIR-relative JSON list file with locked read-modify-write.
"""

import json
import re
import threading
from pathlib import Path

import generator
import state
from sdxl_layout import sdxl_layout_report
from state import MAX_RUNTIME_JSON_BYTES, SAFE_ID_RE, _atomic_write_text

CUSTOM_MODELS_FILE = state.DATA_DIR / "custom_models.json"

_FAMILIES = ("sdxl", "flux2", "flux3", "krea", "zit", "qwen")
_SOURCES = ("local", "hf", "civitai")
_FLUX2_VARIANTS = ("4b", "9b")
_QWEN_VARIANTS = ("base", "turbo")

_custom_models_lock = threading.RLock()

# (signature, validated registry) for the last successful read. Every caller
# holds _custom_models_lock, so no extra locking is needed here.
_custom_models_cache: dict = {}


def _signature():
    try:
        st = CUSTOM_MODELS_FILE.stat()
    except OSError:
        return None
    try:
        resolved = str(CUSTOM_MODELS_FILE.resolve())
    except OSError:
        resolved = ""
    return (st.st_mtime_ns, st.st_size, resolved)


def validate_entry(entry: dict) -> tuple[bool, str | None]:
    """Check a registry entry shape. Returns (True, None) or (False, reason)."""
    if not isinstance(entry, dict):
        return False, "entry must be an object"
    entry_id = entry.get("id")
    if (
        not isinstance(entry_id, str)
        or not SAFE_ID_RE.fullmatch(entry_id)
        or len(entry_id) > 120
    ):
        return False, "invalid id: must match ^[A-Za-z0-9_-]+$, max 120 chars"
    if entry_id.lower() in (key.lower() for key in generator.MODELS.keys()):
        return False, f"invalid id: collides with built-in model '{entry_id}'"
    label = entry.get("label")
    if not isinstance(label, str) or not 1 <= len(label) <= 200:
        return False, "invalid label: required string, 1..200 chars"
    family = entry.get("family")
    if family not in _FAMILIES:
        return False, "invalid family: must be one of sdxl|flux2|flux3|krea|zit|qwen"
    variant = entry.get("variant")
    if family == "flux2":
        if variant not in _FLUX2_VARIANTS:
            return False, "invalid variant: flux2 requires one of 4b|9b"
    elif family == "qwen":
        if variant not in _QWEN_VARIANTS:
            return False, "invalid variant: qwen requires one of base|turbo"
    elif variant is not None:
        return False, "invalid variant: must be absent unless family is flux2 or qwen"
    source = entry.get("source")
    if source not in _SOURCES:
        return False, "invalid source: must be one of local|hf|civitai"
    if source == "local":
        local_path = entry.get("local_path")
        if (
            not isinstance(local_path, str)
            or not local_path
            or len(local_path) > 4096
            or "\x00" in local_path
        ):
            return False, "invalid local_path: required string, max 4096 chars, no NUL"
        if not Path(local_path).is_absolute():
            return False, "invalid local_path: must be absolute when source is local"
    if source == "hf":
        hf_repo = entry.get("hf_repo")
        if not isinstance(hf_repo, str) or not hf_repo or len(hf_repo) > 200:
            return False, "invalid hf_repo: required string, 1..200 chars"
    if source == "civitai":
        civitai_ids = entry.get("civitai_ids")
        if isinstance(civitai_ids, str):
            ids = [civitai_ids]
        elif isinstance(civitai_ids, list):
            ids = civitai_ids
        else:
            return False, "invalid civitai_ids: required non-empty string or list"
        if not ids or any(not isinstance(v, str) or not v for v in ids):
            return False, "invalid civitai_ids: required non-empty string or list"
    is_distilled = entry.get("is_distilled")
    if is_distilled is not None and not isinstance(is_distilled, bool):
        return False, "invalid is_distilled: must be boolean if present"
    return True, None


def _valid_entry(entry: object) -> bool:
    return isinstance(entry, dict) and validate_entry(entry)[0]


def read_custom_models() -> list[dict]:
    with _custom_models_lock:
        signature = _signature()
        cached = _custom_models_cache.get("custom_models")
        if cached is not None and _custom_models_cache.get("signature") == signature:
            return [dict(entry) for entry in cached]
        try:
            if signature is None:
                return []
            if (
                not CUSTOM_MODELS_FILE.resolve().is_relative_to(state.DATA_DIR.resolve())
                or signature[1] > MAX_RUNTIME_JSON_BYTES
            ):
                raise ValueError("Custom model registry exceeds the runtime state size limit")
            data = json.loads(CUSTOM_MODELS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        except ValueError:
            raise
        except (OSError, UnicodeError):
            return []
        if not isinstance(data, list):
            return []
        valid = [dict(entry) for entry in data[:512] if _valid_entry(entry)]
        _custom_models_cache["signature"] = signature
        _custom_models_cache["custom_models"] = valid
        return [dict(entry) for entry in valid]


def write_custom_models(entries: list[dict]):
    with _custom_models_lock:
        valid = [dict(entry) for entry in entries[:512] if _valid_entry(entry)]
        _atomic_write_text(CUSTOM_MODELS_FILE, json.dumps(valid, indent=2, allow_nan=False))
        _custom_models_cache.clear()


# Expected (transformer_blocks, single_transformer_blocks) depths per FLUX.2
# Klein size, measured 2026-10-14 by counting distinct block indices in the
# real HF-cache transformer weight indexes on this machine (negative
# lookbehind so single_transformer_blocks.N never counts as a double block):
#   mlx-community/flux2-klein-4b-4bit -> 5 doubles (0-4) / 20 singles (0-19)
#   mlx-community/flux2-klein-9b-4bit -> 8 doubles (0-7) / 24 singles (0-23)
# Corroborated by the engine itself (mflux ModelConfig transformer_overrides:
# 4b num_layers=5/num_single_layers=20, 9b 8/24). Derived from measurement,
# not memory; the spec's open question is closed by these numbers.
_KLEIN_SIZE_DEPTHS = {
    "4b": (5, 20),
    "9b": (8, 24),
}

_RE_KLEIN_SINGLE = re.compile(r"single_transformer_blocks\.(\d+)")
# single_transformer_blocks.N contains transformer_blocks.N as a substring,
# so the double-block pattern must not match inside a single-block key.
_RE_KLEIN_DOUBLE = re.compile(r"(?<!single_)transformer_blocks\.(\d+)")


def _find_klein_weight_index(model_dir: Path) -> Path | None:
    base = Path(model_dir).expanduser()
    direct = base / "transformer" / "model.safetensors.index.json"
    try:
        if direct.is_file():
            return direct
    except OSError:
        return None
    try:
        candidates = sorted((base / "transformer").glob("*.safetensors.index.json"))
        if candidates:
            return candidates[0]
        candidates = sorted(base.glob("*.safetensors.index.json"))
        if candidates:
            return candidates[0]
    except OSError:
        pass
    return None


def verify_klein_size(model_dir: Path, declared: str) -> tuple[bool, str | None]:
    """Check a model directory's transformer depth against a declared Klein size.

    `declared` is `4b`|`9b` and wins: a disagreeing index refuses, never
    overrides. Counts distinct `transformer_blocks.N` /
    `single_transformer_blocks.N` indices in the weight index — the same
    signal the LoRA checker reads (`lora_compat` parses `weight_map` keys
    the same way). A truncated index (shards named but files absent) reads
    as incomplete, never as either size. Anything unreadable refuses with
    a reason; nothing here raises on bad input.
    """
    if declared not in _KLEIN_SIZE_DEPTHS:
        return False, f"unknown size {declared!r}: must be one of 4b|9b"
    index = _find_klein_weight_index(model_dir)
    if index is None:
        return False, "no transformer weight index (*.safetensors.index.json) found in directory"
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as e:
        return False, f"weight index is not valid JSON: {e}"
    weight_map = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return False, "weight index has no weight_map entries"
    doubles: set[int] = set()
    singles: set[int] = set()
    for name in weight_map:
        match = _RE_KLEIN_SINGLE.search(str(name))
        if match is not None:
            singles.add(int(match.group(1)))
            continue
        match = _RE_KLEIN_DOUBLE.search(str(name))
        if match is not None:
            doubles.add(int(match.group(1)))
    # Completeness before size: a half-downloaded 4B index still names 4B
    # depths, so comparing first would pass what cannot load.
    shards = sorted({v for v in weight_map.values() if isinstance(v, str) and v})
    missing = [s for s in shards if not (index.parent / s).is_file()]
    if missing:
        shown = ", ".join(missing[:3])
        extra = f" and {len(missing) - 3} more" if len(missing) > 3 else ""
        return False, (
            f"incomplete download: weight index names {len(shards)} "
            f"shard(s), {len(missing)} missing: {shown}{extra}"
        )
    expected = _KLEIN_SIZE_DEPTHS[declared]
    found = (len(doubles), len(singles))
    if found != expected:
        return False, (
            f"size mismatch: declared {declared} expects "
            f"{expected[0]} transformer_blocks / {expected[1]} "
            f"single_transformer_blocks, found {found[0]} / {found[1]}"
        )
    return True, None


_CANONICAL_BY_FAMILY = {
    "flux2": "flux2-klein-4b",
    "flux3": "flux3-klein",
    "krea": "krea2-turbo",
    "zit": "z-image-turbo",
    "qwen": "qwen-image-2.1",
}

_ENGINE_BY_FAMILY = {
    "sdxl": "sdxl",
    "flux2": "mflux",
    "flux3": "mflux",
    "krea": "mflux",
    "zit": "mflux",
    "qwen": "qwen",
}


def _slugify_folder_name(folder_name: str) -> str:
    """Derive a registry id from a folder name (ruling, verbatim).

    Lowercase; each run of [^a-z0-9]+ becomes one "-"; strip
    leading/trailing "-"; empty becomes "model"; truncate to 120 chars.
    "My Model v2.0" -> "my-model-v2-0".
    """
    slug = re.sub(r"[^a-z0-9]+", "-", folder_name.lower()).strip("-")
    if not slug:
        return "model"
    return slug[:120].strip("-") or "model"


def verify_and_register(declaration: dict) -> tuple[bool, str | None, dict | None]:
    """Declare -> validate -> write, shared by the onboard route (Task 4)
    and the download flows (Task 5). Never raises; returns
    (True, None, entry) or (False, reason, None). The router maps False
    to 400 with the reason.
    """
    if not isinstance(declaration, dict):
        return False, "invalid declaration: must be an object", None
    source = declaration.get("source", "local")
    if source is None:
        source = "local"
    if source not in ("local", "hf", "civitai"):
        return False, "invalid source: must be one of local|hf|civitai", None
    raw_path = declaration.get("local_path")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or len(raw_path) > 4096
    ):
        if isinstance(raw_path, str) and "\x00" in raw_path:
            return False, "invalid local_path: required string, max 4096 chars, no NUL", None
        return False, "invalid local_path: required string, max 4096 chars, no NUL", None
    if "\x00" in raw_path:
        return False, "invalid local_path: required string, max 4096 chars, no NUL", None
    try:
        candidate_dir = Path(raw_path).expanduser()
    except (OSError, RuntimeError):
        return False, "invalid model path", None
    try:
        is_abs = candidate_dir.is_absolute()
    except (OSError, ValueError):
        return False, "model path must be absolute", None
    if not is_abs:
        return False, "model path must be absolute", None
    raw_id = declaration.get("id")
    if "id" not in declaration or raw_id is None or raw_id == "":
        candidate_name = candidate_dir.stem if candidate_dir.is_file() else candidate_dir.name
        final_id = _slugify_folder_name(candidate_name)
    else:
        final_id = raw_id
    raw_label = declaration.get("label")
    if "label" not in declaration or raw_label is None or raw_label == "":
        final_label = final_id if isinstance(final_id, str) else ""
    else:
        final_label = raw_label
    family = declaration.get("family")
    variant = declaration.get("variant") if "variant" in declaration else None
    entry: dict = {
        "id": final_id,
        "label": final_label,
        "family": family,
        "source": source,
        # Expanded, not resolved: entries may hold ".."/symlinks (install-local stores resolved).
        "local_path": str(candidate_dir),
    }
    if variant is not None:
        entry["variant"] = variant
    if "is_distilled" in declaration and declaration["is_distilled"] is not None:
        entry["is_distilled"] = bool(declaration["is_distilled"])
    if source == "hf":
        # Required-iff-hf is enforced by validate_entry below (verbatim
        # "invalid hf_repo ..." refusal); passed through even when absent.
        entry["hf_repo"] = declaration.get("hf_repo")
    if source == "civitai":
        # validate_entry accepts non-empty strings only, but the download
        # route deals in ints (model_version_id). Normalise ints at this
        # boundary so the ruling's int declaration registers; anything else
        # falls through to validate_entry's verbatim refusal.
        raw_ids = declaration.get("civitai_ids")
        if isinstance(raw_ids, list):
            entry["civitai_ids"] = [
                str(v) if isinstance(v, int) and not isinstance(v, bool) else v
                for v in raw_ids
            ]
        else:
            entry["civitai_ids"] = raw_ids
        for key in ("civitai_model_id", "civitai_version_id"):
            if declaration.get(key) is not None:
                entry[key] = declaration[key]
    try:
        ok, reason = validate_entry(entry)
    except (OSError, ValueError) as e:
        return False, f"entry validation failed: {e}", None
    if not ok:
        return False, reason, None
    try:
        with _custom_models_lock:
            existing = read_custom_models()
    except (OSError, ValueError) as e:
        return False, f"registry read failed: {e}", None
    if isinstance(final_id, str) and any(
        isinstance(e.get("id"), str) and e.get("id").lower() == final_id.lower()
        for e in existing
    ):
        return False, f"custom model id '{final_id}' is already registered", None
    if family == "sdxl":
        try:
            layout_ok, layout_reason = sdxl_layout_report(candidate_dir)
        except Exception as e:
            return False, f"layout check failed: {e}", None
        if not layout_ok:
            return False, layout_reason or "invalid model directory", None
    elif family in _CANONICAL_BY_FAMILY:
        canonical = _CANONICAL_BY_FAMILY[family]
        try:
            layout_ok, layout_reason = generator.validate_local_model_dir(
                canonical, candidate_dir
            )
        except Exception as e:
            return False, f"layout check failed: {e}", None
        if not layout_ok:
            return False, layout_reason or "invalid model directory", None
        if family == "flux2":
            try:
                size_ok, size_reason = verify_klein_size(candidate_dir, entry.get("variant"))
            except Exception as e:
                return False, f"size check failed: {e}", None
            if not size_ok:
                return False, size_reason or "size mismatch", None
    else:
        return False, "invalid family: must be one of sdxl|flux2|flux3|krea|zit|qwen", None
    try:
        from routers.models import _model_is_fully_cached_uncached

        engine = _ENGINE_BY_FAMILY.get(family)
        complete = _model_is_fully_cached_uncached(
            final_id, {"engine": engine}, candidate_dir
        )
    except Exception as e:
        return False, f"completeness check failed: {e}", None
    if not complete:
        return False, "model directory is incomplete", None
    with _custom_models_lock:
        try:
            current = read_custom_models()
        except (OSError, ValueError) as e:
            return False, f"registry read failed: {e}", None
        if isinstance(final_id, str) and any(
            isinstance(e.get("id"), str) and e.get("id").lower() == final_id.lower()
            for e in current
        ):
            return False, f"custom model id '{final_id}' is already registered", None
        try:
            write_custom_models(current + [entry])
        except (OSError, ValueError) as e:
            return False, f"registry write failed: {e}", None
    return True, None, dict(entry)
