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
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

import app_settings
import app_version
import generator

MANIFEST_PATH = Path(__file__).resolve().parent / "model_manifests.json"
_HASH_CHUNK_BYTES = 4 * 1024 * 1024  # 4 MiB streaming chunks
_MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024  # 128 MiB header sanity cap

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
    # Check an explicit override or canonical default dir first so an invalid/corrupt
    # local install is inspected directly instead of silently masked by an HF cache copy.
    configured = app_settings.configured_model_local_path(model_id)
    if configured:
        p = Path(configured).expanduser()
        if p.is_dir():
            return generator._resolve_hf_cache_snapshot(p)
    default_dir = Path(app_settings.ASSET_DIR) / "models" / model_id
    try:
        if default_dir.is_dir() and any(default_dir.rglob("*.safetensors")):
            return default_dir
    except OSError:
        pass
    resolved = generator.resolve_local_model_path(model_id)
    if resolved is not None and resolved.is_dir():
        return resolved
    return None


def _read_safetensors_header(path: Path) -> tuple[bool, dict | None, set[str], str | None]:
    """Read only the 8-byte length prefix and JSON header of a .safetensors file."""
    try:
        size = path.stat().st_size
        if size < 10:
            return False, None, set(), f"file too small ({size} bytes)"
        with path.open("rb") as handle:
            raw_len = handle.read(8)
            if len(raw_len) < 8:
                return False, None, set(), "truncated header length"
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len < 2 or header_len > _MAX_SAFETENSORS_HEADER_BYTES or 8 + header_len > size:
                return False, None, set(), f"invalid safetensors header length ({header_len})"
            raw_header = handle.read(header_len)
        header = json.loads(raw_header.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error) as exc:
        return False, None, set(), f"unreadable safetensors header: {exc}"
    if not isinstance(header, dict):
        return False, None, set(), "safetensors header is not a JSON object"
    meta = header.pop("__metadata__", None)
    if not isinstance(meta, dict):
        meta = None
    keys = {str(k) for k in header.keys()}
    if not keys:
        return False, meta, set(), "safetensors header contains zero tensors"
    return True, meta, keys, None


def _has_mflux_metadata(meta: dict | None) -> bool:
    if not isinstance(meta, dict):
        return False
    return meta.get("quantization_level") is not None or meta.get("mflux_version") is not None


def _resolve_mflux_weight_definition(model_id: str):
    """Return the vendored mflux WeightDefinition class for a built-in model, if any.

    Only called when ``mlx.core`` is already loaded in ``sys.modules`` so pure-Python
    tests in headless/CI/sandboxed environments never trigger Metal initialization.
    """
    if model_id == "flux3-klein":
        from mflux.models.flux3.weights.flux3_weight_definition import Flux3KleinWeightDefinition
        return Flux3KleinWeightDefinition
    if model_id in ("flux2-klein-4b", "flux2-klein-9b"):
        from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
        return Flux2KleinWeightDefinition
    if model_id == "z-image-turbo":
        from mflux.models.z_image.weights.z_image_weight_definition import ZImageWeightDefinition
        return ZImageWeightDefinition
    if model_id == "dreamlite-mobile":
        from mflux.models.dreamlite.weights.dreamlite_weight_definition import DreamLiteWeightDefinition
        return DreamLiteWeightDefinition
    if model_id == "anima-turbo":
        from mflux.models.anima.weights.anima_weight_definition import AnimaWeightDefinition
        return AnimaWeightDefinition
    if model_id == "flux1-schnell":
        from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition
        return FluxWeightDefinition
    return None


def _validate_flux3_headers(
    parsed_components: dict[str, tuple[dict | None, set[str]]],
    local_dir: Path,
) -> tuple[bool, str | None, str | None]:
    """Pure-Python safetensors-header verification for FLUX.3-klein."""
    # 1. VAE: either root BFL ae.safetensors / vae.safetensors or diffusers vae/
    root_vae = next((name for name in ("ae.safetensors", "vae.safetensors") if name in parsed_components), None)
    if root_vae is not None:
        meta, keys = parsed_components[root_vae]
        if not _has_mflux_metadata(meta):
            required_bfl = (
                "bn.running_mean",
                "bn.running_var",
                "decoder.conv_in.weight",
                "decoder.conv_out.weight",
                "encoder.conv_in.weight",
                "encoder.conv_out.weight",
                "decoder.mid.block_1.norm1.weight",
                "encoder.mid.block_1.norm1.weight",
            )
            missing = [k for k in required_bfl if k not in keys]
            if missing:
                return (
                    False,
                    f"FLUX.3 VAE '{root_vae}' missing required BFL weights ({', '.join(missing[:3])})",
                    root_vae,
                )
    else:
        vae_rels = sorted(rel for rel in parsed_components if rel.startswith("vae/"))
        if not vae_rels:
            return False, "FLUX.3 VAE weights missing (expected ae.safetensors or vae/*.safetensors)", None
        vae_keys: set[str] = set()
        vae_mflux = False
        for rel in vae_rels:
            meta, keys = parsed_components[rel]
            vae_mflux = vae_mflux or _has_mflux_metadata(meta)
            vae_keys.update(keys)
        if not vae_mflux:
            required_diffusers = (
                "decoder.conv_in.weight",
                "encoder.conv_in.weight",
                "decoder.mid_block.resnets.0.norm1.weight",
                "encoder.mid_block.resnets.0.norm1.weight",
            )
            missing = [k for k in required_diffusers if k not in vae_keys]
            if missing:
                return (
                    False,
                    f"FLUX.3 VAE '{vae_rels[0]}' missing required Diffusers weights ({', '.join(missing[:3])})",
                    vae_rels[0],
                )

    # 2. Transformer: root model.safetensors / transformer.safetensors or transformer/
    root_tr = next(
        (name for name in ("model.safetensors", "transformer.safetensors") if name in parsed_components),
        None,
    )
    tr_rels = [root_tr] if root_tr is not None else sorted(
        rel for rel in parsed_components if rel.startswith("transformer/")
    )
    if not tr_rels:
        return False, "FLUX.3 transformer weights missing", None
    tr_keys: set[str] = set()
    tr_mflux = False
    for rel in tr_rels:
        meta, keys = parsed_components[rel]
        tr_mflux = tr_mflux or _has_mflux_metadata(meta)
        tr_keys.update(keys)
    if not tr_mflux:
        has_flux3_blocks = any(
            k.startswith(("content_mode_blocks.", "mode_blocks.", "single_blocks.", "single_transformer_blocks."))
            for k in tr_keys
        )
        has_timestep = any(
            k.startswith(("time_in.in_layer.", "timestep_embedder.linear_1."))
            for k in tr_keys
        )
        if not (has_flux3_blocks and has_timestep):
            return (
                False,
                f"FLUX.3 transformer '{tr_rels[0]}' keys do not match FLUX.3 architecture",
                tr_rels[0],
            )

    # 3. Text encoder (if directory is present)
    te_rels = sorted(rel for rel in parsed_components if rel.startswith("text_encoder/"))
    if (local_dir / "text_encoder").is_dir():
        if not te_rels:
            return False, "FLUX.3 text_encoder/ has no valid .safetensors weights", None
        te_keys: set[str] = set()
        te_mflux = False
        for rel in te_rels:
            meta, keys = parsed_components[rel]
            te_mflux = te_mflux or _has_mflux_metadata(meta)
            te_keys.update(keys)
        if not te_mflux:
            has_embed = any(k.endswith("embed_tokens.weight") for k in te_keys)
            has_layer0 = any("layers.0.self_attn.q_proj.weight" in k for k in te_keys)
            has_norm = any(k.endswith(".norm.weight") or k == "norm.weight" for k in te_keys)
            if not (has_embed and has_layer0 and has_norm):
                return (
                    False,
                    f"FLUX.3 text_encoder '{te_rels[0]}' keys do not match Qwen3 layout",
                    te_rels[0],
                )
    return True, None, None


def _validate_model_tokenizers(
    *,
    model_id: str,
    local_dir: Path,
    minfo: dict,
    manifest_files: dict | None = None,
) -> tuple[bool, str | None, str | None]:
    """Validate required tokenizer directories and payload files exist and are readable."""
    files = manifest_files or {}
    if model_id == "anima-turbo" and (local_dir / "split_files").is_dir():
        return True, None, None

    expected_tok_dirs: list[str] = []
    for tok_dir in ("tokenizer", "tokenizer_2", "processor"):
        if any(rel.startswith(f"{tok_dir}/") for rel in files):
            expected_tok_dirs.append(tok_dir)

    if model_id == "anima-turbo":
        for tok_dir in ("tokenizer", "tokenizer_2"):
            if tok_dir not in expected_tok_dirs:
                expected_tok_dirs.append(tok_dir)

    if not expected_tok_dirs:
        return True, None, None

    for tok_dir in expected_tok_dirs:
        tok_path = local_dir / tok_dir
        cfg_rel = f"{tok_dir}/tokenizer_config.json"
        try:
            if tok_path.is_symlink() and not tok_path.exists():
                return False, f"{tok_dir}/ is a broken symlink", cfg_rel
            if not tok_path.is_dir():
                return False, f"missing {tok_dir}/ directory", cfg_rel
        except OSError as exc:
            return False, f"cannot inspect {tok_dir}/: {exc}", cfg_rel

        cfg_path = tok_path / "tokenizer_config.json"
        if not cfg_path.is_file():
            return False, f"missing {cfg_rel}", cfg_rel
        try:
            cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
            if not isinstance(cfg_data, dict):
                return False, f"{cfg_rel} is not a JSON object", cfg_rel
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return False, f"unreadable {cfg_rel}: {exc}", cfg_rel

        payload_names = ("tokenizer.json", "vocab.json", "spiece.model")
        if model_id == "anima-turbo" and tok_dir == "tokenizer":
            required_payloads = ("tokenizer.json",)
        elif model_id == "anima-turbo" and tok_dir == "tokenizer_2":
            required_payloads = ("tokenizer.json", "spiece.model")
        else:
            manifest_payloads = tuple(
                name for name in payload_names if f"{tok_dir}/{name}" in files
            )
            required_payloads = manifest_payloads or payload_names

        has_payload = False
        for name in required_payloads:
            cand = tok_path / name
            try:
                if cand.is_file() and cand.stat().st_size > 0:
                    has_payload = True
                    break
            except OSError:
                continue
        if not has_payload:
            payload_rel = f"{tok_dir}/{required_payloads[0]}"
            return (
                False,
                f"{tok_dir}/ missing readable vocabulary file ({' or '.join(required_payloads)})",
                payload_rel,
            )

    return True, None, None


def _validate_model_loadability(
    *,
    model_id: str,
    local_dir: Path,
    minfo: dict,
    total_bytes: int,
    manifest_files: dict | None = None,
) -> tuple[bool, str | None, str | None]:
    """Check directory layout, tokenizers, safetensors headers, and weight-key loadability.

    Returns ``(ok, error_message, failed_rel_path)``.
    """
    ok, reason = generator.validate_local_model_dir(model_id, local_dir)
    if not ok:
        return False, reason or "model directory failed layout validation", None

    tok_ok, tok_err, tok_rel = _validate_model_tokenizers(
        model_id=model_id,
        local_dir=local_dir,
        minfo=minfo,
        manifest_files=manifest_files,
    )
    if not tok_ok:
        return False, tok_err, tok_rel

    try:
        safetensors_files = sorted(
            f for f in local_dir.rglob("*.safetensors")
            if f.is_file() and not f.name.startswith("._")
        )
    except OSError as exc:
        return False, f"cannot scan weight files: {exc}", None

    parsed_components: dict[str, tuple[dict | None, set[str]]] = {}
    for sf in safetensors_files:
        rel = sf.relative_to(local_dir).as_posix()
        try:
            fsize = sf.stat().st_size
        except OSError as exc:
            return False, f"cannot stat weight file '{rel}': {exc}", rel
        hdr_ok, meta, keys, hdr_err = _read_safetensors_header(sf)
        if hdr_ok:
            parsed_components[rel] = (meta, keys)
        elif total_bytes >= 1024 or fsize >= 256:
            return False, f"corrupted safetensors weight '{rel}': {hdr_err}", rel

    if not parsed_components:
        return True, None, None

    if model_id == "flux3-klein":
        f3_ok, f3_err, f3_rel = _validate_flux3_headers(parsed_components, local_dir)
        if not f3_ok:
            return False, f3_err, f3_rel

    # Live WeightMapper check whenever mlx.core is already imported (never forces Metal init in CI/sandbox).
    if minfo.get("engine") == "mflux" and "mlx.core" in sys.modules:
        try:
            defn = _resolve_mflux_weight_definition(model_id)
        except Exception:
            defn = None
        if defn is not None:
            try:
                from mflux.models.common.weights.mapping.weight_mapper import WeightMapper

                for comp in defn.get_components():
                    if comp.variant_selector is not None:
                        comp = comp.variant_selector(local_dir)
                    comp_dir = local_dir / comp.hf_subdir if comp.hf_subdir else local_dir
                    if not comp_dir.is_dir():
                        continue
                    if comp.weight_files:
                        comp_files = [comp_dir / wf for wf in comp.weight_files if (comp_dir / wf).is_file()]
                    else:
                        comp_files = sorted(
                            f for f in comp_dir.glob("*.safetensors")
                            if f.is_file() and not f.name.startswith("._")
                        )
                    if not comp_files:
                        continue
                    comp_rels = [f.relative_to(local_dir).as_posix() for f in comp_files]
                    comp_entries = [parsed_components[r] for r in comp_rels if r in parsed_components]
                    if not comp_entries:
                        continue
                    if any(_has_mflux_metadata(meta) for meta, _ in comp_entries):
                        # Validate mflux-save weight index shard completeness when present.
                        idx_path = comp_dir / "model.safetensors.index.json"
                        if idx_path.is_file():
                            try:
                                idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
                                wmap = idx_data.get("weight_map") if isinstance(idx_data, dict) else None
                                if isinstance(wmap, dict) and wmap:
                                    missing_shards = [
                                        name for name in sorted(set(wmap.values()))
                                        if not (comp_dir / str(name)).is_file()
                                    ]
                                    if missing_shards:
                                        return (
                                            False,
                                            f"{comp.name} missing indexed shard(s): {', '.join(missing_shards[:3])}",
                                            None,
                                        )
                            except (OSError, UnicodeError, json.JSONDecodeError) as idx_exc:
                                return False, f"{comp.name} unreadable weight index: {idx_exc}", None
                        continue
                    if comp.mapping_getter is None:
                        continue
                    raw_keys: set[str] = set()
                    for _, keys in comp_entries:
                        raw_keys.update(keys)
                    if comp.weight_prefix_filters is not None:
                        raw_keys = {
                            k for k in raw_keys
                            if any(k.startswith(p) for p in comp.weight_prefix_filters)
                        }
                    if comp.key_transform is not None:
                        transformed: set[str] = set()
                        for k in raw_keys:
                            tk = comp.key_transform(k)
                            if tk is not None:
                                transformed.add(tk)
                        raw_keys = transformed
                    raw_weights = {k: None for k in raw_keys}
                    missing = WeightMapper.missing_required_names(
                        hf_weights=raw_weights,
                        mapping=comp.mapping_getter(),
                        num_blocks=comp.num_blocks,
                        num_layers=comp.num_layers,
                    )
                    if missing:
                        return (
                            False,
                            f"{comp.name} weights in '{comp_rels[0]}' do not fit {model_id} "
                            f"(missing required keys such as {', '.join(missing[:3])})",
                            comp_rels[0],
                        )
                if hasattr(defn, "get_tokenizers") and (
                    model_id == "anima-turbo"
                    or any(rel.endswith("tokenizer_config.json") for rel in (manifest_files or {}))
                ):
                    is_upstream_copy = (
                        model_id == "anima-turbo"
                        and hasattr(defn, "is_repository_copy")
                        and defn.is_repository_copy(local_dir)
                    )
                    if not is_upstream_copy:
                        for td in defn.get_tokenizers():
                            candidates = [td.hf_subdir] + list(td.fallback_subdirs or [])
                            if not any(
                                (
                                    (local_dir / sub) if sub and sub != "." else local_dir
                                ).joinpath("tokenizer_config.json").is_file()
                                for sub in candidates
                                if sub is not None
                            ):
                                cfg_rel = (
                                    f"{td.hf_subdir}/tokenizer_config.json"
                                    if td.hf_subdir
                                    else "tokenizer_config.json"
                                )
                                return (
                                    False,
                                    f"missing tokenizer '{td.name}' in {td.hf_subdir}/",
                                    cfg_rel,
                                )
            except Exception as exc:
                print(f"[integrity] WARN WeightMapper check skipped for {model_id}: {exc}", flush=True)

    return True, None, None


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
    is_shared_qwen_tok = generator.is_qwen_flux_tokenizer_model(
        model_id, minfo, manifest_entry.get("repo")
    )
    tok_symlinked = (
        is_shared_qwen_tok
        and (local_dir / "tokenizer").is_symlink()
        and generator._is_valid_qwen_tokenizer_dir(local_dir / "tokenizer")
    )
    proc_symlinked = (
        is_shared_qwen_tok
        and (local_dir / "processor").is_symlink()
        and generator._is_valid_qwen_tokenizer_dir(local_dir / "processor")
    )
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

    load_ok, load_error, failed_rel = _validate_model_loadability(
        model_id=model_id,
        local_dir=local_dir,
        minfo=minfo,
        total_bytes=total_bytes,
        manifest_files=manifest_entry.get("files") or {},
    )
    if not load_ok and failed_rel and not any(
        item.get("path") == failed_rel for item in (missing_files + mismatched_files)
    ):
        expected_meta = main_files.get(failed_rel) or {}
        expected_size = int(expected_meta.get("size") or 0)
        mismatched_files.append({
            "path": failed_rel,
            "size": expected_size,
            "actual_size": (
                (local_dir / failed_rel).stat().st_size
                if (local_dir / failed_rel).is_file()
                else 0
            ),
            "sha256": expected_meta.get("sha256"),
            "target": "main",
            "repo": manifest_entry.get("repo", ""),
            "reason": "layout_incompatible",
            "detail": load_error,
        })
        repair_bytes += expected_size

    if not missing_files and not mismatched_files and load_ok:
        status = "ok"
    elif not load_ok and not missing_files and not mismatched_files:
        status = "legacy_incompatible"
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
        "layout_error": None if load_ok else load_error,
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
