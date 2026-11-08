"""DiffusionBee as a `Source`.

Lists the user's DiffusionBee models and LoRA files from `~/.diffusionbee`.
Unlike Hugging Face or Civitai, this source is local, so it has no remote licence
information -- `Item.license` stays `None`.

**The DiffusionBee directory is strictly read-only.**
Never write, modify, lock, or create files inside `~/.diffusionbee`.
Listing reads only the 288-byte header block and the JSON `keys_info` index,
never decoding tensor payload bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import struct
from typing import Any
import numpy as np

from sources import Item, Provenance, Source

# Magic numbers from DiffusionBee's tdict binary format
_TDICT_FILE_HEAD_MAGIC = 8589935844
_TDICT_BLOCK_HEAD_MAGIC = 8030895855
_TDICT_MAGIC_NUMBER = 4346464


def model_dirs() -> list[Path]:
    """The DiffusionBee asset directories, resolved dynamically on every call.

    Allows overriding via `DIFFUSIONBEE_ROOT` for test isolation.
    """
    custom_root = os.environ.get("DIFFUSIONBEE_ROOT")
    if custom_root:
        root = Path(custom_root)
        return [root / "downloaded_assets", root / "imported_models"]

    home = Path.home()
    return [
        home / ".diffusionbee/downloaded_assets",
        home / ".diffusionbee/imported_models",
    ]


def read_tdict_header(path: Path) -> dict[str, Any] | None:
    """Read the binary header of a .tdict file, strictly read-only.

    Returns header offsets and metadata, or None if the file is invalid or corrupt.
    """
    try:
        with open(path, "rb") as f:
            header_bytes = f.read(64)
            if len(header_bytes) < 64:
                return None
            file_head = struct.unpack("<8Q", header_bytes)
            if file_head[0] != _TDICT_FILE_HEAD_MAGIC:
                return None

            block_head_bytes = f.read(64)
            if len(block_head_bytes) < 64:
                return None
            block_head = struct.unpack("<8Q", block_head_bytes)
            if block_head[0] != _TDICT_BLOCK_HEAD_MAGIC:
                return None

            extra_head_bytes = f.read(160)
            if len(extra_head_bytes) < 160:
                return None
            extra_head = struct.unpack("<20Q", extra_head_bytes)
            if extra_head[0] != _TDICT_MAGIC_NUMBER:
                return None

            return {
                "magic_number": extra_head[0],
                "tdict_format_version": extra_head[1],
                "min_supported_version": extra_head[2],
                "ctdict_version": extra_head[3],
                "weights_json_start": extra_head[4],
                "weights_json_end": extra_head[5],
                "metadata_json_start": extra_head[6],
                "metadata_json_end": extra_head[7],
            }
    except (OSError, struct.error):
        return None


def read_tdict_keys_info(path: Path) -> dict[str, dict]:
    """Read the keys_info metadata dictionary from a .tdict file."""
    header = read_tdict_header(path)
    if not header:
        return {}

    w_start = header["weights_json_start"]
    w_end = header["weights_json_end"]
    if w_end <= w_start or (w_end - w_start) > 50_000_000:
        return {}

    try:
        with open(path, "rb") as f:
            f.seek(w_start)
            raw_json = f.read(w_end - w_start)
            if not raw_json:
                return {}
            data = json.loads(raw_json.decode("utf-8", errors="replace"))
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def read_tdict_keys(path: Path) -> list[str]:
    """Read only tensor key names from a .tdict file without reading tensor payloads."""
    return list(read_tdict_keys_info(path).keys())


_LORA_KEY_MARKERS = (
    "weight_lora_up",
    "weight_lora_down",
    "weight_lora_scale",
    "__up__",
    "__down__",
    "lora_up",
    "lora_down",
)

_FLUX_KEY_MARKERS = (
    "__dit__",
    "transformer.",
    "double_blocks",
    "single_blocks",
)

_SDXL_KEY_MARKERS = (
    "__te2__",
    "text_encoder_2",
    "lora_te2_",
    "conditioner.embedders",
    "label_emb",
)

_UNET_KEY_MARKERS = (
    "model.diffusion_model",
    "__unet__",
    "input_blocks",
    "middle_block",
    "output_blocks",
    "lora_unet",
)


def detect_kind_keys(keys: list[str]) -> str | None:
    """Return 'lora', 'model', or None from a list of tensor keys."""
    if not keys:
        return None
    for k in keys:
        if not isinstance(k, str):
            continue
        for marker in _LORA_KEY_MARKERS:
            if marker in k:
                return "lora"
    return "model"


def detect_base_model_keys(keys: list[str]) -> str | None:
    """Detect the engine/architecture from a list of tensor keys.

    Returns 'flux', 'sdxl', 'sd15', or None. Never guesses.
    """
    if not keys:
        return None

    for k in keys:
        if not isinstance(k, str):
            continue
        for marker in _FLUX_KEY_MARKERS:
            if marker in k:
                return "flux"
        for marker in _SDXL_KEY_MARKERS:
            if marker in k:
                return "sdxl"

    # Specific SD 1.5 markers: input block 1 transformer blocks or blocks 10-11
    # (SDXL has 9 input blocks 0..8 and no transformer in input block 1)
    for k in keys:
        if not isinstance(k, str):
            continue
        if (
            "input_blocks.1.1.transformer_blocks" in k
            or "input_blocks.10." in k
            or "input_blocks.11." in k
        ):
            return "sd15"

    return None



_UPGRADE_HINTS = (
    (
        ("juggernaut", "realistic", "realvis", "photon", "epicphoto", "cyberrealistic"),
        {
            "recommended_model_id": "juggernaut-xl-lightning",
            "recommended_label": "Juggernaut XL Lightning (MLX)",
            "reason": "Replaces legacy photorealistic .tdict checkpoints with 4-step MLX acceleration.",
        },
    ),
    (
        ("pony", "animagine", "illustrious", "counterfeit", "anything", "dreamshaper", "revanimated", "meina"),
        {
            "recommended_model_id": "Pony",
            "recommended_label": "Pony Diffusion V6 XL",
            "reason": "Native SDXL & Pony support with 1-click Civitai LoRA browser.",
        },
    ),
)


def suggest_modern_equivalent(
    stem: str, kind: str | None, base_model: str | None
) -> dict[str, str] | None:
    """Return a suggested native MLX/SDXL replacement for an unconvertible checkpoint."""
    if kind == "lora":
        return None
    low = (stem or "").lower()
    for keywords, suggestion in _UPGRADE_HINTS:
        if any(k in low for k in keywords):
            return dict(suggestion)
    return {
        "recommended_model_id": "flux2-klein-4b",
        "recommended_label": "FLUX.2-klein 4B",
        "reason": "Modern 4-bit MLX replacement for legacy SD 1.5 / .tdict checkpoints.",
    }


@dataclass(frozen=True)
class FileInfo:
    """One DiffusionBee .tdict file, as listed."""

    name: str
    path: Path
    size_bytes: int
    modified_at: float
    base_model: str | None
    kind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "base_model": self.base_model,
            "kind": self.kind,
            "recommended_upgrade": suggest_modern_equivalent(
                Path(self.name).stem, self.kind, self.base_model
            ),
        }


def list_tdict_files(directories: list[Path] | None = None) -> list[FileInfo]:
    """List every .tdict in the DiffusionBee folders, strictly read-only, sorted by name."""
    roots = directories if directories is not None else model_dirs()
    files: list[FileInfo] = []
    seen_names: set[str] = set()

    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for path in sorted(root_path.glob("*.tdict"), key=lambda p: p.name):
            if path.name in seen_names:
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            keys = read_tdict_keys(path)
            kind = detect_kind_keys(keys)
            base_model = detect_base_model_keys(keys) if kind == "lora" else None
            files.append(
                FileInfo(
                    name=path.name,
                    path=path,
                    size_bytes=st.st_size,
                    modified_at=st.st_mtime,
                    base_model=base_model,
                    kind=kind,
                )
            )
            seen_names.add(path.name)

    return sorted(files, key=lambda f: f.name)


class DiffusionBeeSource(Source):
    name = "diffusionbee"

    def search(
        self,
        query: str | None = None,
        *,
        kind: str | None = None,
        sort: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 40,
        directories: list[Path] | None = None,
    ) -> list[Item]:
        """Local DiffusionBee files, as shared Items. Filtered by substring."""
        needle = (query or "").strip().lower()
        items = []
        for info in list_tdict_files(directories):
            if needle and needle not in info.name.lower():
                continue
            if kind and info.kind != kind:
                continue
            items.append(
                Item(
                    id=f"diffusionbee:{info.name}",
                    name=info.name,
                    source="diffusionbee",
                    kind=info.kind or "model",
                    architecture=info.base_model,
                    quantization=None,
                    size_bytes=info.size_bytes,
                    preview_url=None,
                    license=None,  # local files have no remote licence
                    provenance=Provenance(origin="diffusionbee", source_url=None),
                )
            )
            if len(items) >= limit:
                break
        return items


def convert_tdict_lora(
    tdict_path: Path,
    out_path: Path,
    base_model: str,
    allow_root: Path | None = None,
) -> dict[str, Any]:
    """Convert a DiffusionBee .tdict LoRA to safetensors at out_path.

    The DiffusionBee directory is strictly read-only.
    Returns a summary dict with output_path, base_model, tensors, and skipped_tensors.
    """
    from lora_convert import ConversionError, _check_output
    from safetensors.numpy import save_file

    if base_model not in ("sdxl", "flux"):
        raise ConversionError(
            f"unknown base model {base_model!r}; expected 'sdxl' or 'flux'"
        )

    resolved_out = _check_output(out_path, allow_root=allow_root)

    keys_info = read_tdict_keys_info(tdict_path)
    if not keys_info:
        raise ConversionError(f"cannot read keys from {tdict_path}")

    kind = detect_kind_keys(list(keys_info.keys()))
    if kind != "lora":
        raise ConversionError(
            f"{tdict_path.name} is a {kind or 'unknown'} (not a LoRA); full models cannot be converted in v1"
        )

    tensors: dict[str, np.ndarray] = {}
    skipped_tensors = 0

    with open(tdict_path, "rb") as f:
        for orig_key, info in keys_info.items():
            new_key = None
            if orig_key.endswith(".weight_lora_up"):
                new_key = orig_key[:-len(".weight_lora_up")] + ".lora_up.weight"
            elif orig_key.endswith(".weight_lora_down"):
                new_key = orig_key[:-len(".weight_lora_down")] + ".lora_down.weight"
            elif orig_key.endswith("__up__"):
                new_key = orig_key[:-len("__up__")] + ".lora_up.weight"
            elif orig_key.endswith("__down__"):
                new_key = orig_key[:-len("__down__")] + ".lora_down.weight"
            elif orig_key.endswith(".weight_lora_scale"):
                new_key = orig_key[:-len(".weight_lora_scale")] + ".alpha"
            elif (
                orig_key.endswith(".lora_up.weight")
                or orig_key.endswith(".lora_down.weight")
                or orig_key.endswith(".alpha")
            ):
                new_key = orig_key

            if new_key is None:
                skipped_tensors += 1
                continue

            try:
                f.seek(info["start"])
                raw = f.read(info["end"] - info["start"])
                dtype = info.get("dtype", "float32")
                shape = tuple(info.get("shape", [len(raw)]))
                arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
                tensors[new_key] = np.ascontiguousarray(arr, dtype=np.float32)
            except Exception:
                skipped_tensors += 1

    if not tensors:
        raise ConversionError(
            f"no valid LoRA tensors could be extracted from {tdict_path.name}"
        )

    tmp = resolved_out.with_name(resolved_out.name + ".tmp")
    try:
        save_file(tensors, str(tmp))
        os.replace(tmp, resolved_out)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise

    return {
        "output_path": str(resolved_out),
        "base_model": base_model,
        "tensors": len(tensors),
        "skipped_tensors": skipped_tensors,
        "bytes": resolved_out.stat().st_size,
    }


