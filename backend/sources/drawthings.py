"""Draw Things as a `Source`.

Lists the user's own Draw Things LoRA files and works out which engine each one targets.
Unlike the other two sources this one is entirely local, so it has no licence
information at all -- `Item.license` stays `None` and the UI links to Civitai separately
if the user chooses to match one.

**The Draw Things model directory is strictly read-only.** `drawthings_codec.Store.open`
already enforces that with a percent-escaped URI plus `PRAGMA query_only`, and nothing
here writes either. Listing opens the database read-only and reads only the `tensors`
table's `name` column, never a payload, so it stays fast on a 777 MB file.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from sources import Item, Provenance, Source

# Prefixes Draw Things writes. `__dit__` is FLUX; the rest are SD/SDXL components.
_FLUX_PREFIX = "__dit__"
_SDXL_PREFIXES = ("__unet__", "__unet_fixed__", "__text_model__", "__te2__text_model__")


def model_dir() -> Path:
    """The Draw Things models directory, resolved on every call.

    Deliberately a function, not a module constant. `Path.home()` is monkeypatched by an
    existing test -- `test_model_download_path.py` replaces `hf_service.Path.home` with a
    temp dir, which patches the shared `pathlib.Path` for the whole process. Capturing
    the value at import time made this module read a temp dir for the rest of the suite
    whenever that test ran first, which showed up as three of this module's own tests
    failing only in a full `unittest discover` run.
    """
    return Path.home() / (
        "Library/Containers/com.liuliu.draw-things/Data/Documents/Models"
    )

# Names only. Reading payloads here would turn a directory listing into a multi-second
# scan of every LoRA the user owns.
TENSOR_NAME_QUERY = "SELECT name FROM tensors"

# Prefixes Draw Things writes. `__dit__` is FLUX; the rest are SD/SDXL components.
_FLUX_PREFIX = "__dit__"
_SDXL_PREFIXES = ("__unet__", "__unet_fixed__", "__text_model__", "__te2__text_model__")


@dataclass(frozen=True)
class FileInfo:
    """One Draw Things `.ckpt`, as listed."""

    name: str
    path: Path
    size_bytes: int
    modified_at: float
    base_model: str | None
    kind: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "base_model": self.base_model,
            "kind": self.kind,
        }


def _prefix_of(tensor_name: str) -> str | None:
    """Return `__dit__` / `__unet__` etc, or None if the name is not a DT tensor name."""
    if not isinstance(tensor_name, str) or not tensor_name.startswith("__"):
        return None
    bracket = tensor_name.find("[")
    if bracket < 0:
        return None
    return tensor_name[:bracket]


def detect_base_model_names(names) -> str | None:
    """Return `"flux"`, `"sdxl"` or `None` from a list of tensor names.

    A FLUX file is entirely `__dit__`; an SDXL file has none. Measured on real files:
    `bitcoin_lora_flux1d_f16_lora_f16.ckpt` is 1596/1596 `__dit__` and
    `add_detail_xl_lora_f16.ckpt` is 0.

    `None` for an empty or unrecognisable list, never `"sdxl"` -- a default would
    mislabel every file we fail to read, and a mislabelled import produces keys that
    cannot load.
    """
    saw_sdxl = False
    for name in names:
        prefix = _prefix_of(name)
        if prefix == _FLUX_PREFIX:
            return "flux"
        if prefix in _SDXL_PREFIXES:
            saw_sdxl = True
    return "sdxl" if saw_sdxl else None


def detect_kind_names(names) -> str | None:
    """Return `"lora"`, `"model"` or `None` from a list of tensor names.

    The Draw Things folder holds LoRAs, full models and text encoders side by side --
    measured on the real folder: 90 `.ckpt` files spanning `add_detail_xl_lora_f16.ckpt`
    (2112 tensors) through `sd_xl_base_1.0_f16.ckpt` (1820 tensors, 4.9 GB). The Import
    tab converts LoRAs only, so the listing must tell them apart rather than offer a
    4.9 GB model for conversion.

    The discriminator is exact and unambiguous on every file measured: a LoRA's tensors
    are all `__up__`/`__down__` factors, while a model or encoder has none. Any file with
    at least one `__up__`/`__down__` is a LoRA.
    """
    has_lora_parts = False
    for name in names:
        if not isinstance(name, str):
            continue
        if name.endswith("__up__") or name.endswith("__down__"):
            has_lora_parts = True
            break
    if has_lora_parts:
        return "lora"
    return "model" if names else None


def detect_kind(path: Path) -> str | None:
    """Detect whether one `.ckpt` is a LoRA or a model, by reading tensor names."""
    names = _read_names(path)
    return detect_kind_names(names)


def detect_base_model(path: Path) -> str | None:
    """Detect the engine for one `.ckpt` by reading only its tensor names."""
    return detect_base_model_names(_read_names(path))


def _read_names(path: Path) -> list[str]:
    try:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        return [r[0] for r in db.execute(TENSOR_NAME_QUERY)]
    except sqlite3.Error:
        return []
    finally:
        db.close()


def list_ckpt_files(directory: Path | None = None) -> list[FileInfo]:
    """List every `.ckpt` in the Draw Things folder, read-only, sorted by name."""
    root = Path(directory) if directory is not None else model_dir()
    if not root.is_dir():
        return []
    files = []
    for path in sorted(root.glob("*.ckpt"), key=lambda p: p.name):
        try:
            st = path.stat()
        except OSError:
            continue
        names = _read_names(path)
        files.append(FileInfo(
            name=path.name,
            path=path,
            size_bytes=st.st_size,
            modified_at=st.st_mtime,
            # Only a LoRA has an engine. A model's detected "flux" prefix says
            # nothing about which app model it is, so it stays unset there.
            base_model=detect_base_model_names(names) if detect_kind_names(names) == "lora" else None,
            kind=detect_kind_names(names),
        ))
    return files


class DrawThingsSource(Source):
    name = "drawthings"

    def search(self, query, *, kind=None, sort=None, filters=None, limit=40,
               directory: Path | None = None) -> list[Item]:
        """Local files, as items. Filtered by substring because there is no API.

        `kind` filters the real detected kind, so the UI can ask for LoRAs alone and
        list full models separately as not-yet-supported. `directory` defaults to the
        user's Draw Things folder and exists so tests can point at a fixture.
        """
        needle = (query or "").strip().lower()
        items = []
        for info in list_ckpt_files(directory):
            if needle and needle not in info.name.lower():
                continue
            if kind and info.kind != kind:
                continue
            items.append(Item(
                id=f"drawthings:{info.name}",
                name=info.name,
                source="drawthings",
                kind=info.kind or "lora",
                architecture=info.base_model,
                quantization=None,
                size_bytes=info.size_bytes,
                preview_url=None,
                license=None,  # local file: nothing to fetch
                provenance=Provenance(origin="drawthings", source_url=None),
            ))
            if len(items) >= limit:
                break
        return items
