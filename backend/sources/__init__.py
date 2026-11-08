"""The shapes every model source fills and the frontend reads.

One `Item` across Hugging Face, Civitai and Draw Things, so the browse grid renders a
Civitai model and a Hugging Face repo through the same component instead of switching
between two incompatible HTTP shapes.

The field names here are a contract, not an implementation detail. The frontend parses
these keys, and renaming one after that ships breaks it silently. `backend/test_sources.py`
is the record of that contract.

**The licence rule.** `LicenseInfo.identifier` may legitimately be the literal string
`"other"` — Hugging Face reports that for FLUX.1-dev, whose real terms live in the README
card rather than the API field — and Civitai returns no identifier at all, only
`permissions`. So: never synthesise, infer or scrape a licence *name*. A wrong licence
shown next to a model is worse than none shown, because it gets relied on. A field a
source cannot supply is `None`, never a guess.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Provenance:
    """Where an item came from, in enough detail to link back to it.

    `origin` is one of `"huggingface"`, `"civitai"`, `"drawthings"` or `"local"`.
    `model_id` is a Civitai model id or a Hugging Face repo id; `version_id` is a Civitai
    version id or a Hugging Face revision. Both may be `None` for a local file.
    """

    origin: str
    source_url: str | None = None
    model_id: str | None = None
    version_id: str | None = None
    version_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LicenseInfo:
    """Licence and usage terms for one item.

    `identifier` is a source-supplied string and is carried through verbatim, including
    the literal `"other"`. `permissions` is Civitai's four flags and is `None` on sources
    that do not express them. `url` points at licence text, `source_url` at the page the
    creator published, because the creator's own wording is the only authoritative
    licence text either source offers.
    """

    identifier: str | None = None
    url: str | None = None
    source_url: str | None = None
    permissions: dict[str, Any] | None = None
    fetched_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Item:
    """One downloadable model, LoRA or upscaler, from any source."""

    id: str
    name: str
    source: str
    kind: str
    architecture: str | None = None
    quantization: str | None = None
    size_bytes: int | None = None
    preview_url: str | None = None
    license: LicenseInfo | None = None
    provenance: Provenance = field(default_factory=lambda: Provenance("unknown"))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # `license` is the licence identifier for most sources but the dataclass field
        # holds the whole block; nest it as an object so the frontend has one shape.
        d["license"] = self.license.to_dict() if self.license is not None else None
        d["provenance"] = self.provenance.to_dict()
        return d


class Source:
    """Base class for a model source.

    Subclasses set `name` and implement the three methods. `search` returns already
    filtered results: filtering stays in the existing `hf_browse` / `civitai_browse`
    helpers rather than being reimplemented per adapter, because those helpers encode
    decisions (architecture guessing from repo-name substrings, quantisation parsing)
    that have already been got wrong once.
    """

    name: str = ""

    def search(self, query, *, kind=None, sort=None, filters=None, limit=40) -> list[Item]:
        raise NotImplementedError

    def detail(self, item_id: str) -> Item:
        raise NotImplementedError

    def install(self, item_id: str) -> str:
        """Begin an install and return the task id that tracks it."""
        raise NotImplementedError


def _registry():
    """Build the source registry lazily.

    Imported inside the function so `sources` does not import `hf_browse` or
    `civitai_browse` at module load -- those pull in `huggingface_hub` and `requests`,
    which `drawthings_names` and `licensing` do not need.
    """
    from sources import civitai, diffusionbee, drawthings, huggingface
    return {
        huggingface.HuggingFaceSource.name: huggingface.HuggingFaceSource,
        civitai.CivitaiSource.name: civitai.CivitaiSource,
        drawthings.DrawThingsSource.name: drawthings.DrawThingsSource,
        diffusionbee.DiffusionBeeSource.name: diffusionbee.DiffusionBeeSource,
    }


class _LazyRegistry(dict):
    """A dict that populates itself on first access."""

    def _fill(self):
        if not super().__len__():
            super().update(_registry())

    def __contains__(self, key):
        self._fill()
        return super().__contains__(key)

    def __getitem__(self, key):
        self._fill()
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._fill()
        return super().get(key, default)

    def keys(self):
        self._fill()
        return super().keys()

    def items(self):
        self._fill()
        return super().items()

    def __len__(self):
        self._fill()
        return super().__len__()


SOURCES = _LazyRegistry()
