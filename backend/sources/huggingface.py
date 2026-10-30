"""Hugging Face as a `Source`.

A thin adapter over `hf_browse`. It does not filter, guess architecture, or parse
quantisation: `hf_browse.guess_architecture`, `guess_quantization` and `guess_kind`
already encode those decisions and are tested. Duplicating them here is how they drift
apart, which they already did once.
"""

from sources import Item, Provenance, Source


def to_item(raw: dict) -> Item:
    """Map one normalised `hf_browse` entry onto an `Item`."""
    repo_id = raw.get("repo_id") or raw.get("id") or ""
    # `architecture_label` is the human string; `architecture` is the filter key.
    architecture = raw.get("architecture_label") or raw.get("architecture")
    size = raw.get("size_bytes")
    return Item(
        id=f"hf:{repo_id}",
        name=str(raw.get("label") or raw.get("name") or repo_id),
        source="huggingface",
        kind=str(raw.get("kind") or "diffusion"),
        architecture=architecture,
        quantization=raw.get("quantization"),
        size_bytes=int(size) if isinstance(size, (int, float)) and size > 0 else None,
        preview_url=raw.get("preview_url") or raw.get("image_url"),
        license=None,  # fetched on demand; never during a render
        provenance=Provenance(
            origin="huggingface",
            source_url=f"https://huggingface.co/{repo_id}" if repo_id else None,
            model_id=repo_id or None,
        ),
    )


class HuggingFaceSource(Source):
    name = "huggingface"

    def search_raw(
        self,
        search=None,
        *,
        author=None,
        architecture=None,
        quantization=None,
        kind=None,
        limit=40,
        sort="downloads",
        models_only=True,
    ) -> dict:
        """Call `hf_browse.search_models` and return its dict untouched.

        This exists because `/api/hf/models` has always returned the helper's own
        shape -- facets included -- and the frontend reads those keys. Wrapping it
        into `Item`s here would silently change a live response, so the route calls
        this instead and `search()` remains the uniform path for new callers.

        Keeping the call in one place is the point of the split: `downloads.py` no
        longer knows that this source is Hugging Face, only that it is "huggingface".
        """
        import hf_browse
        return hf_browse.search_models(
            search=search,
            author=author,
            architecture=architecture,
            quantization=quantization,
            kind=kind,
            limit=limit,
            sort=sort,
            models_only=models_only,
        )

    def search(self, query, *, kind=None, sort=None, filters=None, limit=40) -> list[Item]:
        filters = filters or {}
        data = self.search_raw(
            query,
            author=filters.get("author"),
            architecture=filters.get("architecture"),
            quantization=filters.get("quantization"),
            kind=kind,
            limit=limit,
            sort=sort or "downloads",
        )
        return [to_item(raw) for raw in data.get("items", [])]
