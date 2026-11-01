"""Civitai as a `Source`.

A thin adapter over `civitai_browse`, on the same terms as the Hugging Face adapter:
filtering stays in the helper that already owns it.
"""

from sources import Item, Provenance, Source


def to_item(raw: dict) -> Item:
    """Map one normalised `civitai_browse` entry onto an `Item`."""
    model_id = raw.get("civitai_model_id")
    version_id = raw.get("civitai_version_id")
    size = raw.get("size_bytes")
    model_id_s = "" if model_id is None else str(model_id)
    version_id_s = "" if version_id is None else str(version_id)
    return Item(
        # The composite id keeps the version in the key, so a model with several
        # versions is addressable at version granularity.
        id=f"civitai:{model_id_s}:{version_id_s}",
        name=str(raw.get("label") or raw.get("name") or model_id_s),
        source="civitai",
        kind=str(raw.get("kind") or "diffusion"),
        architecture=raw.get("architecture_label") or raw.get("architecture"),
        quantization=raw.get("quantization"),
        size_bytes=int(size) if isinstance(size, (int, float)) and size > 0 else None,
        preview_url=raw.get("preview_url"),
        license=None,  # fetched on demand; never during a render
        provenance=Provenance(
            origin="civitai",
            source_url=f"https://civitai.com/models/{model_id_s}" if model_id_s else None,
            model_id=model_id_s or None,
            version_id=version_id_s or None,
        ),
    )


class CivitaiSource(Source):
    name = "civitai"

    def search_raw(self, query=None, *, types=None, sort="Most Downloaded", limit=40,
                   token=None) -> dict:
        """Call `civitai_browse.search_models` and return its dict untouched.

        Same reasoning as the Hugging Face adapter: `/api/civitai/models` has always
        returned the helper's shape, so the route delegates to this and `search()`
        serves callers that want uniform `Item`s.
        """
        import civitai_browse
        return civitai_browse.search_models(
            query=query, types=types, sort=sort, limit=limit, token=token
        )

    def search(self, query, *, kind=None, sort=None, filters=None, limit=40) -> list[Item]:
        filters = filters or {}
        data = self.search_raw(
            query,
            types=filters.get("types"),
            sort=sort or "Most Downloaded",
            limit=limit,
        )
        items = data.get("items", data) if isinstance(data, dict) else data
        return [to_item(raw) for raw in items]
