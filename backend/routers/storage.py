"""Storage manifest endpoints.

Read-only. There is deliberately no delete verb in this router: the manifest's job
is to make 28 GB of weights legible and to name what is unaccounted for, and a
report that can also destroy things is a report people stop opening.
"""

from fastapi import APIRouter

import storage

router = APIRouter(tags=["storage"])


@router.get("/api/storage")
def get_storage(force: bool = False):
    """Everything on disk that the app owns, grouped by category.

    Results are cached for a minute because the UI polls this; pass force=true to
    rescan on demand. Nothing is hashed, so a scan of 28 GB takes milliseconds and
    never contends with generation.
    """
    return storage.get_report(force=force)
