"""Licence and usage terms, fetched per source.

What each source actually offers, measured against the live APIs:

**Civitai** (`GET /api/v1/models/{id}`) returns **no licence name** -- not on the
version, not on the model, not on files. It returns four permissions:

    allowNoCredit          bool
    allowCommercialUse     list, e.g. ["Image","RentCivit","Rent","Sell","SellMerge"]
    allowDerivatives       bool
    allowDifferentLicense  bool

**Hugging Face** (`GET https://huggingface.co/api/models/{id}`) returns
`cardData.license` as an SPDX identifier. It is frequently the literal `"other"` --
`black-forest-labs/FLUX.1-dev` reports `"other"`, because its real terms live in the
README card rather than in that field.

**So this module never produces a licence name.** There is no mapping table here from
permission flags to conventional names, and nothing scrapes the Civitai web page. A wrong
licence displayed next to a model is worse than none displayed, because a user relies on
it. `Global Constraint 1` in the plan, and the reason this docstring is longer than the
code.

The pure `*_from_payload` functions hold all the logic and are unit tested against
recorded fixtures. The `fetch_*` wrappers add a network call and a bounded timeout, and
return `None` on any failure so an unreachable source can never stall a list render --
Civitai's first request to a cold host measured **20.5 s**, which is why the timeout is
short and why nothing here runs during a render.
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import replace

from sources import LicenseInfo

CIVITAI_API_BASE = "https://civitai.com/api/v1"
HF_API_BASE = "https://huggingface.co/api/models"

# Short, because these are never on the critical path of a page render.
DEFAULT_TIMEOUT = 8.0

# Civitai expresses usage as a list of the things that are permitted. An empty or absent
# list means none, which is different from "unknown", so it maps to an empty list.
_CIVITAI_PERMISSION_KEYS = {
    "no_credit": "allowNoCredit",
    "derivatives": "allowDerivatives",
    "different_license": "allowDifferentLicense",
}


def _safe(call):
    """Run `call`, returning None on any network or parse failure.

    Review Focus 4: a source being unreachable must degrade to "no licence shown", not
    to a hung page or a 500.
    """
    try:
        return call()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            OSError, ValueError, json.JSONDecodeError):
        return None


def _get_json(url: str, token: str | None, timeout: float = DEFAULT_TIMEOUT) -> dict | None:
    # Civitai answers 403 to urllib's default User-Agent and 200 to an ordinary browser
    # one. Measured, not guessed: the same request with curl returned 200 and with
    # urllib returned 403, which without this header would degrade silently to "no
    # licence shown" while every unit test stayed green.
    headers = {
        "Accept": "application/json",
        "User-Agent": "DiffusionBear/1.0 (+https://github.com; local app)",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_civitai_license_from_payload(payload: dict) -> LicenseInfo:
    """Build a `LicenseInfo` from a Civitai model payload. Pure.

    Raises `ValueError` when the payload carries no model id, because a licence with no
    subject is not a licence.
    """
    model_id = payload.get("id")
    if model_id is None:
        raise ValueError("Civitai payload has no model id")

    permissions: dict = {}
    for name, key in _CIVITAI_PERMISSION_KEYS.items():
        if key in payload:
            permissions[name] = bool(payload[key])
    if "allowCommercialUse" in payload:
        value = payload["allowCommercialUse"]
        permissions["commercial_use"] = list(value) if isinstance(value, list) else (
            [] if value in (None, "", "None") else [str(value)])

    # identifier stays None on purpose -- see the module docstring.
    return LicenseInfo(
        identifier=None,
        url=None,
        source_url=f"https://civitai.com/models/{model_id}",
        permissions=permissions or None,
        fetched_at=0.0,
    )


def fetch_hf_license_from_payload(payload: dict) -> LicenseInfo:
    """Build a `LicenseInfo` from a Hugging Face model payload. Pure.

    `cardData.license` wins when present, including when it is `"other"`. A
    `license:` tag is used only when `cardData` is absent entirely. Absent both, the
    identifier is `None` -- never a default.
    """
    repo_id = payload.get("id") or payload.get("modelId")
    if not repo_id:
        raise ValueError("Hugging Face payload has no repo id")

    identifier = None
    card = payload.get("cardData")
    if isinstance(card, dict) and "license" in card:
        identifier = card["license"]
    else:
        for tag in payload.get("tags") or []:
            if isinstance(tag, str) and tag.startswith("license:"):
                identifier = tag.split(":", 1)[1].strip() or None
                break

    return LicenseInfo(
        identifier=identifier or None,
        url=None,
        source_url=f"https://huggingface.co/{repo_id}",
        permissions=None,  # HF expresses no equivalent flags
        fetched_at=0.0,
    )


def _stamped(info: LicenseInfo) -> LicenseInfo:
    """Record when this licence was actually fetched.

    The `*_from_payload` builders stay pure and hardcode `fetched_at=0.0`, which keeps
    them deterministic for tests. When the fetch happened is a property of the *call*,
    not of the payload, so it is stamped here, at the boundary where the network call
    has just succeeded.

    This is the only thing that lets a stored licence report its own age. Without it
    every stored block claims epoch 0 and nothing can ever say which badges are stale.
    """
    return replace(info, fetched_at=time.time())


def fetch_civitai_license(model_id, token: str | None = None) -> LicenseInfo | None:
    """Fetch one Civitai model's permissions. Returns None on any failure."""
    if not model_id:
        return None
    payload = _safe(lambda: _get_json(f"{CIVITAI_API_BASE}/models/{int(model_id)}", token))
    if payload is None:
        return None
    try:
        return _stamped(fetch_civitai_license_from_payload(payload))
    except ValueError:
        return None


def fetch_hf_license(repo_id: str, token: str | None = None) -> LicenseInfo | None:
    """Fetch one Hugging Face repo's licence identifier. Returns None on any failure."""
    if not repo_id or "/" not in repo_id:
        return None
    payload = _safe(lambda: _get_json(f"{HF_API_BASE}/{repo_id}", token))
    if payload is None:
        return None
    try:
        return _stamped(fetch_hf_license_from_payload(payload))
    except ValueError:
        return None
