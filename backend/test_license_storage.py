"""Licence storage on the LoRA registry, and the merged inventory endpoint.

Four things are pinned here, three of them the constraints that are easy to break
quietly:

- A `license` block round-trips through the registry verbatim, including the literal
  identifier `"other"` that Hugging Face reports for FLUX.1-dev.
- An entry with no licence omits the key rather than storing a null one. A missing
  licence and an unknown licence are the same thing, and inventing a placeholder
  would make the UI draw a badge that says nothing.
- Refresh touches only entries that name a Civitai model. Everything else has no
  licence to fetch, and a refresh must not invent one for them.
- **No licence fetch happens inside a list read.** `GET /api/loras` is polled by the
  frontend; a network call there would make rendering depend on a third party and
  stall the page when that third party is slow. Licences are refreshed by an
  explicit action, never by looking.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient

import main
import state


_LICENSE = {
    "identifier": "other",
    "url": None,
    "source_url": "https://civitai.com/models/1",
    "permissions": {"allowNoCredit": False, "allowCommercialUse": True,
                    "allowDerivatives": True, "allowDifferentLicense": False},
    "fetched_at": 1.0,
}


class _Registry:
    """A test-local loras.json, so the live registry is never touched.

    The file lives inside DATA_DIR, not /tmp, because `_read_loras` refuses a registry
    outside the data directory and that containment check is deliberate. Writing the
    fixture to /tmp would not be testing the real code path.
    """

    def __init__(self, testcase, entries):
        import shutil
        import tempfile

        self.dir = Path(tempfile.mkdtemp(dir=state.DATA_DIR))
        self.file = self.dir / "loras.json"
        testcase.addCleanup(shutil.rmtree, self.dir, True)
        self.write(entries)

    def write(self, entries):
        import json

        self.file.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        # The registry is cached on (mtime, size); a rewrite with identical content
        # and size would otherwise be served from the previous read.
        state._loras_cache.clear()

    def read(self):
        state._loras_cache.clear()
        return state._read_loras()


class LicenseStorageTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.reg = _Registry(self, [{"name": "x", "path": "/tmp/x.safetensors"}])
        patcher = mock.patch.object(state, "LORAS_FILE", self.reg.file)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_license_block_round_trips_through_the_registry(self):
        state._upsert_lora_entries([
            {"name": "x", "path": "/tmp/x.safetensors", "license": dict(_LICENSE)}
        ])
        stored = self.reg.read()
        self.assertEqual(stored[0]["license"]["identifier"], "other")
        self.assertEqual(stored[0]["license"]["permissions"]["allowCommercialUse"], True)
        self.assertEqual(stored[0]["license"]["source_url"],
                         "https://civitai.com/models/1")

    def test_an_entry_with_no_license_omits_the_key(self):
        """Absent, not null.

        A null licence would be indistinguishable from a source that answered "no
        licence" if anything ever started coercing it, and it would draw a badge.
        """
        stored = self.reg.read()
        self.assertEqual(stored[0]["name"], "x")
        self.assertNotIn("license", stored[0])

    def test_a_refresh_updates_only_entries_that_have_a_civitai_id(self):
        entries = [
            {"name": "with_id", "path": "/tmp/a.safetensors", "civitai_model_id": "123"},
            {"name": "without_id", "path": "/tmp/b.safetensors"},
        ]
        self.reg.write(entries)
        fetched = type("Info", (), {"to_dict": lambda self: {"identifier": "creativeml-openrail-m"}})()
        with mock.patch("licensing.fetch_civitai_license", return_value=fetched) as fetch:
            result = state.refresh_licenses()
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(fetch.call_args.args[0], "123")

        after = {e["name"]: e for e in self.reg.read()}
        self.assertEqual(after["with_id"]["license"]["identifier"],
                         "creativeml-openrail-m")
        self.assertNotIn("license", after["without_id"])
        self.assertEqual(result["updated"], 1)
        self.assertEqual(result["skipped"], 1)

    def test_refresh_leaves_an_entry_alone_when_the_fetch_fails(self):
        """A source being unreachable must cost the badge, not the entry."""
        self.reg.write([
            {"name": "with_id", "path": "/tmp/a.safetensors", "civitai_model_id": "123",
             "license": {"identifier": "creativeml-openrail-m"}}
        ])
        with mock.patch("licensing.fetch_civitai_license", return_value=None):
            result = state.refresh_licenses()
        self.assertEqual(result["updated"], 0)
        stored = self.reg.read()
        self.assertEqual(stored[0]["license"]["identifier"], "creativeml-openrail-m")

    def test_refresh_never_runs_during_a_list_read(self):
        """Global Constraint: no licence fetch inside GET /api/loras.

        The frontend polls that route. A licence fetch there would make a render
        depend on a third party and stall every list view when that party is slow.
        """
        with mock.patch("licensing.fetch_civitai_license") as fetch:
            r = self.client.get("/api/loras")
        self.assertEqual(r.status_code, 200)
        fetch.assert_not_called()

    def test_the_refresh_endpoint_is_an_explicit_action(self):
        self.reg.write([
            {"name": "with_id", "path": "/tmp/a.safetensors", "civitai_model_id": "123"}
        ])
        info = type("Info", (), {"to_dict": lambda self: {"identifier": "cc-by-4.0"}})()
        with mock.patch("licensing.fetch_civitai_license", return_value=info):
            r = self.client.post("/api/loras/refresh-licenses")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["updated"], 1)


class InventoryTests(unittest.TestCase):
    """`GET /api/models/inventory` merges what two routes used to return separately.

    The Models tab currently renders the registry and the on-disk detection list as
    two different things, so a model downloaded from the browser is invisible beside a
    built-in one. One list, each entry marked by origin, is the point.
    """

    def setUp(self):
        self.client = TestClient(main.app)

    def test_inventory_merges_registered_and_detected(self):
        r = self.client.get("/api/models/inventory")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("items", body)

        origins = {i.get("origin") for i in body["items"]}
        self.assertTrue(origins <= {"registry", "detected"}, origins)

    def test_every_inventory_entry_names_its_origin(self):
        """An entry that does not say where it came from cannot be rendered correctly."""
        body = self.client.get("/api/models/inventory").json()
        for item in body["items"]:
            self.assertIn("origin", item)
            self.assertIn(item["origin"], ("registry", "detected"))
            self.assertIn("name", item)

    def test_inventory_does_not_duplicate_a_model_the_registry_knows(self):
        """A registered model is in both lists by design; it must appear once."""
        body = self.client.get("/api/models/inventory?include_registered=true").json()
        names = [i["name"] for i in body["items"]]
        self.assertEqual(len(names), len(set(names)),
                         "the merge duplicated an entry: %s" % names)


if __name__ == "__main__":
    unittest.main()

class LicenseStampTests(unittest.TestCase):
    """A stored licence has to know how old it is.

    `fetched_at` was hardcoded to 0.0 in both fetchers and asserted nowhere, so every
    licence ever stored claimed to have been fetched at the epoch. Nothing could tell
    a fresh badge from a stale one, which is the field's entire purpose.
    """

    def test_a_live_fetch_is_stamped_with_the_time_of_the_call(self):
        import time

        import licensing

        before = time.time()
        with mock.patch("licensing._get_json", return_value={"id": 2890476}):
            info = licensing.fetch_civitai_license(2890476)
        self.assertIsNotNone(info)
        self.assertGreaterEqual(info.fetched_at, before)

    def test_the_payload_builders_stay_pure(self):
        """They hardcode 0.0 so tests on them are deterministic.

        The stamp belongs to the call, not the payload; a test asserting on a fixture
        should not have to freeze the clock to pass.
        """
        from licensing import fetch_civitai_license_from_payload, fetch_hf_license_from_payload

        civitai = fetch_civitai_license_from_payload({"id": 1})
        hf = fetch_hf_license_from_payload({"id": "x", "cardData": {"license": "other"}})
        self.assertEqual(civitai.fetched_at, 0.0)
        self.assertEqual(hf.fetched_at, 0.0)
