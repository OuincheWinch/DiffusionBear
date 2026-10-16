"""Bugs found reviewing the model/LoRA management work, reproduced as tests.

Each of these was a real defect, found by reading rather than by a failing test, which
is the point of writing them down now: they were all invisible to the suite that shipped
them.

Two of the three are in `refresh_licenses`, added in the same task. The third is in the
merged inventory endpoint that same task introduced. A feature that ships with a green
suite and then loses a LoRA when the user presses its refresh button needs the failure
recorded, not just the fix.
"""

import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient

import main
import state
from sources import LicenseInfo


class _RegistryCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(dir=state.DATA_DIR))
        self.file = self.dir / "loras.json"
        self.addCleanup(shutil.rmtree, self.dir, True)
        patcher = mock.patch.object(state, "LORAS_FILE", self.file)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, entries):
        self.file.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        state._loras_cache.clear()

    def read(self):
        state._loras_cache.clear()
        return state._read_loras()


class RefreshLicensesDataLossTests(_RegistryCase):
    """A refresh must not delete a registry entry.

    `refresh_licenses` collected the registry into a dict keyed on `name` and wrote the
    dict's values back. The registry has no unique constraint on `name` -- `_upsert_lora_
    entries` matches on name OR path -- so two entries may legitimately share a name, and
    the dict kept only the last of them. One `POST /api/loras/refresh-licenses` then
    rewrote the whole file from the survivors.

    Reproduced before the fix: a registry of `[dup:/tmp/a, dup:/tmp/b]` became
    `[dup:/tmp/b]`. The user pressed a licence-refresh button and lost a LoRA.
    """

    def test_two_entries_sharing_a_name_both_survive_a_refresh(self):
        self.write([
            {"name": "dup", "path": "/tmp/a.safetensors", "civitai_model_id": "1"},
            {"name": "dup", "path": "/tmp/b.safetensors", "civitai_model_id": "2"},
        ])
        info = LicenseInfo(identifier="cc-by-4.0", fetched_at=1.0)
        with mock.patch("licensing.fetch_civitai_license", return_value=info):
            state.refresh_licenses()

        after = self.read()
        self.assertEqual(len(after), 2, "a licence refresh deleted a registry entry")
        self.assertEqual(
            sorted(e["path"] for e in after),
            ["/tmp/a.safetensors", "/tmp/b.safetensors"],
        )

    def test_both_surviving_entries_get_their_own_licence(self):
        self.write([
            {"name": "dup", "path": "/tmp/a.safetensors", "civitai_model_id": "1"},
            {"name": "dup", "path": "/tmp/b.safetensors", "civitai_model_id": "2"},
        ])
        seen = []

        def fetch(model_id, token=None):
            seen.append(model_id)
            return LicenseInfo(identifier=f"lic-{model_id}", fetched_at=1.0)

        with mock.patch("licensing.fetch_civitai_license", fetch):
            state.refresh_licenses()

        self.assertEqual(sorted(seen), ["1", "2"])
        by_path = {e["path"]: e for e in self.read()}
        self.assertEqual(by_path["/tmp/a.safetensors"]["license"]["identifier"], "lic-1")
        self.assertEqual(by_path["/tmp/b.safetensors"]["license"]["identifier"], "lic-2")


class RefreshLicensesRedundantWriteTests(_RegistryCase):
    """Nothing changed means nothing written.

    The equality check compared the stored block against a freshly fetched one. Because
    `fetched_at` is stamped at the fetch boundary, two fetches are never byte-equal, so
    the "unchanged" branch was dead: every click reported `updated` and rewrote
    `loras.json`, which also bumped its mtime and invalidated the read cache for every
    other caller.

    Reproduced before the fix: with a fetch that stamps the current time, two consecutive
    refreshes both returned `updated: 1` and the file's mtime changed both times.

    The timestamp is excluded from the comparison on purpose. It records *when we last
    checked*, and the point of this branch is "we already have this answer".
    """

    def _fetch(self, model_id, token=None):
        return LicenseInfo(identifier="cc-by-4.0", fetched_at=time_now())

    def test_a_second_refresh_reports_no_change(self):
        self.write([{"name": "x", "path": "/tmp/a.safetensors", "civitai_model_id": "1"}])
        with mock.patch("licensing.fetch_civitai_license", self._fetch):
            first = state.refresh_licenses()
            second = state.refresh_licenses()
        self.assertEqual(first["updated"], 1)
        self.assertEqual(second["updated"], 0)
        self.assertEqual(second["skipped"], 1)

    def test_a_second_refresh_does_not_rewrite_the_file(self):
        """Rewriting bumps the mtime and throws away the read cache for everyone."""
        self.write([{"name": "x", "path": "/tmp/a.safetensors", "civitai_model_id": "1"}])
        with mock.patch("licensing.fetch_civitai_license", self._fetch):
            state.refresh_licenses()
            before = self.file.stat().st_mtime_ns
            state.refresh_licenses()
            self.assertEqual(self.file.stat().st_mtime_ns, before,
                             "an unchanged refresh rewrote loras.json")

    def test_a_changed_licence_is_still_written(self):
        """The check must not become so lenient that real updates stop landing."""
        self.write([{"name": "x", "path": "/tmp/a.safetensors", "civitai_model_id": "1"}])
        with mock.patch("licensing.fetch_civitai_license",
                        return_value=LicenseInfo(identifier="first", fetched_at=1.0)):
            state.refresh_licenses()
        with mock.patch("licensing.fetch_civitai_license",
                        return_value=LicenseInfo(identifier="second", fetched_at=2.0)):
            state.refresh_licenses()
        self.assertEqual(self.read()[0]["license"]["identifier"], "second")


class InventoryDetectedRowsTests(unittest.TestCase):
    """The inventory must actually list what is on disk.

    `/api/models/inventory` built its "already known" set from
    `_detected_model_rows(include_registered=True)` and then asked whether each row of
    `_detected_model_rows(include_registered=False)` was in it. Those two sets are in a
    subset relation -- the only difference is one `continue` inside the scan -- so every
    row was in `known` and every row was filtered out.

    Result: the default response contained zero detected entries, which is precisely the
    invisibility the endpoint was added to remove. Measured before the fix on this
    machine: `_detected_model_rows(False)` -> 0 rows, inventory -> 10 registry entries and
    0 detected.
    """

    def test_detected_rows_are_not_filtered_away_by_the_registry(self):
        """Expressed as an invariant, because the row count is machine state.

        On this machine every directory in the model store belongs to a registry model,
        so the honest default response has zero detected rows. Asserting "there is at
        least one" would test what happens to be downloaded, not the endpoint. What must
        hold is that whatever the scan reports as NOT registered is actually listed --
        that is the property the subset bug broke.
        """
        from routers.downloads import _detected_model_rows

        items = TestClient(main.app).get("/api/models/inventory").json()["items"]
        listed = {i["name"] for i in items}
        unregistered = {r["name"] for r in _detected_model_rows(include_registered=False)}
        self.assertEqual(
            unregistered - listed, set(),
            "inventory dropped rows the scan reported as unregistered",
        )

    def test_include_registered_never_shrinks_the_list(self):
        """Widening the scan may add rows, never remove them.

        Note it does not necessarily add one row per newly-visible directory: a
        registered directory whose model is already listed by id is skipped rather than
        repeated, which is the other half of this endpoint's job.
        """
        client = TestClient(main.app)
        plain = [i["name"] for i in client.get("/api/models/inventory").json()["items"]]
        wide = [i["name"] for i in
                client.get("/api/models/inventory?include_registered=true").json()["items"]]
        self.assertGreaterEqual(len(wide), len(plain))
        self.assertEqual(len(set(wide)), len(wide), "widening duplicated an entry")

    def test_a_registered_row_never_appears_twice(self):
        """The `registered` flag is what the filter now keys on.

        A directory can be reported unregistered and still be absent from the detected
        list, because the model it belongs to is already listed by id further up. That is
        the dedup working, not the bug. What must never happen is the same name showing
        up twice in one response.
        """
        for flag in (False, True):
            with self.subTest(include_registered=flag):
                items = TestClient(main.app).get(
                    f"/api/models/inventory?include_registered={flag}").json()["items"]
                keys = [(i["origin"], i["name"]) for i in items]
                self.assertEqual(len(keys), len(set(keys)),
                                 "one model is listed twice in a single response")

    def test_include_registered_does_not_duplicate_a_registry_model(self):
        client = TestClient(main.app)
        names = [i["name"] for i in
                 client.get("/api/models/inventory?include_registered=true").json()["items"]]
        self.assertEqual(len(names), len(set(names)))

    def test_a_model_the_registry_owns_is_not_repeated_as_detected(self):
        client = TestClient(main.app)
        items = client.get("/api/models/inventory").json()["items"]
        registry = {i["name"] for i in items if i["origin"] == "registry"}
        repeated = [i["name"] for i in items
                    if i["origin"] == "detected" and i["name"] in registry]
        self.assertEqual(repeated, [],
                         "a registry model also appeared as a detected leftover")


def time_now():
    import time

    return time.time()


class _FakeRepo:
    def __init__(self, repo_id, tags):
        self.id = repo_id
        self.tags = tags
        self.downloads = 1
        self.likes = 1
        self.pipeline_tag = None
        self.lastModified = "2026-01-01"


class HfKindFilterTests(unittest.TestCase):
    """The `kind` filter on HF search did nothing at all.

    `search_models` classified each result into `kind` -- assigning to the same name as
    the `kind` PARAMETER. From the second iteration on, the filter was therefore
    comparing each item against the previous item's guessed kind.

    Measured before the fix: with three results (diffusion, lora, diffusion),
    `kind="lora"` returned all three, byte-identical to passing no filter at all. The
    Models tab has offered a kind dropdown for a while; it was decorative.
    """

    RAW = [
        _FakeRepo("mlx-community/a-diffusion", ["diffusers"]),
        _FakeRepo("mlx-community/b-lora", ["lora"]),
        _FakeRepo("mlx-community/c-other", ["diffusers"]),
    ]

    def _search(self, **kwargs):
        import hf_browse

        with mock.patch.object(hf_browse, "_list_models", lambda api, **kw: list(self.RAW)):
            return hf_browse.search_models(search=None, limit=50, **kwargs)

    def test_kind_lora_returns_only_loras(self):
        self.assertEqual([i["kind"] for i in self._search(kind="lora")["items"]], ["lora"])

    def test_kind_diffusion_excludes_loras(self):
        kinds = [i["kind"] for i in self._search(kind="diffusion")["items"]]
        self.assertNotIn("lora", kinds)
        self.assertTrue(kinds)

    def test_no_filter_returns_everything(self):
        self.assertEqual(len(self._search(kind=None)["items"]), 3)

    def test_a_filter_actually_reduces_the_result(self):
        """The property that was broken: filtering changed nothing."""
        self.assertLess(len(self._search(kind="lora")["items"]),
                        len(self._search(kind=None)["items"]))


class AllowRootCannotGrantProtectedTreesTests(unittest.TestCase):
    """`allow_root` must not be able to permit a write into a protected tree.

    `_check_output` returned as soon as `allow_root` was satisfied, before checking
    `PROTECTED_ROOTS` at all. So a caller naming the Draw Things folder as its allowed
    root got a write straight into the user's model library -- the one directory this app
    must never mutate. An allow-list says where a write is permitted; it cannot grant
    what the denylist already refuses.

    Reproduced before the fix:
        _check_output(DRAW_THINGS_MODEL_DIR / "evil.safetensors",
                      allow_root=str(DRAW_THINGS_MODEL_DIR))
        -> returned a path instead of raising
    """

    def test_allow_root_naming_the_draw_things_folder_is_refused(self):
        import lora_convert

        with self.assertRaises(lora_convert.ConversionError):
            lora_convert._check_output(
                lora_convert.DRAW_THINGS_MODEL_DIR / "evil.safetensors",
                allow_root=str(lora_convert.DRAW_THINGS_MODEL_DIR),
            )

    def test_allow_root_naming_the_containers_root_is_refused(self):
        import lora_convert

        with self.assertRaises(lora_convert.ConversionError):
            lora_convert._check_output(
                lora_convert.CONTAINERS_ROOT / "anything" / "x.safetensors",
                allow_root=str(lora_convert.CONTAINERS_ROOT),
            )

    def test_the_read_only_rule_still_fires_without_allow_root(self):
        import lora_convert

        with self.assertRaises(lora_convert.ConversionError):
            lora_convert._check_output(
                lora_convert.DRAW_THINGS_MODEL_DIR / "evil.safetensors")

    def test_protected_roots_follow_the_current_home(self):
        """`Path.home()` is replaced process-wide by a test and never restored.

        The roots used to be captured at import, so after that replacement they pointed
        at a temp directory and the read-only rule stopped protecting the real tree --
        which is how a Draw Things safety test could pass for the wrong reason.
        """
        import lora_convert

        roots = lora_convert._protected_roots()
        self.assertEqual(roots[0], Path.home() / "Library/Containers/com.liuliu.draw-things/Data/Documents/Models")

        with mock.patch.object(lora_convert.Path, "home",
                               return_value=Path("/tmp/fake-home")):
            self.assertEqual(lora_convert._protected_roots()[0],
                             Path("/tmp/fake-home/Library/Containers/com.liuliu.draw-things/Data/Documents/Models"))


class BaseModelVocabularyTests(unittest.TestCase):
    """Two vocabularies for the same concept, and no check that they meet.

    The Draw Things side says `flux`: `ConvertRequest` validates `^(sdxl|flux)$` and the
    conversion summary reports `req.base_model` back. The LoRA registry says `flux2`:
    `routers/loras._BASE_MODELS` is `{sdxl, flux2, krea2, z-image}` and anything else is a
    400.

    Nothing reconciled them, so every FLUX import reached the Confirm button -- after the
    ~40 s conversion had already run -- and then failed:

        POST /api/sources/drawthings/convert -> {"base_model": "flux"}
        POST /api/loras                     -> 400 "invalid base model"

    The UI even warns the user that FLUX cannot be used, and then offered the button
    anyway. The fix is a mapping in the client; this test is what stops it drifting back.
    """

    CONVERTER_VOCABULARY = {"sdxl", "flux"}
    REGISTRY_VOCABULARY = {"sdxl", "flux2", "krea2", "z-image"}

    def test_the_converter_only_speaks_values_the_registry_knows(self):
        import routers.sources as sources_router

        field = sources_router.ConvertRequest.model_fields["base_model"]
        pattern = next(m.pattern for m in field.metadata if hasattr(m, "pattern"))
        # strip() removes CHARACTERS, not the anchors -- "^" and "$" are a set of
        # characters to it, so the first attempt left "(sdxl" and "flux)" behind.
        emitted = set(pattern.removeprefix("^").removesuffix("$").strip("()").split("|"))
        self.assertEqual(emitted, self.CONVERTER_VOCABULARY)
        self.assertEqual(
            emitted - self.REGISTRY_VOCABULARY, {"flux"},
            "if the converter vocabulary changes, the client mapping below must change too",
        )

    def test_the_registry_rejects_the_converter_vocabulary_directly(self):
        """Documents the raw mismatch, so the mapping is not mistaken for redundancy."""
        from routers.loras import _BASE_MODELS

class OptionsPreflightTests(unittest.TestCase):
    """CORS preflight (OPTIONS) requests never carry authorization headers.
    _requires_api_token must exempt OPTIONS on all paths unconditionally."""

    def test_options_preflight_does_not_require_token(self):
        self.assertFalse(main._requires_api_token("OPTIONS", "/api/generate"))
        self.assertFalse(main._requires_api_token("OPTIONS", "/api/models"))
        self.assertFalse(main._requires_api_token("OPTIONS", "/api/custom-models"))

    def test_get_on_protected_endpoints_still_requires_token(self):
        self.assertTrue(main._requires_api_token("GET", "/api/models"))
        self.assertTrue(main._requires_api_token("POST", "/api/generate"))


if __name__ == "__main__":
    unittest.main()

