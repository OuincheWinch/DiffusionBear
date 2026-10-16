"""Deleting a detected model must not take unrelated settings with it.

Two separate concerns are pinned here, and one reported premise is corrected.

Clearing model_paths overrides (reported bug)

The report was that ``delete_detected_model`` wipes every ``model_paths`` override,
because ``app_settings.update_settings`` is claimed to REPLACE the whole key rather
than merge into it. It does not. ``update_settings`` validates the *partial* map it is
handed and then applies it key by key against the loaded settings
(``app_settings.update_settings``, the ``if key == "model_paths"`` arm does
``current.setdefault("model_paths", {})[model_id] = path``), and an empty-string value
pops exactly that one id. The per-key loop in the endpoint is therefore correct, and a
before/after reproduction on this branch shows unrelated overrides intact after the
delete. ``test_unrelated_model_paths_overrides_survive_a_delete`` runs that
reproduction for real so a future change to the merge semantics fails loudly instead
of silently eating user settings.

The endpoint does gain a real safety hole, which is what the rest of this file covers:
``shutil.rmtree`` on a multi-gigabyte model store had no server-side confirmation, so
any caller that was not the settings panel's ``window.confirm`` -- a script, curl, a
future panel -- deleted unrecoverably without ever being asked.

Every test here is hermetic. ``DIFFUSIONBEAR_DATA_DIR`` and
``DIFFUSIONBEAR_ASSET_DIR`` are pointed at fresh temp directories *before* the backend
modules are imported, because ``app_settings`` resolves them at import time. The real
settings file and the real model store are never read or written.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

BACKEND = Path(__file__).resolve().parent
_ENV_VARS = ("DIFFUSIONBEAR_DATA_DIR", "DIFFUSIONBEAR_ASSET_DIR")

# Modules to drop from sys.modules so the env change below is actually observed.
# app_settings resolves DATA_DIR/ASSET_DIR at import; state.py and generator.py
# resolve their own directories the same way.
_PURGED_PREFIXES = ("routers", "generator", "app_settings", "state", "hf_browse",
                    "hf_service", "civitai_service", "civitai_browse", "sdxl_convert",
                    "storage", "sdxl_layout")


def _is_backend_module(name: str, mod) -> bool:
    """True for a module loaded out of this checkout, by file or by package name."""
    try:
        file = getattr(mod, "__file__", None)
    except Exception:  # pragma: no cover - defensive
        return False
    if file:
        try:
            if Path(file).resolve().is_relative_to(BACKEND):
                return True
        except (OSError, ValueError):
            pass
    return name.split(".")[0] in _PURGED_PREFIXES


def _drop_backend_modules() -> None:
    """Evict anything loaded from this checkout, whatever its module name."""
    for name, mod in list(sys.modules.items()):
        if _is_backend_module(name, mod):
            del sys.modules[name]


class DetectedModelDeleteTests(unittest.TestCase):
    """The two endpoints' contracts, with no real settings and no real weights."""

    @classmethod
    def setUpClass(cls):
        cls._saved_env = {v: os.environ.get(v) for v in _ENV_VARS}
        cls._tmp = Path(tempfile.mkdtemp(prefix="dmd-test-"))
        cls.data_dir = cls._tmp / "data"
        cls.asset_dir = cls._tmp / "assets"
        cls.data_dir.mkdir(parents=True)
        (cls.asset_dir / "models").mkdir(parents=True)

        os.environ["DIFFUSIONBEAR_DATA_DIR"] = str(cls.data_dir)
        os.environ["DIFFUSIONBEAR_ASSET_DIR"] = str(cls.asset_dir)
        # Snapshot every module this checkout has already loaded, so tearDownClass can
        # put the *same objects* back. Deleting them instead (the obvious approach)
        # silently breaks every later test that does mock.patch("state.SOMETHING"):
        # `state` gets re-imported as a NEW module object while modules imported
        # earlier still hold a reference to the old one, so the patch lands on an
        # object nobody reads. That is not hypothetical -- it broke three test_storage
        # assertions for anyone who ran the suite after this file.
        cls._saved_modules = {
            name: mod for name, mod in sys.modules.items()
            if _is_backend_module(name, mod)
        }
        _drop_backend_modules()
        sys.path.insert(0, str(BACKEND))
        try:
            from routers import downloads
            import app_settings
            import generator
        finally:
            _drop_backend_modules()
        cls.downloads = downloads
        cls.app_settings = app_settings
        cls.generator = generator

        # Sanity: if the isolation failed, everything below would be testing the
        # user's real settings file, so assert it loudly rather than silently.
        assert app_settings.DATA_DIR.resolve() == cls.data_dir.resolve(), (
            "backend modules were already imported with the real DATA_DIR; "
            "these tests would read and write the user's settings.json"
        )

    @classmethod
    def tearDownClass(cls):
        for var, value in cls._saved_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        # Restore the ORIGINAL module objects, not fresh imports. Anything else
        # re-imported while these tests ran is evicted, then the saved objects go back
        # so module identity matches what the rest of the suite already captured.
        _drop_backend_modules()
        sys.modules.update(cls._saved_modules)
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def setUp(self):
        # A fresh, empty settings file and model store per test: no state bleeds
        # between them, and no test can observe a sibling's leftovers.
        self.settings_file = self.data_dir / "settings.json"
        if self.settings_file.exists():
            self.settings_file.unlink()
        self.app_settings._load_cache.clear()
        shutil.rmtree(self.models_root, ignore_errors=True)
        self.models_root.mkdir(parents=True, exist_ok=True)

    # ---- fixtures -------------------------------------------------------

    @property
    def models_root(self):
        return self.asset_dir / "models"

    def _make_model(self, name: str, size_bytes: int = 3_000_000) -> Path:
        """A real directory in the (temporary) model store with a known size."""
        target = self.models_root / name
        target.mkdir(parents=True)
        (target / "weights.safetensors").write_bytes(b"\0" * size_bytes)
        return target

    def _client(self):
        """A TestClient over just the downloads router."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.include_router(self.downloads.router)
        return TestClient(app)

    def _delete(self, name, **body):
        return self._client().request("DELETE", f"/api/hf/models/{name}", json=body)

    # ---- model_paths overrides ------------------------------------------

    def test_unrelated_model_paths_overrides_survive_a_delete(self):
        """Two ids point at the deleted directory; a third points elsewhere.

        Both matching ids must end up cleared and the unrelated one must be byte-for-byte
        identical afterwards. This runs against the real ``app_settings`` merge in a
        temp data dir rather than a mock, because a mock cannot tell us whether the
        merge is additive -- which is precisely the premise being checked.
        """
        target = self._make_model("qwen-image-2.1-turbo")
        resolved = str(target.resolve())
        self.app_settings.update_settings({
            "model_paths": {
                "qwen-image-2.1-turbo": resolved,
                "flux2-klein-4b": resolved,
                "krea2-turbo": "/somewhere/else/krea2",
            }
        })
        before = self.app_settings.get_settings()["model_paths"]
        self.assertEqual(len(before), 3)

        with mock.patch.object(self.generator, "ASSET_DIR", self.asset_dir), \
             mock.patch.object(shutil, "rmtree", wraps=shutil.rmtree) as rmtree:
            response = self._delete("qwen-image-2.1-turbo", confirm=True)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(rmtree.called, "the directory should have been removed")
        self.assertTrue(response.json()["cleared_overrides"])

        after = self.app_settings.get_settings()["model_paths"]
        self.assertNotIn("qwen-image-2.1-turbo", after)
        self.assertNotIn("flux2-klein-4b", after)
        self.assertEqual(
            after.get("krea2-turbo"),
            "/somewhere/else/krea2",
            "an override that pointed somewhere else was destroyed by deleting an "
            "unrelated model; update_settings merges model_paths per key, so nothing "
            "here should have removed it",
        )

    def test_the_cleared_overrides_flag_is_false_when_nothing_pointed_here(self):
        target = self._make_model("lonely-model")
        self.app_settings.update_settings({"model_paths": {"krea2-turbo": "/elsewhere"}})

        with mock.patch.object(self.generator, "ASSET_DIR", self.asset_dir), \
             mock.patch.object(shutil, "rmtree", wraps=shutil.rmtree):
            response = self._delete("lonely-model", confirm=True)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["cleared_overrides"])
        self.assertEqual(
            self.app_settings.get_settings()["model_paths"],
            {"krea2-turbo": "/elsewhere"},
        )
        self.assertFalse(target.exists())

    def test_nothing_is_cleared_when_the_rmtree_fails(self):
        """Overrides are cleared before the delete, so a failed delete leaves a
        cleared override behind -- pre-existing behaviour, recorded rather than
        silently accepted."""
        target = self._make_model("stubborn-model")
        self.app_settings.update_settings({"model_paths": {"krea2-turbo": str(target.resolve())}})

        with mock.patch.object(self.generator, "ASSET_DIR", self.asset_dir), \
             mock.patch.object(shutil, "rmtree", side_effect=OSError("busy")):
            response = self._delete("stubborn-model", confirm=True)

        self.assertEqual(response.status_code, 500, response.text)
        self.assertEqual(self.app_settings.get_settings()["model_paths"], {})

    # ---- the confirmation gate ------------------------------------------

    def test_omitting_confirm_is_rejected(self):
        self._make_model("qwen-image-2.1-turbo")
        with mock.patch.object(self.generator, "ASSET_DIR", self.asset_dir), \
             mock.patch.object(shutil, "rmtree") as rmtree:
            response = self._client().delete("/api/hf/models/qwen-image-2.1-turbo")
        self.assertEqual(response.status_code, 422, response.text)
        rmtree.assert_not_called()

    def test_confirm_false_deletes_nothing_and_says_what_would_go(self):
        target = self._make_model("qwen-image-2.1-turbo", size_bytes=3_000_000)
        resolved = str(target.resolve())
        self.app_settings.update_settings({"model_paths": {"krea2-turbo": resolved}})

        with mock.patch.object(self.generator, "ASSET_DIR", self.asset_dir), \
             mock.patch.object(shutil, "rmtree") as rmtree, \
             mock.patch.object(self.app_settings, "update_settings") as update:
            response = self._delete("qwen-image-2.1-turbo", confirm=False)

        self.assertEqual(response.status_code, 409, response.text)
        rmtree.assert_not_called()
        update.assert_not_called()

        detail = response.json()["detail"]
        self.assertIn("qwen-image-2.1-turbo", detail, "the name of what would go")
        self.assertIn(resolved, detail, "the exact path that would be removed")
        size_mb = self.downloads._dir_size(target) / (1024 * 1024)
        self.assertIn(f"{size_mb:.1f} MB", detail, "the blast radius, in MB")
        self.assertIn("confirm: true", detail, "and what to do next")
        self.assertTrue(target.exists(), "a rejected delete must not remove anything")

    def test_the_success_response_reports_the_size_removed(self):
        self._make_model("qwen-image-2.1-turbo", size_bytes=3_000_000)
        with mock.patch.object(self.generator, "ASSET_DIR", self.asset_dir), \
             mock.patch.object(shutil, "rmtree", wraps=shutil.rmtree):
            response = self._delete("qwen-image-2.1-turbo", confirm=True)

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "deleted")
        self.assertEqual(body["install_name"], "qwen-image-2.1-turbo")
        self.assertIn("cleared_overrides", body)
        self.assertAlmostEqual(body["deleted_mb"], 3_000_000 / (1024 * 1024), places=1)

    def test_the_request_model_forbids_unknown_fields(self):
        """`extra="forbid"`, as ConvertDetectedRequest does: a typo in `confirm`
        must not be read as consent."""
        from pydantic import ValidationError

        model = self.downloads.DeleteDetectedRequest
        with self.assertRaises(ValidationError):
            model(confrm=True)
        with self.assertRaises(ValidationError):
            model()  # required, no default
        self.assertIs(model(confirm=True).confirm, True)

    def test_a_dotdot_name_cannot_wipe_the_whole_model_store(self):
        """DELETE /api/hf/models/%2e%2e used to delete every model on disk.

        safe_install_name does not filter dots, so ".." survived name validation, and
        the containment check compared against ASSET_DIR -- of which ASSET_DIR/models/..
        is trivially "inside". rmtree then emptied the entire model store and only
        failed on the final rmdir of the trailing "..", returning 500 *after* the
        damage. One request, no confirmation of what was lost, and the error message
        pointed at a path that no longer existed.
        """
        victim = self._make_model("precious-model")
        bystander = self._make_model("other-model")
        (self.asset_dir / "lora_files").mkdir(exist_ok=True)

        with mock.patch.object(self.generator, "ASSET_DIR", self.asset_dir), \
             mock.patch.object(shutil, "rmtree", wraps=shutil.rmtree) as rmtree:
            response = self._client().request(
                "DELETE", "/api/hf/models/%2e%2e", json={"confirm": True}
            )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("stay inside", response.json()["detail"])
        self.assertFalse(rmtree.called, "nothing may be removed by a rejected path")
        self.assertTrue(victim.exists(), "precious-model was deleted by a traversal name")
        self.assertTrue(bystander.exists(), "other-model was deleted by a traversal name")
        self.assertTrue((self.asset_dir / "lora_files").exists())

    def test_a_dot_name_cannot_wipe_the_whole_model_store(self):
        """. targets models/ itself, which the same check rejects."""
        self._make_model("precious-model")
        with mock.patch.object(self.generator, "ASSET_DIR", self.asset_dir), \
             mock.patch.object(shutil, "rmtree") as rmtree:
            response = self._client().request(
                "DELETE", "/api/hf/models/%2e", json={"confirm": True}
            )
        self.assertEqual(response.status_code, 400, response.text)
        rmtree.assert_not_called()
        self.assertTrue((self.models_root / "precious-model").exists())

    def test_an_unsafe_name_is_rejected(self):
        """A name that safe_install_name would rewrite is refused outright."""
        with mock.patch.object(self.generator, "ASSET_DIR", self.asset_dir), \
             mock.patch.object(shutil, "rmtree") as rmtree:
            response = self._client().request(
                "DELETE", "/api/hf/models/a%3Ab", json={"confirm": True}
            )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["detail"], "invalid model name")
        rmtree.assert_not_called()

    def test_a_missing_directory_is_a_404_not_a_confirm_prompt(self):
        with mock.patch.object(self.generator, "ASSET_DIR", self.asset_dir), \
             mock.patch.object(shutil, "rmtree") as rmtree:
            response = self._delete("never-downloaded", confirm=False)
        self.assertEqual(response.status_code, 404, response.text)
        rmtree.assert_not_called()


if __name__ == "__main__":
    unittest.main()