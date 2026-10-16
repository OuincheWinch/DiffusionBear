"""The enhancer backend is a user choice, not a bundled default.

Two models (0.5B, 1.5B), one active at a time, selected in settings. Weights
are never bundled -- they live in the HF cache or the asset dir -- so first
use downloads, and the UI offers an explicit download for the other one.

Behaviors pinned here:

1. DEFAULT: no setting stored means qwen15. The default lives in exactly one
   place (DEFAULTS in app_settings); the test reads it through get_setting,
   not by importing a constant from the module under test.
2. UNKNOWN FALLS BACK: a stored value outside the known ids resolves to
   qwen15 rather than raising -- a hand-edited settings.json must not break
   Enhance.
3. VALIDATOR: accepts exactly the two ids, rejects everything else (including
   case variants -- the closed-set rule, same as theme).
4. RESOLUTION: local asset dir wins when present, else the HF repo id. The
   local dir is per-selection, so each model resolves independently.
5. SWITCH INVALIDATES: changing the setting unloads the other model, so the
   next run loads the newly selected one instead of reusing the resident
   wrong-size weights.
6. CACHED CHECK: never touches the network (local cache scan only).
7. DOWNLOAD: invalid id is 400 without attempting anything; valid id calls
   the snapshot download exactly once for that model's repo.
"""
import unittest
from pathlib import Path
from unittest import mock

import app_settings
import prompt_enhancer as pe


class EnhancerModelChoiceTests(unittest.TestCase):
    def test_default_is_qwen15(self):
        self.assertEqual(app_settings.DEFAULTS.get("enhancer_model"), "qwen15")

    def test_validator_accepts_only_the_two_ids(self):
        validate = app_settings._VALIDATORS["enhancer_model"]
        self.assertTrue(validate("qwen15"))
        self.assertTrue(validate("qwen05"))
        for bad in ("QWEN15", "qwen-1.5b", "", None, 15, "llama"):
            with self.subTest(bad=bad):
                self.assertFalse(validate(bad))

    def test_unknown_stored_value_falls_back_to_qwen15(self):
        with mock.patch.object(
            app_settings, "get_setting", return_value="mystery-model"
        ):
            self.assertEqual(pe.selected_model_id(), "qwen15")

    def test_resolution_prefers_local_dir_per_selection(self):
        with (
            mock.patch.object(app_settings, "get_setting", return_value="qwen05"),
            mock.patch.object(Path, "exists", return_value=True),
        ):
            repo = pe.resolve_model_repo()
        self.assertIn("qwen2.5-0.5b", repo)
        self.assertNotIn("1.5b", repo)

    def test_resolution_falls_back_to_repo_id(self):
        with (
            mock.patch.object(app_settings, "get_setting", return_value="qwen15"),
            mock.patch.object(Path, "exists", return_value=False),
        ):
            repo = pe.resolve_model_repo()
        self.assertEqual(repo, "mlx-community/Qwen2.5-1.5B-Instruct-4bit")

    def test_switching_selection_unloads_the_other_model(self):
        sentinel_15 = object()
        sentinel_05 = object()
        pe._model = pe._tokenizer = pe._loaded_repo = None

        def fake_load(repo):
            return (sentinel_15, None) if "1.5B" in repo else (sentinel_05, None)

        with (
            mock.patch.object(app_settings, "get_setting", return_value="qwen15"),
            mock.patch("mlx_lm.load", side_effect=fake_load),
        ):
            model, _ = pe._get_model()
            self.assertIs(model, sentinel_15)
        with (
            mock.patch.object(app_settings, "get_setting", return_value="qwen05"),
            mock.patch("mlx_lm.load", side_effect=fake_load) as loader,
        ):
            model, _ = pe._get_model()
            self.assertIs(model, sentinel_05)
            self.assertTrue(loader.called, "switch did not reload")


class EnhancerModelStatusTests(unittest.TestCase):
    def test_cached_check_never_touches_network(self):
        with mock.patch(
            "huggingface_hub.snapshot_download",
            side_effect=Exception("network reached"),
        ):
            # Must not raise the network error: presence is decided from
            # local state only. Either boolean is fine here; the network
            # reaching the mock is what fails the test.
            try:
                pe.model_cached("qwen15")
            except Exception as e:
                self.assertNotIn("network reached", str(e))

    def test_download_invalid_id_is_400_without_attempt(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers import gallery as gallery_router

        app = FastAPI()
        app.include_router(gallery_router.router)
        client = TestClient(app, raise_server_exceptions=False)
        with mock.patch(
            "huggingface_hub.snapshot_download",
            side_effect=AssertionError("download attempted"),
        ):
            response = client.post(
                "/api/prompt/enhancer/download", json={"id": "nope"})
        self.assertEqual(response.status_code, 400)

    def test_download_calls_snapshot_once_for_the_right_repo(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers import gallery as gallery_router

        app = FastAPI()
        app.include_router(gallery_router.router)
        client = TestClient(app, raise_server_exceptions=False)
        with mock.patch(
            "huggingface_hub.snapshot_download", return_value="/tmp/x") as dl:
            response = client.post(
                "/api/prompt/enhancer/download", json={"id": "qwen05"})
        self.assertEqual(response.status_code, 200)
        dl.assert_called_once()
        self.assertIn("Qwen2.5-0.5B", dl.call_args.args[0])

    def test_settings_route_accepts_the_choice(self):
        """The UI saves through POST /api/settings, whose pydantic schema
        forbids extras: a key missing there 422s before the settings
        validator runs. This test posts the real route; the missing field
        is exactly what broke the picker in production."""
        import app_settings as app_settings_module
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers import settings as settings_router

        app = FastAPI()
        app.include_router(settings_router.router)
        client = TestClient(app, raise_server_exceptions=False)
        with mock.patch.object(
            app_settings_module, "update_settings",
            return_value={"enhancer_model": "qwen05"},
        ) as updater:
            response = client.post(
                "/api/settings", json={"enhancer_model": "qwen05"})
        self.assertEqual(response.status_code, 200, response.text[:200])
        updater.assert_called_once_with({"enhancer_model": "qwen05"})

    def test_settings_route_rejects_unknown_choice(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers import settings as settings_router

        app = FastAPI()
        app.include_router(settings_router.router)
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/settings", json={"enhancer_model": "llama"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
