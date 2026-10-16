"""Final review fixes for custom-model onboarding (findings 1-5).

Hermetic: synthetic weight dirs, temp-file registry via module-constant
patching, no network, no weights. One class per review finding.
"""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient

import civitai_service
import custom_models
import generator
import state
import main
from routers import downloads as dl
from routers.models import (
    _fully_cached_cache,
    _model_is_fully_cached_uncached,
)


def _make_sdxl_dir(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"unet": ["diffusers", "UNet2DConditionModel"]}),
        encoding="utf-8",
    )
    unet = model_dir / "unet"
    unet.mkdir(parents=True, exist_ok=True)
    (unet / "diffusion_pytorch_model.safetensors").write_bytes(b"\x00")


def _make_generic_dir(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.safetensors").write_bytes(b"\x00")


class _RegistryTempDir(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.saved_file = custom_models.CUSTOM_MODELS_FILE
        self.saved_cache = custom_models._custom_models_cache
        self.temp = tempfile.TemporaryDirectory(dir=state.DATA_DIR)
        custom_models._custom_models_cache = {}
        custom_models.CUSTOM_MODELS_FILE = Path(self.temp.name) / "custom_models.json"
        self.dirs = tempfile.TemporaryDirectory()
        _fully_cached_cache.clear()

    def tearDown(self):
        custom_models.CUSTOM_MODELS_FILE = self.saved_file
        custom_models._custom_models_cache = self.saved_cache
        _fully_cached_cache.clear()
        self.temp.cleanup()
        self.dirs.cleanup()
        super().tearDown()

    def _weights(self, name: str, kind: str = "sdxl") -> Path:
        d = Path(self.dirs.name) / name
        if kind == "sdxl":
            _make_sdxl_dir(d)
        else:
            _make_generic_dir(d)
        return d


class Finding1LocalPathHonoredTests(_RegistryTempDir):
    def test_sdxl_translation_drops_builtin_model_dir(self):
        d = self._weights("f1-sdxl-drop")
        custom_models.write_custom_models([{
            "id": "f1-sdxl-drop", "label": "F1 SDXL", "family": "sdxl",
            "source": "local", "local_path": str(d),
        }])
        minfo = generator.get_model_info("f1-sdxl-drop")
        self.assertIsNotNone(minfo)
        self.assertNotIn("model_dir", minfo)
        self.assertEqual(minfo["local_path"], str(d))

    def test_sdxl_local_dir_counts_as_cached_and_sized(self):
        d = self._weights("f1-sdxl-cached")
        custom_models.write_custom_models([{
            "id": "f1-sdxl-cached", "label": "F1 SDXL", "family": "sdxl",
            "source": "local", "local_path": str(d),
        }])
        minfo = generator.get_model_info("f1-sdxl-cached")
        self.assertTrue(_model_is_fully_cached_uncached("f1-sdxl-cached", minfo))
        usage = generator.model_disk_usage("f1-sdxl-cached", minfo)
        self.assertGreater(usage, 0)
        # The entry dir holds bytes, not gigabytes: anything near a built-in
        # tree means disk usage followed the inherited model_dir instead.
        self.assertLess(usage, 1 << 20)

    def test_flux_local_dir_counts_as_cached_and_sized(self):
        d = self._weights("f1-flux-cached", kind="generic")
        custom_models.write_custom_models([{
            "id": "f1-flux-cached", "label": "F1 Flux", "family": "flux2",
            "variant": "4b", "source": "local", "local_path": str(d),
        }])
        minfo = generator.get_model_info("f1-flux-cached")
        self.assertTrue(_model_is_fully_cached_uncached("f1-flux-cached", minfo))
        self.assertGreater(generator.model_disk_usage("f1-flux-cached", minfo), 0)

    def test_models_listing_shows_custom_installed_with_local_path(self):
        d = self._weights("f1-sdxl-listed")
        custom_models.write_custom_models([{
            "id": "f1-sdxl-listed", "label": "F1 Listed", "family": "sdxl",
            "source": "local", "local_path": str(d),
        }])
        items = TestClient(main.app).get("/api/models").json()
        row = next((i for i in items if i.get("id") == "f1-sdxl-listed"), None)
        self.assertIsNotNone(row)
        self.assertTrue(row["installed"])
        self.assertEqual(row.get("local_path"), str(d))
        self.assertLess(row["disk_usage_bytes"], 1 << 20)


class Finding2UninstallRefusalTests(_RegistryTempDir):
    def test_custom_sdxl_uninstall_refused_without_touching_disk(self):
        d = self._weights("f2-sdxl")
        custom_models.write_custom_models([{
            "id": "f2-sdxl", "label": "F2 SDXL", "family": "sdxl",
            "source": "local", "local_path": str(d),
        }])
        # The 13 GB Juggernaut tree is live in this checkout: record, never run.
        with mock.patch("shutil.rmtree") as rm:
            ok, reason = generator.uninstall_model("f2-sdxl")
        self.assertFalse(ok)
        self.assertIn("custom", (reason or "").lower())
        rm.assert_not_called()
        self.assertTrue(d.exists())

    def test_remove_model_route_404s_custom_id(self):
        d = self._weights("f2-sdxl-route")
        custom_models.write_custom_models([{
            "id": "f2-sdxl-route", "label": "F2 Route", "family": "sdxl",
            "source": "local", "local_path": str(d),
        }])
        r = TestClient(main.app).delete("/api/models/f2-sdxl-route")
        self.assertEqual(r.status_code, 404)


class Finding3PostTimeEnumGateTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.client = TestClient(main.app)
        self.asset_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.asset_tmp.cleanup)
        asset_root = Path(self.asset_tmp.name)
        (asset_root / "models").mkdir(parents=True, exist_ok=True)
        self._asset_patch = mock.patch.object(generator, "ASSET_DIR", asset_root)
        self._asset_patch.start()
        self.addCleanup(self._asset_patch.stop)
        self._stream_patch = mock.patch.object(
            civitai_service, "open_public_https_stream",
            side_effect=RuntimeError("no network in tests"),
        )
        self._stream_patch.start()
        self.addCleanup(self._stream_patch.stop)
        self._before_tasks = set(dl.MODEL_DOWNLOAD_TASKS)
        self.addCleanup(self._drop_new_tasks)

    def _drop_new_tasks(self):
        with dl._MODEL_DOWNLOAD_LOCK:
            for tid in set(dl.MODEL_DOWNLOAD_TASKS) - self._before_tasks:
                dl.MODEL_DOWNLOAD_TASKS.pop(tid, None)

    def _preinstalled(self, name: str) -> None:
        (Path(self.asset_tmp.name) / "models" / name).mkdir(parents=True, exist_ok=True)

    def _post_hf(self, **fields):
        payload = {
            "model_id": "f3-new-model",
            "repo_id": "org/probe",
            "install_name": "probe-hf-f3",
            "family": "flux2",
            "variant": "4b",
        }
        payload.update(fields)
        self._preinstalled(payload["install_name"])
        return self.client.post("/api/models/download", json=payload)

    def _post_civitai(self, **fields):
        payload = {"model_id": 7, "model_version_id": 12345, "family": "flux2",
                   "variant": "4b"}
        payload.update(fields)
        return self.client.post("/api/civitai/models/download", json=payload)

    def test_hf_bad_family_400s_with_onboard_string(self):
        r = self._post_hf(family="pony")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(
            r.json()["detail"],
            "invalid family: must be one of sdxl|flux2|krea|zit|qwen",
        )

    def test_hf_bad_flux_variant_400s_with_onboard_string(self):
        r = self._post_hf(family="flux2", variant="xx")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(
            r.json()["detail"], "invalid variant: flux2 requires one of 4b|9b")

    def test_hf_qwen_missing_variant_400s_with_onboard_string(self):
        payload = {"model_id": "f3-new-model", "repo_id": "org/probe",
                   "install_name": "probe-hf-f3q", "family": "qwen"}
        self._preinstalled(payload["install_name"])
        r = self.client.post("/api/models/download", json=payload)
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(
            r.json()["detail"], "invalid variant: qwen requires one of base|turbo")

    def test_hf_variant_on_sdxl_400s_with_onboard_string(self):
        r = self._post_hf(family="sdxl", variant="4b")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(
            r.json()["detail"],
            "invalid variant: must be absent unless family is flux2 or qwen",
        )

    def test_civitai_bad_family_400s_with_onboard_string(self):
        r = self._post_civitai(family="pony")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(
            r.json()["detail"],
            "invalid family: must be one of sdxl|flux2|krea|zit|qwen",
        )

    def test_civitai_bad_variant_400s_with_onboard_string(self):
        r = self._post_civitai(family="flux2", variant="xx")
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(
            r.json()["detail"], "invalid variant: flux2 requires one of 4b|9b")


class Finding4NulPathReasonTests(unittest.TestCase):
    def test_nul_in_local_path_returns_invalid_local_path_reason(self):
        ok, reason, entry = custom_models.verify_and_register(
            {"family": "sdxl", "local_path": "/tmp/ok\x00evil"})
        self.assertFalse(ok)
        self.assertIsNone(entry)
        self.assertIn("invalid local_path", reason)
        self.assertIn("NUL", reason)

    def test_empty_local_path_returns_invalid_local_path_reason(self):
        ok, reason, entry = custom_models.verify_and_register(
            {"family": "sdxl", "local_path": ""})
        self.assertFalse(ok)
        self.assertIsNone(entry)
        self.assertIn("invalid local_path", reason)


class Finding5SidecarModelFallbackTests(unittest.TestCase):
    def test_repo_wins_when_present(self):
        self.assertEqual(
            generator._sidecar_model_value({"repo": "org/m", "id": "x"}, "x"),
            "org/m",
        )

    def test_id_used_when_repo_absent(self):
        self.assertEqual(
            generator._sidecar_model_value({"repo": None, "id": "my-sdxl"}, "my-sdxl"),
            "my-sdxl",
        )

    def test_id_used_when_repo_key_missing(self):
        self.assertEqual(
            generator._sidecar_model_value({"id": "my-sdxl"}, "my-sdxl"),
            "my-sdxl",
        )

    def test_model_arg_used_when_no_id(self):
        self.assertEqual(
            generator._sidecar_model_value({}, "fallback-id"), "fallback-id")


if __name__ == "__main__":
    unittest.main()
