"""Task 4: local onboard endpoint — declare, verify, register.

Hermetic: fake dirs, synthetic indexes, TestClient routes — no network,
no weights; temp DATA_DIR via module-constant patching.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient

import custom_models
import state
import main


def _write_flux_index(model_dir: Path, doubles: int, singles: int,
                      shards: int = 2, create_shards: bool = True) -> None:
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for i in range(doubles):
        shard = f"model-{i % shards:05d}-of-{shards:05d}.safetensors"
        weight_map[f"transformer.transformer_blocks.{i}.attn.to_q.weight"] = shard
    for i in range(singles):
        shard = f"model-{(i + doubles) % shards:05d}-of-{shards:05d}.safetensors"
        weight_map[f"transformer.single_transformer_blocks.{i}.attn.to_q.weight"] = shard
    (transformer_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    if create_shards:
        for shard in set(weight_map.values()):
            (transformer_dir / shard).write_bytes(b"\x00")


def _make_sdxl_dir(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"unet": ["diffusers", "UNet2DConditionModel"]}), encoding="utf-8")
    unet = model_dir / "unet"
    unet.mkdir(parents=True, exist_ok=True)
    (unet / "diffusion_pytorch_model.safetensors").write_bytes(b"\x00")


def _make_generic_dir(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.safetensors").write_bytes(b"\x00")


class OnboardEndpointTests(unittest.TestCase):
    def setUp(self):
        self.saved_file = custom_models.CUSTOM_MODELS_FILE
        self.saved_cache = custom_models._custom_models_cache
        self.temp = tempfile.TemporaryDirectory(dir=state.DATA_DIR)
        custom_models._custom_models_cache = {}
        custom_models.CUSTOM_MODELS_FILE = Path(self.temp.name) / "custom_models.json"
        self.dirs = tempfile.TemporaryDirectory()
        self.client = TestClient(main.app)

    def tearDown(self):
        custom_models.CUSTOM_MODELS_FILE = self.saved_file
        custom_models._custom_models_cache = self.saved_cache
        self.temp.cleanup()
        self.dirs.cleanup()

    def _post(self, payload: dict):
        return self.client.post("/api/models/onboard", json=payload)

    def test_valid_sdxl_persists_readable_entry(self):
        d = Path(self.dirs.name) / "sdxl-valid"
        _make_sdxl_dir(d)
        resp = self._post({"id": "my-sdxl", "label": "My SDXL",
                           "family": "sdxl", "local_path": str(d)})
        self.assertEqual(resp.status_code, 200, resp.text)
        entry = resp.json()
        self.assertEqual(entry["id"], "my-sdxl")
        self.assertEqual(entry["source"], "local")
        self.assertEqual(entry["local_path"], str(d))
        stored = custom_models.read_custom_models()
        self.assertIn(entry, stored)

    def test_valid_flux2_4b_and_9b(self):
        for variant, doubles, singles in (("4b", 5, 20), ("9b", 8, 24)):
            d = Path(self.dirs.name) / f"flux2-{variant}"
            d.mkdir(parents=True, exist_ok=True)
            _write_flux_index(d, doubles, singles)
            resp = self._post({"id": f"my-flux-{variant}", "label": f"Flux {variant}",
                               "family": "flux2", "variant": variant,
                               "local_path": str(d)})
            self.assertEqual(resp.status_code, 200, resp.text)
            self.assertEqual(resp.json()["variant"], variant)

    def test_slug_and_label_defaults_from_folder(self):
        d = Path(self.dirs.name) / "My Model v2.0"
        _make_sdxl_dir(d)
        resp = self._post({"family": "sdxl", "local_path": str(d)})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["id"], "my-model-v2-0")
        self.assertEqual(resp.json()["label"], "my-model-v2-0")

    def test_empty_string_id_and_label_default(self):
        d = Path(self.dirs.name) / "Some Folder"
        _make_sdxl_dir(d)
        resp = self._post({"id": "", "label": "", "family": "sdxl",
                           "local_path": str(d)})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["id"], "some-folder")
        self.assertEqual(resp.json()["label"], "some-folder")

    def test_relative_path_refused(self):
        resp = self._post({"id": "rel", "label": "Rel",
                           "family": "sdxl", "local_path": "relative/path"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("model path must be absolute", resp.json()["detail"])

    def test_builtin_collision_refused_case_insensitive(self):
        d = Path(self.dirs.name) / "collide"
        _make_sdxl_dir(d)
        for bad in ("flux2-klein-4b", "FLUX2-KLEIN-4B"):
            resp = self._post({"id": bad, "label": "X",
                               "family": "sdxl", "local_path": str(d)})
            self.assertEqual(resp.status_code, 400, bad)
            self.assertIn(bad, resp.json()["detail"])

    def test_custom_collision_refused(self):
        d = Path(self.dirs.name) / "dup"
        _make_sdxl_dir(d)
        first = self._post({"id": "dup-id", "label": "A",
                            "family": "sdxl", "local_path": str(d)})
        self.assertEqual(first.status_code, 200, first.text)
        second = self._post({"id": "dup-id", "label": "B",
                             "family": "sdxl", "local_path": str(d)})
        self.assertEqual(second.status_code, 400)
        self.assertIn("dup-id", second.json()["detail"])

    def test_unknown_family_refused(self):
        d = Path(self.dirs.name) / "fam"
        _make_generic_dir(d)
        resp = self._post({"id": "x", "label": "X",
                           "family": "pony", "local_path": str(d)})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("family", resp.json()["detail"])

    def test_flux2_dir_declared_sdxl_is_layout_mismatch(self):
        d = Path(self.dirs.name) / "flux-as-sdxl"
        d.mkdir(parents=True, exist_ok=True)
        _write_flux_index(d, 5, 20)
        resp = self._post({"id": "m1", "label": "M",
                           "family": "sdxl", "local_path": str(d)})
        self.assertEqual(resp.status_code, 400)
        self.assertTrue(resp.json()["detail"])

    def test_sdxl_dir_declared_flux2_refused(self):
        d = Path(self.dirs.name) / "sdxl-as-flux"
        _make_sdxl_dir(d)
        resp = self._post({"id": "m2", "label": "M", "family": "flux2",
                           "variant": "4b", "local_path": str(d)})
        self.assertEqual(resp.status_code, 400)
        self.assertTrue(resp.json()["detail"])

    def test_size_mismatch_names_both(self):
        d = Path(self.dirs.name) / "sizem"
        d.mkdir(parents=True, exist_ok=True)
        _write_flux_index(d, 5, 20)
        resp = self._post({"id": "sz", "label": "S", "family": "flux2",
                           "variant": "9b", "local_path": str(d)})
        self.assertEqual(resp.status_code, 400)
        detail = resp.json()["detail"]
        self.assertIn("size mismatch", detail)
        self.assertIn("5", detail)
        self.assertIn("8", detail)

    def test_missing_shards_read_as_incomplete(self):
        d = Path(self.dirs.name) / "trunc"
        d.mkdir(parents=True, exist_ok=True)
        _write_flux_index(d, 5, 20)
        shards = sorted((d / "transformer").glob("*.safetensors"))
        self.assertTrue(shards)
        shards[0].unlink()
        resp = self._post({"id": "tr", "label": "T", "family": "flux2",
                           "variant": "4b", "local_path": str(d)})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("incomplete", resp.json()["detail"])

    def test_stray_partial_fails_completeness(self):
        d = Path(self.dirs.name) / "partial"
        d.mkdir(parents=True, exist_ok=True)
        _write_flux_index(d, 5, 20)
        (d / "x.incomplete").write_bytes(b"\x00")
        resp = self._post({"id": "pc", "label": "P", "family": "flux2",
                           "variant": "4b", "local_path": str(d)})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("model directory is incomplete", resp.json()["detail"])

    def test_missing_variant_refused(self):
        d = Path(self.dirs.name) / "novar"
        d.mkdir(parents=True, exist_ok=True)
        _write_flux_index(d, 5, 20)
        resp = self._post({"id": "nv", "label": "N",
                           "family": "flux2", "local_path": str(d)})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("variant", resp.json()["detail"])

    def test_valid_krea_and_qwen_via_canonical_layout(self):
        d = Path(self.dirs.name) / "krea-valid"
        _make_generic_dir(d)
        resp = self._post({"id": "my-krea", "label": "K",
                           "family": "krea", "local_path": str(d)})
        self.assertEqual(resp.status_code, 200, resp.text)
        q = Path(self.dirs.name) / "qwen-valid"
        _make_generic_dir(q)
        resp = self._post({"id": "my-qwen", "label": "Q", "family": "qwen",
                           "variant": "base", "local_path": str(q)})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["variant"], "base")


class VerifyAndRegisterTests(unittest.TestCase):
    def setUp(self):
        self.saved_file = custom_models.CUSTOM_MODELS_FILE
        self.saved_cache = custom_models._custom_models_cache
        self.temp = tempfile.TemporaryDirectory(dir=state.DATA_DIR)
        custom_models._custom_models_cache = {}
        custom_models.CUSTOM_MODELS_FILE = Path(self.temp.name) / "custom_models.json"
        self.dirs = tempfile.TemporaryDirectory()

    def tearDown(self):
        custom_models.CUSTOM_MODELS_FILE = self.saved_file
        custom_models._custom_models_cache = self.saved_cache
        self.temp.cleanup()
        self.dirs.cleanup()

    def test_nonlocal_sources_register_via_download_declarations(self):
        # Task 5 retired the "local folders only" refusal: hf/civitai
        # declarations (local_path = the worker's install dir) register
        # through this same helper. Missing source fields refuse with the
        # validate_entry message, verbatim.
        d = Path(self.dirs.name) / "dl-ok"
        _make_sdxl_dir(d)
        ok, reason, entry = custom_models.verify_and_register(
            {"id": "dl-hf", "label": "DL", "family": "sdxl",
             "source": "hf", "hf_repo": "org/model",
             "local_path": str(d)})
        self.assertTrue(ok, reason)
        self.assertEqual(entry["source"], "hf")
        self.assertEqual(entry["hf_repo"], "org/model")
        ok, reason, entry = custom_models.verify_and_register(
            {"id": "dl-no-repo", "label": "DL", "family": "sdxl",
             "source": "hf", "local_path": str(d)})
        self.assertFalse(ok)
        self.assertIsNone(entry)
        self.assertIn("hf_repo", reason)
        ok, reason, entry = custom_models.verify_and_register(
            {"id": "dl-no-ids", "label": "DL", "family": "sdxl",
             "source": "civitai", "local_path": str(d)})
        self.assertFalse(ok)
        self.assertIsNone(entry)
        self.assertIn("civitai_ids", reason)
        ok, reason, entry = custom_models.verify_and_register(
            {"id": "dl-bad-source", "label": "DL", "family": "sdxl",
             "source": "url", "local_path": str(d)})
        self.assertFalse(ok)
        self.assertIsNone(entry)
        self.assertIn("source", reason)

    def test_helper_returns_entry_on_success(self):
        d = Path(self.dirs.name) / "helper-ok"
        _make_sdxl_dir(d)
        ok, reason, entry = custom_models.verify_and_register(
            {"family": "sdxl", "local_path": str(d)})
        self.assertTrue(ok, reason)
        self.assertIsNone(reason)
        self.assertEqual(entry["id"], "helper-ok")
        self.assertEqual(entry["source"], "local")
        self.assertIn(entry, custom_models.read_custom_models())


if __name__ == "__main__":
    unittest.main()
