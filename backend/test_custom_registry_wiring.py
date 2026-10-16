"""Task 2: registry fall-through — info, listing, LoRA compat.

Hermetic: temp-DATA_DIR registry via module-constant patching (Task 1
precedent), no network, no weights.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient

import custom_models
import generator
import lora_compat
import state
import main

ENTRY = {
    "id": "my-klein",
    "label": "My Klein",
    "family": "flux2",
    "variant": "4b",
    "source": "local",
    "local_path": "/models/my-klein",
}

_CIVITAI_IDENTITY_KEYS = (
    "civitai_model_id",
    "civitai_version_id",
    "civitai_model_name",
    "civitai_version_name",
    "sha256",
)


class CustomRegistryWiringTests(unittest.TestCase):
    def setUp(self):
        self.saved_file = custom_models.CUSTOM_MODELS_FILE
        self.saved_cache = custom_models._custom_models_cache
        self.temp = tempfile.TemporaryDirectory(dir=state.DATA_DIR)
        custom_models._custom_models_cache = {}
        custom_models.CUSTOM_MODELS_FILE = Path(self.temp.name) / "custom_models.json"
        custom_models.write_custom_models([dict(ENTRY)])

    def tearDown(self):
        custom_models.CUSTOM_MODELS_FILE = self.saved_file
        custom_models._custom_models_cache = self.saved_cache
        self.temp.cleanup()

    def test_builtin_ids_resolve_exactly_as_before(self):
        self.assertIs(
            generator.get_model_info("flux2-klein-4b"),
            generator.MODELS["flux2-klein-4b"],
        )

    def test_custom_id_returns_translated_dict(self):
        info = generator.get_model_info("my-klein")
        self.assertIsNotNone(info)
        self.assertEqual(info["id"], "my-klein")
        self.assertEqual(info["label"], "My Klein")
        self.assertEqual(info["engine"], "mflux")
        self.assertEqual(info["family"], "flux2")
        self.assertEqual(info["variant"], "4b")
        self.assertEqual(info["local_path"], "/models/my-klein")
        # A local-source custom model has no remote: it must not inherit the
        # canonical entry's repo or Civitai identity.
        self.assertIsNone(info["repo"])
        for key in _CIVITAI_IDENTITY_KEYS:
            self.assertNotIn(key, info)
        # Family behavior rides along from the canonical entry, not literals.
        canonical = generator.MODELS["flux2-klein-4b"]
        self.assertEqual(info["supports_loras"], canonical["supports_loras"])
        self.assertEqual(info["default_steps"], canonical["default_steps"])
        self.assertEqual(info["presets"], canonical["presets"])

    def test_unknown_id_still_returns_none(self):
        self.assertIsNone(generator.get_model_info("nope"))

    def test_mapping_for_resolves_custom_id_via_entry_family(self):
        resolved = lora_compat.mapping_for("my-klein")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved, lora_compat.mapping_for("flux2-klein-4b"))

    def test_models_listing_contains_custom_id_with_enrichment(self):
        items = TestClient(main.app).get("/api/models").json()
        row = next((i for i in items if i.get("id") == "my-klein"), None)
        self.assertIsNotNone(row)
        for key in ("installed", "download_repo", "disk_usage_bytes"):
            self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
