"""Custom model registry: round-trip, corrupt-file tolerance, entry validation.

Hermetic: temp file + module FILE-constant patch, no network, no weights.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import custom_models
import state


def _valid_entry(**overrides):
    entry = {
        "id": "my-model",
        "label": "My Model",
        "family": "sdxl",
        "source": "local",
        "local_path": "/models/my-model",
    }
    entry.update(overrides)
    return entry


class CustomModelsRegistryTests(unittest.TestCase):
    def setUp(self):
        self.saved_file = custom_models.CUSTOM_MODELS_FILE
        self.saved_cache = custom_models._custom_models_cache
        # Temp dir lives inside DATA_DIR: the containment check is deliberate,
        # and a /tmp fixture would not exercise the real code path.
        self.temp = tempfile.TemporaryDirectory(dir=state.DATA_DIR)
        custom_models._custom_models_cache = {}
        custom_models.CUSTOM_MODELS_FILE = Path(self.temp.name) / "custom_models.json"

    def tearDown(self):
        custom_models.CUSTOM_MODELS_FILE = self.saved_file
        custom_models._custom_models_cache = self.saved_cache
        self.temp.cleanup()

    def test_round_trip_preserves_entries(self):
        entries = [
            _valid_entry(),
            _valid_entry(id="flux-custom", label="Flux Custom", family="flux2",
                         variant="4b", source="hf", hf_repo="org/model",
                         local_path=None),
        ]
        # local_path=None would be invalid; drop it for the hf entry
        del entries[1]["local_path"]
        custom_models.write_custom_models(entries)
        self.assertEqual(custom_models.read_custom_models(), entries)

    def test_corrupt_json_reads_as_empty(self):
        custom_models.CUSTOM_MODELS_FILE.write_text("{not json", encoding="utf-8")
        custom_models._custom_models_cache.clear()
        self.assertEqual(custom_models.read_custom_models(), [])

    def test_missing_file_reads_as_empty(self):
        self.assertEqual(custom_models.read_custom_models(), [])

    def test_non_list_json_reads_as_empty(self):
        custom_models.CUSTOM_MODELS_FILE.write_text(json.dumps({"id": "x"}), encoding="utf-8")
        custom_models._custom_models_cache.clear()
        self.assertEqual(custom_models.read_custom_models(), [])


class ValidateEntryTests(unittest.TestCase):
    def assert_rejected(self, entry, field):
        ok, reason = custom_models.validate_entry(entry)
        self.assertFalse(ok, f"expected rejection for {entry}")
        self.assertIsNotNone(reason)
        self.assertIn(field, reason)

    def assert_accepted(self, entry):
        ok, reason = custom_models.validate_entry(entry)
        self.assertTrue(ok, f"expected accept, got reason: {reason}")
        self.assertIsNone(reason)

    def test_accepts_valid_entries_per_family(self):
        self.assert_accepted(_valid_entry())
        self.assert_accepted(_valid_entry(id="f2", family="flux2", variant="4b",
                                         source="hf", hf_repo="org/m",
                                         local_path=None))
        self.assert_accepted(_valid_entry(id="f9", family="flux2", variant="9b",
                                         source="hf", hf_repo="org/m",
                                         local_path=None))
        self.assert_accepted(_valid_entry(id="q", family="qwen", variant="base",
                                         source="civitai", civitai_ids="123",
                                         local_path=None))
        self.assert_accepted(_valid_entry(id="qt", family="qwen", variant="turbo",
                                         source="civitai", civitai_ids=["1", "2"],
                                         local_path=None))
        self.assert_accepted(_valid_entry(id="k", family="krea"))
        self.assert_accepted(_valid_entry(id="z", family="zit"))

    def test_bad_ids(self):
        import generator
        builtin = next(iter(generator.MODELS.keys()))
        for bad in ["", "has space", "bad/id", "x" * 121]:
            self.assert_rejected(_valid_entry(id=bad), "id")
        self.assert_rejected(_valid_entry(id=builtin.upper()), "id")
        self.assert_rejected({"label": "x"}, "id")

    def test_bad_label(self):
        self.assert_rejected(_valid_entry(label=""), "label")
        self.assert_rejected(_valid_entry(label="x" * 201), "label")
        self.assert_rejected(_valid_entry(label=None), "label")

    def test_bad_family(self):
        self.assert_rejected(_valid_entry(family="pony"), "family")
        self.assert_rejected(_valid_entry(family="FLUX2"), "family")

    def test_variant_rules(self):
        self.assert_rejected(_valid_entry(family="flux2", source="hf",
                                         hf_repo="o/m", local_path=None), "variant")
        self.assert_rejected(_valid_entry(family="flux2", variant="base", source="hf",
                                         hf_repo="o/m", local_path=None), "variant")
        self.assert_rejected(_valid_entry(family="qwen", local_path="/m"), "variant")
        self.assert_rejected(_valid_entry(family="qwen", variant="4b",
                                         local_path="/m"), "variant")
        for family in ("sdxl", "krea", "zit"):
            self.assert_rejected(_valid_entry(family=family, variant="4b"), "variant")

    def test_bad_source(self):
        self.assert_rejected(_valid_entry(source="url"), "source")

    def test_local_path_rules(self):
        self.assert_rejected(_valid_entry(local_path=None), "local_path")
        self.assert_rejected(_valid_entry(local_path="relative/path"), "local_path")
        self.assert_rejected(_valid_entry(local_path="/m\x00x"), "local_path")
        self.assert_rejected(_valid_entry(local_path="/" + "x" * 4096), "local_path")

    def test_hf_repo_rules(self):
        base = _valid_entry(source="hf", hf_repo="org/model", local_path=None)
        del base["local_path"]
        self.assert_accepted(base)
        bad = dict(base)
        del bad["hf_repo"]
        self.assert_rejected(bad, "hf_repo")
        self.assert_rejected(dict(base, hf_repo=""), "hf_repo")
        self.assert_rejected(dict(base, hf_repo="x" * 201), "hf_repo")

    def test_civitai_ids_rules(self):
        base = _valid_entry(source="civitai", civitai_ids="123", local_path=None)
        del base["local_path"]
        self.assert_accepted(base)
        bad = dict(base)
        del bad["civitai_ids"]
        self.assert_rejected(bad, "civitai_ids")
        self.assert_rejected(dict(base, civitai_ids=""), "civitai_ids")
        self.assert_rejected(dict(base, civitai_ids=[]), "civitai_ids")
        self.assert_rejected(dict(base, civitai_ids=[""]), "civitai_ids")
        self.assert_rejected(dict(base, civitai_ids=["1", 2]), "civitai_ids")


if __name__ == "__main__":
    unittest.main()
