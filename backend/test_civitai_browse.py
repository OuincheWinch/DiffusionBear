"""Tests for Civitai discovery (backend/civitai_browse.py).

The reason this module exists is that Civitai is a *better* source than Hugging Face for
two of the fields the browser filters on, so these lock in the preference for structured
data over the name-parsing heuristics hf_browse has to fall back on.
"""
import unittest

import civitai_browse


class BaseModelTests(unittest.TestCase):
    def test_structured_base_model_beats_the_name(self):
        for base_model, expected in [
            ("SDXL 1.0", "sdxl"),
            ("Pony", "pony"),
            ("Illustrious XL", "illustrious"),
            ("SD 1.5", "sd15"),
            ("Flux.1", "flux1"),
            ("Z-Image", "z-image"),
        ]:
            key, _ = civitai_browse._architecture_from(base_model, "some-unrelated-name")
            self.assertEqual(key, expected, base_model)

    def test_falls_back_to_name_when_base_model_is_absent(self):
        key, label = civitai_browse._architecture_from(None, "FLUX.2-Klein-4B-4bit")
        self.assertEqual(key, "flux2")
        self.assertTrue(label)

    def test_unknown_base_model_is_reported_not_guessed(self):
        key, _ = civitai_browse._architecture_from("Something Custom", "mystery-model")
        self.assertEqual(key, "other")


class QuantizationTests(unittest.TestCase):
    def test_precision_comes_from_file_metadata(self):
        self.assertEqual(civitai_browse._quantization_from_file({"fp": "fp8"}), "FP8")
        self.assertEqual(civitai_browse._quantization_from_file({"fp": "bf16"}), "BF16")

    def test_absent_precision_is_none_not_inferred(self):
        """A single-file fp16 checkpoint must not be labelled with a bit depth."""
        self.assertIsNone(civitai_browse._quantization_from_file({"format": "SafeTensor"}))
        self.assertIsNone(civitai_browse._quantization_from_file({}))


class FileSelectionTests(unittest.TestCase):
    def test_primary_safetensor_wins(self):
        version = {"files": [
            {"name": "a.safetensors", "sizeKB": 10, "metadata": {"format": "SafeTensor"}},
            {"name": "b.safetensors", "sizeKB": 99_000, "primary": True, "metadata": {"format": "SafeTensor"}},
        ]}
        self.assertEqual(civitai_browse._pick_file(version)["name"], "b.safetensors")

    def test_largest_wins_when_nothing_is_primary(self):
        version = {"files": [
            {"name": "small.safetensors", "sizeKB": 10, "metadata": {"format": "SafeTensor"}},
            {"name": "big.safetensors", "sizeKB": 5000, "metadata": {"format": "SafeTensor"}},
        ]}
        self.assertEqual(civitai_browse._pick_file(version)["name"], "big.safetensors")

    def test_non_safetensor_files_are_never_picked(self):
        version = {"files": [{"name": "model.ckpt", "sizeKB": 9000, "metadata": {"format": "PickleTensor"}}]}
        self.assertIsNone(civitai_browse._pick_file(version))

    def test_version_without_files_is_handled(self):
        self.assertIsNone(civitai_browse._pick_file({"files": []}))


class SortTests(unittest.TestCase):
    def test_unknown_sort_falls_back_instead_of_passing_through(self):
        self.assertEqual(civitai_browse._VALID_SORTS.get("; DROP TABLE", "Most Downloaded"), "Most Downloaded")

    def test_known_sorts_survive(self):
        for name in ("Most Downloaded", "Highest Rated", "Newest", "Most Liked"):
            self.assertEqual(civitai_browse._VALID_SORTS.get(name), name)


if __name__ == "__main__":
    unittest.main()
