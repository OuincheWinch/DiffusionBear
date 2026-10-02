"""Tests for Hugging Face model discovery (backend/hf_browse.py).

The classification rules exist because there is no structured metadata to lean on:
quantisation and architecture are only ever substrings of the repo id. These lock in the
cases that actually appear in `mlx-community`, including the ones that are easy to get
wrong (FLUX.2 vs FLUX.1, "4bit" vs "fp16", LoRAs named after their base model).
"""

import unittest
from unittest import mock

import hf_browse


class QuantizationTests(unittest.TestCase):
    def test_bit_depth_suffixes(self):
        for name, expected in [
            ("FLUX.2-Klein-4B-4bit", "4bit"),
            ("FLUX.2-Klein-4B-3bit", "3bit"),
            ("FLUX.2-Klein-4B-6bit", "6bit"),
            ("model-16bit", "16bit"),
        ]:
            self.assertEqual(hf_browse.guess_quantization(f"org/{name}"), expected, name)

    def test_short_and_literal_schemes(self):
        for name, expected in [
            ("m-q4", "4bit"),
            ("m-int8", "8bit"),
            ("Krea-mflux-bf16", "BF16"),
            ("x-fp16", "FP16"),
            ("y-mxfp4", "MXFP4"),
            ("z-nf4", "NF4"),
        ]:
            self.assertEqual(hf_browse.guess_quantization(f"org/{name}"), expected, name)

    def test_absent_quantisation_is_none_not_a_guess(self):
        self.assertIsNone(hf_browse.guess_quantization("org/FLUX.2-klein-9B"))

    def test_ignores_the_organisation_segment(self):
        self.assertEqual(
            hf_browse.guess_quantization("mlx-community/A-4bit"),
            hf_browse.guess_quantization("someone-else/A-4bit"),
        )


class ArchitectureTests(unittest.TestCase):
    def test_flux2_is_not_confused_with_flux1(self):
        self.assertEqual(hf_browse.guess_architecture("org/FLUX.2-Klein-4B-4bit")[0], "flux2")
        self.assertEqual(hf_browse.guess_architecture("org/flux1-schnell")[0], "flux1")

    def test_named_families(self):
        for name, key in [
            ("Krea-2-Turbo-mflux-bf16", "krea"),
            ("Ming-Image-0.1-Design", "ming"),
            ("Qwen-Image-2.1-8bit", "qwen-image"),
            ("z-image-turbo", "z-image"),
            ("Illustrious-XL-v2", "illustrious"),
            ("Pony-v6", "pony"),
        ]:
            self.assertEqual(hf_browse.guess_architecture(f"org/{name}")[0], key, name)

    def test_unknown_family_is_reported_as_other_not_dropped(self):
        self.assertEqual(hf_browse.guess_architecture("org/some-new-model")[0], "other")


class KindTests(unittest.TestCase):
    def test_upscalers_are_distinct_from_generators(self):
        self.assertEqual(hf_browse.guess_kind("org/Restormer-4bit"), "upscaler")
        self.assertEqual(hf_browse.guess_kind("org/Real-ESRGAN-4bit"), "upscaler")

    def test_lora_detected_despite_being_named_after_a_base_model(self):
        self.assertEqual(hf_browse.guess_kind("org/Illustrious-XL-LoRA"), "lora")
        self.assertEqual(hf_browse.guess_kind("org/Krea-2-Turbo-Distill-4step-LoRA"), "lora")

    def test_plain_generator(self):
        self.assertEqual(hf_browse.guess_kind("org/FLUX.2-Klein-4B-4bit"), "diffusion")

    def test_ming_is_flagged_for_native_alpha(self):
        _, label = hf_browse.guess_architecture("org/Ming-Image-0.1-Design")
        self.assertTrue(hf_browse.supports_alpha("org/Ming-Image-0.1-Design", label))
        _, other = hf_browse.guess_architecture("org/FLUX.2-Klein-4B-4bit")
        self.assertFalse(hf_browse.supports_alpha("org/FLUX.2-Klein-4B-4bit", other))


class InstallNameTests(unittest.TestCase):
    def test_path_traversal_is_neutralised(self):
        name = hf_browse.safe_install_name("../../etc/passwd")
        self.assertNotIn("/", name)
        self.assertNotIn("..", name)

    def test_organisation_is_dropped(self):
        self.assertEqual(hf_browse.safe_install_name("mlx-community/FLUX.2-Klein-4B-4bit"), "FLUX.2-Klein-4B-4bit")

    def test_never_empty(self):
        self.assertTrue(hf_browse.safe_install_name("org/###"))


class StemTests(unittest.TestCase):
    def test_quantisation_suffix_is_stripped(self):
        self.assertEqual(hf_browse._stem("mlx-community/FLUX.2-Klein-4B-4bit"), "flux.2-klein-4b")

    def test_org_and_agnostic_same_model_match(self):
        self.assertEqual(
            hf_browse._stem("mlx-community/FLUX.2-Klein-4B-4bit"),
            hf_browse._stem("black-forest-labs/FLUX.2-klein-4B"),
        )

    def test_bf16_suffix_stripped(self):
        self.assertEqual(hf_browse._stem("org/Krea-2-Turbo-mflux-bf16"), "krea-2-turbo-mflux")


class ListModelsCompatTests(unittest.TestCase):
    def test_kwargs_are_filtered_to_the_installed_hub_signature(self):
        """huggingface_hub 1.28.0 dropped `direction`; passing it raises TypeError."""
        import inspect

        def list_models(**kwargs):
            return iter([])

        api = mock.Mock()
        api.list_models = list_models
        api.list_models.__signature__ = inspect.Signature(
            [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
             inspect.Parameter("author", inspect.Parameter.POSITIONAL_OR_KEYWORD),
             inspect.Parameter("search", inspect.Parameter.POSITIONAL_OR_KEYWORD),
             inspect.Parameter("limit", inspect.Parameter.POSITIONAL_OR_KEYWORD),
             inspect.Parameter("sort", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        )
        # must not raise
        hf_browse._list_models(api, author="mlx-community", search="klein", limit=10, sort="downloads")

    def test_unknown_sort_falls_back(self):
        import inspect

        captured = {}

        def list_models(**kwargs):
            captured.update(kwargs)
            return iter([])

        api = mock.Mock()
        api.list_models = list_models
        api.list_models.__signature__ = inspect.Signature(
            [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
             inspect.Parameter("author", inspect.Parameter.POSITIONAL_OR_KEYWORD),
             inspect.Parameter("search", inspect.Parameter.POSITIONAL_OR_KEYWORD),
             inspect.Parameter("limit", inspect.Parameter.POSITIONAL_OR_KEYWORD),
             inspect.Parameter("sort", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        )
        hf_browse._list_models(api, author="a", search=None, limit=5, sort="nonsense")
        self.assertEqual(captured["sort"], "downloads")


if __name__ == "__main__":
    unittest.main()

class UsableAsTests(unittest.TestCase):
    """Which downloads this app can actually RUN.

    Every generator build is a `model_id == ...` dispatch onto a fixed mflux class, so a
    repo is runnable only if it maps onto one of those ids. Getting this wrong is what
    makes a downloaded model look selectable when it is not.
    """

    def test_alternative_quantisations_map_onto_the_engine(self):
        for repo, expected in [
            ("mlx-community/FLUX.2-Klein-4B-4bit", "flux2-klein-4b"),
            ("mlx-community/FLUX.2-Klein-4B-3bit", "flux2-klein-4b"),
            ("mlx-community/FLUX.2-klein-9B-4bit", "flux2-klein-9b"),
            ("mlx-community/Krea-2-Turbo-mflux-q4", "krea2-turbo"),
            ("mlx-community/Qwen-Image-2.1-8bit", "qwen-image-2.1"),
        ]:
            model_id, _ = hf_browse.usable_as(repo, "diffusion")
            self.assertEqual(model_id, expected, repo)

    def test_alternative_spellings_match(self):
        """Regression: alternatives were being AND-ed, which broke z-image entirely."""
        for repo in ["mlx-community/z-image-turbo-6bit", "mlx-community/zimageturbo-4bit"]:
            model_id, _ = hf_browse.usable_as(repo, "diffusion")
            self.assertEqual(model_id, "z-image-turbo", repo)

    def test_klein_size_selects_the_right_engine(self):
        """9B must not fall through to the 4B engine -- they are separate dispatches."""
        self.assertEqual(hf_browse.usable_as("mlx-community/FLUX.2-klein-9B-4bit", "diffusion")[0], "flux2-klein-9b")
        self.assertEqual(hf_browse.usable_as("mlx-community/FLUX.2-Klein-4B-4bit", "diffusion")[0], "flux2-klein-4b")

    def test_upscalers_are_not_runnable_and_say_why(self):
        model_id, reason = hf_browse.usable_as("mlx-community/Restormer-real-denoising-fp32", "upscaler")
        self.assertIsNone(model_id)
        self.assertIn("Lanczos", reason)

    def test_lora_is_not_a_model(self):
        model_id, _ = hf_browse.usable_as("org/Illustrious-XL-LoRA", "lora")
        self.assertIsNone(model_id)

    def test_unknown_architecture_is_not_guessed(self):
        model_id, reason = hf_browse.usable_as("cagliostrolab/Illustrious-XL-v2", "diffusion")
        self.assertIsNone(model_id)
        self.assertTrue(reason)
