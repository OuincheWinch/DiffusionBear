"""FLUX.1-schnell runs natively: registry entry, dispatch, LoRA mapping.

Draw Things trains FLUX.1 layouts, so its LoRAs apply to a real FLUX.1
pipeline with zero conversion math -- which is the entire point of running
schnell instead of fusing weights onto Klein. Three properties:

1. REGISTRY: a flux1-schnell entry with the mflux-native q4 repo,
   distilled defaults (4 steps, guidance-distilled so no negative prompt),
   and LoRA support on. Values mirror the klein-9b entry field-for-field;
   anything unowned (e.g. someone else's Civitai ids) is omitted, not guessed.
2. DISPATCH: _get_pipeline routes the id to mflux Flux1 with the resolved
   model path, LoRA paths/scales, and no baking -- asserted via a stubbed
   constructor, so no weights load. A dispatch that falls through to the
   custom/unknown branch fails here.
3. COMPAT: mapping_for resolves the id to FluxLoRAMapping, so converted
   Draw Things outputs and native FLUX.1 adapters both check. Without this,
   every schnell LoRA reports "unsupported" and the picker blocks it.
"""
import unittest
from pathlib import Path
from unittest import mock

import generator
import lora_compat


class SchnellRegistryTests(unittest.TestCase):
    def test_entry_exists_with_distilled_defaults(self):
        entry = generator.MODELS.get("flux1-schnell")
        self.assertIsNotNone(entry, "flux1-schnell missing from MODELS")
        self.assertEqual(entry["engine"], "mflux")
        self.assertEqual(
            entry["repo"], "mflux-community/flux-1-schnell-mflux-q4")
        self.assertTrue(entry["is_distilled"])
        self.assertEqual(entry["default_steps"], 4)
        self.assertFalse(entry["supports_negative"])
        self.assertTrue(entry["supports_loras"])
        self.assertEqual(entry["lora_format"], "FLUX.1")

    def test_entry_has_no_guessed_civitai_identity(self):
        entry = generator.MODELS["flux1-schnell"]
        for key in ("civitai_model_id", "civitai_version_id",
                    "civitai_model_name", "civitai_version_name", "sha256"):
            self.assertNotIn(
                key, entry,
                f"{key} would attribute someone else's Civitai page to stock weights",
            )


class SchnellDispatchTests(unittest.TestCase):
    def test_routes_to_flux1_with_paths_and_no_bake(self):
        with mock.patch(
            "mflux.models.flux.variants.txt2img.flux.Flux1"
        ) as flux_cls:
            generator._get_pipeline(
                "flux1-schnell", 4,
                [{"path": "/tmp/a.safetensors", "scale": 1.0}],
                variant="standard",
            )
        _, kwargs = flux_cls.call_args
        self.assertIn("flux-1-schnell-mflux-q4", kwargs["model_path"])
        # Resolved: the pipeline normalises /tmp to /private/tmp on macOS
        # before the constructor ever sees the paths.
        self.assertEqual(
            kwargs["lora_paths"],
            [str(Path("/tmp/a.safetensors").resolve())],
        )
        self.assertEqual(kwargs["lora_scales"], [1.0])
        self.assertFalse(kwargs["bake_lora"])


class SchnellCompatTests(unittest.TestCase):
    def test_mapping_resolves_to_flux_lora_mapping(self):
        from mflux.models.flux.weights.flux_lora_mapping import FluxLoRAMapping

        self.assertIs(lora_compat.mapping_for("flux1-schnell"), FluxLoRAMapping)


if __name__ == "__main__":
    unittest.main()
