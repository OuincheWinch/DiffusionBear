import unittest

import generator
from state import GenerateRequest


TINY_SIZES = [(128, 128), (256, 256)]

# Sizes that already worked before the 128 floor existed: dropdown entries and
# model presets. None of these may regress.
KNOWN_GOOD_SIZES = [
    (1024, 1024), (768, 768), (512, 512), (512, 768), (1216, 832), (832, 1216),
    (1280, 720), (576, 1024), (720, 1280), (1536, 640), (1152, 896), (1008, 672),
    (1248, 832), (1152, 768), (768, 512), (832, 1248), (896, 1152), (768, 1152),
    (672, 1008), (768, 1344), (640, 1536), (1024, 576), (1344, 768),
]


class DimensionValidationTests(unittest.TestCase):
    def test_tiny_sizes_accepted_for_every_model(self):
        for model_id, minfo in generator.MODELS.items():
            alignment = 8 if minfo.get("engine") == "sdxl" else 16
            for size in TINY_SIZES:
                self.assertEqual(size[0] % alignment, 0, f"{model_id} alignment")
                self.assertEqual(size[1] % alignment, 0, f"{model_id} alignment")
                self.assertEqual(generator._validate_dimensions(*size, minfo), size, model_id)

    def test_known_good_sizes_still_accepted(self):
        for model_id, minfo in generator.MODELS.items():
            for size in KNOWN_GOOD_SIZES:
                self.assertEqual(generator._validate_dimensions(*size, minfo), size, f"{model_id} {size}")

    def test_below_floor_rejected(self):
        for model_id, minfo in generator.MODELS.items():
            for size in ((64, 64), (96, 128), (128, 96), (0, 128), (127, 128)):
                with self.assertRaises(ValueError, msg=model_id):
                    generator._validate_dimensions(*size, minfo)

    def test_alignment_still_enforced(self):
        self.assertEqual(
            generator._validate_dimensions(520, 520, generator.MODELS["juggernaut-xl-lightning"]),
            (520, 520),
        )
        for model_id in ("flux2-klein-4b", "flux2-klein-9b", "z-image-turbo", "krea2-turbo", "qwen-image-2.1"):
            with self.assertRaises(ValueError, msg=model_id):
                generator._validate_dimensions(520, 520, generator.MODELS[model_id])

    def test_upper_bound_unchanged(self):
        for model_id, minfo in generator.MODELS.items():
            for size in ((128, 2176), (2176, 128), (2049, 2049)):
                with self.assertRaises(ValueError, msg=model_id):
                    generator._validate_dimensions(*size, minfo)

    def test_request_schema_bounds(self):
        for model_id in ("flux2-klein-4b", "z-image-turbo", "juggernaut-xl-lightning", "krea2-turbo", "qwen-image-2.1"):
            for size in TINY_SIZES:
                request = GenerateRequest(prompt="t", model=model_id, width=size[0], height=size[1])
                self.assertEqual((request.width, request.height), size)
        with self.assertRaises(ValueError):
            GenerateRequest(prompt="t", model="flux2-klein-4b", width=64, height=64)
        with self.assertRaises(ValueError):
            GenerateRequest(prompt="t", model="flux2-klein-4b", width=4096, height=1024)

    def test_model_registry_has_no_test_sizes_leftover(self):
        for model_id, minfo in generator.MODELS.items():
            self.assertIsNone(minfo.get("test_sizes"), model_id)


if __name__ == "__main__":
    unittest.main()
