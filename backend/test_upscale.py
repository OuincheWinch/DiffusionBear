import json
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

try:
    import upscale
except ModuleNotFoundError as exc:
    if exc.name not in {"mlx", "mflux", "mlx_taef"}:
        raise
    upscale = None


@unittest.skipUnless(upscale is not None, "MLX runtime is unavailable")
class UpscaleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous_dir = upscale.GENERATED_DIR
        upscale.GENERATED_DIR = Path(self.temp.name)
        self.image_id = "source-image"
        Image.new("RGB", (4, 4), "white").save(upscale.GENERATED_DIR / f"{self.image_id}.png")
        (upscale.GENERATED_DIR / f"{self.image_id}.json").write_text(
            json.dumps(
                {
                    "id": self.image_id,
                    "file": f"{self.image_id}.png",
                    "format": "png",
                    "model": "",
                    "artist": "Test Artist",
                    "width": 4,
                    "height": 4,
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        upscale.GENERATED_DIR = self.previous_dir
        self.temp.cleanup()

    def test_upscale_writes_image_and_json(self):
        result = upscale.upscale_image(self.image_id, scale=2)
        self.assertEqual((result["width"], result["height"]), (8, 8))
        self.assertTrue((upscale.GENERATED_DIR / result["file"]).exists())
        sidecar = json.loads(
            (upscale.GENERATED_DIR / f"{result['id']}.json").read_text(encoding="utf-8")
        )
        self.assertEqual(sidecar["upscaled_from"], self.image_id)
        self.assertEqual(sidecar["upscale_factor"], 2)

    def test_upscale_rejects_output_over_pixel_limit(self):
        with patch.object(upscale, "MAX_UPSCALE_PIXELS", 60):
            with self.assertRaisesRegex(ValueError, "exceeds 60 pixel limit"):
                upscale.upscale_image(self.image_id, scale=4)

    def test_concurrent_upscales_are_serialized(self):
        active = 0
        peak = 0
        state_lock = threading.Lock()
        original_save = upscale.save_image_with_metadata

        def tracked_save(*args, **kwargs):
            nonlocal active, peak
            with state_lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.05)
                return original_save(*args, **kwargs)
            finally:
                with state_lock:
                    active -= 1

        with patch.object(upscale, "MAX_UPSCALE_PIXELS", 100), patch.object(
            upscale, "save_image_with_metadata", side_effect=tracked_save
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: upscale.upscale_image(self.image_id, 2), range(2)))

        self.assertNotEqual(results[0]["id"], results[1]["id"])
        self.assertEqual(peak, 1)


if __name__ == "__main__":
    unittest.main()
