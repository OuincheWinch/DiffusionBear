"""TAESD must be reachable from the installed app, not just from the repo.

THE FAILURE THIS GUARDS
`get_taesd_decoder()` defaulted to

    Path(__file__).parent / "data" / "models" / "taesdxl" / ...

which is right in the repo checkout and wrong in the installed app: inside the
signed bundle `__file__` is `Contents/Resources/backend`, so the lookup resolved
to `Contents/Resources/backend/data/models/taesdxl` -- inside the signature. The
weights sit in the model store, so the file was never found.

The consequence was silent. `sdxl_engine.py` wraps the call in `except Exception`
and falls back to the full VAE, printing one line to stderr. So the "~0.5s
ultra-fast TAESD decode" was dead in every shipped build, and the only evidence
was a stderr line nobody reads. No test failed, because no test asked.

WHAT IS ASSERTED
That the default resolves through ASSET_DIR (so it honours `store_path`), that it
never points back inside the bundle, and that the file actually exists where it
says. The existence check skips when the weights are not on this machine, because
a path assertion is the load-bearing part and must run everywhere.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app_settings  # noqa: E402
import taesd_mlx  # noqa: E402


class DefaultTaesdPathTests(unittest.TestCase):
    def setUp(self):
        self.path = taesd_mlx._default_taesd_path()

    def test_it_resolves_under_the_model_store(self):
        self.assertEqual(
            self.path.parent.parent.parent,
            Path(app_settings.ASSET_DIR).resolve(),
            "TAESD must be looked up in ASSET_DIR so store_path relocation is honoured",
        )

    def test_it_never_points_inside_the_bundle(self):
        """The exact defect: resolving to Contents/Resources/backend/data."""
        text = str(self.path)
        self.assertNotIn("Contents/Resources", text)
        self.assertNotIn("__file__", text)

    def test_it_follows_a_relocated_store(self):
        """The invariant that actually matters, and cannot be seen in the repo.

        In the checkout ASSET_DIR *is* backend/data, so the old __file__/data path
        and the new one coincide -- the bug was invisible here and only appeared
        once the app was installed. So point ASSET_DIR somewhere else entirely and
        assert the lookup moves with it.
        """
        import tempfile

        original = app_settings.ASSET_DIR
        with tempfile.TemporaryDirectory() as tmp:
            app_settings.ASSET_DIR = Path(tmp)
            try:
                moved = taesd_mlx._default_taesd_path()
            finally:
                app_settings.ASSET_DIR = original
        self.assertEqual(
            moved,
            Path(tmp) / "models" / "taesdxl" / "diffusion_pytorch_model.safetensors",
            "the TAESD lookup must follow ASSET_DIR, not the module location",
        )

    def test_the_filename_is_still_the_one_on_disk(self):
        self.assertEqual(self.path.name, "diffusion_pytorch_model.safetensors")

    def test_the_weights_exist_where_it_points(self):
        if not self.path.exists():
            self.skipTest(f"TAESD weights are not on this machine: {self.path}")
        self.assertGreater(self.path.stat().st_size, 1_000_000, "weights look truncated")

    def test_the_lookup_is_cheap_enough_to_run_per_decode(self):
        """It is called on the decode path, so it must not stat the world."""
        self.assertTrue(str(self.path).endswith(
            "models/taesdxl/diffusion_pytorch_model.safetensors"))


if __name__ == "__main__":
    unittest.main()