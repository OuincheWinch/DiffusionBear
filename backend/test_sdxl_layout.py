"""SDXL needs the diffusers layout, not merely "some weights somewhere".

THE FAILURE
A user pointed an SDXL engine at a model directory and generation died with

    FileNotFoundError: No .safetensors files in
    .../data/models/juggernaut-xl-lightning/unet

`backend/sdxl_engine.py` calls `StableDiffusionXLPipeline.from_diffusers(...)`,
which needs a diffusers directory: `model_index.json` plus a populated `unet/`.
Almost every SDXL checkpoint published on Hugging Face is a SINGLE FILE, not a
diffusers directory, so this is the common case and not an edge case.

WHY IT REACHED GENERATION
`_has_weights()` in routers/downloads.py searches with `rglob("*")`, so ANY
.safetensors anywhere in the tree satisfies it -- including one sitting at the
top level, and including an `unet/` that was never populated. A single-file
checkpoint therefore passed validation at download/adopt time and only exploded
at generation time, as a forty-line Python traceback in the UI.

WHAT IS ASSERTED
That the SDXL layout check accepts a real diffusers directory and rejects each
way a directory can look plausible but be unusable, with the reason naming the
missing part rather than raising a bare FileNotFoundError later.
"""

import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from routers.downloads import sdxl_layout_report  # noqa: E402

from app_settings import ASSET_DIR  # noqa: E402


def _write(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class SdxlLayoutTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    # --- accepts -----------------------------------------------------------
    def test_a_real_diffusers_directory_is_accepted(self):
        d = self.root / "good"
        _write(d / "model_index.json", b"{}")
        _write(d / "unet" / "diffusion_pytorch_model-00001-of-00002.safetensors")
        _write(d / "unet" / "diffusion_pytorch_model.safetensors.index.json")
        _write(d / "vae" / "diffusion_pytorch_model.safetensors")
        ok, reason = sdxl_layout_report(d)
        self.assertTrue(ok, reason)

    def test_unet_may_be_a_single_unsharded_file(self):
        """Lightning SDXL checkpoints are frequently not sharded."""
        d = self.root / "unsharded"
        _write(d / "model_index.json", b"{}")
        _write(d / "unet" / "diffusion_pytorch_model.safetensors")
        ok, reason = sdxl_layout_report(d)
        self.assertTrue(ok, reason)

    # --- rejects -----------------------------------------------------------
    def test_a_single_file_checkpoint_is_rejected(self):
        """The overwhelmingly common Hugging Face SDXL layout."""
        d = self.root / "single"
        _write(d / "juggernautXL_v9.safetensors", b"0" * 64)
        ok, reason = sdxl_layout_report(d)
        self.assertFalse(ok)
        self.assertIn("diffusers", reason.lower())

    def test_an_empty_unet_directory_is_rejected(self):
        """Exactly what the user hit: unet/ present, nothing in it."""
        d = self.root / "empty-unet"
        _write(d / "model_index.json", b"{}")
        (d / "unet").mkdir(parents=True, exist_ok=True)
        ok, reason = sdxl_layout_report(d)
        self.assertFalse(ok)
        self.assertIn("unet", reason.lower())

    def test_a_missing_directory_is_rejected_not_raised(self):
        ok, reason = sdxl_layout_report(self.root / "does-not-exist")
        self.assertFalse(ok)
        self.assertTrue(reason)

    def test_pytorch_bin_only_is_rejected_with_a_reason_about_bin(self):
        """mlx_diffuser reads safetensors; a .bin UNet cannot be loaded."""
        d = self.root / "bin-only"
        _write(d / "model_index.json", b"{}")
        _write(d / "unet" / "diffusion_pytorch_model.bin")
        ok, reason = sdxl_layout_report(d)
        self.assertFalse(ok)
        self.assertIn("safetensors", reason.lower())

    def test_the_reason_never_leaks_an_absurd_path(self):
        ok, reason = sdxl_layout_report(self.root / "nope")
        self.assertLess(len(reason), 200)


if __name__ == "__main__":
    unittest.main()