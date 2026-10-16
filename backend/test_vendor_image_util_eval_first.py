"""The untiled VAE output must evaluate before numpy conversion, too.

Same bug class as test_vae_tiler_eval_first.py, third site: at 512x320
(max dimension 512, NOT > 512) tiling stays off, so VAE decode runs full-frame
and the result converts in ImageUtil._to_numpy -- which never got the
eval-first guard either. A Metal OOM there aborts inside numpy's C buffer
callback (std::terminate, whole engine process gone) instead of raising.

The module is loaded straight from packaging/vendor/mflux-src under a
synthetic name (see test_vae_tiler_eval_first.py for why); its mflux-relative
imports resolve against the installed fork. Skipped where the fork is absent.
"""
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

try:
    import mlx.core as mx

    _MLX_AVAILABLE = True
except ModuleNotFoundError:
    _MLX_AVAILABLE = False

try:
    import mflux.models.qwen21  # noqa: F401 -- fork marker, not used directly

    _FORK_PRESENT = True
except (ModuleNotFoundError, ImportError):
    _FORK_PRESENT = False

VENDOR_IMAGE_UTIL = (
    Path(__file__).resolve().parents[1]
    / "packaging" / "vendor" / "mflux-src"
    / "mflux" / "utils" / "image_util.py"
)


def _load_vendor_image_util():
    spec = importlib.util.spec_from_file_location(
        "dbx_image_util_under_test", VENDOR_IMAGE_UTIL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(_MLX_AVAILABLE and _FORK_PRESENT, "MLX fork unavailable")
class VendorImageUtilEvalFirstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_util = _load_vendor_image_util()
        cls._real_eval = mx.eval

    def test_to_numpy_evaluates_before_converting(self):
        """A full-frame decode output reaches PIL through _to_numpy."""
        decoded = mx.random.uniform(shape=(1, 3, 64, 40)) * 2 - 1
        with mock.patch.object(mx, "eval", wraps=self._real_eval) as spy:
            image = self.image_util.ImageUtil.to_pil(decoded)
        self.assertTrue(
            spy.called,
            "the full-frame output was converted with np.array() without "
            "mx.eval() first -- under memory pressure that aborts the "
            "process via std::terminate instead of raising",
        )
        import PIL.Image

        self.assertIsInstance(image, PIL.Image.Image)
        self.assertEqual(image.size, (40, 64))


if __name__ == "__main__":
    unittest.main()
