"""Tiled VAE conversion must evaluate on the MLX side before touching numpy.

The failure this pins down killed a real generation: after 4/4 denoise steps,
VAE decode died with `libc++abi: terminating due to uncaught exception of
type std::runtime_error: [METAL] Command buffer execution failed:
Insufficient Memory`, aborting the entire engine process. No catchable error,
no partial image, nothing in tmp -- the latents only ever lived in Metal
memory.

The mechanism: `np.array()` on an UNEVALUATED mlx array forces evaluation
inside numpy's C buffer-protocol callback. A Metal failure there escapes
pybind11's exception translator (it only wraps Python-to-C calls) and lands in
`std::terminate`. The identical OOM under `mx.eval()` in Python scope raises a
catchable RuntimeError instead -- the engine would have lived and reported
"decode ran out of memory". The vendored fork's tiler never got the eval-first
guard the main backend already has for the same pattern.

What this asserts: `mx.eval` runs before every `np.array` conversion in both
tiled paths, and the math is unchanged (identity decode round-trips). What it
does NOT do: reproduce an OOM. At test scale nothing fails either way, so the
ordering assertion -- via a wrapping spy, removed afterwards -- is the whole
test. A small-scale run passing tells you nothing; the spy failing pre-patch
tells you the guard is absent.

The module under test is loaded straight from packaging/vendor/mflux-src
under a synthetic name: importing it as `mflux...` would collide with the
installed mflux in sys.modules. Its only imports are mlx, numpy and typing,
so a file-location load is faithful.
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

VENDOR_TILER = (
    Path(__file__).resolve().parents[1]
    / "packaging" / "vendor" / "mflux-src"
    / "mflux" / "models" / "common" / "vae" / "vae_tiler.py"
)


def _load_tiler():
    spec = importlib.util.spec_from_file_location(
        "dbx_vae_tiler_under_test", VENDOR_TILER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(_MLX_AVAILABLE, "MLX runtime is unavailable")
class TilerEvalFirstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiler = _load_tiler()
        cls._real_eval = mx.eval

    def test_decode_evaluates_before_numpy(self):
        """The exact line that killed the Qwen run (decode path, line 166)."""
        latent = mx.random.uniform(shape=(1, 4, 1, 12, 12))
        with mock.patch.object(mx, "eval", wraps=self._real_eval) as spy:
            out = self.tiler.VAETiler.decode_image_tiled(
                latent=latent,
                # Nearest-neighbour stand-in for a real x8 decoder: lazy, so
                # the tile reaching np.array is still unevaluated pre-patch.
                decode_fn=lambda tile: mx.repeat(mx.repeat(tile, 8, 3), 8, 4),
                tile_size=(32, 32),
                tile_overlap=(8, 8),
                spatial_scale=8,
            )
        self.assertTrue(
            spy.called,
            "a tile was converted with np.array() without mx.eval() first -- "
            "under memory pressure that aborts the process via std::terminate "
            "instead of raising",
        )
        self.assertEqual(tuple(out.shape), (1, 4, 96, 96))
        # Overlapping tiles decode identical values, and blending weights sum
        # to one, so the interior must round-trip. (Edges can hit the
        # sliver-skip rule; the interior cannot.)
        import numpy as np

        self._real_eval(out, latent)
        expected = np.array(latent[:, :, 0, 4:8, 4:8]).repeat(8, axis=2).repeat(8, axis=3)
        np.testing.assert_allclose(
            np.array(out[:, :, 32:64, 32:64]), expected, rtol=1e-4, atol=1e-5)

    def test_encode_evaluates_before_numpy(self):
        """Same pattern on the encode path (line 75)."""
        image = mx.random.uniform(shape=(1, 3, 24, 24))
        with mock.patch.object(mx, "eval", wraps=self._real_eval) as spy:
            out = self.tiler.VAETiler.encode_image_tiled(
                image=image,
                encode_fn=lambda tile: mx.zeros((1, 4, 1) + tile.shape[3:]),
                latent_channels=4,
                tile_size=(16, 16),
                tile_overlap=(8, 8),
                spatial_scale=8,
            )
        self.assertTrue(
            spy.called,
            "a tile was converted with np.array() without mx.eval() first",
        )
        self.assertEqual(tuple(out.shape), (1, 4, 1, 3, 3))

    def _assert_allclose(self, actual, expected):
        import numpy as np

        self._real_eval(actual, expected)
        np.testing.assert_allclose(
            np.array(actual), np.array(expected), rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
