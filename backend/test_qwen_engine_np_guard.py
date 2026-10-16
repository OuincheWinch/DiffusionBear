"""The Qwen engine process must run with the numpy eval guard installed.

Same abort class as the tiled and untiled conversion sites, but those are
fixed call sites in vendored code. This pins the process-wide backstop in OUR
code: any np.array() on an MLX array anywhere in engine scope evaluates first
in Python scope, so a future unevaluated conversion degrades to a catchable
error instead of std::terminate. generator.py has carried the identical guard
since the 2026-10-02 crash; qwen_engine.py never got one.

Two behaviors, both observable without an OOM:

1. IDEMPOTENCY: installing twice wraps once. A stacked wrapper would eval
   twice and print two stacks per conversion -- harmless but sloppy, and the
   marker is what keeps a re-exec from stacking.
2. DELEGATION: conversions through the wrapped np.array return correct
   values, for MLX and non-MLX inputs alike.

The installer patches np.array process-wide while active, so the original is
restored afterwards: a leaked wrapper would print a stack trace into every
later test's stderr on each conversion.
"""
import unittest
from unittest import mock

import numpy as np

try:
    import mlx.core as mx

    _MLX_AVAILABLE = True
except ModuleNotFoundError:
    _MLX_AVAILABLE = False

try:
    import qwen_engine

    _ENGINE_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _ENGINE_AVAILABLE = False


@unittest.skipUnless(
    _MLX_AVAILABLE and _ENGINE_AVAILABLE, "MLX engine unavailable")
class QwenEngineNpGuardTests(unittest.TestCase):
    def setUp(self):
        self._orig_array = np.array
        self.addCleanup(self._restore_np_array)

    def _restore_np_array(self):
        np.array = self._orig_array

    def test_install_is_idempotent(self):
        qwen_engine.install_np_eval_guard()
        first = np.array
        qwen_engine.install_np_eval_guard()
        self.assertIs(
            np.array, first,
            "installing twice stacked the wrapper -- each conversion would "
            "eval and print twice",
        )

    def test_conversions_still_return_correct_values(self):
        qwen_engine.install_np_eval_guard()
        lazy = mx.arange(6).reshape(2, 3) + 1
        self.assertEqual(np.array(lazy).tolist(), [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(np.array([7, 8]).tolist(), [7, 8])


if __name__ == "__main__":
    unittest.main()
