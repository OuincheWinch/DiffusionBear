"""The np.array() eval guard added after the 2026-10-02 SIGABRT.

The backend aborted with:

    mlx::core::eval_impl -> array::eval -> _PyManagedBuffer_FromObject
      -> numpy PyArray_FromAny -> array_array

numpy pulled the buffer protocol off an MLX array, and because pybind11 only wraps
Python->C calls, the Metal error escaped into std::terminate. Measured in a subprocess,
the same failure is a catchable RuntimeError when eval is forced in Python scope first
and an abort when it is not -- so the guard is a correctness fix, not a guess.

These tests pin the two properties that matter: MLX arrays are evaluated before numpy
sees them, and everything else is passed through untouched.
"""

import unittest
from unittest import mock

import numpy as np


class NpArrayEvalGuardTests(unittest.TestCase):
    def _guard(self):
        import generator

        return generator._np_array_eval_guard

    def test_non_mlx_input_is_untouched(self):
        """The common path must not change: fill.py and friends feed PIL/list data in."""
        guard = self._guard()
        out = guard([1, 2, 3])
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.tolist(), [1, 2, 3])

    def test_numpy_array_passes_through_unchanged(self):
        """np.array copies, so compare by value -- but the guard must not transform it."""
        guard = self._guard()
        src = np.arange(6, dtype=np.float32)
        out = guard(src)
        self.assertEqual(out.dtype, src.dtype)
        self.assertTrue(np.array_equal(out, src))

    def test_mlx_array_is_evaluated_before_numpy_sees_it(self):
        """The whole point: eval must happen while a Python exception can still be raised."""
        import generator

        guard = self._guard()

        class FakeMlxArray:
            __module__ = "mlx.core"

        # The guard captured the real np.array at import time, so patch that reference.
        with mock.patch("mlx.core.eval") as fake_eval, mock.patch.object(
            generator, "_orig_np_array", return_value="converted"
        ) as fake_array:
            result = guard(FakeMlxArray())

        fake_eval.assert_called_once()
        fake_array.assert_called_once()
        self.assertEqual(result, "converted")

    def test_mlx_eval_failure_surfaces_as_a_python_exception(self):
        """If eval raises, it must propagate as Python -- never reach np.array()."""
        import generator

        guard = self._guard()

        class FakeMlxArray:
            __module__ = "mlx.core"

        with mock.patch("mlx.core.eval", side_effect=RuntimeError("Insufficient Memory")):
            with mock.patch.object(generator, "_orig_np_array") as fake_array:
                with self.assertRaises(RuntimeError):
                    guard(FakeMlxArray())
        fake_array.assert_not_called()

    def test_lookalike_modules_are_not_treated_as_mlx(self):
        """Only real mlx arrays take the guarded path."""
        guard = self._guard()

        class Impostor:
            __module__ = "numpy"

        with mock.patch("mlx.core.eval") as fake_eval:
            guard(Impostor())
        fake_eval.assert_not_called()

    def test_submodule_prefix_matches(self):
        """mlx.core.array must match too, not just exactly 'mlx'."""
        guard = self._guard()

        class Deep:
            __module__ = "mlx.core.something"

        with mock.patch("mlx.core.eval") as fake_eval:
            guard(Deep())
        fake_eval.assert_called_once()


if __name__ == "__main__":
    unittest.main()