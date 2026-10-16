"""Regression guard for the rope memo cache under mx.compile.

The cache keys on `ids.reshape(-1).tolist()`, which forces an evaluation. Inside a
compiled trace MLX refuses that outright:

    ValueError: [eval] Attempting to eval an array during function transformations
    like compile or vmap is not allowed.

That killed FLUX 4B *edit* at denoise step 3 of 4: the reference-image graph reaches
the rope embedder while `mx.compile` is tracing, and the generation died. It was first
mistaken for an mflux bug and "fixed" by not compiling the edit path at all, which
silently cost the speedup. The real fix is here, in the cache: detect the tracing case
and fall through to the uncached call.

These tests exercise the actual failure rather than grepping for a comment.
"""

import unittest

import mlx.core as mx

import generator


class RopeMemoCacheTraceTests(unittest.TestCase):
    def _fresh_cache(self):
        """Install the real installer onto a throwaway class; return (cls, call_count)."""
        calls = {"n": 0}

        class FakeRope:
            def __call__(self, ids):
                calls["n"] += 1
                return ids.astype(mx.float32) + 1.0

        generator._install_rope_cache_on_class(FakeRope)
        return FakeRope, calls

    def test_eager_calls_are_cached(self):
        cls, calls = self._fresh_cache()
        rope = cls()
        ids = mx.arange(4)
        first = rope(ids)
        second = rope(ids)
        self.assertEqual(calls["n"], 1, "second identical call should hit the memo")
        self.assertTrue(mx.array_equal(first, second).item())

    def test_call_survives_being_traced_by_mx_compile(self):
        """The regression: this raised ValueError and killed FLUX edit mid-generation."""
        cls, _ = self._fresh_cache()
        rope = cls()

        def body(x):
            return rope(x) * 2

        try:
            out = mx.compile(body)(mx.arange(4))
            mx.eval(out)
        except RuntimeError as e:
            if "Unable to build metal library" in str(e) or "monolithic_metal.pcm" in str(e):
                self.skipTest("Metal compiler cache not writable in sandboxed test environment")
            raise
        self.assertEqual(out.shape, (4,))
        self.assertTrue(mx.allclose(out, mx.arange(4).astype(mx.float32) * 2 + 2).item())

    def test_a_different_ids_array_is_not_served_from_the_memo(self):
        cls, calls = self._fresh_cache()
        rope = cls()
        rope(mx.arange(4))
        rope(mx.arange(4) + 10)
        self.assertEqual(calls["n"], 2, "distinct ids must not collide in the cache")

    def test_cache_key_does_not_silently_swallow_other_errors(self):
        """Only the tracing ValueError may be bypassed; anything else must propagate."""
        import types

        class Exploding:
            def __call__(self, ids):
                if ids.reshape(-1).tolist():
                    raise RuntimeError("boom")
                return ids

        generator._install_rope_cache_on_class(Exploding)
        with self.assertRaises(RuntimeError):
            Exploding()(mx.arange(4))


if __name__ == "__main__":
    unittest.main()
