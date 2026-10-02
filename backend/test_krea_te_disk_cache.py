"""Tests for the on-disk Krea 2 q4 text-encoder cache.

The bug these guard: the 7.5GB bf16 encoder was loaded on every pipeline build because
mflux sets skip_quantization=True for that component, so the only cache was in-process
and died with the app. Peak measured 14.96GB against a 10.88GB budget, which produced a
flat image rather than an error.
"""

import types
import unittest
from unittest import mock

import generator
from generator import _krea_te_disk_path


class DiskPathTests(unittest.TestCase):
    def test_path_is_stable_for_same_model(self):
        a = _krea_te_disk_path("/models/krea2-turbo-q4", 4, 64)
        b = _krea_te_disk_path("/models/krea2-turbo-q4", 4, 64)
        self.assertEqual(a, b)

    def test_path_changes_with_model_path(self):
        """A different checkpoint must re-derive rather than load stale weights."""
        a = _krea_te_disk_path("/models/krea2-turbo-q4", 4, 64)
        b = _krea_te_disk_path("/models/krea2-raw-q4", 4, 64)
        self.assertNotEqual(a, b)

    def test_path_encodes_quant_settings(self):
        a = _krea_te_disk_path("/models/krea2-turbo-q4", 4, 64)
        b = _krea_te_disk_path("/models/krea2-turbo-q4", 8, 64)
        self.assertNotEqual(a, b)
        self.assertIn("q4-g64", a.name)
        self.assertIn("q8-g64", b.name)

    def test_path_lives_under_asset_dir(self):
        p = _krea_te_disk_path("/models/krea2-turbo-q4", 4, 64)
        self.assertEqual(p.parent, generator.ASSET_DIR / "models")
        self.assertEqual(p.suffix, ".safetensors")


class SaveAndLoadTests(unittest.TestCase):
    def test_save_is_atomic_and_uses_no_partial_suffix(self):
        """A crash mid-write must never leave a file the loader would accept."""
        saved = {}

        class FakeMx:
            @staticmethod
            def save_safetensors(path, tensors):
                with open(path, "w") as fh:
                    fh.write("payload")
                saved["path"] = path

        fake_module = types.SimpleNamespace(parameters=lambda: {"w": 1})
        final = _krea_te_disk_path("/models/krea2-turbo-q4", 4, 64)
        with mock.patch("mlx.core.save_safetensors", FakeMx.save_safetensors), mock.patch(
            "mlx.utils.tree_flatten", lambda t: [("w", 1)]
        ):
            generator._krea_te_save_disk(fake_module, "/models/krea2-turbo-q4", 4, 64)

        import os

        # written to a .partial.safetensors then renamed into place.
        # The temp name must keep the real suffix: mx.save_safetensors appends
        # ".safetensors" to anything that does not already end in it, which would
        # silently redirect the write and make the rename fail with ENOENT.
        self.assertTrue(saved["path"].endswith(".partial.safetensors"))
        self.assertFalse(os.path.exists(saved["path"]), "partial must be gone after rename")
        self.assertTrue(os.path.exists(final), "final cache file must exist")
        os.unlink(final)

    def test_save_failure_is_swallowed_and_leaves_no_partial(self):
        """A failed write must not break the build that just succeeded."""

        def boom(path, tensors):
            raise OSError("disk full")

        with mock.patch("mlx.core.save_safetensors", boom), mock.patch(
            "mlx.utils.tree_flatten", lambda t: [("w", 1)]
        ):
            generator._krea_te_save_disk(
                types.SimpleNamespace(parameters=lambda: {"w": 1}),
                "/models/krea2-turbo-q4",
                4,
                64,
            )


class Bf16SkipTests(unittest.TestCase):
    """The patch that stops mflux materialising bf16 at all."""

    def _original(self):
        seen = []

        def _orig(weights, models, components=None):
            seen.append(dict(models))
            return "ok"

        return _orig, seen

    def test_encoder_is_dropped_only_for_krea2(self):
        generator._krea_te_skip_installed = False
        orig, seen = self._original()
        fake_applier = types.SimpleNamespace(_set_weights=staticmethod(orig))
        module = types.ModuleType("mflux.models.common.weights.loading.weight_applier")
        module.WeightApplier = fake_applier

        generator._krea_te_skip_bf16["on"] = True
        try:
            with mock.patch.dict(
                "sys.modules",
                {
                    "mflux.models.common.weights.loading.weight_applier": module,
                },
            ):
                generator._krea_te_skip_installed = False
                generator._install_krea_te_bf16_skip()
                patched = fake_applier._set_weights

                krea = type("Krea2TextEncoder", (), {})()
                other = type("QwenImage21", (), {})()

                patched({}, {"text_encoder": krea, "transformer": other}, {})
                self.assertEqual(list(seen[-1]), ["transformer"])

                patched({}, {"text_encoder": krea, "transformer": other}, {})
                # flag off -> both loaded again, i.e. no leakage into other engines
                generator._krea_te_skip_bf16["on"] = False
                patched({}, {"text_encoder": krea, "transformer": other}, {})
                self.assertEqual(sorted(seen[-1]), ["text_encoder", "transformer"])

                generator._krea_te_skip_bf16["on"] = True
                patched({}, {"transformer": other}, {})
                self.assertEqual(list(seen[-1]), ["transformer"])
        finally:
            generator._krea_te_skip_bf16["on"] = False
            generator._krea_te_skip_installed = False

    def test_install_is_idempotent(self):
        generator._krea_te_skip_installed = True
        generator._install_krea_te_bf16_skip()  # must not raise or double-wrap
        self.assertTrue(generator._krea_te_skip_installed)


if __name__ == "__main__":
    unittest.main()