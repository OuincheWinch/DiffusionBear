"""Tests for writing the converted LoRA to safetensors.

Review Focus #4: output written outside the asset directory, or a partial file
left behind on error. Both are pinned here -- the Draw Things model directory
is read-only, and a half-written safetensors file is worse than no file because
it loads and then fails deep inside a model load.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lora_convert as convert  # noqa: E402


def down(rank, width):
    return np.arange(rank * width, dtype=np.float32).reshape(rank, width)


def up(rows, rank):
    return np.arange(rows * rank, dtype=np.float32).reshape(rows, rank) + 0.5


def read_keys(path):
    from safetensors.numpy import load_file
    return load_file(str(path))


class TensorsForTests(unittest.TestCase):
    """Pins key spelling and orientation. No transpose, ever."""

    def setUp(self):
        self.pairs = [
            convert.Pair("unet.x.attn.to_q", down(16, 3072), up(3072, 16)),
            convert.Pair("text_encoder_2.y.mlp.c_fc", down(8, 3072), up(3072, 8)),
        ]

    def test_keys_use_the_kohya_suffixes(self):
        tensors = convert._tensors_for(self.pairs)
        self.assertEqual(sorted(tensors), [
            "text_encoder_2.y.mlp.c_fc.lora_down.weight",
            "text_encoder_2.y.mlp.c_fc.lora_up.weight",
            "unet.x.attn.to_q.lora_down.weight",
            "unet.x.attn.to_q.lora_up.weight",
        ])

    def test_no_transpose(self):
        tensors = convert._tensors_for(self.pairs)
        self.assertEqual(tensors["unet.x.attn.to_q.lora_down.weight"].shape,
                         (16, 3072))
        self.assertEqual(tensors["unet.x.attn.to_q.lora_up.weight"].shape,
                         (3072, 16))

    def test_everything_is_float32(self):
        for array in convert._tensors_for(self.pairs).values():
            self.assertEqual(array.dtype, np.float32)

    def test_alpha_is_omitted_by_default(self):
        # mflux and the SDXL engine both default alpha to rank, which makes the
        # effective scale 1.0. DT stores no alpha, so emitting one would
        # silently rescale the whole LoRA.
        self.assertNotIn("unet.x.attn.to_q.alpha",
                         convert._tensors_for(self.pairs))

    def test_alpha_is_emitted_when_asked_for(self):
        tensors = convert._tensors_for(self.pairs, alpha=32.0)
        self.assertEqual(float(tensors["unet.x.attn.to_q.alpha"]), 32.0)
        self.assertIn("text_encoder_2.y.mlp.c_fc.alpha", tensors)

    def test_two_pairs_with_one_base_are_refused(self):
        pairs = [convert.Pair("a", down(4, 4), up(4, 4)),
                 convert.Pair("a", down(4, 4), up(4, 4))]
        with self.assertRaises(convert.ConversionError):
            convert._tensors_for(pairs)


class WriteTests(unittest.TestCase):
    """Pins atomicity and the refusal to write anywhere unsafe."""

    def setUp(self):
        self.pairs = [convert.Pair("unet.x.attn.to_q", down(16, 8), up(8, 16))]

    def test_writes_a_readable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "lora.safetensors"
            summary = convert.write_pairs(self.pairs, out)
            self.assertTrue(out.exists())
            keys = read_keys(out)
            self.assertEqual(len(keys), 2)
            self.assertEqual(summary["pairs"], 1)
            self.assertEqual(summary["tensors"], 2)
            self.assertGreater(summary["bytes"], 0)

    def test_no_temp_file_is_left_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "lora.safetensors"
            convert.write_pairs(self.pairs, out)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["lora.safetensors"])

    def test_a_failed_write_leaves_nothing_behind(self):
        class Boom(Exception):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "lora.safetensors"
            original = convert._save_file
            try:
                def explode(tensors, path):
                    # Write the temp file, then fail, so the cleanup path is
                    # actually exercised rather than skipped.
                    from safetensors.numpy import save_file
                    save_file(tensors, str(path))
                    raise Boom("disk full")
                convert._save_file = explode
                with self.assertRaises(Boom):
                    convert.write_pairs(self.pairs, out)
            finally:
                convert._save_file = original
            self.assertEqual(os.listdir(tmp), [])

    def test_refuses_to_write_inside_the_draw_things_model_directory(self):
        with self.assertRaises(convert.ConversionError) as cm:
            convert.write_pairs(
                self.pairs,
                convert.DRAW_THINGS_MODEL_DIR / "converted.safetensors")
        self.assertIn("read-only", str(cm.exception))

    def test_refuses_to_write_inside_the_containers_directory(self):
        with self.assertRaises(convert.ConversionError):
            convert.write_pairs(
                self.pairs,
                Path.home() / "Library" / "Containers" / "x.safetensors")

    def test_refusal_happens_before_anything_is_written(self):
        with self.assertRaises(convert.ConversionError):
            convert.write_pairs(
                self.pairs,
                convert.DRAW_THINGS_MODEL_DIR / "sub" / "converted.safetensors")

    def test_a_relative_path_inside_the_model_directory_is_refused(self):
        # Resolve before comparing, or `models/../../elsewhere` walks out.
        with tempfile.TemporaryDirectory() as tmp:
            sneaky = os.path.join(convert.DRAW_THINGS_MODEL_DIR, "..", "..",
                                  "sneaky.safetensors")
            with self.assertRaises(convert.ConversionError):
                convert.write_pairs(self.pairs, Path(sneaky))


class AllowRootTests(unittest.TestCase):
    """A denylist only stops the paths it names, so containment is also offered."""

    def setUp(self):
        self.pairs = [convert.Pair("unet.x.attn.to_q", down(16, 8), up(8, 16))]

    def test_output_inside_the_permitted_root_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "loras"
            root.mkdir()
            out = root / "a.safetensors"
            convert.write_pairs(self.pairs, out, allow_root=tmp)
            self.assertTrue(out.exists())

    def test_output_outside_the_permitted_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            allowed = Path(tmp) / "allowed"
            allowed.mkdir()
            other = Path(tmp) / "other"
            other.mkdir()
            with self.assertRaises(convert.ConversionError) as cm:
                convert.write_pairs(self.pairs, other / "a.safetensors",
                                    allow_root=allowed)
            self.assertIn("outside", str(cm.exception))
            self.assertFalse((other / "a.safetensors").exists())

    def test_a_traversal_out_of_the_permitted_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "allowed"
            root.mkdir()
            sneaky = Path(tmp) / "allowed" / ".." / "escape.safetensors"
            with self.assertRaises(convert.ConversionError):
                convert.write_pairs(self.pairs, sneaky, allow_root=root)


class ConvertTests(unittest.TestCase):
    """End to end against a synthetic store."""

    def test_unknown_base_model_is_refused_before_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "x.safetensors"
            with self.assertRaises(convert.ConversionError):
                convert.convert(None, out, "wan")
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()