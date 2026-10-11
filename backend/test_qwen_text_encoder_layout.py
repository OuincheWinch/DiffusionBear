"""Qwen21 text_encoder layout detection.

The patch in qwen_engine.py overrides mflux's text_encoder variant_selector so the two
single-file community layouts load as `single`. The failure it exists to prevent is
subtle: mflux's weight_loader does `component.variant_selector(root)` and then
dereferences the result, so a selector that returns None does not raise a useful error --
it raises "AttributeError: 'NoneType' object has no attribute 'download_url'", which
names neither the model, the path, nor the component. That single line cost a debugging
session. These tests pin the invariant that makes it impossible to regress.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qwen_engine  # noqa: E402  (installs the patch at import)


def _selector():
    from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition
    comp = next(c for c in Qwen21WeightDefinition.get_components() if c.name == "text_encoder")
    return comp.variant_selector


class TextEncoderLayoutTests(unittest.TestCase):
    """Every layout must resolve to a usable component. None is never an answer."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _make(self, name, files, index_map=None):
        d = self.root / name
        for f in files:
            p = d / "text_encoder" / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")
        if index_map is not None:
            (d / "text_encoder" / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": index_map})
            )
        return d

    def test_single_file_without_index(self):
        """Sawfwair Turbo layout."""
        d = self._make("a", ["model.safetensors"])
        self.assertEqual(_selector()(d).loading_mode, "single")

    def test_single_file_with_index_naming_it(self):
        """mlx-community/Qwen-Image-2.1-MLX-4bit layout: the file AND an index that names it."""
        d = self._make(
            "b",
            ["model.safetensors", "model.safetensors.index.json"],
            {"a.weight": "model.safetensors", "b.weight": "model.safetensors"},
        )
        self.assertEqual(_selector()(d).loading_mode, "single")

    def test_genuinely_sharded_falls_through(self):
        """A real shard set must keep mflux's own multi_json path, not be forced to single."""
        d = self._make(
            "c",
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
            {
                "a.weight": "model-00001-of-00002.safetensors",
                "b.weight": "model-00002-of-00002.safetensors",
            },
        )
        self.assertEqual(_selector()(d).loading_mode, "multi_json")

    def test_numbered_shard_without_index(self):
        """`0.safetensors` -- a layout that once shipped under backend/data/models."""
        d = self._make("d", ["0.safetensors"])
        self.assertEqual(_selector()(d).loading_mode, "multi_json")

    def test_index_naming_a_different_file_is_not_single(self):
        """A single .safetensors alongside an index pointing at another shard is sharded."""
        d = self._make(
            "e",
            ["model.safetensors", "other-00001-of-00002.safetensors"],
            {"a.weight": "other-00001-of-00002.safetensors"},
        )
        self.assertNotEqual(_selector()(d).loading_mode, "single")

    def test_selector_never_returns_none(self):
        """The invariant. Any layout, including an empty text_encoder, must stay usable."""
        for name in ("f_empty", "g_missing", "h_single"):
            d = self._make(name, [] if name == "f_empty" else ["model.safetensors"])
            self.assertIsNotNone(
                _selector()(d),
                f"selector returned None for {name}: mflux would raise the opaque "
                f"'NoneType' has no attribute 'download_url'",
            )

    def test_single_mode_preserves_inherited_fields(self):
        """replace(_base, ...) must keep precision/skip_quantization/key_transform.

        A hand-built ComponentDefinition silently resets them to defaults, which would
        quietly change how the text encoder is loaded.
        """
        from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition
        base = next(c for c in Qwen21WeightDefinition.get_components() if c.name == "text_encoder")
        d = self._make("i", ["model.safetensors"])
        patched = _selector()(d)
        for field in ("precision", "skip_quantization", "key_transform", "mapping_getter", "hf_subdir"):
            self.assertEqual(
                getattr(patched, field, None), getattr(base, field, None),
                f"single-mode component lost '{field}'",
            )

    def test_malformed_index_does_not_raise(self):
        """A corrupt index must degrade to multi_json, not crash the selector."""
        d = self._make("j", ["model.safetensors"])
        (d / "text_encoder" / "model.safetensors.index.json").write_text("{not json")
        self.assertEqual(_selector()(d).loading_mode, "multi_json")


class PatchHygieneTests(unittest.TestCase):
    def test_repeated_calls_are_stable_and_never_none(self):
        """get_components() builds a fresh closure per call; that is fine.

        What must hold is that resolution is deterministic and never degrades to None --
        the recursion/accumulation failure mode this guards is a selector that wraps
        itself on each call.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "text_encoder").mkdir()
            (root / "text_encoder" / "model.safetensors").write_text("x")
            modes = [_selector()(root).loading_mode for _ in range(3)]
        self.assertEqual(modes, ["single"] * 3)


if __name__ == "__main__":
    unittest.main()
