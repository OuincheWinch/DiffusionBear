"""Klein block-count verifier: declared 4B-vs-9B size checked against the
transformer weight index.

Hermetic: synthetic indexes in temp dirs only — never the real HF cache.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from custom_models import verify_klein_size


def _write_index(model_dir: Path, doubles: int, singles: int,
                 shards: int = 2, create_shards: bool = True) -> Path:
    """Synthetic transformer weight index with exact block depths.

    Keys mirror the real mlx-community index layout
    (transformer.transformer_blocks.N... / transformer.single_transformer_blocks.N...).
    """
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for i in range(doubles):
        shard = f"model-{i % shards:05d}-of-{shards:05d}.safetensors"
        weight_map[f"transformer.transformer_blocks.{i}.attn.to_q.weight"] = shard
    for i in range(singles):
        shard = f"model-{(i + doubles) % shards:05d}-of-{shards:05d}.safetensors"
        weight_map[f"transformer.single_transformer_blocks.{i}.attn.to_q.weight"] = shard
    index = transformer_dir / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}),
                     encoding="utf-8")
    if create_shards:
        for shard in set(weight_map.values()):
            (transformer_dir / shard).write_bytes(b"\x00")
    return index


class VerifyKleinSizeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temp.name) / "model"
        self.model_dir.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_4b_passes_for_4b(self):
        _write_index(self.model_dir, 5, 20)
        ok, reason = verify_klein_size(self.model_dir, "4b")
        self.assertTrue(ok, reason)
        self.assertIsNone(reason)

    def test_exact_4b_fails_for_9b_naming_both(self):
        _write_index(self.model_dir, 5, 20)
        ok, reason = verify_klein_size(self.model_dir, "9b")
        self.assertFalse(ok)
        self.assertIn("5", reason)
        self.assertIn("8", reason)

    def test_exact_9b_passes_for_9b(self):
        _write_index(self.model_dir, 8, 24)
        ok, reason = verify_klein_size(self.model_dir, "9b")
        self.assertTrue(ok, reason)
        self.assertIsNone(reason)

    def test_exact_9b_fails_for_4b_naming_both(self):
        _write_index(self.model_dir, 8, 24)
        ok, reason = verify_klein_size(self.model_dir, "4b")
        self.assertFalse(ok)
        self.assertIn("8", reason)
        self.assertIn("5", reason)

    def test_named_but_missing_shards_read_as_incomplete(self):
        _write_index(self.model_dir, 5, 20, create_shards=False)
        ok, reason = verify_klein_size(self.model_dir, "4b")
        self.assertFalse(ok)
        self.assertIn("incomplete", reason)

    def test_garbage_json_is_refusal_not_traceback(self):
        transformer_dir = self.model_dir / "transformer"
        transformer_dir.mkdir(parents=True, exist_ok=True)
        (transformer_dir / "model.safetensors.index.json").write_text(
            "{not json", encoding="utf-8")
        ok, reason = verify_klein_size(self.model_dir, "4b")
        self.assertFalse(ok)
        self.assertTrue(reason)

    def test_missing_index_is_refusal(self):
        ok, reason = verify_klein_size(self.model_dir, "4b")
        self.assertFalse(ok)
        self.assertTrue(reason)


if __name__ == "__main__":
    unittest.main()
