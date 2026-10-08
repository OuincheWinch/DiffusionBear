"""Orphaned partial downloads must not fail every future download forever.

huggingface_hub uses a fresh temp name per download attempt, so a partial left
behind by a killed run is never resumed -- yet the layout verification failed on
it every time (seen live: an 18-day-old 13 MB orphan failing a fresh 28-file
download of realvis-xl-v5, with all 28 files at 100%).
"""

import sys
import tempfile
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

from routers.downloads import _drop_orphan_partials  # noqa: E402
from sdxl_layout import sdxl_layout_report  # noqa: E402


class DropOrphanPartialsTests(unittest.TestCase):
    def _mk(self, root):
        d = Path(root) / "model"
        (d / "unet").mkdir(parents=True)
        (d / "model_index.json").write_text("{}", encoding="utf-8")
        (d / "unet" / "w.safetensors").write_bytes(b"0" * 32)
        partial_dir = d / ".cache" / "huggingface" / "download" / "unet"
        partial_dir.mkdir(parents=True)
        orphan = partial_dir / "ab12cd34.incomplete"
        orphan.write_bytes(b"0" * 32)
        (partial_dir / "w.safetensors.metadata").write_bytes(b"meta")
        (partial_dir / "w.safetensors.lock").write_bytes(b"")
        return d, orphan

    def test_orphan_is_removed_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, orphan = self._mk(tmp)
            dropped, freed = _drop_orphan_partials(d)
            self.assertEqual(dropped, 1)
            self.assertEqual(freed, 32)
            self.assertFalse(orphan.exists())

    def test_sidecars_and_weights_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, _ = self._mk(tmp)
            _drop_orphan_partials(d)
            partial_dir = d / ".cache" / "huggingface" / "download" / "unet"
            self.assertTrue((partial_dir / "w.safetensors.metadata").is_file())
            self.assertTrue((partial_dir / "w.safetensors.lock").is_file())
            self.assertTrue((d / "unet" / "w.safetensors").is_file())

    def test_layout_passes_once_the_orphan_is_gone(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, _ = self._mk(tmp)
            ok_before, _ = sdxl_layout_report(d)
            self.assertFalse(ok_before, "the orphan must fail verification first")
            _drop_orphan_partials(d)
            ok_after, reason = sdxl_layout_report(d)
            self.assertTrue(ok_after, reason)

    def test_missing_cache_dir_is_a_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(_drop_orphan_partials(Path(tmp)), (0, 0))
            self.assertEqual(_drop_orphan_partials(None), (0, 0))


if __name__ == "__main__":
    unittest.main()
