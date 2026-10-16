"""A settings override pointing at a weightless directory must not strand a model.

The 4B failure, 2026-10-14: the live settings.json pointed flux2-klein-4b at
a corpse HF snapshot (config.json only -- an interrupted 22 GB upstream pull),
while the complete mlx-community snapshot sat unused in the same cache.
`resolve_local_model_path` trusted the override because the directory EXISTS,
and the engine died scanning it for weights: "No safetensors files found in
.../transformer". Correct code plus stale settings equalled a wrong model,
forever -- the same stale-state class as a settings pin outliving its revert.

Three tests:

1. WEIGHTLESS OVERRIDE IGNORED: override -> tmp corpse dir, model
   flux2-klein-4b. The result must never be the corpse. It falls through to
   the HF-cache branch: a complete snapshot where one exists, else None.
   Either way the assertion (result != corpse) holds on any machine, so this
   needs no cache, no network, no gate.
2. FALL-THROUGH FINDS REAL WEIGHTS (gated): same setup, but only runs where
   this machine's HF cache holds a complete mlx-community 4B snapshot; the
   result must be one of the complete snapshots found by an independent
   glob (membership, not identity -- reimplementing the newest-by-mtime
   ordering would test a copy of the code, not the behavior).
3. CONTROL: an override pointing at a directory WITH weights is returned
   untouched. Without this, the fix could nuke every valid override and the
   suite would still be green.

`app_settings.model_local_path` is patched because the point is precisely
that the real settings file is not involved: whatever string it yields,
weightless means ignored. The identity comparison at the heart of the fix
stays real.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app_settings
import generator


def _corpse(directory: Path) -> Path:
    """A directory shaped like the real failure: exists, config present,
    zero weights -- what an interrupted snapshot download leaves behind."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return directory


class StaleOverrideTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def enterContext(self, cm):
        result = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return result

    def _override(self, path: Path):
        self.enterContext(mock.patch.object(
            app_settings, "model_local_path", return_value=str(path)))

    def test_weightless_override_is_never_returned(self):
        corpse = _corpse(self.tmp / "corpse-snapshot" / "transformer")
        self._override(corpse.parent)
        result = generator.resolve_local_model_path("flux2-klein-4b")
        # None is an honest answer when no complete copy exists anywhere;
        # the corpse is never one. Pre-fix this returned the corpse itself.
        if result is not None:
            self.assertNotEqual(
                Path(result), corpse.parent,
                "the weightless override was trusted -- the engine would die "
                "scanning it for weights",
            )

    def test_partially_downloaded_snapshot_is_rejected(self):
        """The real 4B failure shape, not just an empty dir: text_encoder fully
        downloaded (shards + index), transformer/ holding only config.json.
        any-*.safetensors passes on the text_encoder shards, so a presence
        check alone trusts it -- and the engine then dies scanning the empty
        transformer/. The transformer rule must reject what the presence
        check accepts."""
        snap = self.tmp / "partial-snapshot"
        (snap / "transformer").mkdir(parents=True)
        (snap / "transformer" / "config.json").write_text("{}", encoding="utf-8")
        (snap / "text_encoder").mkdir(parents=True)
        (snap / "text_encoder" / "model-00001-of-00002.safetensors").write_bytes(b"")
        self._override(snap)
        result = generator.resolve_local_model_path("flux2-klein-4b")
        if result is not None:
            self.assertNotEqual(
                Path(result), snap,
                "a snapshot missing transformer weights was trusted",
            )

    def test_transformer_dir_with_weights_validates(self):
        """The rule's positive side, pinned so it cannot drift into rejecting
        healthy flux layouts: transformer/ holding shards is valid even when
        that is the ONLY weights present."""
        healthy = self.tmp / "flux-layout"
        (healthy / "transformer").mkdir(parents=True)
        (healthy / "transformer" / "0.safetensors").write_bytes(b"")
        ok, reason = generator.validate_local_model_dir("flux2-klein-4b", healthy)
        self.assertTrue(ok, f"healthy flux layout rejected: {reason}")

    def test_fall_through_finds_the_complete_snapshot(self):
        hub = Path.home() / ".cache" / "huggingface" / "hub"
        snapshots = hub / "models--mlx-community--flux2-klein-4b-4bit" / "snapshots"
        complete = [
            c for c in snapshots.iterdir() if c.is_dir()
            and any((c / "transformer").glob("*.safetensors"))
        ] if snapshots.is_dir() else []
        if not complete:
            self.skipTest("no complete mlx-community 4B snapshot in this HF cache")
        corpse = _corpse(self.tmp / "corpse-snapshot" / "transformer")
        self._override(corpse.parent)
        result = generator.resolve_local_model_path("flux2-klein-4b")
        self.assertIn(
            Path(result), complete,
            "fell through, but not onto real weights",
        )

    def test_healthy_override_is_returned_untouched(self):
        healthy = _corpse(self.tmp / "good-model")
        (healthy / "weights.safetensors").write_bytes(b"")
        self._override(healthy)
        result = generator.resolve_local_model_path("flux2-klein-4b")
        self.assertEqual(Path(result), healthy)


if __name__ == "__main__":
    unittest.main()
