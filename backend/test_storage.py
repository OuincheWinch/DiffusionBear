"""Tests for the storage manifest.

The properties that matter are the ones that could hurt someone if they broke:
a symlink must not be followed, a credential must never be read, and a model the
app actually uses must never be reported as unaccounted for.
"""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import state
import storage


class StorageScanTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data = self.root / "data"
        self.assets = self.root / "assets"
        for sub in ("models/known-model", "models/hf-cache", "generated", "lora_files",
                    "uploads", "lora_files/sdxl"):
            (self.assets / sub if sub.startswith("models") else self.data / sub).mkdir(
                parents=True, exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _patched_dirs(self):
        # build_report() reads app_settings.DATA_DIR / app_settings.ASSET_DIR at call
        # time, so those are the attributes to patch -- not names in this module.
        return (
            patch.object(storage.app_settings, "DATA_DIR", self.data),
            patch.object(storage.app_settings, "ASSET_DIR", self.assets),
            patch.object(storage.generator, "ASSET_DIR", self.assets),
            patch.object(storage, "LORA_FILES_DIR", self.data / "lora_files"),
            patch.object(storage, "SDXL_LORA_DIR", self.data / "lora_files" / "sdxl"),
            patch.object(storage, "UPLOADS_DIR", self.data / "uploads"),
            patch.object(storage, "SETTINGS_FILE", self.data / "settings.json"),
            patch.object(storage, "LORAS_FILE", self.data / "loras.json"),
            # _read_loras() reads state.LORAS_FILE, not the name imported here, and it
            # memoises on a signature -- so the memo has to go too or a previous real
            # read is returned.
            patch("state.LORAS_FILE", self.data / "loras.json"),
            patch("state._loras_signature", return_value=("test", 1)),
            # _read_loras() also asserts LORAS_FILE is inside state.DATA_DIR; without
            # this it raises and the registry silently reads as empty.
            patch("state.DATA_DIR", self.data),
        )

    def _report(self):
        patches = self._patched_dirs()
        for p in patches:
            p.start()
        state._loras_cache.clear()
        try:
            storage._CACHE._value = None
            storage._CACHE._at = 0.0
            return storage.get_report(force=True)
        finally:
            for p in reversed(patches):
                p.stop()

    def test_totals_add_up(self):
        (self.assets / "models/known-model/a.safetensors").write_bytes(b"x" * 1000)
        (self.data / "generated/img.png").write_bytes(b"y" * 500)
        (self.data / "generated/img_thumb.png").write_bytes(b"z" * 100)
        report = self._report()
        self.assertEqual(report["totals"]["by_category"]["models"], 1000)
        self.assertEqual(report["totals"]["by_category"]["gallery"], 600)
        self.assertEqual(
            report["totals"]["bytes"], sum(report["totals"]["by_category"].values()))

    def test_symlinks_are_not_followed(self):
        """A symlink out of the tree must not be walked, or this becomes a
        whole-filesystem size report."""
        secret = self.root / "outside.bin"
        secret.write_bytes(b"z" * 10_000_000)
        link_dir = self.assets / "models/linked"
        link_dir.mkdir()
        os.symlink(secret, link_dir / "big.bin")
        report = self._report()
        linked = next(m for m in report["models"] if m["name"] == "linked")
        self.assertEqual(linked["bytes"], 0, "symlink target must not be counted")
        self.assertLess(report["totals"]["by_category"]["models"], 10_000_000)

    def test_secret_contents_are_never_read(self):
        secret = self.data / "hf_token.txt"
        secret.write_text("hf_SECRET", encoding="utf-8")
        with patch("pathlib.Path.read_text", side_effect=AssertionError("read a secret")):
            report = self._report()
        entry = next(s for s in report["secrets"] if s["name"] == "hf_token.txt")
        self.assertTrue(entry["present"])
        self.assertEqual(entry["bytes"], len("hf_SECRET"))
        self.assertNotIn("hf_SECRET", json.dumps(report), "no secret value may appear anywhere")

    def test_internal_assets_are_not_unaccounted_for(self):
        for name in ("qwen2.5-0.5b-instruct-4bit", "taesdxl"):
            (self.assets / "models" / name).mkdir(parents=True, exist_ok=True)
            (self.assets / "models" / name / "w.bin").write_bytes(b"a" * 10)
        report = self._report()
        unrecognised = {m["name"] for m in report["unrecognised"]}
        for name in ("qwen2.5-0.5b-instruct-4bit", "taesdxl"):
            self.assertNotIn(name, unrecognised, f"{name} is loaded at runtime and must be accounted for")
        # The two dirs setUp creates are genuinely unreferenced, so they belong here.
        self.assertEqual(unrecognised, {"known-model", "hf-cache"})

    def test_nothing_is_ever_claimed_reclaimable(self):
        # This module does not delete. The flag must stay false so no future UI can
        # render a delete button off it.
        (self.assets / "models/leftover").mkdir(parents=True, exist_ok=True)
        (self.assets / "models/leftover/x.bin").write_bytes(b"b" * 100)
        report = self._report()
        for group in ("models", "unrecognised"):
            for entry in report[group]:
                self.assertFalse(entry.get("reclaimable", False))

    def test_lora_entries_with_no_file_are_counted(self):
        (self.data / "lora_files").mkdir(parents=True, exist_ok=True)
        (self.data / "loras.json").write_text(json.dumps([
            {"name": "present", "path": str(self.data / "lora_files/present.safetensors")},
            {"name": "gone", "path": str(self.data / "lora_files/gone.safetensors")},
        ]), encoding="utf-8")
        (self.data / "lora_files/present.safetensors").write_bytes(b"c" * 32)
        report = self._report()
        self.assertEqual(report["loras"]["files"], 1)
        self.assertEqual(report["loras"]["missing_entries"], 1)
        self.assertEqual(report["loras"]["missing_names"], ["gone"])

    def test_walk_is_bounded(self):
        # A runaway tree must not hang the scan.
        deep = self.assets / "models/deep"
        deep.mkdir(parents=True)
        for i in range(40):
            sub = deep / f"d{i}"
            sub.mkdir()
            (sub / "f.bin").write_bytes(b"d")
        report = self._report()
        self.assertIn("deep", {m["name"] for m in report["models"]})


class StorageEndpointTests(unittest.TestCase):
    def test_endpoint_returns_a_report(self):
        from fastapi.testclient import TestClient
        import main
        client = TestClient(main.app)
        response = client.get("/api/storage")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        for key in ("totals", "models", "loras", "gallery", "secrets", "config", "unrecognised"):
            self.assertIn(key, body)

    def test_no_delete_verb_exists(self):
        """A guard on intent: this router must stay read-only."""
        from fastapi.testclient import TestClient
        import main
        client = TestClient(main.app)
        for method, path in (
            ("post", "/api/storage/delete"),
            ("delete", "/api/storage"),
            ("post", "/api/storage/purge"),
        ):
            response = getattr(client, method)(path)
            self.assertGreaterEqual(response.status_code, 400, f"{method} {path} must not succeed")


if __name__ == "__main__":
    unittest.main()
