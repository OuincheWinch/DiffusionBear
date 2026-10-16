import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import app_settings
import state


class SettingsMemoTests(unittest.TestCase):
    """app_settings._load() memoises on (mtime_ns, size, resolved path)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.previous_file = app_settings.SETTINGS_FILE
        self.previous_dir = app_settings.DATA_DIR
        app_settings.SETTINGS_FILE = self.root / "settings.json"
        app_settings.DATA_DIR = self.root
        app_settings._load_cache.clear()

    def tearDown(self):
        app_settings.SETTINGS_FILE = self.previous_file
        app_settings.DATA_DIR = self.previous_dir
        app_settings._load_cache.clear()
        self.temp.cleanup()

    def _write(self, payload: dict):
        app_settings.SETTINGS_FILE.write_text(json.dumps(payload), encoding="utf-8")

    def test_missing_file_yields_defaults(self):
        self.assertEqual(app_settings.get_settings()["artist_name"], app_settings.DEFAULTS["artist_name"])

    def test_returns_independent_copies(self):
        self._write({"artist_name": "First"})
        first = app_settings.get_settings()
        first["artist_name"] = "Mutated"
        first["model_defaults"]["flux2-klein-4b"] = {"steps": 99}
        second = app_settings.get_settings()
        self.assertEqual(second["artist_name"], "First")
        self.assertNotIn("flux2-klein-4b", second["model_defaults"])

    def test_picks_up_external_write(self):
        self._write({"artist_name": "First"})
        self.assertEqual(app_settings.get_settings()["artist_name"], "First")
        self._write({"artist_name": "Second"})
        self.assertEqual(app_settings.get_settings()["artist_name"], "Second")

    def test_picks_up_deletion(self):
        self._write({"artist_name": "First"})
        self.assertEqual(app_settings.get_settings()["artist_name"], "First")
        app_settings.SETTINGS_FILE.unlink()
        self.assertEqual(app_settings.get_settings()["artist_name"], app_settings.DEFAULTS["artist_name"])

    def test_update_settings_visible_immediately(self):
        self._write({"artist_name": "First"})
        self.assertEqual(app_settings.get_settings()["artist_name"], "First")
        updated = app_settings.update_settings({"artist_name": "Written"})
        self.assertEqual(updated["artist_name"], "Written")
        self.assertEqual(app_settings.get_settings()["artist_name"], "Written")
        reloaded = json.loads(app_settings.SETTINGS_FILE.read_text(encoding="utf-8"))
        self.assertEqual(reloaded["artist_name"], "Written")

    def test_invalid_values_still_rejected(self):
        self._write({"artist_name": 12345})
        self.assertEqual(app_settings.get_settings()["artist_name"], app_settings.DEFAULTS["artist_name"])

    def test_second_read_does_not_touch_the_file(self):
        self._write({"artist_name": "First"})
        self.assertEqual(app_settings.get_settings()["artist_name"], "First")
        with patch.object(app_settings.Path, "read_text", side_effect=AssertionError("re-read on cache hit")):
            self.assertEqual(app_settings.get_settings()["artist_name"], "First")


class LoraRegistryMemoTests(unittest.TestCase):
    """state._read_loras() memoises on (mtime_ns, size, resolved path)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.previous_file = state.LORAS_FILE
        self.previous_dir = state.DATA_DIR
        state.LORAS_FILE = self.root / "loras.json"
        state.DATA_DIR = self.root
        state._loras_cache.clear()

    def tearDown(self):
        state.LORAS_FILE = self.previous_file
        state.DATA_DIR = self.previous_dir
        state._loras_cache.clear()
        self.temp.cleanup()

    def _write(self, entries: list[dict]):
        state.LORAS_FILE.write_text(json.dumps(entries), encoding="utf-8")

    def test_missing_file_yields_empty(self):
        self.assertEqual(state._read_loras(), [])

    def test_returns_independent_copies(self):
        self._write([{"name": "alpha", "path": "/tmp/alpha.safetensors"}])
        first = state._read_loras()
        first[0]["name"] = "mutated"
        self.assertEqual(state._read_loras()[0]["name"], "alpha")

    def test_picks_up_external_write(self):
        self._write([{"name": "alpha", "path": "/tmp/alpha.safetensors"}])
        self.assertEqual(len(state._read_loras()), 1)
        self._write([
            {"name": "alpha", "path": "/tmp/alpha.safetensors"},
            {"name": "beta", "path": "/tmp/beta.safetensors"},
        ])
        self.assertEqual(len(state._read_loras()), 2)

    def test_picks_up_deletion(self):
        self._write([{"name": "alpha", "path": "/tmp/alpha.safetensors"}])
        self.assertEqual(len(state._read_loras()), 1)
        state.LORAS_FILE.unlink()
        self.assertEqual(state._read_loras(), [])

    def test_write_then_read_reflects_new_entry(self):
        self._write([{"name": "alpha", "path": "/tmp/alpha.safetensors"}])
        state._upsert_lora_entries([{"name": "beta", "path": "/tmp/beta.safetensors"}])
        names = {entry["name"] for entry in state._read_loras()}
        self.assertEqual(names, {"alpha", "beta"})

    def test_oversized_registry_still_raises(self):
        state.LORAS_FILE.write_text("[]", encoding="utf-8")
        original = state.MAX_RUNTIME_JSON_BYTES
        state.MAX_RUNTIME_JSON_BYTES = 1
        try:
            with self.assertRaises(ValueError):
                state._read_loras()
        finally:
            state.MAX_RUNTIME_JSON_BYTES = original


if __name__ == "__main__":
    unittest.main()
