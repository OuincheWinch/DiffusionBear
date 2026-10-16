"""The `theme` setting: defaults, validation and round-trip.

Modelled directly on `language`, which already solves the same problem: `auto` means
follow the OS, and a picker overrides it. Reusing that convention means the theme
behaves like a setting the user has already met somewhere else in this app.

Why these tests exist rather than a manual check: a validator that accepts junk writes
junk into settings.json, and the frontend then falls back to something the user never
chose. `theme: "lignt"` failing silently to dark is exactly the bug worth a test.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app_settings
import unittest.mock

from routers.settings import SettingsUpdate


class ThemeSettingTests(unittest.TestCase):
    def setUp(self):
        # Both DATA_DIR and SETTINGS_FILE are patched, not just the file. Other tests
        # repoint DATA_DIR at their own temporary directory and delete it on teardown,
        # so a test that only patches SETTINGS_FILE later resolves a directory that no
        # longer exists and fails with FileNotFoundError from mkdtemp -- a failure about
        # someone else's cleanup, in a test about themes.
        self.dir = Path(tempfile.mkdtemp())
        self.file = self.dir / "settings.json"
        self.addCleanup(shutil.rmtree, self.dir, True)
        for attr, value in (("DATA_DIR", self.dir),
                            ("SETTINGS_FILE", self.dir / "settings.json")):
            patcher = unittest.mock.patch.object(app_settings, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        app_settings._load_cache.clear()
        self.addCleanup(app_settings._load_cache.clear)

    def test_the_default_is_auto(self):
        """`auto` means follow the OS. Defaulting to dark would ignore the setting
        on every machine whose owner prefers light, until they found the toggle."""
        self.assertEqual(app_settings.DEFAULTS["theme"], "auto")

    def test_the_three_legal_values_are_accepted(self):
        validate = app_settings._VALIDATORS["theme"]
        for value in ("auto", "light", "dark"):
            with self.subTest(value=value):
                self.assertTrue(validate(value))

    def test_anything_else_is_refused(self):
        validate = app_settings._VALIDATORS["theme"]
        for value in ("", "Dark", "DARK", "lignt", "system", "true", None, 1, ["dark"]):
            with self.subTest(value=value):
                self.assertFalse(validate(value),
                                 f"{value!r} was accepted as a theme")

    def test_it_round_trips_through_save_and_load(self):
        """The value the user picked is the value the app reads back."""
        app_settings.update_settings({"theme": "light"})
        self.assertEqual(app_settings.get_setting("theme"), "light")
        self.assertEqual(
            json.loads(self.file.read_text(encoding="utf-8"))["theme"], "light",
            "the value did not reach disk")
        app_settings.update_settings({"theme": "dark"})
        self.assertEqual(app_settings.get_setting("theme"), "dark")

    def test_an_invalid_theme_is_refused_rather_than_written(self):
        """A typo must not reach settings.json and silently become dark."""
        with self.assertRaises(Exception):
            app_settings.update_settings({"theme": "lignt"})

    def test_a_settings_file_without_the_key_loads_cleanly(self):
        self.file.write_text(json.dumps({"artist_name": "someone"}), encoding="utf-8")
        app_settings._load_cache.clear()
        settings = app_settings.get_settings()
        self.assertEqual(settings["theme"], "auto")
        self.assertEqual(settings["artist_name"], "someone")


if __name__ == "__main__":
    unittest.main()

class ThemeRouteTests(unittest.TestCase):
    """The HTTP surface, because the model whitelist is a separate gate from the
    validator.

    `SettingsUpdate` is `extra="forbid"` with an explicit field list. A key that exists
    in `DEFAULTS` and `_VALIDATORS` but not in that model is silently unreachable over
    HTTP: the POST returns 422 and nothing explains why. This happened -- the setting
    validated, persisted when called directly, and could not be set from the UI at all.
    """

    def setUp(self):
        from fastapi.testclient import TestClient

        import main

        self.client = TestClient(main.app)
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)
        for attr, value in (("DATA_DIR", self.dir),
                            ("SETTINGS_FILE", self.dir / "settings.json")):
            patcher = unittest.mock.patch.object(app_settings, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        app_settings._load_cache.clear()
        self.addCleanup(app_settings._load_cache.clear)

    def test_the_theme_key_is_reachable_over_http(self):
        self.assertIn("theme", SettingsUpdate.model_fields,
                      "the theme can validate but can never be set: the update model "
                      "does not list it")

    def test_it_round_trips(self):
        self.assertEqual(self.client.get("/api/settings").json()["theme"], "auto")
        self.assertEqual(
            self.client.post("/api/settings", json={"theme": "light"}).status_code, 200)
        self.assertEqual(self.client.get("/api/settings").json()["theme"], "light")

    def test_a_bad_value_is_refused_and_changes_nothing(self):
        self.client.post("/api/settings", json={"theme": "light"})
        for bad in ("Dark", "lignt", "system", ""):
            with self.subTest(value=bad):
                self.assertEqual(
                    self.client.post("/api/settings", json={"theme": bad}).status_code, 422)
                self.assertEqual(self.client.get("/api/settings").json()["theme"], "light",
                                 "a rejected value still changed the stored theme")
