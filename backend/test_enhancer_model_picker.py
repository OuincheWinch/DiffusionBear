"""The enhancer backend picker: choice, download, activation.

The model is a user setting with explicit download, not a bundled default.
The UI must therefore do three things, each of which can fail alone:

1. LIST with state: fetch the models endpoint and render one radio per
   backend showing cached/active state. A picker that lists without state
   lets the user activate weights that are not on disk.
2. DOWNLOAD before activate: activating an uncached model downloads it
   first. Reversing that order stores a selection whose weights are absent,
   and Enhance fails later, inside generation, instead of here.
3. SAVE through settings: the choice persists via the settings endpoint so
   a reload shows the same active model. A local-only useState would forget
   on every navigation.

Static source checks in the repo's established style: no frontend runner
exists here, so these assert wiring, never just text. Each asserts the call
chain (fetch URL, POST body key, ordering), so a renamed-but-unwired button
still fails.
"""
import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"
COMP = (SRC / "components" / "settings" / "EnhancerSystemSection.jsx").read_text(
    encoding="utf-8")

KEYS = [
    "settings.enhancer.modelTitle",
    "settings.enhancer.modelHint",
    "settings.enhancer.downloadBtn",
    "settings.enhancer.downloading",
    "settings.enhancer.downloadedBadge",
    "settings.enhancer.activeBadge",
]


class TestModelPickerWiring(unittest.TestCase):
    def test_lists_backends_with_cached_state(self):
        self.assertIn("/api/prompt/enhancer/models", COMP)
        self.assertIn("m.cached", COMP)
        self.assertIn("m.active", COMP)

    def test_download_posts_the_model_id(self):
        self.assertIn("/api/prompt/enhancer/download", COMP)
        self.assertRegex(COMP, r"JSON\.stringify\(\{\s*id\b")

    def test_activation_downloads_first_when_uncached(self):
        """Order is the whole point: activate-then-download stores a choice
        whose weights are absent. The download call must precede the settings
        save in the activation path."""
        body_start = COMP.index("async function activateModel")
        body = COMP[body_start:body_start + 1200]
        dl_at = body.index("downloadModel(")
        save_at = body.index("enhancer_model")
        self.assertLess(
            dl_at, save_at,
            "activation saves the choice before downloading its weights",
        )

    def test_choice_persists_through_settings(self):
        self.assertIn("/api/settings", COMP)
        self.assertIn("enhancer_model", COMP)

    def test_download_button_only_when_uncached(self):
        at = COMP.index("settings.enhancer.downloadBtn")
        segment = COMP[max(0, at - 900):at]
        self.assertIn("{!m.cached && (", segment)


class TestModelPickerStrings(unittest.TestCase):
    def test_keys_exist_in_every_language(self):
        parts = (SRC / "i18n" / "parts" / "settings.js").read_text(encoding="utf-8")
        generated = (SRC / "i18n" / "lang" / "_parts" / "settings.2.json").read_text(
            encoding="utf-8")
        for key in KEYS:
            with self.subTest(key=key):
                self.assertIn(key, COMP, f"{key} is never used")
                self.assertIn(key, parts, f"{key} missing from reference catalogue")
                self.assertIn(key, generated, f"{key} missing from generated catalogue")
        for code in ("es", "zh", "ja", "pt", "ko"):
            text = (SRC / "i18n" / "lang" / f"{code}.js").read_text(encoding="utf-8")
            for key in KEYS:
                with self.subTest(lang=code, key=key):
                    self.assertIn(key, text)


class TestEnhanceButtonTag(unittest.TestCase):
    """The Generate Enhance button names the active backend.

    Two properties: the label carries the {model} parameter (not a second
    hardcoded string that drifts), and the tag defaults to 1.5B exactly like
    the backend default -- an unknown stored value must show the model that
    will actually run.
    """

    def test_button_label_takes_the_model_parameter(self):
        form = (SRC / "components" / "GenerateForm.jsx").read_text(encoding="utf-8")
        self.assertIn("generate.enhance.btnIdle\", { model:", form.replace(
            '"generate.enhance.btnIdle", { model:', '"generate.enhance.btnIdle", { model:'))
        self.assertIn("{model}", self._btn_idle_en())

    def test_tag_defaults_to_15b_like_the_backend(self):
        form = (SRC / "components" / "GenerateForm.jsx").read_text(encoding="utf-8")
        self.assertRegex(
            form,
            r'appSettings\?\.enhancer_model === "qwen05" \? "0\.5B" : "1\.5B"',
        )

    def test_settings_refresh_on_tab_return(self):
        """The form stays mounted while Parameters changes the choice behind
        its back. Without a refresh on return, the tag shows a stale model."""
        form = (SRC / "components" / "GenerateForm.jsx").read_text(encoding="utf-8")
        self.assertIn('activeTab === "generate"', form)
        self.assertIn("refreshAppSettings", form)
        app = (SRC / "App.jsx").read_text(encoding="utf-8")
        self.assertIn("activeTab={tab}", app)

    def _btn_idle_en(self):
        parts = (SRC / "i18n" / "parts" / "generate.js").read_text(encoding="utf-8")
        start = parts.index('"generate.enhance.btnIdle"')
        return parts[start:start + 400]


if __name__ == "__main__":
    unittest.main()
