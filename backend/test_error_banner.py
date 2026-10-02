"""A generation failure must be impossible to miss.

Reported: the [METAL] ... Insufficient Memory error sat too deep in the UI to notice. It
rendered as a 0.85rem red paragraph at the BOTTOM of the Generate form, after the prompt,
the canvas controls and the advanced panels -- so after a two-minute wait that produced
nothing, the reason was the last thing on the page.
"""
import re
import unittest
from pathlib import Path

FORM = Path(__file__).resolve().parents[1] / "frontend" / "src" / "components" / "GenerateForm.jsx"
CSS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "App.css"


class ErrorBannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.form = FORM.read_text(encoding="utf-8")
        cls.css = CSS.read_text(encoding="utf-8")

    def _rule(self, selector: str) -> str:
        css = re.sub(r"/\*.*?\*/", "", self.css, flags=re.S)
        m = re.search(rf"(?m)^{re.escape(selector)}\s*\{{", css)
        self.assertIsNotNone(m, f"rule not found: {selector}")
        depth, i = 1, m.end()
        while depth and i < len(css):
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
            i += 1
        return css[m.end(): i - 1]

    def test_the_banner_is_the_first_child_of_the_form(self):
        """First, so it is above the prompt and the model picker."""
        form = re.sub(r"//[^\n]*", "", self.form)
        banner = form.index('className="error-banner"')
        toolbar = form.index('className="prompt-toolbar"')
        self.assertLess(banner, toolbar, "the banner must precede the prompt toolbar")

    def test_it_is_the_only_renderer_of_the_top_level_error(self):
        """The old bottom-of-form paragraph must be gone, or there are two errors."""
        self.assertNotIn('<p className="error" role="alert">\n          {error}', self.form)
        self.assertEqual(self.form.count('className="error-banner"'), 1)

    def test_the_nested_reference_image_error_is_left_alone(self):
        """It belongs next to the reference controls it refers to."""
        self.assertIn('className="error" role="alert"', self.form)

    def test_it_sticks_to_the_top(self):
        """The failure arrives after a long wait, by which point the user has scrolled
        away from the top of the form."""
        self.assertRegex(self._rule(".error-banner"), r"position:\s*sticky")
        self.assertRegex(self._rule(".error-banner"), r"top:\s*0")

    def test_it_is_announced_assertively(self):
        self.assertIn('role="alert"', self.form)
        self.assertIn('aria-live="assertive"', self.form)

    def test_the_message_cannot_widen_the_banner(self):
        """MLX errors carry long unbreakable tokens such as
        kiOGPUCommandBufferCallbackErrorOutOfMemory, which would otherwise push the form
        sideways -- the same class of bleed the gallery detail panel had."""
        body = self._rule(".error-banner-body")
        self.assertRegex(body, r"overflow-wrap:\s*break-word")
        self.assertRegex(body, r"white-space:\s*pre-wrap")

    def test_it_is_visually_a_banner_not_a_line_of_text(self):
        rule = self._rule(".error-banner")
        self.assertRegex(rule, r"border-left-width:\s*4px")
        self.assertRegex(rule, r"box-shadow:")
        self.assertRegex(rule, r"border-radius:")

    def test_it_can_be_dismissed(self):
        """An error you cannot clear stays on screen and trains people to ignore it."""
        self.assertIn("error-banner-dismiss", self.form)
        self.assertRegex(self.form, r'onClick=\{\(\) => setError\(null\)\}')

    def test_it_respects_reduced_motion(self):
        self.assertIn("@media (prefers-reduced-motion: reduce)", self.css)

    def test_the_heading_string_exists_in_every_language(self):
        """A hard-coded heading would ship untranslated in nine languages."""
        # The shipped catalogue lives in a per-component part file, not strings.js, which only
        # merges the parts.
        strings = (FORM.parents[1] / "i18n" / "parts" / "generate.js").read_text(encoding="utf-8")
        self.assertIn('"generate.error.title"', strings)
        self.assertIn("generate.error.title", self.form)
