"""Contrast guard for the app's design tokens.

WCAG 1.4.3 (4.5:1 for normal text) and 1.4.11 (3:1 for control boundaries and
non-text graphics) are not observable by eye in a screenshot, so they are measured
here instead. Four pairs were genuinely failing when this file was written:

    --accent      #7c5cff  4.29:1 on --bg, 3.88:1 on --panel   (needs 4.5:1 as text)
    primary fill  #3b82f6  3.68:1 behind a white 15px label   (needs 4.5:1)
    checkbox border #4b5563 2.23:1 against --panel            (needs 3:1)

The fix was not to nudge the old values but to separate the two jobs a colour was
doing: --accent stays the brand hue for borders and fills, and --accent-text is the
same hue light enough to carry text. --primary replaces #3b82f6 for button fills.

If a token is changed, this test decides whether the change is legal.
"""

import re
import unittest
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "App.css"
FORM_CSS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "settings-form.css"


def _linear(channel: float) -> float:
    c = channel / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def tokens() -> dict:
    """Read the :root custom properties out of App.css."""
    text = CSS.read_text(encoding="utf-8")
    root = text[text.index(":root {") : text.index("\n}", text.index(":root {"))]
    return dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6})", root))


class ContrastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = tokens()
        cls.bg = cls.t["bg"]
        cls.panel = cls.t["panel"]

    def test_tokens_parsed(self):
        for name in ("bg", "panel", "text", "muted", "accent", "accent-text", "primary"):
            self.assertIn(name, self.t, f"--{name} missing from :root")

    def test_body_text(self):
        for label, surface in (("bg", self.bg), ("panel", self.panel)):
            with self.subTest(surface=label):
                self.assertGreaterEqual(contrast(self.t["text"], surface), 4.5)

    def test_muted_text(self):
        # --muted carries every hint, label and unit in the app, so it is text
        # and not a decoration. It passed at 5.52:1 but is close enough to the
        # limit to be worth pinning.
        for label, surface in (("bg", self.bg), ("panel", self.panel)):
            with self.subTest(surface=label):
                self.assertGreaterEqual(contrast(self.t["muted"], surface), 4.5)

    def test_accent_is_not_used_as_text(self):
        """--accent is a non-text accent. If it is ever safe as text, split it."""
        for label, surface in (("bg", self.bg), ("panel", self.panel)):
            with self.subTest(surface=label):
                self.assertLess(
                    contrast(self.t["accent"], surface),
                    4.5,
                    "--accent now clears 4.5:1 as text; --accent-text can be retired",
                )

    def test_accent_text_is_legible(self):
        for label, surface in (("bg", self.bg), ("panel", self.panel)):
            with self.subTest(surface=label):
                self.assertGreaterEqual(contrast(self.t["accent-text"], surface), 4.5)

    def test_primary_button_label(self):
        self.assertGreaterEqual(contrast("#ffffff", self.t["primary"]), 4.5)
        self.assertGreaterEqual(contrast("#ffffff", self.t["primary-hover"]), 4.5)

    def test_status_text(self):
        for name in ("danger", "success"):
            with self.subTest(status=name):
                self.assertGreaterEqual(contrast(self.t[name], self.panel), 4.5)

    def test_control_boundaries(self):
        """3:1 for a control's edge against its own surface (1.4.11)."""
        for name, surface in (
            ("checkbox-border", self.panel),
            ("switch-knob", self.panel),
            ("focus-ring", self.bg),
            ("focus-ring", self.panel),
        ):
            with self.subTest(token=name, surface=surface):
                self.assertGreaterEqual(contrast(self.t[name], surface), 3.0)

    def test_tick_on_filled_control(self):
        """The checkmark sits on --primary, so it needs 3:1 against that fill."""
        self.assertGreaterEqual(contrast("#ffffff", self.t["primary"]), 3.0)


class NoHardcodedContrastColoursTests(unittest.TestCase):
    """The tokens exist so fills come from one place.

    A raw #3b82f6 in a component is how the failing button label shipped in the
    first place: the token was correct and the component was not.
    """

    FORBIDDEN_IN_FORM = ("3b82f6", "4b5563")

    def test_settings_form_uses_tokens(self):
        text = FORM_CSS.read_text(encoding="utf-8")
        for value in self.FORBIDDEN_IN_FORM:
            with self.subTest(colour=value):
                self.assertNotIn(
                    value, text, f"#{value} is hardcoded in settings-form.css; use a token"
                )


if __name__ == "__main__":
    unittest.main()
