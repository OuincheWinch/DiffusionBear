"""The window must open wide enough for the aspect-ratio row to fit on one line.

Reported from the Generate tab: the nine ratio buttons wrapped onto a second line.
Measured across viewport widths, the two-column layout engages at >=1024px and squeezes
the parameter column to 450-570px, while the nine chips need about 625px. So the row was
WRAPPED from 1024 to 1280 and only fitted from ~1366 up -- and the window's default width
was 1280, i.e. exactly the failing case.
"""
import re
import unittest
from pathlib import Path

SWIFT = Path(__file__).resolve().parents[1] / "packaging" / "shell" / "main.swift"
CSS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "App.css"
COMPONENT = Path(__file__).resolve().parents[1] / "frontend" / "src" / "components" / "SizeSelector.jsx"


class WindowWidthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.swift = SWIFT.read_text(encoding="utf-8")
        cls.css = CSS.read_text(encoding="utf-8")
        cls.component = COMPONENT.read_text(encoding="utf-8")

    def test_the_default_width_clears_the_wrap_point(self):
        """The measured wrap band is 1024-1280; one line from ~1366."""
        match = re.search(r"min\(CGFloat\((\d+)\),\s*screenFrame\.width\)", self.swift)
        self.assertIsNotNone(match, "the content width must be clamped to the screen")
        self.assertGreaterEqual(
            int(match.group(1)), 1366,
            "below ~1366 the nine ratio buttons wrap onto a second line",
        )

    def test_the_width_is_clamped_to_the_screen(self):
        """A 1400pt default must not open off the edge of a 1280pt display."""
        self.assertIn("NSScreen.main?.visibleFrame", self.swift)
        self.assertRegex(self.swift, r"min\(CGFloat\(1400\), screenFrame\.width\)")

    def test_the_window_constructor_no_longer_hardcodes_1280(self):
        """Scoped to the NSWindow call on purpose.

        Two other 1280x840 literals remain and are harmless: the AppWebView and
        DropHostView are constructed with placeholder frames and immediately given the
        real bounds by `webView.frame = content.bounds`. Asserting on the string
        anywhere in the file would fail on those and train us to ignore the test."""
        call = re.search(r"window = NSWindow\((.*?)\)\s*\n", self.swift, re.S)
        self.assertIsNotNone(call, "NSWindow construction not found")
        self.assertNotIn("1280", call.group(1), "the window default is the width that wrapped")

    def test_the_placeholder_view_frames_are_overwritten(self):
        """Documented above: harmless, but assert it so nobody removes the reassignment."""
        self.assertIn("webView.frame = content.bounds", self.swift)

    def test_the_window_is_still_resizable(self):
        """Resizing down must still work; it degrades to a wrapped row, not a broken one."""
        self.assertIn(".resizable", self.swift)

    def test_the_ratio_row_can_wrap_when_it_has_to(self):
        """Narrow windows are legitimate. The row must degrade gracefully rather than
        overflow, so flex-wrap stays on the row."""
        # Selector order is base-then-ratio in the stylesheet, so match either way round.
        rule = re.search(
            r"\.size-(?:base|ratio)-row,\s*\.size-(?:base|ratio)-row\s*\{([^}]*)\}", self.css
        )
        self.assertIsNotNone(rule)
        self.assertIn("flex-wrap: wrap", rule.group(1))
