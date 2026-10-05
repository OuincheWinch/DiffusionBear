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

    def test_the_preset_grid_wraps_instead_of_overflowing(self):
        """Narrow windows are legitimate; the control must degrade, not overflow.

        The row this used to guard (.size-base-row/.size-ratio-row) is gone: the
        control is now four orientation tabs over an auto-fit preset grid. The
        same failure mode survives in a new shape -- four tab labels sharing a
        450-570px parameter column -- so the invariant is asserted against what
        is actually in the stylesheet now.
        """
        # auto-fit + minmax wraps by construction: a preset drops to the next row
        # rather than pushing the container wider.
        grid = re.search(r"\.size-presets\s*\{([^}]*)\}", self.css)
        self.assertIsNotNone(grid, ".size-presets rule is missing")
        self.assertIn("repeat(auto-fit, minmax(", grid.group(1))
        self.assertIn("1fr", grid.group(1))

        # A long tab label must ellipsize, not stretch the row.
        tab = re.search(r"\.size-selector-tab\s*\{([^}]*)\}", self.css)
        self.assertIsNotNone(tab, ".size-selector-tab rule is missing")
        for prop in ("min-width: 0", "overflow: hidden", "text-overflow: ellipsis"):
            self.assertIn(prop, tab.group(1), f"tab must degrade gracefully: {prop}")

        # The tabs are a 4-column grid matching the four orientation options.
        tabs = re.search(r"\.size-selector-tabs\s*\{([^}]*)\}", self.css)
        self.assertIn("repeat(4, 1fr)", tabs.group(1))
        self.assertIn('grid-template-columns: repeat(4', tabs.group(1))

        # And the component really does render exactly those four tabs.
        self.assertIn("ORIENTATIONS.map", self.component)
