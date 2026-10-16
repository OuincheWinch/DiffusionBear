"""Progress bars must not animate layout properties.

Found by the impeccable detector's `layout-transition` rule (4 warnings): transitioning
`width` forces a layout pass on every progress tick, which is exactly the jank a progress
bar is most noticeable for. The three fills now paint full width and reveal it with
scaleX(), which runs on the compositor.

The subtlety this file exists to pin down: the fill used by the generative-fill brush has
TWO states sharing one element, and only one of them may own `transform`.
"""
import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"
CSS = SRC / "App.css"

FILLS = [
    ".civitai-progress-fill",
    ".progress-fill",
    ".fill-brush-progress-fill",
]


def rule(selector: str) -> str:
    css = re.sub(r"/\*.*?\*/", "", CSS.read_text(encoding="utf-8"), flags=re.S)
    m = re.search(rf"(?m)^{re.escape(selector)}\s*\{{", css)
    assert m, f"rule not found: {selector}"
    depth, i = 1, m.end()
    while depth and i < len(css):
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
        i += 1
    return css[m.end(): i - 1]


class NoLayoutTransitionsTests(unittest.TestCase):
    def test_nothing_transitions_width_anywhere(self):
        css = re.sub(r"/\*.*?\*/", "", CSS.read_text(encoding="utf-8"), flags=re.S)
        for m in re.finditer(r"transition:\s*([^;]+);", css):
            self.assertNotRegex(
                m.group(1), r"\bwidth\b",
                f"App.css transitions a layout property: transition: {m.group(1).strip()[:60]}",
            )

    def test_nothing_transitions_padding_anywhere(self):
        css = re.sub(r"/\*.*?\*/", "", CSS.read_text(encoding="utf-8"), flags=re.S)
        for m in re.finditer(r"transition:\s*([^;]+);", css):
            self.assertNotRegex(m.group(1), r"\bpadding\b", f"transition: {m.group(1).strip()[:60]}")

    def test_each_fill_paints_full_width_and_uses_transform(self):
        for sel in FILLS:
            with self.subTest(selector=sel):
                body = rule(sel)
                self.assertRegex(body, r"width:\s*100%")
                self.assertRegex(body, r"transition:\s*transform\b")
                self.assertNotRegex(body, r"transition:[^;]*\bwidth\b")

    def test_each_fill_grows_from_the_left(self):
        """Without transform-origin the bar would scale about its centre and appear to
        start half-way along, which is wrong for a progress indicator."""
        for sel in FILLS:
            with self.subTest(selector=sel):
                self.assertRegex(rule(sel), r"transform-origin:\s*left center")


class ScaleXCallSitesTests(unittest.TestCase):
    """Every width-producing call site must emit scaleX, or the bar renders at full width
    (scaleX absent) or not at all (width now 100% in CSS)."""

    SITES = [
        ("GenerationStack.jsx", "scaleX(${pct / 100})"),
        ("ResultCanvas.jsx", "scaleX(${Math.min(100, (progress.step / progress.steps) * 100) / 100})"),
        ("ModelInstaller.jsx", 'scaleX(${(task?.status === "done" ? 100 : pct) / 100})'),
        ("UniversalDownloader.jsx", "scaleX(${Math.round((dl.progress || 0) * 100) / 100})"),
    ]

    def test_each_site_uses_scale_x(self):
        for name, fragment in self.SITES:
            with self.subTest(component=name):
                text = (SRC / "components" / name).read_text(encoding="utf-8")
                self.assertIn("scaleX(", text, f"{name} must reveal the bar with scaleX")
                self.assertIn(fragment, text)

    # A percentage width specifically. A bare `width:` match was far too loose: it fired on
    # an unrelated width prop belonging to a later element on the same line range, which
    # sent me looking for a bug that did not exist.
    PCT_WIDTH = re.compile(r'width:\s*(?:\$\{)?`?[^,}\n]{0,40}%')

    def test_no_fill_call_site_still_sets_an_inline_width_percentage(self):
        for path in SRC.rglob("*.jsx"):
            text = path.read_text(encoding="utf-8")
            for cls in ("progress-fill", "civitai-progress-fill", "fill-brush-progress-fill"):
                for m in re.finditer(rf'className="[^"]*{cls}[^"]*"[\s\S]{{0,240}}?', text):
                    hit = self.PCT_WIDTH.search(m.group(0))
                    with self.subTest(file=path.name, cls=cls):
                        if hit:
                            self.fail(f"{path.name}: {cls} still sets width {hit.group(0).strip()[:50]}")


class IndeterminateStateTests(unittest.TestCase):
    """The one place width must stay.

    The fill brush shows an indeterminate bar whose keyframes animate
    transform: translateX(). One element cannot carry both a scaleX and a translateX --
    whichever is applied last wins -- so the indeterminate state keeps width and only the
    determinate state uses scaleX.
    """

    def test_the_keyframes_use_translate(self):
        css = CSS.read_text(encoding="utf-8")
        m = re.search(r"@keyframes fill-indeterminate\s*\{([^}]*\}[^}]*)\}", css)
        self.assertIsNotNone(m, "fill-indeterminate keyframes missing")
        self.assertIn("translateX", m.group(1))

    def test_the_brush_keeps_width_for_the_indeterminate_branch(self):
        text = (SRC / "components" / "FillBrush.jsx").read_text(encoding="utf-8")
        self.assertIn("scaleX(", text)
        self.assertRegex(text, r':\s*\{\s*width:\s*"100%"\s*\}', "the indeterminate branch must keep width")

    def test_the_two_states_are_not_fused(self):
        """Guards against someone 'simplifying' this into a single transform channel."""
        text = (SRC / "components" / "FillBrush.jsx").read_text(encoding="utf-8")
        self.assertRegex(text, r"\.\.\.\(\s*progress\.steps", "the two states must stay a conditional")
