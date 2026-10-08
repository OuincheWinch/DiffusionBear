"""The Generate-tab download panel must not break page layout while running.

Reported with screenshots: with a model download in progress, the Generate tab
showed the ModelInstaller panel squeezed into the model-picker flex row -- long
status text, MB counters and the local-install panel pushed the canvas into a
horizontal scrollbar and shifted layout on every file change.

Root causes, all in CSS:
- `.preset-row` was a non-wrapping flex row while `.model-installer` demanded
  `flex-basis: 100%`: guaranteed horizontal overflow.
- Flex children default to `min-width: auto`, so the installer, the select group
  and the path input refused to shrink below their content.
- `.model-installer-local` expanded inline, pushing everything below it.

Screen capture is forbidden, so these are asserted as CSS contracts, in the
style of test_error_banner.py.
"""

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CSS = REPO / "frontend" / "src" / "App.css"
INSTALLER = REPO / "frontend" / "src" / "components" / "ModelInstaller.jsx"


def _rule(css, selector):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    m = re.search(rf"{re.escape(selector)}\s*\{{([^}}]*)\}}", css)
    assert m, f"{selector} rule missing"
    return m.group(1)


class ModelInstallerLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.css = CSS.read_text(encoding="utf-8")
        cls.jsx = INSTALLER.read_text(encoding="utf-8")

    def test_picker_row_wraps_so_the_panel_gets_its_own_line(self):
        self.assertIn("flex-wrap: wrap", _rule(self.css, ".preset-row"))

    def test_installer_can_shrink_below_its_content(self):
        self.assertIn("min-width: 0", _rule(self.css, ".model-installer"))

    def test_select_group_can_shrink_below_its_content(self):
        self.assertIn("min-width: 0", _rule(self.css, ".model-select-group"))

    def test_status_line_keeps_ellipsis_truncation(self):
        rule = _rule(self.css, ".model-installer-text")
        self.assertIn("overflow: hidden", rule)
        self.assertIn("text-overflow: ellipsis", rule)
        self.assertIn("white-space: nowrap", rule)

    def test_status_line_exposes_full_text_on_hover(self):
        self.assertIn("title={statusText}", self.jsx)

    def test_local_panel_is_an_overlay_not_inline_expansion(self):
        rule = _rule(self.css, ".model-installer-local")
        self.assertIn("position: absolute", rule)

    def test_overlay_has_an_anchor(self):
        self.assertIn("position: relative", _rule(self.css, ".model-installer"))

    def test_path_input_cannot_force_overflow(self):
        rule = _rule(self.css, ".model-installer-select,\n.model-installer-path")
        self.assertIn("min-width: 0", rule)
        self.assertIn("max-width: 100%", rule)


if __name__ == "__main__":
    unittest.main()
