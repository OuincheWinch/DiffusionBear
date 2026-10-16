"""Tabs must not render at half the available width.

Reported by the user as "Models and Parameters only use half of the screen". The cause was
a single line of CSS: `.parameters-tab { max-width: 760px }`, applied to ModelsTab,
ParametersTab and LicencesTab. The app container is 1400px, so those tabs drew at 55% of
the available width while the Generate tab -- which has no cap -- used all of it.

On the Models tab it was not cosmetic: a browse list and an inventory table were being
squeezed into a column measured for a settings form.

The cap exists now, as a regression test, because nothing about it looked like a bug. It
is the kind of value that gets written once, for one tab, and then inherited silently by
every tab that reuses the class.
"""

import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"
CSS = SRC / "App.css"

# The app shell. Anything a tab may reasonably exceed is derived from this.
APP_MAX_WIDTH = 1400
APP_PADDING = 32  # 1rem each side

# Widths below this leave a tab visibly under-filling the shell. Not zero: a form can
# legitimately be narrower than the shell, but then it is a deliberate choice, not an
# inherited constant measured for a different tab.
MIN_FRACTION = 0.8


def _css() -> str:
    """The stylesheet with comments removed.

    Comments carry the reasoning for these values, and several of them mention the very
    colours and widths being asserted -- so an unstripped file makes a guard match its
    own documentation.
    """
    return re.sub(r"/\*.*?\*/", "", CSS.read_text(encoding="utf-8"), flags=re.S)


def _merged(css: str) -> dict[str, dict]:
    """Every top-level selector with all its properties merged, in source order.

    This parses independently of `_rules` rather than merging that one's output.
    `_rules` is last-declaration-wins, so by the time its result reaches a merge there
    is only one declaration left to merge -- which is exactly the information needed
    here. The inventory sets `table-layout` in one block and `width` in another, and the
    first version of this helper reported the width missing because it had already been
    discarded upstream.
    """
    merged: dict[str, dict] = {}
    i, depth, buf = 0, 0, ""
    while i < len(css):
        ch = css[i]
        if ch == "{":
            selector = " ".join(buf.split())
            buf = ""
            if depth == 0 and selector and not selector.startswith("@"):
                j, d = i + 1, 1
                while j < len(css) and d:
                    if css[j] == "{":
                        d += 1
                    elif css[j] == "}":
                        d -= 1
                        if not d:
                            break
                    j += 1
                props: dict[str, str] = {}
                for decl in css[i + 1:j].split(";"):
                    if ":" in decl:
                        k, _, v = decl.partition(":")
                        props[k.strip()] = v.strip()
                for one in selector.split(","):
                    merged.setdefault(one.strip(), {}).update(props)
                i = j + 1
                continue
            depth += 1
        elif ch == "}":
            depth -= 1
            buf = ""
        elif depth == 0:
            buf += ch
        i += 1
    return merged


def _rules(css: str | None = None) -> dict[str, dict]:
    """Every TOP-LEVEL rule in the stylesheet, as {selector: {prop: value}}.

    Depth matters here. A naive regex pass also picks up the rules nested inside
    `@media` blocks, and because those come later they overwrite the base rule in the
    dict -- so `.parameters-tab { max-width: 760px }` was silently replaced by the
    media query's `{ grid-template-columns: repeat(2, ...) }`, which carries no
    max-width at all. The width test then read "no cap" from a stylesheet that had one.

    That was measured, not theorised: injecting the old 760px cap back into the file
    left the test passing. A guard that cannot fail is worse than no guard.
    """
    css = re.sub(r"/\*.*?\*/", "", CSS.read_text(encoding="utf-8"), flags=re.S)
    out: dict[str, dict] = {}
    if css is None:
        css = _css()
    i, depth, buf = 0, 0, ""
    while i < len(css):
        ch = css[i]
        if ch == "{":
            selector = " ".join(buf.split())
            buf = ""
            if depth == 0 and selector and not selector.startswith("@"):
                # Read this rule's body, tracking nesting so an inner brace (as in
                # a nested @media inside a rule) cannot end it early.
                j, d = i + 1, 1
                while j < len(css) and d:
                    if css[j] == "{":
                        d += 1
                    elif css[j] == "}":
                        d -= 1
                        if not d:
                            break
                    j += 1
                body = css[i + 1:j]
                props = {}
                for decl in body.split(";"):
                    if ":" in decl:
                        k, _, v = decl.partition(":")
                        props[k.strip()] = v.strip()
                for one in selector.split(","):
                    out[one.strip()] = props
                i = j + 1
                depth = 0
                continue
            depth += 1
        elif ch == "}":
            depth -= 1
            buf = ""
        elif depth == 0:
            buf += ch
        i += 1
    return out


class TabWidthTests(unittest.TestCase):
    def setUp(self):
        self.rules = _rules()

    def _max_width_px(self, selector: str) -> int | None:
        raw = self.rules.get(selector, {}).get("max-width")
        if raw is None or raw in ("none", "100%", "unset", "initial"):
            return None
        m = re.match(r"(\d+)px$", raw)
        return int(m.group(1)) if m else None

    def test_the_tab_classes_this_bug_lived_in_still_exist(self):
        """If these are renamed the test below is vacuous, so make that loud."""
        for selector in (".parameters-tab", ".models-tab"):
            self.assertIn(selector, self.rules, f"{selector} rule disappeared")

    def test_no_tab_is_capped_below_the_available_width(self):
        available = APP_MAX_WIDTH - APP_PADDING
        for selector in (".parameters-tab", ".models-tab"):
            with self.subTest(selector=selector):
                width = self._max_width_px(selector)
                # `max-width: none` parses to None, which is the state we want.
                self.assertTrue(
                    width is None or width >= available * MIN_FRACTION,
                    f"{selector} is capped at {width}px, which is "
                    f"{width / available:.0%} of the {available}px available"
                    if width is not None else "",
                )

    def test_models_is_not_grid_split(self):
        """A browse list and a wide table stay one readable column."""
        display = self.rules.get(".models-tab", {}).get("display")
        self.assertEqual(display, "flex", "models-tab is no longer a single column")


if __name__ == "__main__":
    unittest.main()

class ImportRowActionVisibilityTests(unittest.TestCase):
    """The Import button must be reachable without scrolling or widening the window.

    Reported as "the Import button is too far to the right, off the screen".

    Two causes, both the same flexbox trap:

    - `.import-name` is `white-space: nowrap`, and a flex item's default
      `min-width: auto` resolves to its MIN-CONTENT width -- the entire filename.
      So it refuses to shrink, `text-overflow: ellipsis` never engages, and the row
      grows past its container.
    - `.import-row-actions` had no rule at all, so nothing stopped the action from
      shrinking or being pushed out.

    Draw Things names LoRA files after the Civitai prompt, so these are long: the
    longest filename in this machine's folder is 210 characters, roughly 1512px,
    against about 1344px of available row. Measured from the real folder, not
    sampled.
    """

    def test_the_filename_can_shrink_so_the_ellipsis_engages(self):
        props = _rules().get(".import-name", {})
        self.assertEqual(props.get("min-width"), "0",
                         ".import-name keeps min-width:auto, so a long Draw Things "
                         "filename cannot shrink and the row overflows")
        self.assertEqual(props.get("white-space"), "nowrap")

    def test_the_action_column_cannot_shrink_or_be_pushed_out(self):
        props = _rules().get(".import-row-actions", {})
        self.assertTrue(props, ".import-row-actions has no rule at all")
        self.assertEqual(props.get("flex"), "0 0 auto",
                         "the action column must not shrink")

    def test_the_row_body_can_clip(self):
        props = _rules().get(".import-row-main", {})
        self.assertEqual(props.get("min-width"), "0")
        self.assertEqual(props.get("overflow"), "hidden",
                         ".import-row-main needs a clipping backstop")

    def test_the_browser_name_is_bounded_too(self):
        """Same trap, different row. The cap alone was doing the work."""
        props = _rules().get(".hf-browser-name", {})
        self.assertEqual(props.get("min-width"), "0")
        self.assertIn("min(", props.get("max-width", ""),
                      "the name cap should be responsive, not a fixed rem width")


class InventoryNameColumnTests(unittest.TestCase):
    """A model name you cannot read is a name you cannot match to a file.

    The inventory is a table of names, several of which run past 200 characters --
    model folders inherit Draw Things' habit of naming files after the prompt. The
    table scrolls sideways rather than hiding the name behind an ellipsis, and the
    name column gets the slack instead of a fixed share.
    """

    def test_the_table_is_wrapped_in_a_horizontal_scroller(self):
        src = (SRC / "components" / "InventoryTable.jsx").read_text(encoding="utf-8")
        self.assertIn("inventory-scroll", src,
                      "the inventory table has no horizontal scroller")

    def test_the_scroller_is_defined(self):
        props = _merged(_css()).get(
            ".inventory-scroll", {})
        self.assertEqual(props.get("overflow-x"), "auto",
                         "without overflow-x the wrapper does nothing")

    def test_the_name_column_is_not_fixed_width(self):
        """table-layout:fixed with equal columns would clip the name again.

        Read through the merged view: `.inventory-grid` is declared more than once, so
        `table-layout` and `width` live in different blocks and a last-rule-wins lookup
        reports one of them missing.
        """
        css = _merged(_css())
        grid = css.get(".inventory-grid", {})
        self.assertEqual(grid.get("table-layout"), "fixed")
        self.assertEqual(grid.get("width"), "100%")
        first = css.get(".inventory-grid td:first-child", {})
        self.assertEqual(first.get("width"), "auto",
                         "the name column must take the slack")

    def test_a_long_name_wraps_rather_than_truncating(self):
        """The override lives on a more specific selector than `.import-name`.

        Checking the base class alone would pass while the inventory still truncated,
        because the base class keeps its ellipsis for the browse list, where it is
        correct -- a repo name is short and a model folder name is not.
        """
        css = _merged(_css())
        block = css.get(".inventory-row td:first-child .import-name", {})
        self.assertEqual(block.get("white-space"), "normal")
        self.assertEqual(block.get("text-overflow"), "clip",
                         "an ellipsis on the model name hides which model it is")
        self.assertEqual(block.get("overflow-wrap"), "anywhere")


class NoSecondColumnTests(unittest.TestCase):
    """No tab may leave half the screen blank.

    The two- and three-column grid introduced to fill the width was worse than the
    problem it solved: a tab whose root is a single grid item puts all its content in
    the first column and leaves the rest empty. Reverted to one full-width column.
    """

    def test_the_tab_layout_is_a_single_column(self):
        props = _merged(_css()).get(
            ".parameters-tab", {})
        self.assertEqual(props.get("display"), "flex")
        self.assertNotIn("grid-template-columns", props)

    def test_no_media_query_makes_the_tab_multi_column(self):
        css = re.sub(r"/\*.*?\*/", "", CSS.read_text(encoding="utf-8"), flags=re.S)
        offenders = re.findall(
            r"@media[^{]*\{\s*\.parameters-tab\s*\{[^}]*grid-template-columns", css)
        self.assertEqual(offenders, [],
                         "a wide-screen rule still splits the tab into columns")
