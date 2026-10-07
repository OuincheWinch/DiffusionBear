"""The gallery detail panel must contain its own contents.

Reported as: the metadata panel overflowed sideways and the right-hand buttons
(Finder, Upscale 4x, Export) bled out of the dark container over the app
background and the window scrollbar. The same complaint had a second half --
opening a card showed the image too small to judge, even though the file was large.

These are static guards on the stylesheet, in the spirit of test_drag_contract.py:
they assert the properties that produce containment, so a well-meaning later edit
that reintroduces the bug fails here instead of in front of a user.
"""
import re
import unittest
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "App.css"
SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"


def rule(css: str, selector: str) -> str:
    """The body of one top-level rule, braces balanced, comments stripped.

    Comments are removed first because these rules EXPLAIN the old values ("not the old
    min-width: 260px"), and a naive search finds the prose in the comment instead of the
    declaration. Stripping keeps the assertions honest about what the browser sees."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    match = re.search(rf"(?m)^{re.escape(selector)}\s*\{{", css)
    assert match, f"rule not found: {selector}"
    depth, i = 1, match.end()
    while depth and i < len(css):
        if css[i] == "{":
            depth += 1
        elif css[i] == "}":
            depth -= 1
        i += 1
    return css[match.end(): i - 1]


class DetailPanelContainmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.css = CSS.read_text(encoding="utf-8")
        # Later rules win, so every lookup has to see the final declaration.
        cls.last = {}
        for match in re.finditer(r"(?m)^([.#][A-Za-z0-9_.\- >:[\]()-]+?)\s*\{([^}]*)\}", cls.css):
            cls.last.setdefault(match.group(1).strip(), match.group(2))

    def body(self, selector: str) -> str:
        return rule(self.css, selector)

    def body_of_group(self, first: str, last: str) -> str:
        """A rule whose selector spans several lines, e.g.

            .detail-actions button,
            .detail-actions a.btn,
            .tag-editor button {
        """
        css = re.sub(r"/\*.*?\*/", "", self.css, flags=re.S)
        match = re.search(
            rf"(?m)^{re.escape(first)}\s*\n(?:[^\{{]*\n)*?{re.escape(last)}\s*\{{", css
        )
        self.assertIsNotNone(match, f"grouped rule not found: {first} .. {last}")
        depth, i = 1, match.end()
        while depth and i < len(css):
            if css[i] == "{":
                depth += 1
            elif css[i] == "}":
                depth -= 1
            i += 1
        return css[match.end(): i - 1]

    # ------------------------------------------------------------- the panel
    def test_the_panel_is_bordered_box(self):
        self.assertRegex(
            self.body(".modal-body"), r"box-sizing:\s*border-box",
            "the global * rule exists, but the panel must not depend on it",
        )

    def test_the_panel_uses_two_bounded_tracks(self):
        """Not flex, and not a bare 1fr.

        A `1fr` grid track has an automatic MINIMUM of min-content, so one long
        unbroken value in the metadata widened the track and dragged the panel off
        screen. minmax(0, 1fr) is the load-bearing part."""
        body = self.body(".modal-body")
        self.assertRegex(body, r"display:\s*grid")
        self.assertRegex(body, r"minmax\(0,\s*1fr\)")
        # "0, 1fr" inside minmax() would match any naive bare-1fr test, so remove the
        # bounded forms first and require that no 1fr track is left unbounded.
        tracks = re.search(r"grid-template-columns:\s*([^;]+);", body).group(1)
        leftover = tracks.replace("minmax(0, 1fr)", "")
        self.assertNotIn("1fr", leftover, f"unbounded 1fr track in {tracks.strip()!r}")

    def test_the_panel_clips_what_escapes(self):
        """Every other rule here makes a child shrinkable. This is the one that
        actually guarantees nothing paints over the app background."""
        self.assertRegex(self.body(".modal-body"), r"overflow:\s*hidden")

    # ------------------------------------------------------------- the image
    def test_the_detail_image_is_bounded(self):
        """It previously had NO size rule at all.

        The only rule that matched was `.modal-body > img`, and the <img> sits inside
        .gallery-detail-frame, so it never applied -- it rendered at intrinsic size
        (4000x3000 in testing) and was itself a major source of the overflow."""
        body = self.body(".gallery-detail-img")
        self.assertRegex(body, r"max-width:\s*100%")
        self.assertRegex(body, r"max-height:\s*\d")
        self.assertRegex(body, r"object-fit:\s*contain",
                         "a small image must not be stretched to fill the column")

    def test_no_rule_is_aimed_at_a_selector_the_markup_does_not_have(self):
        """The bug in one line: `.modal-body > img` cannot match anything.

        Asserted so the next person does not re-add a selector by eye without
        checking the DOM."""
        gallery = (SRC / "components" / "Gallery.jsx").read_text(encoding="utf-8")
        detail_block = gallery[gallery.index('className="modal-body"'):]
        frame_at = detail_block.index('className="gallery-detail-frame"')
        img_at = detail_block.index("gallery-detail-img")
        self.assertLess(frame_at, img_at, "the <img> is nested inside the frame, not a direct child")

    def test_the_image_is_allowed_to_grow(self):
        """The second half of the report: a large image should open large."""
        self.assertRegex(self.body(".gallery-detail-frame"), r"max-width:\s*100%")
        self.assertRegex(self.body(".gallery-detail-frame"), r"justify-self:\s*center")

    # ------------------------------------------------------------- the metadata column
    def test_the_metadata_column_can_shrink(self):
        """min-width: 0, not min-width: 260px.

        A minimum sets a FLOOR; it never grants permission to shrink, so long
        content could not be contained."""
        body = self.body(".detail")
        self.assertRegex(body, r"min-width:\s*0")
        self.assertNotRegex(body, r"min-width:\s*2\d\dpx")

    def test_the_metadata_column_clips_and_wraps(self):
        body = self.body(".detail")
        self.assertRegex(body, r"overflow-x:\s*hidden")
        self.assertRegex(body, r"overflow-wrap:\s*break-word")

    def test_the_value_track_cannot_be_widened_by_its_content(self):
        self.assertRegex(
            self.body("dl"), r"grid-template-columns:[^;]*minmax\(0,\s*1fr\)",
            "the 1fr track's automatic min-content minimum was the actual bleed",
        )

    def test_long_single_tokens_can_break(self):
        """Model ids, sampler lists and file paths have no spaces in them, so they need
        an explicit break opportunity or they push the track."""
        self.assertRegex(self.body("dd"), r"overflow-wrap:\s*break-word")
        self.assertRegex(self.body("dd"), r"word-break:\s*break-word")

    # ------------------------------------------------------------- the controls
    def test_action_buttons_wrap_and_stay_capped(self):
        actions = self.body(".detail-actions")
        self.assertRegex(actions, r"flex-wrap:\s*wrap")
        self.assertRegex(actions, r"min-width:\s*0")
        buttons = self.body_of_group(".detail-actions button,", ".tag-editor button")
        self.assertRegex(buttons, r"max-width:\s*100%")
        # Nine languages, several materially longer than English.
        self.assertRegex(buttons, r"white-space:\s*normal")
        self.assertRegex(buttons, r"overflow-wrap:\s*break-word")

    def test_the_tag_editor_can_give_up_space(self):
        """A second unbounded row: `flex: 1` leaves min-width at auto, which is the
        input's intrinsic width, and that pushed Save out of the panel."""
        self.assertRegex(self.body(".tag-editor"), r"flex-wrap:\s*wrap")
        self.assertRegex(self.body(".tag-editor label"), r"min-width:\s*0")
        self.assertRegex(self.body(".tag-editor input"), r"width:\s*100%")


class DetailPanelBalanceTests(unittest.TestCase):
    """A stray or missing brace here silently reverts every fix above."""

    def test_the_stylesheet_braces_balance(self):
        css = CSS.read_text(encoding="utf-8")
        self.assertEqual(css.count("{"), css.count("}"), "unbalanced braces in App.css")

    def test_the_detail_image_rule_is_not_shadowed(self):
        css = CSS.read_text(encoding="utf-8")
        base = re.findall(r"(?m)^\.gallery-detail-img\s*\{", css)
        self.assertEqual(len(base), 1, "a second bare .gallery-detail-img rule would fight the fix")
