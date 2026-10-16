"""Contrast guards for the light-mode readability pass.

Twelve failures were found by looking at a screenshot, measured from the stylesheet, and
fixed. Each ratio below is recomputed here from the token values in the CSS, so a later
edit that changes a colour is caught rather than re-discovered by a user in daylight.

`test_theme_css` already guarantees that no hardcoded colour survives on a surface. This
file is the other half: it guarantees the TOKENS are usable -- that a text colour
actually clears its background, in both themes.

The canvas-overlay numbers are bounds across pure-white and pure-black image content.
A scrim sits on a photograph, so the fix has to hold across the whole range rather than
for one sample, and that is what is asserted.
"""

import re
import unittest
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "App.css"

TEXT_MIN = 4.5   # WCAG 1.4.3 normal text
UI_MIN = 3.0     # WCAG 1.4.11 non-text boundaries


def _lum(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = f(r), f(g), f(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ratio(a: str, b: str) -> float:
    la, lb = _lum(a), _lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _over(fg: str, alpha: float, bg: str) -> str:
    """Composite `fg` at `alpha` over `bg` -- what the browser actually paints."""
    f = [int(fg.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(bg.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(round(f[i] * alpha + b[i] * (1 - alpha)) for i in range(3))


_CSS = re.sub(r"/\*.*?\*/", "", CSS.read_text(encoding="utf-8"), flags=re.S)
_TOKENS = {
    "dark": dict(re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);",
                            re.search(r"(?ms)^:root\s*\{(.*?)^\}", _CSS).group(1))),
    "light": dict(re.findall(r"(--[a-z0-9-]+)\s*:\s*([^;]+);",
                             re.search(r'(?ms)^:root\[data-theme="light"\]\s*\{(.*?)^\}', _CSS).group(1))),
}


def _token(theme, name):
    """A token's value in one theme.

    Module level rather than a method on one TestCase: the role tests below need it too,
    and a second class inheriting for one helper would inherit four unrelated tests.
    """
    table = _TOKENS[theme]
    assert name in table, f"{name} is not defined in the {theme} theme"
    return table[name].strip()


class ContrastCase(unittest.TestCase):
    def _tok(self, theme, name):
        return _token(theme, name)

    # --- the core tokens, both themes ---------------------------------------

    def test_body_text_clears_its_panel(self):
        for theme in ("dark", "light"):
            with self.subTest(theme=theme):
                self.assertGreaterEqual(_ratio(self._tok(theme, "--text"),
                                              self._tok(theme, "--panel")), TEXT_MIN)

    def test_muted_text_clears_its_panel(self):
        for theme in ("dark", "light"):
            with self.subTest(theme=theme):
                self.assertGreaterEqual(_ratio(self._tok(theme, "--muted"),
                                              self._tok(theme, "--panel")), TEXT_MIN)

    def test_accent_text_clears_its_panel(self):
        for theme in ("dark", "light"):
            with self.subTest(theme=theme):
                self.assertGreaterEqual(_ratio(self._tok(theme, "--accent-text"),
                                              self._tok(theme, "--panel")), TEXT_MIN)

    def test_no_rule_stacks_a_text_token_on_a_fill_token(self):
        """`.size-selector-tab.is-active` did exactly this: 1.60 dark / 1.50 light.

        --accent-text is a TEXT token and --accent is a FILL token; putting one on the
        other is the single worst contrast pair in the app and it is entirely silent,
        because the rule reads correctly -- both variables are called "accent".

        Asserted on RULES, not on the tokens. The two tokens are allowed to coexist in
        the stylesheet; what is forbidden is one rule using them together. An earlier
        version of this test compared the token values directly and failed for as long
        as both existed, which is not the same thing and would have forced a rename.
        """
        css = re.sub(r"/\*.*?\*/", "", CSS.read_text(encoding="utf-8"), flags=re.S)
        offenders = re.findall(
            r"\{[^{}]*background:\s*var\(--accent\)[^{}]*[\s;]color:\s*var\(--accent-text\)[^{}]*\}", css)
        offenders += re.findall(
            r"\{[^{}]*[\s;]color:\s*var\(--accent-text\)[^{}]*background:\s*var\(--accent\)[^{}]*\}", css)
        self.assertEqual(offenders, [], "a rule puts the TEXT token on the FILL token")

    # --- the scrim family ----------------------------------------------------

    def test_on_scrim_text_holds_across_the_whole_image_range(self):
        """A scrim sits on a photograph, so it must work on ANY image.

        Asserted against the extreme bounds rather than a sample: the scrim is
        composite, and the correct fix has to clear the bar for both a pure white and
        a pure black pixel, not for whatever image happened to be on screen.
        """
        scrim = self._tok("dark", "--scrim")
        m = re.search(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)", scrim)
        self.assertIsNotNone(m, "--scrim must be an rgba so it can be composited")
        rgb, alpha = "#%02x%02x%02x" % tuple(int(m.group(i)) for i in (1, 2, 3)), float(m.group(4))
        for theme in ("dark", "light"):
            for image in ("#ffffff", "#000000"):
                with self.subTest(theme=theme, image=image):
                    bg = _over(rgb, alpha, image)
                    self.assertGreaterEqual(
                        _ratio(self._tok(theme, "--on-scrim"), bg), TEXT_MIN,
                        "text on a scrim fails over one end of the image range")
                    self.assertGreaterEqual(
                        _ratio(self._tok(theme, "--on-scrim-muted"), bg), TEXT_MIN)

    def test_the_scrim_is_identical_in_both_themes(self):
        """If the scrim inverted, white text would land on a light scrim."""
        self.assertEqual(self._tok("dark", "--scrim"), self._tok("light", "--scrim"))

    # --- fills that carry white text ----------------------------------------

    def test_every_white_text_fill_is_theme_invariant(self):
        """--fill-* deliberately do not change with the theme.

        A fill that lightened while its label stayed white is how a hover state ends up
        at 1.67:1.
        """
        for name in ("--fill-accent", "--fill-brand-civitai", "--fill-brand-direct"):
            with self.subTest(token=name):
                self.assertEqual(self._tok("dark", name), self._tok("light", name),
                                 f"{name} must be identical in both themes")

    def test_white_on_the_invariant_fills_clears_text_minimum(self):
        for name in ("--fill-accent", "--fill-brand-civitai",
                     "--fill-brand-direct", "--fill-hover"):
            with self.subTest(token=name):
                self.assertGreaterEqual(
                    _ratio(self._tok("light", "--on-scrim"), self._tok("light", name)),
                    TEXT_MIN)

    def test_fills_do_not_invert_with_the_panel(self):
        """A fill must stay a fill: the same value cannot be --panel in light."""
        for theme in ("dark", "light"):
            for name in ("--fill-accent",):
                with self.subTest(theme=theme, token=name):
                    self.assertNotEqual(self._tok(theme, name),
                                        self._tok(theme, "--panel"),
                                        "a text-bearing fill has become the panel colour")

    # --- disabled states -----------------------------------------------------

    def test_disabled_text_is_not_produced_by_opacity_alone(self):
        """`opacity` composites text and fill together, so it destroys the label.

        Three rules used it and the light-mode label fell to 3.44:1. A disabled control
        that still carries a message needs its own colour tokens.
        """
        css = re.sub(r"/\*.*?\*/", "", CSS.read_text(encoding="utf-8"), flags=re.S)
        offenders = re.findall(
            r"\.action-btn:disabled[^{]*\{[^}]*opacity:\s*0?\.[0-6]", css)
        self.assertEqual(offenders, [], "a disabled action button is dimmed by opacity")

    def test_muted_still_clears_a_disabled_surface(self):
        """The replacement pair must actually work, not merely differ from opacity."""
        for theme in ("dark", "light"):
            with self.subTest(theme=theme):
                self.assertGreaterEqual(
                    _ratio(self._tok(theme, "--muted"), self._tok(theme, "--surface-2")),
                    TEXT_MIN,
                    "disabled labels would be unreadable in the replacement tokens")


if __name__ == "__main__":
    unittest.main()


class SurfaceFamilyRoleTests(unittest.TestCase):
    """The `--surface-*` family must not be used for three unrelated jobs again.

    In the light theme, 40 of the 43 numbered tokens were assigned the SAME value
    (#dedee8), and the family was used as a background fill, a control boundary AND a
    text colour. A token that is always the same colour cannot carry meaning, and the
    measured consequences were: error text at 1.34:1, control edges at 1.34:1, and a
    failed download bar at 1.00:1 against its own empty track.

    What is forbidden is not the family -- it is a token standing in for more than one
    ROLE. The families that are allowed to be shared are the ones this file already
    checks for contrast.
    """

    ROLE_TOKENS = {"--control-border", "--border", "--checkbox-border", "--focus-ring"}

    def _rules(self):
        css = re.sub(r"/\*.*?\*/", "", CSS.read_text(encoding="utf-8"), flags=re.S)
        out = []
        for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
            selector = " ".join(m.group(1).split()).split(",")[0]
            out.append((selector, m.group(2)))
        return out

    def _role(self, prop: str) -> str:
        if prop == "color":
            return "text"
        if prop.startswith("background"):
            return "fill"
        if "border" in prop or prop in ("outline", "outline-color"):
            return "boundary"
        return prop

    def test_no_numbered_surface_token_is_used_as_a_text_colour(self):
        """A numbered surface token is a fill or a boundary, never a text colour."""
        offenders = []
        for selector, body in self._rules():
            for m in re.finditer(r"([a-z-]+)\s*:\s*([^;{}]*var\(--surface-\d+\)[^;{}]*)", body):
                prop = m.group(1).strip()
                if self._role(prop) != "text":
                    continue
                line_m = CSS.read_text(encoding="utf-8")
                offenders.append(f"{selector} {{{prop}: var(--surface-*)}}")
        self.assertEqual(
            offenders, [],
            "a --surface-* token used as TEXT colour -- it is a fill token, and the "
            "light value measured 1.34:1 on white: " + ", ".join(offenders))

    def test_a_control_edge_uses_a_control_boundary_token(self):
        """Interactive boundaries must clear 3:1, not 1.34:1.

        Scoped to the rules that carry a button or field affordance. Decorative
        hairlines are allowed to stay low-contrast -- that is what `--border` is for --
        so the list is explicit rather than guessed.
        """
        interactive = (".action-btn", ".btn-mini", ".color-tool-btn", ".enhancer-save-btn",
                       ".ref-remove-btn", ".language-option", ".error-banner-dismiss",
                       ".universal-input-row", ".trigger-reminder", ".queue-recovery-error")
        offenders = []
        for selector, body in self._rules():
            if not any(selector.startswith(sel) for sel in interactive):
                continue
            for m in re.finditer(r"([a-z-]*border[a-z-]*)\s*:\s*([^;{}]*var\(--surface-\d+\)[^;{}]*)", body):
                offenders.append(f"{selector} {{{m.group(1).strip()}}}")
        self.assertEqual(
            offenders, [],
            "an interactive edge using a fill token, which is 1.34:1 in light mode: "
            + ", ".join(offenders))

    def test_the_control_border_clears_the_ui_threshold(self):
        for theme in ("dark", "light"):
            with self.subTest(theme=theme):
                self.assertGreaterEqual(
                    _ratio(_token(theme, "--control-border"),
                           _token(theme, "--panel")), UI_MIN)

    def test_the_control_border_is_not_the_decorative_border(self):
        """Otherwise "control" means nothing and the distinction is decorative."""
        for theme in ("dark", "light"):
            with self.subTest(theme=theme):
                self.assertGreater(
                    _ratio(_token(theme, "--control-border"),
                           _token(theme, "--panel")),
                    _ratio(_token(theme, "--border"), _token(theme, "--panel")))
