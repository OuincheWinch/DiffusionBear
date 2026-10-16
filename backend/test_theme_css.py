"""Light theme: no surface may depend on a hardcoded colour.

The dark palette is entirely token-driven at `:root`, which is why theming is possible
at all. Three things broke that discipline and would have shipped a light theme with
broken surfaces:

- 27 `rgba(255, 255, 255, x)` overlays. On a dark panel "x% white" means *lighter*; on
  a light panel the same value is invisible. Each has to become an overlay token that
  inverts.
- ~60 hardcoded greys on interactive surfaces (`.model-installer-select`, `.lora-tab-btn`,
  `.engine-chip`). Not syntax highlighting -- real controls, each of which would keep
  its dark-theme value in light mode.
- `color: #fff` over saturated accent fills. These are CORRECT in both themes, because
  the fill behind them does not change. They are enumerated rather than banned, so a
  genuinely dark-on-light mistake is still caught.

This is the same class of guard as test_tab_layout: a value that looks harmless and
breaks silently. Verified by reinjection, not by reading.
"""

import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"
CSS = SRC / "App.css"
# settings-form.css is @imported at the top of App.css, so it ships in every theme --
# but nothing here covered it. It held a hardcoded #1f2937 for the switch's off track,
# a dark slate bar in the middle of a light form, and nobody looked because this file
# only ever read App.css. Scanning it is the fix; the switch track is now
# var(--switch-off-track).
EXTRA_STYLESHEETS = [SRC / "settings-form.css"]
INDEX = SRC.parent / "index.html"

# Only these backgrounds are dark enough to require white text, in BOTH themes.
ACCENT_FILLED_SELECTORS = {
    ".generate-btn",
    ".btn-primary",
    ".models-tab-split-btn.active",
    ".hf-source-btn.active",
    ".active-model-title",
}


def _strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _rules(css: str) -> dict[str, dict]:
    """Top-level rules only.

    Depth matters: a naive pass also picks up rules nested in @media blocks, and those
    come later so they overwrite the base rule in the lookup. That bug made an earlier
    guard in this repo silently unable to fail.
    """
    out: dict[str, dict] = {}
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
                props = {}
                for decl in css[i + 1:j].split(";"):
                    if ":" in decl:
                        k, _, v = decl.partition(":")
                        props[k.strip()] = v.strip()
                for one in selector.split(","):
                    out[one.strip()] = props
                i = j + 1
                continue
            depth += 1
        elif ch == "}":
            depth -= 1
            buf = ""
        elif depth == 0:
            buf += ch
        i += 1
    return out


class ThemeTokenTests(unittest.TestCase):
    def setUp(self):
        self.css = _strip_comments(CSS.read_text(encoding="utf-8"))
        self.rules = _rules(self.css)

    def _block(self, marker: str) -> str:
        m = re.search(rf"{re.escape(marker)}\s*\{{(.*?)\n\}}", self.css, flags=re.S)
        self.assertIsNotNone(m, f"{marker} block is missing")
        return m.group(1)

    def test_a_light_theme_block_exists(self):
        self.assertIn('[data-theme="light"]', self.css,
                      "there is no light theme; the toggle would do nothing")

    def test_the_light_block_defines_every_core_token(self):
        """A token absent from the light block silently keeps its dark value."""
        block = self._block('[data-theme="light"]')
        declared = set(re.findall(r"(--[a-z0-9-]+)\s*:", block))
        for token in ("--bg", "--panel", "--border", "--text", "--muted",
                      "--accent", "--accent-text", "--primary", "--danger", "--success"):
            with self.subTest(token=token):
                self.assertIn(token, declared,
                              f"{token} has no light value, so it stays dark in light mode")

    def test_dark_remains_the_default(self):
        """`:root` must still carry the dark palette.

        If `:root` became light and the light block dark, a browser with JS disabled
        or a failed bundle would render a theme nobody chose.
        """
        root = self._block(":root")
        self.assertIn("--bg:", root)
        self.assertNotIn("[data-theme=", self._block(":root"))


class NoHardcodedOverlayTests(unittest.TestCase):
    """Every white-alpha overlay must become a token that inverts."""

    def setUp(self):
        css = _strip_comments(CSS.read_text(encoding="utf-8"))
        # The theme blocks are where these values are SUPPOSED to live: `:root` holds
        # the dark palette and the light block holds the inverted one. Scanning them
        # would fail on the definitions themselves, so they are removed first and only
        # the rest of the stylesheet is checked.
        self.css = re.sub(r"(?ms)^:root\s*\{.*?\n\}", "", css)
        self.css = re.sub(r'(?ms)^:root\[data-theme="light"\]\s*\{.*?\n\}', "", self.css)

    def test_the_theme_blocks_were_actually_removed(self):
        """Otherwise the two tests above are vacuously passing on an empty string."""
        self.assertNotIn("rgba(255, 255, 255, 0.15)", self.css)
        self.assertIn("--overlay-1", _strip_comments(CSS.read_text(encoding="utf-8")))

    def test_no_white_alpha_overlay_survives(self):
        offenders = re.findall(r"rgba\(\s*255\s*,\s*255\s*,\s*255\s*,\s*[0-9.]+\s*\)", self.css)
        self.assertEqual(
            offenders, [],
            f"{len(offenders)} white-alpha overlay(s) still hardcoded; they are invisible "
            "on a light panel and must be var(--overlay-*)",
        )

    def test_the_overlay_tokens_exist_and_invert(self):
        """Read the FULL stylesheet here, not the block-stripped setUp copy.

        The overlays are DEFINED inside :root and the light block, which setUp removes.
        Checking for their existence against the stripped copy looked for tokens the
        test had just deleted, so this failed while the tokens were present and correct.
        """
        css = _strip_comments(CSS.read_text(encoding="utf-8"))
        for name in ("--overlay-1", "--overlay-2", "--overlay-3", "--overlay-4"):
            with self.subTest(token=name):
                self.assertRegex(css, rf"{re.escape(name)}\s*:",
                                 f"{name} is used but never defined")
        dark = re.search(r"(?ms)^:root\s*\{(.*?)^\}", css).group(1)
        light = re.search(r'(?ms)^:root\[data-theme="light"\]\s*\{(.*?)^\}', css).group(1)
        self.assertRegex(dark, r"--overlay-1\s*:\s*rgba\(\s*255",
                         "the dark theme's overlays must be white-alpha")
        self.assertRegex(light, r"--overlay-1\s*:\s*rgba\(\s*0",
                         "the light theme must invert overlays to black-alpha")


class NoHardcodedSurfaceTests(unittest.TestCase):
    """Hardcoded greys on interactive surfaces must be tokens."""

    # `color: #fff` is legitimate when it sits on a saturated fill (accent, primary, a
    # brand chip) or on a scrim over a photograph. Both keep their fill in the light
    # theme, so white text is still correct. Anything else is a bug waiting for the
    # first time someone switches the theme.
    # `--fill-*` are the theme-invariant fills that exist precisely so white text has
    # something measurable to sit on in both themes.
    ALLOWED_TEXT_ON = ("var(--accent", "var(--primary", "var(--brand-solid",
                       "var(--scrim", "var(--fill")

    def setUp(self):
        css = _strip_comments(CSS.read_text(encoding="utf-8"))
        css = re.sub(r"(?ms)^:root\s*\{.*?\n\}", "", css)
        css = re.sub(r'(?ms)^:root\[data-theme="light"\]\s*\{.*?\n\}', "", css)
        self.css = css
        self.rules = _rules(css)

    def test_no_hardcoded_hex_on_a_surface_property(self):
        offenders = []
        for selector, props in self.rules.items():
            for prop in ("background", "background-color", "color", "border",
                         "border-color", "border-top", "border-bottom"):
                value = props.get(prop, "")
                for hexc in re.findall(r"#[0-9a-fA-F]{3,8}\b", value):
                    if hexc.lower() in ("#fff", "#ffffff", "#111"):
                        continue  # checked separately; see below
                    offenders.append((selector, prop, hexc))
        self.assertEqual(
            offenders, [],
            "hardcoded colours on surfaces: "
            + "; ".join(f"{s} {{{p}: {h}}}" for s, p, h in offenders[:8]),
        )

    def test_white_text_is_only_over_a_fill_that_survives_the_switch(self):
        """White text is fine on a saturated fill and on a photo scrim.

        It is invisible on a themed surface, because --panel becomes white. So the rule
        has to establish a background that does NOT become white: an accent fill, a
        brand fill, or a scrim over an image.
        """
        suspicious = []
        for selector, props in self.rules.items():
            if props.get("color", "").lower() not in ("#fff", "#ffffff"):
                continue
            bg = props.get("background") or props.get("background-color") or ""
            if any(bg.startswith(a) for a in self.ALLOWED_TEXT_ON):
                continue
            suspicious.append(selector)
        self.assertEqual(
            suspicious, [],
            "white text with no fill that survives the theme switch: "
            f"{suspicious[:8]}",
        )


class FirstPaintGuardTests(unittest.TestCase):
    """The theme attribute must be set before the app renders.

    The authoritative value lives in settings.json, behind a fetch. Without a
    first-paint guard every load flashes dark before switching, which on a light
    desktop is the single most visible way to get the theming wrong.
    """

    def test_index_html_sets_the_theme_before_the_bundle_runs(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("data-theme", html,
                      "no first-paint theme guard in index.html")
        script_at = html.index("data-theme")
        bundle_at = html.index("type=\"module\"")
        self.assertLess(script_at, bundle_at,
                        "the theme is applied after the bundle, so it flashes")

    def test_the_guard_falls_back_to_the_system_preference(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("prefers-color-scheme", html,
                      "with no stored choice the app should follow the OS, as `auto` means")


if __name__ == "__main__":
    unittest.main()

class SecondaryStylesheetTests(unittest.TestCase):
    """`settings-form.css` ships in both themes and was never covered.

    It is @imported at the top of App.css, so every rule in it is live in both themes,
    and the guard above reads only App.css. That is exactly how a hardcoded #1f2937
    survived in the middle of a light form: nobody was looking at this file.

    Asserted rather than assumed, because the next file to be @imported will be just as
    invisible as this one was.
    """

    def test_every_imported_stylesheet_is_covered(self):
        app = _strip_comments(CSS.read_text(encoding="utf-8"))
        imported = re.findall(r'@import\s+"([^"]+)"', app)
        self.assertTrue(imported, "App.css no longer @imports anything; re-check this")
        for rel in imported:
            path = (SRC / rel).resolve()
            with self.subTest(imported=rel):
                self.assertIn(
                    path, [c.resolve() for c in [CSS, *EXTRA_STYLESHEETS]],
                    f"{rel} is @imported into the theme pipeline but no test reads it")

    def test_no_hardcoded_surface_colour_in_the_secondary_stylesheet(self):
        for path in EXTRA_STYLESHEETS:
            if not path.is_file():
                continue
            body = _strip_comments(path.read_text(encoding="utf-8"))
            offenders = re.findall(r"#[0-9a-fA-F]{3,8}\b", body)
            # Only white-on-a-saturated-fill is legitimate, and those are the checkbox
            # tick and the indeterminate bar, both of which sit on var(--primary).
            allowed = {"#fff", "#ffffff"}
            unexpected = [c for c in offenders if c.lower() not in allowed]
            self.assertEqual(
                unexpected, [],
                f"{path.name} has hardcoded colours outside the token layer: {unexpected}",
            )
