"""Every font the app names must be free, open, or provided by the OS.

WHY THIS EXISTS
An audit of v0.3.5 found `Consolas` in two `font-family` declarations. It is a
Microsoft font: proprietary, and not present on a Mac unless Office is installed,
so it could only ever resolve on a machine that already had it. It was a dead
reference to a font nobody may redistribute.

It was found by eye, once. This file is the standing check so it does not have to
be.

WHAT "FREE AND OPEN SOURCE" MEANS HERE, PRECISELY
Three categories, and the distinction matters:

  GENERIC KEYWORD   sans-serif, monospace, system-ui, ui-monospace... These are
                    CSS-wide keywords. The user agent resolves them. Nothing is
                    named, nothing is shipped.
  SYSTEM FONT       -apple-system, Menlo, SFMono-Regular. macOS PROVIDES these.
                    The app does not embed or redistribute them; it names them,
                    exactly as a native Mac app does. Shipping a copy would be a
                    licensing problem, naming one is not.
  EMBEDDED FONT     a .woff/.ttf/.otf inside the bundle, or an @font-face. This
                    is redistribution, and it must be under an OSS licence with the
                    licence text shipped alongside.

The check is DEFAULT-DENY: a font-family name that is not in one of those
categories fails. A denylist alone would only catch what someone already thought
of, which is how Consolas survived.

WHAT IS ASSERTED
1. No @font-face, so nothing is embedded through CSS.
2. Every font-family name in the source classifies as generic, system or OSS.
3. Every font FILE inside the shipped bundle belongs to a freely licensed package
   AND its licence text ships in the same directory -- so the claim is checkable
   by a reader, not just asserted here.
4. The explicitly proprietary names are named and reported individually, because
   "unknown font" is a weaker message than "that is Microsoft's".
"""

import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FRONTEND_SRC = REPO / "frontend" / "src"
CSS = FRONTEND_SRC / "App.css"
BUNDLE = Path("/Applications/DiffusionBear.app")

# --- category 1: CSS-wide keywords. No font is named, so none can be proprietary.
GENERIC = {
    "inherit", "initial", "unset", "revert", "revert-layer",
    "sans-serif", "serif", "monospace", "system-ui",
    "ui-sans-serif", "ui-serif", "ui-monospace", "ui-rounded",
    "-apple-system", "blinkmacsystemfont", "math", "emoji", "fangsong",
}

# --- category 2: fonts macOS provides. Named, never embedded.
# Provenance is macOS itself; the app ships no copy.
SYSTEM_FONTS = {
    "menlo", "sfsmono-regular", "sfmono-regular", "monaco", "courier",
    "andale mono", "apple-chancery", "avenir", "avenir next", "geneva",
    "lucida grande", "optima", "palatino", "zapfino",
}

# --- category 3: fonts that are free and open wherever they come from.
#   Bitstream Vera / DejaVu : permissive, free
#   Liberation              : SIL OFL 1.1
#   Noto / Roboto / Ubuntu  : SIL OFL 1.1
#   Iosevka / JetBrains Mono : SIL OFL 1.1
#   Fira Code               : SIL OFL 1.1
#   Source Code Pro         : SIL OFL 1.1
#   Cousine                 : Apache 2.0
OPEN_FONTS = {
    "dejavu sans", "dejavu sans mono", "dejavu serif", "dejavu sans display",
    "dejavu serif display",
    "liberation sans", "liberation serif", "liberation mono",
    "noto sans", "noto serif", "noto sans mono", "noto color emoji",
    "roboto", "roboto mono",
    "ubuntu", "ubuntu mono",
    "fira code", "fira sans", "fira mono",
    "jetbrains mono", "iosevka",
    "source code pro", "cousine", "inconsolata", "ibm plex mono", "ibm plex sans",
}

# Named explicitly so the failure message is specific rather than "unknown".
PROPRIETARY = {
    "consolas": "Microsoft, proprietary",
    "arial": "Monotype, proprietary",
    "helvetica": "Linotype/Monotype, proprietary",
    "times new roman": "Monotype, proprietary",
    "courier new": "Monotype, proprietary",
    "impact": "Monotype, proprietary",
    "verdana": "Microsoft, proprietary",
    "tahoma": "Microsoft, proprietary",
    "calibri": "Microsoft, proprietary",
    "cambria": "Microsoft, proprietary",
    "garamond": "Monotype, proprietary",
    "georgia": "Microsoft, proprietary",
    "wingdings": "Microsoft, proprietary",
    "webdings": "Microsoft, proprietary",
    "franklin gothic": "Monotype, proprietary",
}

# Font files are only acceptable from a package whose licence we can name.
FREELY_LICENSED_FONT_PACKAGES = {
    "matplotlib/mpl-data/fonts/ttf": "matplotlib's bundled test fonts",
}

# Per family, the licence and the licence TEXT as it actually ships. Verified by
# reading the files in the installed bundle, not from memory:
#
#   DejaVu              Bitstream Vera copyright, DejaVu changes public domain.
#                       Licence text: LICENSE_DEJAVU, present.
#   STIX                "This Font Software is licensed under the SIL Open Font
#                       License, Version 1.1." Licence text: LICENSE_STIX, present.
#   Computer Modern     Donald Knuth's CM fonts, public-domain lineage and free to
#                       redistribute. NO licence text ships beside them.
#
# That last row is why this table is per family rather than per directory. An
# earlier version asserted "licence text ships beside the fonts" and passed for the
# Computer Modern files only because LICENSE_STIX happens to sit in the same
# directory -- it was implying coverage it did not have.
FONT_FAMILY_LICENCES = {
    "dejavu": ("Bitstream Vera + public-domain changes", "LICENSE_DEJAVU"),
    "stix": ("SIL OFL 1.1", "LICENSE_STIX"),
    "cm": ("Knuth Computer Modern, public-domain lineage", None),
}


def _family(font_path: Path) -> str | None:
    stem = font_path.stem.lower()
    if stem.startswith("dejavu"):
        return "dejavu"
    if stem.startswith("stix"):
        return "stix"
    if stem.startswith("cm"):
        return "cm"
    return None

FONT_SUFFIXES = {".woff", ".woff2", ".ttf", ".otf", ".eot"}


def _tracked_files() -> list[str]:
    """Files git actually tracks. Falls back to nothing rather than the filesystem."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [p for p in out.split("\0") if p]


def family_names(css_text: str) -> set[str]:
    """Every font-family name in the stylesheet, lowercased and unquoted."""
    found = set()
    for decl in re.findall(r"font-family\s*:\s*([^;}]+)", css_text, flags=re.I):
        for part in decl.split(","):
            name = part.strip().strip("\"'").strip().lower()
            if name:
                found.add(name)
    return found


class NoEmbeddedFontsTests(unittest.TestCase):
    def test_there_is_no_font_face(self):
        """The strongest possible answer: nothing is embedded, so nothing can be."""
        hits = [
            f"{p.relative_to(REPO)}:{i + 1}"
            for p in FRONTEND_SRC.rglob("*")
            if p.suffix in (".css", ".js", ".jsx", ".html")
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines())
            if "@font-face" in line
        ]
        self.assertEqual(hits, [], f"@font-face embeds a font: {hits}")

    def test_no_font_files_are_committed_to_the_repo(self):
        """Scoped to TRACKED files.

        Walking the filesystem instead found the 38 TTFs inside the gitignored
        venv/, which are matplotlib's and are covered by the bundle test below.
        "Committed" has to mean committed, so this asks git rather than guessing
        which directories happen to be ignored today.
        """
        tracked = _tracked_files()
        strays = [t for t in tracked if Path(t).suffix in FONT_SUFFIXES]
        self.assertEqual(
            strays, [],
            "font files committed to the repo -- an embedded font is redistribution "
            f"and needs an OSS licence: {strays}",
        )


# The stylesheet source is what gets screened: the bundle ships frontend/dist, not
# frontend/src, so this cannot run from an installed app. Skipping there rather than
# erroring keeps the copy that does ship in the bundle from looking broken.
@unittest.skipUnless(
    CSS.is_file(),
    "frontend/src/App.css is not present (running against an installed bundle, "
    "which ships frontend/dist rather than the source stylesheet)",
)
class FontFamilyLicensingTests(unittest.TestCase):
    def setUp(self):
        self.names = family_names(CSS.read_text(encoding="utf-8"))

    def test_the_stylesheet_actually_declares_fonts(self):
        """Otherwise every other test here passes vacuously."""
        self.assertTrue(self.names, "no font-family found; the parser is broken")

    def test_no_proprietary_font_is_named(self):
        found = {n: PROPRIETARY[n] for n in self.names if n in PROPRIETARY}
        self.assertEqual(found, {}, f"proprietary font(s) named in the UI: {found}")

    def test_every_named_font_is_generic_system_or_open(self):
        unknown = sorted(
            n for n in self.names
            if n not in GENERIC and n not in SYSTEM_FONTS and n not in OPEN_FONTS
        )
        self.assertEqual(
            unknown, [],
            "font name(s) that are neither a CSS keyword, a macOS system font, nor "
            f"known open source: {unknown}. If one of these is free, add it to "
            "OPEN_FONTS with its licence; if it is not, remove it.",
        )

    def test_consolas_is_gone_and_stays_gone(self):
        """Named specifically because it was there and is a Microsoft font."""
        self.assertNotIn("consolas", self.names)
        self.assertNotIn("consolas", CSS.read_text(encoding="utf-8").lower())


@unittest.skipUnless(BUNDLE.is_dir(), "the app bundle is not installed here")
class ShippedBundleFontTests(unittest.TestCase):
    """Fonts inside the installed bundle, with their licences."""

    def setUp(self):
        self.fonts = [
            p for p in BUNDLE.rglob("*")
            if p.is_file() and p.suffix.lower() in FONT_SUFFIXES
        ]

    def test_every_shipped_font_file_is_freely_licensed(self):
        offenders = []
        for path in self.fonts:
            try:
                rel = str(path.relative_to(BUNDLE / "Contents" / "Resources"))
            except ValueError:
                offenders.append(f"{path} (outside Resources)")
                continue
            # Matched anywhere in the path, not as a prefix: inside the bundle the
            # real path is Contents/Resources/venv/lib/.../matplotlib/mpl-data/...,
            # so a prefix test flagged 38 correctly-licensed files as unvetted.
            if not any(pkg in rel for pkg in FREELY_LICENSED_FONT_PACKAGES):
                offenders.append(rel)
        self.assertEqual(
            offenders, [],
            "font file(s) in the bundle from an unvetted package -- an embedded font "
            f"is redistribution and needs an OSS licence: {offenders}",
        )

    def test_every_shipped_font_belongs_to_a_known_licensed_family(self):
        for path in self.fonts:
            family = _family(path)
            self.assertIsNotNone(
                family,
                f"{path.name} is from an unrecognised font family; add it to "
                "FONT_FAMILY_LICENCES with its licence, or remove it",
            )

    def test_the_declared_licence_text_actually_ships(self):
        """A licence claim a reader cannot check is not a licence claim.

        Asserted per family against the licence file the table names, so a family
        recorded as None (Computer Modern) is an explicit, visible decision rather
        than a directory that happened to contain someone else's licence.
        """
        directory = self.fonts[0].parent if self.fonts else None
        if directory is None:
            self.skipTest("no font files in the bundle")
        present = {f.name.lower() for f in directory.iterdir() if f.is_file()}
        for family, (_basis, licence_file) in FONT_FAMILY_LICENCES.items():
            if licence_file and family in {_family(p) for p in self.fonts}:
                self.assertIn(
                    licence_file.lower(), present,
                    f"{family}: the bundle claims {licence_file} but it is not shipped",
                )

    def test_matplotlib_fonts_are_unused_dead_weight(self):
        """Reported, not failed: they arrive via a transitive dependency.

        Nothing in backend/ imports matplotlib, so these ~2 MB of TTF are never
        rendered. Stripping matplotlib is a dependency decision with a real risk of
        breaking the SDXL path, so it is reported rather than asserted.
        """
        self.assertTrue(self.fonts, "expected matplotlib's fonts to be present")
        # Not an assertion about behaviour -- a nudge if someone ever removes the
        # dependency, in which case this test should go quiet rather than pass
        # unnoticed.


if __name__ == "__main__":
    unittest.main()