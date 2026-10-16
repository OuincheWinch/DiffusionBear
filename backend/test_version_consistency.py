"""The app version must be defined in exactly one place.

It used to live in four:

    backend/app_version.py       APP_VERSION = "0.2.1"   <- the real one
    frontend/src/version.js      APP_VERSION = "0.1.2"   <- stale
    frontend/package.json        "version": "0.1.2"       <- stale
    Info.plist                   read from app_version.py by the build

The bundle therefore shipped as 0.2.1 while the badge in the header -- the one a
user actually reads -- said 0.1.2. Only app_version.py was ever bumped.

packaging/build_app.sh now regenerates both frontend copies from app_version.py
before building, so the drift cannot come back through a normal build. This test
catches the case where someone edits the frontend copies by hand, or builds with
`npm run build` and never runs the packaging script at all.
"""

import json
import re
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_VERSION_PY = ROOT / "backend" / "app_version.py"
VERSION_JS = ROOT / "frontend" / "src" / "version.js"
PACKAGE_JSON = ROOT / "frontend" / "package.json"
BUILD_APP = ROOT / "packaging" / "build_app.sh"
README = ROOT / "README.md"
CHANGELOG = ROOT / "CHANGELOG.md"

# The label carries "(beta)" in the UI, which app_version.py does not include.
BETA_SUFFIX = " (beta)"


def read_source_version() -> tuple[str, str]:
    text = APP_VERSION_PY.read_text(encoding="utf-8")
    version = re.search(r'^APP_VERSION\s*=\s*"([^"]+)"', text, re.M)
    label = re.search(r'^APP_VERSION_LABEL\s*=\s*"([^"]+)"', text, re.M)
    assert version, "APP_VERSION missing from backend/app_version.py"
    assert label, "APP_VERSION_LABEL missing from backend/app_version.py"
    return version.group(1), label.group(1)


def read_js_version() -> tuple[str, str]:
    text = VERSION_JS.read_text(encoding="utf-8")
    version = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', text)
    label = re.search(r'APP_VERSION_LABEL\s*=\s*"([^"]+)"', text)
    assert version, "APP_VERSION missing from frontend/src/version.js"
    assert label, "APP_VERSION_LABEL missing from frontend/src/version.js"
    return version.group(1), label.group(1)


class VersionConsistencyTests(unittest.TestCase):
    def test_frontend_matches_backend(self):
        src_version, src_label = read_source_version()
        js_version, js_label = read_js_version()
        self.assertEqual(
            js_version,
            src_version,
            f"frontend/src/version.js says {js_version} but backend/app_version.py "
            f"says {src_version}. Run packaging/build_app.sh, or edit "
            f"backend/app_version.py and rebuild -- do not edit version.js by hand.",
        )
        self.assertEqual(
            js_label,
            src_label + BETA_SUFFIX,
            f"version.js label {js_label!r} should be "
            f"{src_label + BETA_SUFFIX!r} (the UI adds the beta suffix)",
        )

    def test_package_json_matches_backend(self):
        src_version, _ = read_source_version()
        pkg = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
        self.assertEqual(
            pkg.get("version"),
            src_version,
            f"package.json says {pkg.get('version')!r} but backend/app_version.py "
            f"says {src_version!r}",
        )

    def test_version_is_a_plain_semver_triple(self):
        src_version, _ = read_source_version()
        self.assertRegex(
            src_version,
            r"^\d+\.\d+\.\d+$",
            f"APP_VERSION {src_version!r} is not a plain MAJOR.MINOR.PATCH; the "
            f"Info.plist and the badge both assume that shape",
        )

    def test_readme_quotes_the_current_version(self):
        """The README is the one version surface no build step can rewrite.

        0.3.8 shipped with the badge, the download link, the asset name and the
        checksum instructions all still reading 0.3.6 -- the build regenerates
        version.js and package.json, so neither the build nor the two tests above
        could have noticed. This is the file a newcomer reads first, and a dead
        download link is the worst possible first impression.

        Loopback addresses (127.0.0.1) are excluded: they are not versions.
        """
        src_version, _ = read_source_version()
        # The shields.io badge percent-encodes the space in "DiffusionBear 0.3.8",
        # so the literal text reads "...%200.3.8%20beta". Undo that before
        # scanning, otherwise the badge is read as version "200.3.8".
        text = README.read_text(encoding="utf-8").replace("%20", " ")
        # A historical caption is not drift. "Captured from v0.3.7" describes when
        # the *picture* was taken, not what to download; the release links below it
        # are asserted separately and are the part that must match. Excluded so the
        # caption cannot be "fixed" to agree with the release and become a lie.
        caption = re.search(r"^Captured from \*\*v[\d.]+\*\*.*$", text, re.M)
        if caption:
            text = text.replace(caption.group(0), "")
        # Anything shaped like a release version: optional "v", MAJOR.MINOR.PATCH.
        found = {
            m.group(0)
            for m in re.finditer(r"(?<![\d.])v?(\d+\.\d+\.\d+)(?![\d.])", text)
            if m.group(1) != src_version
        }
        self.assertEqual(
            found,
            set(),
            f"backend/app_version.py says {src_version!r} but README.md still quotes "
            f"{sorted(found)}. The README is hand-written and no build step rewrites "
            f"it, so it has to be bumped by hand -- do it before tagging.",
        )

    def test_readme_download_links_point_at_the_current_release(self):
        """The badge and the call-to-action must not 404.

        Checks the two URLs that actually resolve: the direct asset download and
        the release tag page.
        """
        src_version, _ = read_source_version()
        text = README.read_text(encoding="utf-8")
        repo = "https://github.com/OuincheWinch/DiffusionBear"
        for pattern in (
            rf"{re.escape(repo)}/releases/download/v(\d+\.\d+\.\d+)/",
            rf"{re.escape(repo)}/releases/tag/v(\d+\.\d+\.\d+)",
        ):
            for m in re.finditer(pattern, text):
                self.assertEqual(
                    m.group(1),
                    src_version,
                    f"README.md links to release v{m.group(1)} but the current "
                    f"release is v{src_version}; that link will 404.",
                )

    def test_changelog_has_an_entry_for_the_current_version(self):
        """The release notes must exist for what is being shipped."""
        src_version, _ = read_source_version()
        text = CHANGELOG.read_text(encoding="utf-8")
        # re.M: "^" must mean start-of-line, not start-of-file, or any heading that
        # is not on line 1 will never match.
        self.assertRegex(
            text,
            rf"(?m)^## \[{re.escape(src_version)}\]",
            f"CHANGELOG.md has no '## [{src_version}]' heading. Every release "
            f"needs its notes, including the fixes.",
        )

    def test_changelog_has_exactly_one_entry_per_version(self):
        """Two headings for one release means a botched merge, not a richer changelog.

        The 0.3.8 notes were written twice and shipped duplicated, so the entry
        described each fix twice at different lengths. Nothing caught it, because
        every other assertion here only asks whether a heading *exists*.
        """
        text = CHANGELOG.read_text(encoding="utf-8")
        counts = Counter(re.findall(r"^## \[(\d+\.\d+\.\d+)\]", text, re.M))
        dupes = {v: n for v, n in counts.items() if n > 1}
        self.assertEqual(
            dupes,
            {},
            f"CHANGELOG.md has more than one section for {dupes}; merge them.",
        )

    def test_version_js_is_marked_generated(self):
        """So the next person knows the file is output, not a source."""
        head = VERSION_JS.read_text(encoding="utf-8")[:400]
        self.assertIn(
            "GENERATED",
            head,
            "version.js should carry a GENERATED header pointing at "
            "backend/app_version.py",
        )

    def test_build_script_generates_the_frontend_copies(self):
        """The mechanism that stops the drift, not just the current state."""
        script = BUILD_APP.read_text(encoding="utf-8")
        self.assertIn(
            "write_frontend_version",
            script,
            "build_app.sh must regenerate the frontend version files, otherwise the "
            "next bump drifts again",
        )
        self.assertIn(
            "frontend/src/version.js",
            script,
            "build_app.sh should reference frontend/src/version.js explicitly",
        )

    def test_build_script_has_no_absolute_python_path(self):
        """A hardcoded venv path made the build depend on one machine's layout."""
        script = BUILD_APP.read_text(encoding="utf-8")
        self.assertNotRegex(
            script,
            r"/Volumes/[^/]+/venv/bin/python",
            "build_app.sh must not hardcode an absolute interpreter path; "
            "use ${PYTHON:-python3}",
        )


if __name__ == "__main__":
    unittest.main()
