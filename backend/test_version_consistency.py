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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_VERSION_PY = ROOT / "backend" / "app_version.py"
VERSION_JS = ROOT / "frontend" / "src" / "version.js"
PACKAGE_JSON = ROOT / "frontend" / "package.json"
BUILD_APP = ROOT / "packaging" / "build_app.sh"

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
