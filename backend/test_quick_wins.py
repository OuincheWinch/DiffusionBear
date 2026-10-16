"""Quick-wins batch: four small defects, each previously verified in code.

H4: authHeaders() used Object.keys() to detect an existing Authorization
header, which returns [] on a Headers instance -- a caller-passed token on
one was always overwritten. Fixed with a Headers.has() branch.

H5: detectLanguage(null) never reached the navigator.language branch
(null !== "auto"), so first launch ignored the OS locale. Fixed by treating
null like "auto".

M6: ModelsTab.handleFeedback returned a cleanup function from a plain event
handler, which React never runs -- the timer leaked and a second success
left two timers racing. Fixed with a ref-held timer cleared on unmount and
on every new feedback.

H6: install_app() hardcoded dest to /Applications while copying to
$DEST_PARENT: with an override it deleted the production app, installed
elsewhere, and verified the wrong path. Fixed by deriving dest from
DEST_PARENT.

H7 from the same review is deliberately NOT fixed here: defaulting
LSMinimumSystemVersion to 15.0 would understate the real 26.2 floor the mlx
wheels declare (verified by otool), shipping a dyld death instead of a
clean version refusal.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = (ROOT / "frontend" / "src" / "api.js").read_text(encoding="utf-8")
LANG = (ROOT / "frontend" / "src" / "i18n" / "languages.js").read_text(encoding="utf-8")
MODELS_TAB = (ROOT / "frontend" / "src" / "components" / "ModelsTab.jsx").read_text(
    encoding="utf-8")
BUILD_APP = (ROOT / "packaging" / "build_app.sh").read_text(encoding="utf-8")
GENERATOR = (ROOT / "backend" / "generator.py").read_text(encoding="utf-8")
DOWNLOADS = (ROOT / "backend" / "routers" / "downloads.py").read_text(encoding="utf-8")


class TestAuthHeaderPreservation(unittest.TestCase):
    def test_headers_instances_checked_with_has(self):
        self.assertIn("headers.has(API_TOKEN_HEADER)", API)

    def test_plain_objects_still_checked_with_keys(self):
        self.assertIn("Object.keys(headers)", API)


class TestFirstLaunchLocale(unittest.TestCase):
    def test_null_reaches_the_os_locale_branch(self):
        self.assertRegex(
            LANG, r'if\s*\(\s*!stored\s*\|\|\s*stored\s*===\s*"auto"\s*\)')


class TestFeedbackTimer(unittest.TestCase):
    def test_timer_lives_in_a_ref_with_unmount_cleanup(self):
        self.assertIn("feedbackTimer = useRef(null)", MODELS_TAB)
        self.assertRegex(
            MODELS_TAB,
            r"useEffect\(\(\) => \(\) => \{\s*\n\s*if \(feedbackTimer\.current")

    def test_no_inert_cleanup_return(self):
        self.assertNotIn("return () => clearTimeout(timer)", MODELS_TAB)


class TestInstallDest(unittest.TestCase):
    def test_dest_derives_from_dest_parent(self):
        self.assertIn('local dest="$DEST_PARENT/$APP_NAME.app"', BUILD_APP)


class TestOsFloor(unittest.TestCase):
    """H7 proposed defaulting LSMinimumSystemVersion to 15.0. Declined with
    evidence: both shipped mlx wheels declare minos 26.2 (verified by otool
    against the bundle), and understating the floor launches the shell fine
    then dies in dyld with "backend exited during startup". The fallback
    stays at the mlx floor (fails safe), and a scan that finds nothing must
    say so instead of silently assuming."""

    def test_fallback_is_the_mlx_floor_not_sequoia(self):
        self.assertIn('MIN_MACOS="${MIN_MACOS:-26.2}"', BUILD_APP)
        self.assertNotIn('MIN_MACOS="${MIN_MACOS:-15.0}"', BUILD_APP)

    def test_empty_scan_warns_out_loud(self):
        self.assertIn("no LC_BUILD_VERSION found in payload", BUILD_APP)


class TestSafePromptCacheTempfiles(unittest.TestCase):
    def test_no_mktemp_in_generator(self):
        self.assertNotIn("tempfile.mktemp", GENERATOR)

    def test_named_temporary_file_used_in_generator(self):
        self.assertIn("tempfile.NamedTemporaryFile", GENERATOR)


class TestSingleSafeMfluxToNumpyPatch(unittest.TestCase):
    def test_only_one_safe_mflux_to_numpy_patch(self):
        self.assertEqual(GENERATOR.count("def _safe_mflux_to_numpy"), 1)
        self.assertEqual(
            GENERATOR.count("_mflux_img_util.ImageUtil._to_numpy = _safe_mflux_to_numpy"), 1
        )


class TestContentDispositionParsing(unittest.TestCase):
    def test_downloads_does_not_use_naive_split(self):
        self.assertNotIn('disposition.split("filename=", 1)', DOWNLOADS)

    def test_downloads_uses_content_disposition_helper(self):
        self.assertIn('_content_disposition_filename(disposition)', DOWNLOADS)


class TestBuildAppSecretScan(unittest.TestCase):
    def test_audits_secret_patterns_in_content(self):
        self.assertIn("SUSPECT SECRET CONTENT", BUILD_APP)
        self.assertIn("grep -rEIq", BUILD_APP)


if __name__ == "__main__":
    unittest.main()
