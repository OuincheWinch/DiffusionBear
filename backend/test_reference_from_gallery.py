"""A browser card offers "use as reference", and a full reference row asks first.

The Generate canvas had the button; the browser card did not, so a gallery
image could only become a reference by downloading and re-uploading it. Both
buttons now render from the SAME ImageActions prop, so the labels cannot
drift apart again.

Three properties, each of which can fail alone:

  * the button exists on BOTH surfaces through the shared prop -- a Gallery
    that never passes onSetReference renders no button, silently;
  * the card's choice flows through addRefImage, not around it -- the model
    support check, duplicate filter and over-limit dialog live there, and a
    direct setRefImages would bypass all three;
  * a full row OPENS A DIALOG (Yes replaces the last slot, No cancels)
    instead of erroring -- the old addRefImage errored, and the dialog state
    is what proves the replacement path exists.

Static source checks, in the style of test_frontend_no_silent_failures.py:
no frontend test runner exists here, so these assert structure rather than
behaviour. Each asserts the wiring, never just the presence of a string:
a button that calls the wrong handler passes a text grep and still does
nothing.
"""
import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"

FORM = SRC / "components" / "GenerateForm.jsx"
CANVAS = SRC / "components" / "ResultCanvas.jsx"
GALLERY = SRC / "components" / "Gallery.jsx"
APP = SRC / "App.jsx"
ACTIONS = SRC / "components" / "ImageActions.jsx"

FORM_TEXT = FORM.read_text(encoding="utf-8")
CANVAS_TEXT = CANVAS.read_text(encoding="utf-8")
GALLERY_TEXT = GALLERY.read_text(encoding="utf-8")
APP_TEXT = APP.read_text(encoding="utf-8")
ACTIONS_TEXT = ACTIONS.read_text(encoding="utf-8")


def _function_body(text: str, name: str) -> str:
    """Source of a function or callback, by brace matching from its opening."""
    start = text.index(f"function {name}(")
    depth, i = 0, text.index("{", start)
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    raise AssertionError(f"unterminated function {name}")


class TestSharedButton(unittest.TestCase):
    def test_image_actions_owns_the_reference_button(self):
        self.assertIn("onSetReference", ACTIONS_TEXT)
        self.assertIn('t("canvas.action.useAsRef")', ACTIONS_TEXT)

    def test_canvas_uses_the_shared_prop_not_a_bespoke_button(self):
        """The canvas used to hand-roll its own button beside ImageActions.
        A bespoke button with the same label would pass a text grep while
        drifting again on the next edit."""
        self.assertIn("onSetReference={", CANVAS_TEXT)
        self.assertNotIn('t("canvas.action.useAsRef")', CANVAS_TEXT)

    def test_gallery_forwards_card_choice_through_a_prop(self):
        self.assertIn("onSetReference", GALLERY_TEXT)
        self.assertRegex(GALLERY_TEXT, r"onSetReference=\{[^}]*selected")


class TestReferenceFlow(unittest.TestCase):
    def test_app_carries_the_pending_reference_and_switches_tab(self):
        self.assertIn("pendingReference", APP_TEXT)
        self.assertIn('setTab("generate")', APP_TEXT)

    def test_form_consumes_pending_through_add_ref_image(self):
        """The card's choice must go through addRefImage -- the model support
        check, duplicate filter and over-limit dialog live there. A direct
        setRefImages would bypass all three."""
        body_start = FORM_TEXT.index("pendingReference == null")
        segment = FORM_TEXT[body_start:body_start + 600]
        self.assertIn("addRefImage({", segment)
        self.assertNotIn("setRefImages([", segment)

    def test_limit_report_sits_below_the_limit_definition(self):
        """The onReferenceLimit effect reads maxRefImages in its dependency
        array, which evaluates during render. Placed above the const, it
        throws a TDZ ReferenceError that takes down the whole tab behind the
        recovery boundary -- which is exactly what shipped once."""
        define_at = FORM_TEXT.index("const maxRefImages =")
        report_at = FORM_TEXT.index("onReferenceLimit?.(maxRefImages)")
        self.assertLess(
            define_at, report_at,
            "the limit-reporting effect runs before maxRefImages is initialized",
        )

    def test_model_without_reference_support_still_errors(self):
        """The model check ("check model loaded") stays an error, not a
        dialog: there is nothing to replace and no choice to offer."""
        body = _function_body(FORM_TEXT, "addRefImage")
        self.assertIn("generate.error.refNotSupported", body)


class TestOverLimitDialog(unittest.TestCase):
    def test_full_row_opens_a_dialog_not_an_error(self):
        body = _function_body(FORM_TEXT, "addRefImage")
        self.assertIn("setRefLimitItem(item)", body)
        self.assertNotIn("maxReferencesOne", body)
        self.assertNotIn("maxReferencesMany", body)

    def test_dialog_wires_yes_replace_and_no_cancel(self):
        self.assertIn("confirmRefLimitReplace", FORM_TEXT)
        self.assertIn("generate.refLimit.yesBtn", FORM_TEXT)
        self.assertIn("generate.refLimit.noBtn", FORM_TEXT)
        self.assertIn("generate.refLimit.title", FORM_TEXT)
        self.assertIn("generate.refLimit.body", FORM_TEXT)

    def test_yes_replaces_only_the_last_slot(self):
        body = _function_body(FORM_TEXT, "confirmRefLimitReplace")
        self.assertIn("previous.slice(0,", body)
        self.assertNotIn("setRefImages([]", body)

    def test_dialog_state_clears_on_both_answers(self):
        body = _function_body(FORM_TEXT, "confirmRefLimitReplace")
        self.assertIn("setRefLimitItem(null)", body)
        self.assertGreaterEqual(FORM_TEXT.count("setRefLimitItem(null)"), 2)

    def test_ref_limit_strings_exist_in_every_language(self):
        parts = (SRC / "i18n" / "parts" / "generate.js").read_text(encoding="utf-8")
        generated = (SRC / "i18n" / "lang" / "_parts" / "generate.json").read_text(
            encoding="utf-8")
        for key in ("generate.refLimit.title", "generate.refLimit.body",
                    "generate.refLimit.yesBtn", "generate.refLimit.noBtn"):
            self.assertIn(key, FORM_TEXT, f"{key} is never used")
            self.assertIn(key, parts, f"{key} missing from the reference catalogue")
            self.assertIn(key, generated, f"{key} missing from the generated catalogue")


class TestCanvasExportUsesNativeBridge(unittest.TestCase):
    """The canvas Export must try the shell's save panel before the blob path.

    Shipped bug: handleDownload went straight to downloadImage, whose object
    URL WKWebView cannot download -- it hands the blob: link to macOS, which
    shows "no application set to open the URL". The gallery card already
    prefers the native bridge; the canvas must do the same, in the same
    order: bridge first, blob fallback only when the bridge is absent or
    throws.
    """

    def test_bridge_is_tried_before_blob_download(self):
        canvas = (SRC / "components" / "ResultCanvas.jsx").read_text(encoding="utf-8")
        body_start = canvas.index("async function handleDownload()")
        body = canvas[body_start:body_start + 900]
        bridge_at = body.index("exportImageNatively(currentImage.id)")
        # Not just before the downloadImage call: before ANY blob work
        # (spinner, fetch). A bridge attempt after setDownloading still
        # fires the fetch first, which is the defect.
        for marker in ("setDownloading(true)", "downloadImage(currentImage.id"):
            self.assertLess(
                bridge_at, body.index(marker),
                f"canvas Export reaches {marker} before trying the native bridge",
            )

    def test_bridge_absence_falls_through_to_blob(self):
        canvas = (SRC / "components" / "ResultCanvas.jsx").read_text(encoding="utf-8")
        self.assertIn("messageHandlers?.native", canvas)


if __name__ == "__main__":
    unittest.main()
