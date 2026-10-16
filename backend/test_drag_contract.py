"""Regression guards for drag-and-drop out of the app.

Ten attempts were needed to make dragging a gallery cell copy the real PNG to
Finder, and every failure had the same shape: the code looked right, and nothing
told anyone it was inert until a human tried a drag. These tests pin the
specific mistakes, because each one has been made at least once.

They are static source guards rather than behavioural ones on purpose. The real
gesture needs a mouse and a pasteboard, which a test cannot have. What it *can*
do is refuse the changes that broke it:

  * mouseDown/mouseDragged overrides on the WKWebView subclass. They never fire,
    because the view hit-tests to an internal content subview. This cost several
    attempts that could never have worked.
  * Any element marked draggable in the page. WebKit builds a pasteboard from the
    DOM element under the cursor, so a draggable element competes with the shell's
    monitor and produces an http:// link instead of the file.
  * A cached rect index, or a hit test driven by one.
  * requestAnimationFrame in the freshness path. rAF stops when the window is
    occluded, which silently freezes the cursor position and resurrects the
    wrong-image bug.
  * Array.isArray on a NodeList, which made the page report nothing at all while
    looking correct.

The Swift-side behavioural check is a small headless WKWebView probe that loads
the real page and reports what the bridge receives; see tools/.
"""

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHELL = REPO / "packaging" / "shell" / "main.swift"
CSS = REPO / "frontend" / "src" / "App.css"
FRONTEND = REPO / "frontend" / "src"

DANGEROUS_SAMPLERS = ("mouseDown", "mouseDragged")


def strip_comments(text: str) -> str:
    """Source with // and /* */ comments removed.

    Without this the guards match their own explanatory comments -- the rAF test
    found the sentence explaining why not to use rAF, and the draggable test found
    prose about a draggable attribute that no longer exists. A guard that can be
    satisfied by a comment is worse than none.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


class ShellDragContractTests(unittest.TestCase):
    """The gesture must be observed at the AppKit level, not on the web view."""

    @classmethod
    def setUpClass(cls):
        cls.src = SHELL.read_text(encoding="utf-8")

    def test_the_file_exists(self):
        self.assertTrue(SHELL.is_file(), "the shell source moved; update these guards")

    def test_no_mouse_overrides_on_the_web_view(self):
        """WKWebView hit-tests to an internal content subview.

        A press on the page is delivered there and never reaches the subclass, so
        an override cannot observe it. This is why several attempts produced no
        drag at all rather than a wrong one.
        """
        for selector in DANGEROUS_SAMPLERS:
            with self.subTest(selector=selector):
                self.assertNotRegex(
                    self.src,
                    r"override\s+func\s+" + selector,
                    f"{selector} on the WKWebView subclass never fires; "
                    "use an NSEvent local monitor instead",
                )

    def test_a_local_event_monitor_is_installed(self):
        self.assertIn(
            "addLocalMonitorForEvents",
            self.src,
            "the drag gesture must be watched with an NSEvent local monitor, "
            "which sees events before dispatch and does not depend on hit-testing",
        )
        self.assertIn(".leftMouseDragged", self.src, "the monitor must watch dragged events")

    def test_the_monitor_never_consumes_the_event(self):
        """Returning the event unchanged is what keeps clicks and scrolling working."""
        monitor = self.src.split("addLocalMonitorForEvents", 1)[-1]
        window = monitor[: monitor.find("private func beginDrag")] if "private func beginDrag" in monitor else monitor[:4000]
        self.assertRegex(
            window,
            r"return\s+event\b",
            "the monitor must return the event unchanged; consuming it would break "
            "every click, text selection and scroll in the app",
        )

    def test_the_pasteboard_gets_a_real_file(self):
        self.assertIn(
            "NSDraggingItem(pasteboardWriter:",
            self.src,
            "the drag must put a real file URL on the pasteboard",
        )
        self.assertIn(
            "isFileURL",
            self.src,
            "a drag candidate must be checked for the file scheme before it is used",
        )


class NoCompetingDragSourceTests(unittest.TestCase):
    """WebKit must not be able to start a drag at all inside the shell.

    Two sources raced for the pasteboard: the shell's monitor (file://) and
    WebKit's DOM-element drag (the last http:// the img loaded). Whichever began
    first won, so results alternated between the right file and a link stub.
    """

    @classmethod
    def setUpClass(cls):
        cls.css = CSS.read_text(encoding="utf-8")

    def test_nothing_in_the_shell_is_draggable(self):
        self.assertRegex(
            self.css,
            r"body\.native-shell\s*\*?\s*\{[^}]*webkit-user-drag:\s*none",
            "the blanket rule that disables element dragging inside the shell is "
            "missing; WebKit will otherwise compete for the pasteboard",
        )

    def test_the_rule_is_scoped_to_the_native_shell(self):
        """The browser dev server has no native replacement, so it keeps web drag."""
        self.assertIn("body.native-shell", self.css)
        blanket = re.search(r"(body\.native-shell[^{]*)\{[^}]*webkit-user-drag:\s*none", self.css)
        self.assertIsNotNone(blanket)
        self.assertNotIn(
            "webkit-user-drag",
            self.css.split("body.native-shell")[0].split("/* Drag source")[-1][-2000:],
            "an unscoped -webkit-user-drag rule would disable dragging in the browser "
            "dev server too, where there is no native session to fall back to",
        )

    def test_no_element_in_the_frontend_is_marked_draggable(self):
        """A draggable attribute is all WebKit needs to become a second source."""
        offenders = []
        for path in FRONTEND.rglob("*.jsx"):
            text = strip_comments(path.read_text(encoding="utf-8"))
            for match in re.finditer(r"\bdraggable\b", text):
                line = text.count("\n", 0, match.start()) + 1
                # draggable={...} is fine -- it resolves to false inside the shell.
                context = text[match.start(): match.start() + 40].splitlines()[0]
                if context.lstrip().startswith("draggable={"):
                    continue
                offenders.append(f"{path.relative_to(REPO)}:{line} {context.strip()[:40]}")
        self.assertEqual(
            offenders, [],
            "these elements are draggable by WebKit and will race the shell's monitor:\n"
            + "\n".join(offenders),
        )

    def test_the_transparent_anchor_overlay_is_gone(self):
        """It was added as 'inert' but carried draggable, which is all WebKit needs."""
        for path in FRONTEND.rglob("*.jsx"):
            self.assertNotIn(
                "drag-anchor",
                strip_comments(path.read_text(encoding="utf-8")),
                f"{path.relative_to(REPO)} still renders a drag anchor; it competes "
                "with the native session and drops an http:// link instead of the file",
            )


class CursorResolutionTests(unittest.TestCase):
    """The dragged file is resolved from the cursor, not from cached geometry."""

    @classmethod
    def setUpClass(cls):
        cls.gallery = (FRONTEND / "components" / "Gallery.jsx").read_text(encoding="utf-8")
        cls.dragdrop = (FRONTEND / "utils" / "dragDrop.js").read_text(encoding="utf-8")
        cls.shell = SHELL.read_text(encoding="utf-8")

    def test_the_page_reports_the_cursor_target(self):
        self.assertIn(
            "dragCandidate",
            self.gallery,
            "the page must report which file is under the cursor; the shell has no "
            "public way to convert a mouse point into page coordinates",
        )
        self.assertIn("elementFromPoint", self.gallery)

    def test_the_hit_test_is_throttled_with_a_timer_not_raf(self):
        """rAF stops entirely when the window is occluded.

        A cursor position that has silently stopped updating resolves to whatever
        was last seen, which is precisely the wrong-image bug this replaced.
        """
        code = strip_comments(self.gallery)
        self.assertIn("setTimeout", code)
        self.assertNotIn(
            "requestAnimationFrame",
            code,
            "rAF is throttled to a stop when the window is occluded, which would "
            "freeze the cursor position; use a short timer",
        )

    def test_there_is_no_cached_rect_index_in_the_resolution_path(self):
        """A rect index drifted from the live layout: grab N dropped N+4."""
        # The index may still exist as a fallback for a press with no preceding
        # pointermove, but it must never be the primary answer.
        self.assertIn("currentDragCandidate", self.shell)
        shell_code = strip_comments(self.shell)
        self.assertRegex(
            shell_code,
            r"guard let url = self\.nativeBridge\.currentDragCandidate",
            "the drag must take its file from the cursor hit-test the page pushed",
        )
        self.assertNotIn(
            "imageIdAtPoint?(point)",
            shell_code,
            "resolving the image from the cached rect index produced 'grab N, drop N+4'; "
            "the page's cursor hit-test must be the only source",
        )

    def test_nodelist_is_not_treated_as_an_array(self):
        """This one failed silently: every publish returned early and nothing broke
        visibly, the shell simply had no information at all."""
        self.assertNotRegex(
            self.dragdrop,
            r"Array\.isArray\s*\(\s*nodes\s*\)",
            "querySelectorAll returns a NodeList; Array.isArray(nodes) is always false "
            "and silently stops the page reporting anything",
        )
        self.assertIn("typeof nodes.length", self.dragdrop)


class ImagesAreTaggedForResolutionTests(unittest.TestCase):
    """Everything the user might grab must carry its on-disk path."""

    def test_gallery_and_canvas_images_expose_their_file_url(self):
        for name in ("Gallery.jsx", "ResultCanvas.jsx"):
            with self.subTest(component=name):
                text = (FRONTEND / "components" / name).read_text(encoding="utf-8")
                self.assertIn("data-mlx-file-url", text)
                self.assertIn("data-mlx-image-id", text)


if __name__ == "__main__":
    unittest.main()