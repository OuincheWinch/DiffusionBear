"""A button that fails silently is indistinguishable from a broken app.

Ten handlers had a bare `catch {}` and eight of them were clipboard writes. The modern
clipboard API only exists in a SECURE context -- localhost counts, but the same app
opened from another machine on the LAN does not -- and the rejection was swallowed, so
the button did nothing and said nothing. That is the "some buttons don't work" report.

This bans the bare form in the components and requires the shared helper instead. It is
deliberately narrow: an empty catch is occasionally correct, but not on a user action.
"""

import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"


def _comment_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of every comment, so a match inside one can be skipped."""
    spans = []
    for m in re.finditer(r"/\*.*?\*/|//[^\n]*", text, flags=re.S):
        spans.append((m.start(), m.end()))
    return spans


def _inside(spans, pos: int) -> bool:
    return any(a <= pos < b for a, b in spans)


def _strip_js_comments(text: str) -> str:
    """Remove comments while preserving offsets, so reported line numbers stay right.

    Without this the test matched the phrase "catch {}" inside a comment explaining why
    the empty catches were removed -- a comment written two commits earlier to explain
    this very fix. A guard that reads its own documentation as a violation is noise.
    """
    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))
    text = re.sub(r"/\*.*?\*/", blank, text, flags=re.S)
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return text


class NoSilentCatchTests(unittest.TestCase):
    def _components(self):
        return sorted(p for p in SRC.rglob("*.jsx") if "node_modules" not in str(p))

    def test_the_scan_found_components(self):
        self.assertGreater(len(self._components()), 10)

    def test_no_component_has_an_UNDOCUMENTED_empty_catch(self):
        """A catch with nothing in it is fine IF it says why.

        Polling loops legitimately swallow a transient error and keep the last known
        state -- the comment is the part that matters, because without it the next
        reader cannot tell a decision from an oversight.

        The check therefore looks at the ORIGINAL text: an empty body that carries a
        comment passes, an empty body with nothing in it fails. An earlier version
        stripped comments first and then flagged exactly those documented ones -- three
        false failures, and it would have driven someone to delete good comments.
        """
        offenders = []
        for path in self._components():
            text = path.read_text(encoding="utf-8")
            spans = _comment_spans(text)
            for m in re.finditer(r"catch\s*(?:\(\s*\w+\s*\))?\s*\{([^{}]*)\}", text):
                # A match found INSIDE a comment is prose, not code. The comment that
                # documents removing these catches quotes the pattern verbatim.
                if _inside(spans, m.start()):
                    continue
                body = m.group(1)
                if not _strip_js_comments(body).strip() and not body.strip():
                    line = text[: m.start()].count("\n") + 1
                    offenders.append(f"{path.name}:{line}")
        self.assertEqual(
            offenders, [],
            "empty catch with no explanation -- add a comment saying why it is safe, "
            "or handle it: " + ", ".join(offenders),
        )

    def test_a_catch_must_at_least_say_something(self):
        """A `catch (e) {}` with a renamed binding is the same defect."""
        offenders = []
        for path in self._components():
            text = _strip_js_comments(path.read_text(encoding="utf-8"))
            spans = _comment_spans(text)
            for m in re.finditer(r"catch\s*\(\s*\w+\s*\)\s*\{\s*\}", text):
                if _inside(spans, m.start()):
                    continue
                line = text[: m.start()].count("\n") + 1
                offenders.append(f"{path.name}:{line}")
        self.assertEqual(offenders, [], "unused catch binding with an empty body: "
                         + ", ".join(offenders))

    def test_clipboard_writes_go_through_the_helper(self):
        """Not `navigator.clipboard.writeText` inline.

        The helper is what falls back to the legacy path and reports failure; a direct
        call in a component reintroduces the silent version.
        """
        offenders = []
        for path in self._components():
            text = path.read_text(encoding="utf-8")
            if path.name == "clipboard.js":
                continue
            if re.search(r"navigator\.clipboard\.writeText", text):
                offenders.append(path.name)
        self.assertEqual(offenders, [],
                         "components must use copyText() from src/clipboard.js, not "
                         f"navigator.clipboard directly: {offenders}")

    def test_the_helper_reports_failure_rather_than_throwing(self):
        src = (SRC / "clipboard.js").read_text(encoding="utf-8")
        self.assertIn("export async function copyText", src)
        self.assertIn("legacyCopy", src,
                      "the helper lost its fallback, so it fails outside a secure context")
        self.assertRegex(src, r"return false",
                         "the helper must report failure to the caller")


if __name__ == "__main__":
    unittest.main()
