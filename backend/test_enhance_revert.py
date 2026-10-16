"""After enhancing, the user must be able to put their prompt back.

The enhancer overwrites the textarea in place. Before this, the text the user typed was
gone the moment it ran -- and because the 0.5B model intermittently returns its own
instructions or half a word, "restore what I wrote" is the escape hatch that makes
enhancing safe to try at all.

Three properties, each of which can fail alone:

  * the ORIGINAL prompt is captured before the box is overwritten, not after;
  * a rejection does NOT overwrite the box, so there is nothing to restore;
  * the control disappears once the prompt is back, rather than offering to revert to
    the text already in the box.

Static source checks, in the same style as test_frontend_no_silent_failures.py: the
frontend has no unit-test runner here, so these assert structure rather than behaviour.
They are narrow on purpose -- a broad "the feature exists" grep would pass on a revert
button wired to the wrong state.
"""
import re
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"
FORM = SRC / "components" / "GenerateForm.jsx"

TEXT = FORM.read_text(encoding="utf-8")


def _function_body(name: str) -> str:
    """Source of a top-level function, by brace matching from its `def`."""
    start = TEXT.index(f"function {name}(")
    depth, i = 0, TEXT.index("{", start)
    while i < len(TEXT):
        if TEXT[i] == "{":
            depth += 1
        elif TEXT[i] == "}":
            depth -= 1
            if depth == 0:
                return TEXT[start:i + 1]
        i += 1
    raise AssertionError(f"unterminated function {name}")


class TestRevertControl(unittest.TestCase):
    def test_original_is_captured_before_the_box_is_overwritten(self):
        """Order is the whole point. Capturing after setPrompt would store the
        enhancement and 'revert' would be a no-op that looks like it works."""
        body = _function_body("runEnhancePrompt")
        capture = body.index("setPreEnhancePrompt(originalPrompt)")
        overwrite = body.index("setPrompt(res.enhanced)")
        self.assertLess(capture, overwrite,
                        "the original prompt is captured after it is overwritten")

    def test_original_prompt_is_snapshotted_from_the_live_prompt(self):
        body = _function_body("runEnhancePrompt")
        self.assertRegex(body, r"const originalPrompt = prompt;")

    def test_revert_restores_the_snapshot_and_clears_it(self):
        body = _function_body("revertEnhancedPrompt")
        self.assertIn("setPrompt(preEnhancePrompt)", body)
        # Clearing is what hides the control; leaving it set would offer to revert to
        # the text already in the box.
        self.assertIn("setPreEnhancePrompt(null)", body)

    def test_control_is_hidden_once_there_is_nothing_to_restore(self):
        self.assertRegex(
            TEXT,
            r"preEnhancePrompt != null && preEnhancePrompt !== prompt",
        )

    def test_rejected_output_leaves_the_prompt_box_alone(self):
        """A refusal must not write an empty string into the box: that reads as the app
        deleting the user's work, and it destroys the text revert would restore."""
        body = _function_body("runEnhancePrompt")
        rejected = body.index("res?.rejected")
        overwrite = body.index("setPrompt(res.enhanced)")
        self.assertLess(rejected, overwrite)
        segment = body[rejected:overwrite]
        self.assertNotIn("setPrompt(", segment,
                         "the rejection branch writes to the prompt box")

    def test_rejection_is_surfaced_not_swallowed(self):
        body = _function_body("runEnhancePrompt")
        self.assertIn("generate.enhanceFeedback.rejected", body)

    def test_revert_strings_exist_in_every_language(self):
        """A missing key renders as the raw identifier, which is how this was noticed
        the first time: test_i18n.py exists precisely so that cannot ship."""
        for key in ("generate.enhanceFeedback.revert",
                    "generate.enhanceFeedback.originalLabel",
                    "generate.enhanceFeedback.reverted",
                    "generate.enhanceFeedback.rejected"):
            self.assertIn(key, TEXT, f"{key} is never used")
            parts = (SRC / "i18n" / "parts" / "generate.js").read_text(encoding="utf-8")
            self.assertIn(key, parts, f"{key} missing from the reference catalogue")
            generated = SRC / "i18n" / "lang" / "_parts" / "generate.json"
            self.assertIn(key, generated.read_text(encoding="utf-8"),
                          f"{key} missing from the generated catalogue")


if __name__ == "__main__":
    unittest.main()