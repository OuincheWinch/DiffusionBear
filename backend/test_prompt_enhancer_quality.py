"""The enhancer must not hand back its own instructions, or half a word.

Two failures, both seen in real output on 2026-10-13, both invisible to the suite that
shipped them:

1. PROMPT LEAK. On a 0.5B instruct model the system prompt sometimes comes back as the
   answer. The user saw, in their textarea:

       ## FLUX.2, Klein 4B (mflux/MLX; guidance-distilled flow-matching DiT with a T5
       text encoder). FLUX.2 attends most to the START. Order elements Main subject ->
       ...

   Every one of those sentences is a line of `ENGINE_PROFILES["flux2"]["instructions"]`.
   The model was reciting its instructions instead of writing a prompt. It also
   hallucinated `flux2kldm_lorafriendly.jpg?Expiries=0&Signature=bE16bDpwm...`, a
   presigned-URL fragment that exists nowhere in the registry -- `triggers` for that LoRA
   is exactly `['LOGO', 'logoredmaf']`.

   Nothing detected it, because nothing was looking. A leak is not a parse error.

2. TRUNCATION WITHOUT AN ENDING. The generation budget stops mid-token, so the prompt
   ended `...at the middle part of a helmet), or adding LOGO to the pendant. TheLOG`.
   That fragment was then stored as the prompt and used to generate. A word cut in half
   is not a prompt, and the caller had no way to know.

Both are silent by construction: `enhance_prompt` returns whatever the model produced.
The checks belong where the output is accepted, not in the UI.

What is deliberately NOT here: a test asserting the model produces good prose. That is
not a property a 0.5B model reliably has, and a test that fails intermittently teaches
everyone to ignore the suite. These tests assert only that garbage is refused rather
than passed on.
"""
import unittest
from unittest import mock

import prompt_enhancer as pe


class TestSystemPromptLeak(unittest.TestCase):
    """Refuse output that quotes the instructions back at the user."""

    def test_engine_header_is_detected_as_a_leak(self):
        leak = (
            "## FLUX.2, Klein 4B (mflux/MLX; guidance-distilled flow-matching DiT with "
            "a T5 text encoder).\nThe user concept is a helmet."
        )
        self.assertTrue(pe._looks_like_prompt_leak(leak))

    def test_ordering_rule_verbatim_is_detected(self):
        leak = ("FLUX.2 attends most to the START. Order elements Main subject -> Key "
                "action -> Critical style -> Essential context -> Secondary details.")
        self.assertTrue(pe._looks_like_prompt_leak(leak))

    def test_forbidden_clause_is_detected(self):
        leak = "STRICTLY FORBIDDEN: raw tag soup / comma chains, generic buzzwords."
        self.assertTrue(pe._looks_like_prompt_leak(leak))

    def test_a_real_prompt_is_not_a_leak(self):
        good = ("A weathered brass diving helmet rests on a weathered wooden dock in "
                "early morning fog, brass gone green at the seams, LOGO picked out on "
                "the front plate.")
        self.assertFalse(pe._looks_like_prompt_leak(good))

    def test_leak_check_actually_reads_the_live_profile(self):
        """A leak detector tuned against a hardcoded copy of the prompt is a detector
        for that copy. This asserts the phrases it looks for exist in the profile it
        ships with, so the two cannot drift apart unnoticed."""
        instructions = pe.ENGINE_PROFILES["flux2"]["instructions"]
        self.assertTrue(
            any(p in instructions for p in pe.LEAK_MARKERS),
            "LEAK_MARKERS share nothing with the live flux2 profile",
        )

    def test_engine_label_follows_the_model_not_a_hardcoded_4b(self):
        """The profile told the model 'Klein 4B' while the user ran the 9B."""
        self.assertNotIn(
            "Klein 4B", pe._engine_target_line("flux2", "flux2-klein-9b"),
            "a 9B request must not be described to the model as a 4B",
        )
        self.assertIn("9B", pe._engine_target_line("flux2", "flux2-klein-9b"))
        self.assertIn("4B", pe._engine_target_line("flux2", "flux2-klein-4b"))

    def test_model_id_passed_in_the_engine_slot_is_not_silently_ignored(self):
        """A caller that swaps the two arguments gets a plausible line with no size at
        all, rather than an error. That is how the 4B label survived: the wrong thing
        looked right."""
        line = pe._engine_target_line("flux2-klein-9b")
        self.assertNotIn("9B", line)
        self.assertIn("Target Engine:", line)


class TestTruncatedOutput(unittest.TestCase):
    """A prompt ending mid-word must not reach the generator."""

    def test_mid_word_cut_is_detected(self):
        self.assertTrue(pe._ends_mid_word("... adding LOGO to the pendant. TheLOG"))

    def test_clean_sentence_end_is_not(self):
        self.assertFalse(pe._ends_mid_word("... adding LOGO to the pendant."))

    def test_trailing_quote_is_a_clean_end(self):
        self.assertFalse(pe._ends_mid_word('A sign reading "OPEN"'))

    def test_unpunctuated_tail_is_treated_as_possibly_cut(self):
        """Deliberately conservative: no terminal punctuation means 'probably cut'. The
        consequence is only a trim to the last full sentence, and a text with no
        sentence boundary is returned untouched rather than emptied."""
        self.assertTrue(pe._ends_mid_word("A brass helmet above the wooden"))

    def test_truncation_is_trimmed_to_the_last_real_sentence(self):
        out = pe._trim_to_complete_sentence(
            "A brass diving helmet on a wooden dock in early fog. The dock is wet. TheLOG"
        )
        self.assertTrue(out.endswith("The dock is wet."))
        self.assertNotIn("TheLOG", out)

    def test_nothing_complete_leaves_the_text_alone_rather_than_empty(self):
        """Better a rough fragment than an empty box the user cannot see."""
        out = pe._trim_to_complete_sentence("TheLOG")
        self.assertTrue(out)

    def test_triggers_are_not_lost_when_trimming(self):
        """Trimming to a sentence boundary can cut a trailing trigger. It is prepended
        back, because losing the trigger silently changes what the LoRA does."""
        out = pe._trim_to_complete_sentence(
            "A brass diving helmet on a wooden dock. LOGO", mandatory_triggers=["LOGO"]
        )
        self.assertIn("LOGO", out)


class TestRejection(unittest.TestCase):
    """What enhance_prompt does when the output is unusable.

    The old behaviour was to return it. The user cannot tell a leaked system prompt from
    a good one, so the backend has to refuse and say so.
    """

    def test_leaked_output_is_not_returned_as_a_prompt(self):
        with mock.patch.object(pe, "_generate_raw", return_value=(
            "## FLUX.2, Klein 4B (mflux/MLX; guidance-distilled flow-matching DiT)."
        )):
            result = pe.enhance_prompt("a diving helmet", engine="flux2-klein-9b")
        self.assertTrue(result.get("rejected"), result)
        self.assertEqual(result["reason"], "prompt_leak")
        self.assertNotIn("flow-matching DiT", str(result.get("enhanced", "")))
        # The original must survive, so the UI can leave the box untouched.
        self.assertEqual(result["original"], "a diving helmet")

    def test_a_leaked_prompt_does_not_become_the_box_contents(self):
        """The specific damage: the textarea filled with instructions."""
        with mock.patch.object(pe, "_generate_raw", return_value=(
            "Order elements Main subject -> Key action -> Critical style."
        )):
            result = pe.enhance_prompt("a diving helmet", engine="flux2")
        self.assertEqual(result.get("enhanced", "").strip(), "")

    def test_truncated_output_is_trimmed_not_passed_through(self):
        with mock.patch.object(pe, "_generate_raw", return_value=(
            "A brass diving helmet on a wooden dock in fog. The dock is wet. TheLOG"
        )):
            result = pe.enhance_prompt("a diving helmet", engine="flux2")
        self.assertNotIn("TheLOG", result["enhanced"])
        self.assertTrue(result["enhanced"].strip())


class TestJsonFallback(unittest.TestCase):
    """When JSON parsing fails, the fallback must be prose or nothing.

    The fallback took the model's raw output and, if it was not parseable JSON, shipped
    it as a plain text prompt. The user got this in the box:

        logoredmaf, LOGO, { "subject": "A weathered brass diving helmet ...

    i.e. a broken JSON document with trigger words glued to the front of it, which the
    diffusion model then renders as literal characters. `parse_error` was set, so the app
    KNEW, and displayed the rubble anyway.
    """

    BROKEN = 'logoredmaf, LOGO, { "subject": "A brass helmet", "appearance": "rusty",'

    def test_json_fallback_is_not_the_broken_json(self):
        self.assertIsNone(pe._json_fallback_prose(self.BROKEN))

    def test_genuine_prose_survives_as_fallback(self):
        prose = "A weathered brass diving helmet rests on a wooden dock in early morning fog."
        self.assertEqual(pe._json_fallback_prose(prose), prose)

    def test_unparseable_json_output_is_rejected_not_rendered(self):
        with mock.patch.object(pe, "_generate_raw", return_value=self.BROKEN):
            result = pe.enhance_prompt("a diving helmet", engine="flux2-klein-9b",
                                       output_format="json")
        self.assertTrue(result.get("rejected"), result)
        self.assertEqual(result.get("reason"), "json_unparseable")
        self.assertNotIn('"subject"', str(result.get("enhanced", "")))


if __name__ == "__main__":
    unittest.main()