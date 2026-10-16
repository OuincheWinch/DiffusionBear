"""Tests for tools/harvest_model_mapping.py.

The harvester reads Draw Things' own `ModelMapping.swift` and is the only thing
standing between a silently partial name table and a converter that drops
weights without saying so. It had no tests, which is how a fused-row regex and
a missed assignment form both shipped. These tests pin the two defect classes
and the loudness rule, not the happy path.

Row-count pins for the committed tables live in `test_drawthings_names.py`;
these use synthetic sources and test the parser itself.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
TOOLS = BACKEND.parent / "tools"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _load_harvester():
    """Import the tool by path: `tools/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "harvest_model_mapping", TOOLS / "harvest_model_mapping.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harvest = _load_harvester()

# A closure-built text table in the shape ModelMapping.swift actually uses: a
# fused row carrying three ids, rows inside a loop, and rows outside it.
TEXT_TABLE = '''
  public static let OpenCLIPTextModelG: ModelWeightMapping = {
    var textModel = ModelWeightMapping()
    textModel["cond_stage_model.model.positional_embedding"] = [
      "t-0-0"
    ]
    for i in 0..<2 {
      textModel["...resblocks.\\(i).ln_1.weight"] = ["t-\\(i * 8 + 2)-0"]
      textModel[
        "...resblocks.\\(i).attn.in_proj_weight"] = [
          "t-\\(i * 8 + 3)-0", "t-\\(i * 8 + 4)-0", "t-\\(i * 8 + 5)-0",
        ]
      textModel["...resblocks.\\(i).mlp.c_proj.bias"] = ["t-\\(i * 8 + 9)-1"]
    }
    textModel["cond_stage_model.model.ln_final.weight"] = ["t-258-0"]
    return textModel
  }()
'''

LITERAL_TABLE = '''
  public static let UNetXLBase: ModelWeightMapping = [
    "model.diffusion_model.time_embed.0.weight": ["t-0-0"],
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight": [
      "t-34-0", "t-33-0",
    ],
  ]
'''


class FusedRowTests(unittest.TestCase):
    """A fused row carries several ids; dropping all but one loses weights."""

    def setUp(self):
        self.rows = harvest.harvest(TEXT_TABLE)["OpenCLIPTextModelG"]

    def test_fused_row_keeps_every_id(self):
        self.assertEqual(
            self.rows["...resblocks.0.attn.in_proj_weight"],
            ["t-3-0", "t-4-0", "t-5-0"],
        )

    def test_fused_row_expands_for_every_loop_value(self):
        self.assertEqual(
            self.rows["...resblocks.1.attn.in_proj_weight"],
            ["t-11-0", "t-12-0", "t-13-0"],
        )

    def test_single_id_row_is_unchanged(self):
        self.assertEqual(self.rows["...resblocks.0.ln_1.weight"], ["t-2-0"])

    def test_every_loop_value_produces_a_row(self):
        # 2 loop values x 3 assignments inside the loop.
        self.assertEqual(len([k for k in self.rows if "resblocks" in k]), 6)


class NonLoopRowTests(unittest.TestCase):
    """The text tables are subscript assignments, not dictionary literals.

    `ENTRY_RE` only matches the `"key": [...]` literal form, so every row
    outside the `for` loop was silently lost -- four per text table.
    """

    def setUp(self):
        self.rows = harvest.harvest(TEXT_TABLE)["OpenCLIPTextModelG"]

    def test_row_before_the_loop_is_harvested(self):
        self.assertEqual(
            self.rows["cond_stage_model.model.positional_embedding"], ["t-0-0"]
        )

    def test_row_after_the_loop_is_harvested(self):
        self.assertEqual(self.rows["cond_stage_model.model.ln_final.weight"], ["t-258-0"])

    def test_text_table_row_count_includes_both_outside_rows(self):
        self.assertEqual(len(self.rows), 8)


class LiteralTableTests(unittest.TestCase):
    """The UNet tables are dictionary literals, and already worked."""

    def setUp(self):
        self.rows = harvest.harvest(LITERAL_TABLE)["UNetXLBase"]

    def test_literal_rows_are_harvested(self):
        self.assertEqual(
            self.rows["model.diffusion_model.time_embed.0.weight"], ["t-0-0"]
        )

    def test_literal_fused_row_keeps_every_id(self):
        key = (
            "model.diffusion_model.input_blocks.4.1."
            "transformer_blocks.0.ff.net.0.proj.weight"
        )
        self.assertEqual(self.rows[key], ["t-34-0", "t-33-0"])


class LoudnessTests(unittest.TestCase):
    """Standing rule: the harvester must fail loudly on anything it skips.

    A harvest that drops rows must never exit 0. Silent partial data is worse
    than no data -- the converter then emits a file that loads and quietly
    ignores part of the LoRA.
    """

    def test_assignment_with_no_recognised_id_fails(self):
        source = TEXT_TABLE.replace('["t-\\(i * 8 + 2)-0"]', '["t-\\(i * 8 + 2)-x"]')
        with self.assertRaises(SystemExit) as cm:
            harvest.harvest(source)
        self.assertIn("ln_1.weight", str(cm.exception))

    def test_interpolated_row_outside_any_loop_fails(self):
        # An interpolated key no loop can expand would be dropped silently.
        source = TEXT_TABLE.replace(
            'textModel["cond_stage_model.model.positional_embedding"] = [\n'
            '      "t-0-0"\n    ]',
            'textModel["cond_stage_model.model.ghost.\\(i)"] = ["t-0-0"]',
        )
        with self.assertRaises(SystemExit) as cm:
            harvest.harvest(source)
        self.assertIn("ghost", str(cm.exception))

    def test_two_different_ids_for_one_key_fails(self):
        # Swift would let the second overwrite the first; the harvest must not.
        source = LITERAL_TABLE.replace(
            '"t-34-0", "t-33-0",', '"t-34-0", "t-33-0", "t-32-0",'
        ).replace(
            '  ]\n', '    "model.diffusion_model.time_embed.0.weight": ["t-9-9"],\n  ]\n'
        )
        with self.assertRaises(SystemExit) as cm:
            harvest.harvest(source)
        self.assertIn("time_embed.0.weight", str(cm.exception))

    def test_a_clean_source_does_not_fail(self):
        self.assertTrue(harvest.harvest(TEXT_TABLE))
        self.assertTrue(harvest.harvest(LITERAL_TABLE))


class MissingTableTests(unittest.TestCase):
    """A wanted table with no rows is an error, not an empty dict."""

    def test_missing_table_is_reported(self):
        tables = harvest.harvest(LITERAL_TABLE)
        self.assertEqual(harvest.missing_tables(tables, ("UNet",)), ["UNet"])

    def test_present_table_is_not_reported(self):
        tables = harvest.harvest(LITERAL_TABLE)
        self.assertEqual(harvest.missing_tables(tables, ("UNetXLBase",)), [])


class LoopFormTests(unittest.TestCase):
    """Revision 060f99d has exactly four loops, all `for i in 0..<N`.

    A future revision adding a form `LOOP_RE` cannot see would lose every row
    inside it, so the harvester reports unsupported loops rather than assuming.
    """

    def test_a_supported_loop_is_not_reported(self):
        self.assertEqual(harvest.unrecognised_loops(TEXT_TABLE), [])

    def test_an_unsupported_loop_form_is_reported(self):
        source = TEXT_TABLE.replace(
            "for i in 0..<2 {", "for case i in 0..<2 {"
        )
        self.assertEqual(harvest.unrecognised_loops(source), ["for case i in 0..<2 {"])

    def test_an_unsupported_loop_stops_the_harvest(self):
        source = TEXT_TABLE.replace(
            "for i in 0..<2 {", "for case i in 0..<2 {"
        )
        with self.assertRaises(SystemExit) as cm:
            harvest.harvest(source)
        self.assertIn("loop", str(cm.exception))

    def test_a_third_assignment_spelling_is_reported(self):
        # Symmetric to the loop guard. Only `"key": [...]` and
        # `var["key"] = [...]` are read; a third spelling would drop its row
        # silently, which is what this tool exists to prevent.
        source = TEXT_TABLE.replace(
            'textModel["cond_stage_model.model.ln_final.weight"] = ["t-258-0"]',
            'textModel[ghostKey] = ["t-258-0"]',
        )
        self.assertTrue(harvest.unrecognised_assignments(source))

    def test_an_unsupported_assignment_stops_the_harvest(self):
        source = TEXT_TABLE.replace(
            'textModel["cond_stage_model.model.ln_final.weight"] = ["t-258-0"]',
            'textModel[ghostKey] = ["t-258-0"]',
        )
        with self.assertRaises(SystemExit) as cm:
            harvest.harvest(source)
        self.assertIn("ghostKey", str(cm.exception))

    def test_both_known_spellings_are_read(self):
        self.assertEqual(harvest.unrecognised_assignments(TEXT_TABLE), [])
        self.assertEqual(harvest.unrecognised_assignments(LITERAL_TABLE), [])


if __name__ == "__main__":
    unittest.main()