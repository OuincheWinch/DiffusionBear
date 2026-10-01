"""The size selector's dimension arithmetic.

Every number produced here reaches the sampler, and mflux rejects a dimension
that is not a multiple of 16 -- a bad value is a failed generation, not a
cosmetic glitch.

The functions are EXTRACTED FROM THE COMPONENT AND RUN UNDER NODE rather than
re-implemented in Python. A re-implementation would test the copy rather than the
shipped code, which is precisely how the original bug survived review: the copy
had the arithmetic, the component had a different line, and only the component
runs in the app.

The case that matters most is base 256 with a non-square ratio. A first pass
clamped the short edge up to the 256 floor, which turned 16:9 at base 256 into
256x256 -- silently discarding the aspect the user had just chosen, in a control
whose entire purpose is choosing an aspect.
"""

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[1] / "frontend" / "src" / "components" / "SizeSelector.jsx"

MULTIPLE = 16
FLOOR = 256
BASES = [256, 512, 768, 1024]
RATIOS = [
    ("1:1", 1, 1),
    ("16:9", 16, 9),
    ("3:2", 3, 2),
    ("9:16", 9, 16),
    ("2:3", 2, 3),
]

HARNESS = """
const out = { cases: [], budget: [] };
for (const base of %(bases)s) {
  for (const r of %(ratios)s) {
    const d = dimensionsFor(base, { w: r[1], h: r[2] });
    out.cases.push({ base, ratio: r[0], width: d.width, height: d.height });
    const cap = 1024 * 768;
    if (d.width * d.height > cap) {
      const c = fitToBudget(d.width, d.height, cap);
      out.budget.push({ base, ratio: r[0], width: c.width, height: c.height, cap });
    }
  }
}
out.uncapped = fitToBudget(1024, 1024, null);
console.log(JSON.stringify(out));
"""


def _extract(name):
    src = COMPONENT.read_text(encoding="utf-8")
    match = re.search(rf"(?:const|function) {name}\s*(?:=\s*)?\(?.*?\n\}}", src, flags=re.S)
    if not match:
        raise AssertionError(f"{name} not found in {COMPONENT.name}")
    return match.group(0)


class SizeArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not COMPONENT.is_file():
            raise unittest.SkipTest("SizeSelector.jsx is not present")
        node = shutil.which("node")
        if not node:
            raise unittest.SkipTest("node is unavailable; cannot execute the component's own maths")

        constants = re.search(r"const MULTIPLE = \d+;\s*\nconst FLOOR = \d+;", COMPONENT.read_text())
        if not constants:
            raise AssertionError("MULTIPLE/FLOOR constants are missing")

        program = (
            constants.group(0)
            + "\n"
            + _extract("snap")
            + "\n"
            + _extract("dimensionsFor")
            + "\n"
            + _extract("fitToBudget")
            + "\n"
            + HARNESS % {"bases": json.dumps(BASES), "ratios": json.dumps(RATIOS)}
        )
        proc = subprocess.run([node, "-e", program], capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise AssertionError(f"the component's maths did not run:\n{proc.stderr[:600]}")
        cls.data = json.loads(proc.stdout.strip().splitlines()[-1])

    def test_every_base_and_ratio_is_a_multiple_of_16(self):
        for case in self.data["cases"]:
            with self.subTest(base=case["base"], ratio=case["ratio"]):
                self.assertEqual(case["width"] % MULTIPLE, 0, f"width {case['width']}")
                self.assertEqual(case["height"] % MULTIPLE, 0, f"height {case['height']}")

    def test_nothing_falls_below_the_floor(self):
        for case in self.data["cases"]:
            with self.subTest(base=case["base"], ratio=case["ratio"]):
                self.assertGreaterEqual(case["width"], FLOOR)
                self.assertGreaterEqual(case["height"], FLOOR)

    def test_the_chosen_ratio_survives(self):
        """The regression: clamping the short edge collapsed every ratio at base 256
        into a square, silently discarding what the user had just picked."""
        for case in self.data["cases"]:
            _, w, h = next(r for r in RATIOS if r[0] == case["ratio"])
            with self.subTest(base=case["base"], ratio=case["ratio"]):
                long = max(case["width"], case["height"])
                short = min(case["width"], case["height"])
                want = max(w, h) / min(w, h)
                self.assertAlmostEqual(
                    long / short, want, delta=want * 0.10,
                    msg=f"{case['base']} {case['ratio']} produced {case['width']}x{case['height']}",
                )

    def test_base_256_is_not_square_for_every_non_square_ratio(self):
        """Named explicitly because it is the case that broke."""
        square = next(c for c in self.data["cases"] if c["base"] == 256 and c["ratio"] == "1:1")
        self.assertEqual(square["width"], square["height"], "1:1 at 256 should be square")
        for case in self.data["cases"]:
            if case["base"] != 256 or case["ratio"] == "1:1":
                continue
            with self.subTest(ratio=case["ratio"]):
                self.assertNotEqual(
                    case["width"], case["height"],
                    f"{case['ratio']} at base 256 must not collapse to a square",
                )

    def test_base_applies_to_the_long_edge(self):
        for case in self.data["cases"]:
            if case["base"] != 1024:
                continue
            with self.subTest(ratio=case["ratio"]):
                self.assertEqual(max(case["width"], case["height"]), 1024)

    def test_budget_clamp_stays_within_the_cap(self):
        self.assertTrue(self.data["budget"], "expected at least one case to exceed the cap")
        for case in self.data["budget"]:
            with self.subTest(base=case["base"], ratio=case["ratio"]):
                self.assertLessEqual(
                    case["width"] * case["height"], case["cap"] * 1.02,
                    f"{case['base']} {case['ratio']} clamped to {case['width']}x{case['height']}",
                )

    def test_a_missing_budget_does_not_clamp(self):
        self.assertEqual(self.data["uncapped"]["width"], 1024)
        self.assertEqual(self.data["uncapped"]["height"], 1024)


class SizeSelectorStructureTests(unittest.TestCase):
    """The control must keep the two decisions separate and Custom reachable."""

    @classmethod
    def setUpClass(cls):
        raw = COMPONENT.read_text(encoding="utf-8")
        cls.raw = raw
        # Strip comments without letting DOTALL turn "//" into "to end of file".
        cls.code = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
        cls.code = re.sub(r"//[^\n]*", "", cls.code)

    def test_custom_is_reachable_and_switches_mode(self):
        self.assertIn("params.size.customShort", self.raw)
        self.assertIn('mode === "custom"', self.code)

    def test_custom_exposes_two_numeric_fields(self):
        self.assertIn("params.size.widthLabel", self.raw)
        self.assertIn("params.size.heightLabel", self.raw)
        self.assertEqual(self.code.count('type="number"'), 2)

    def test_the_resolved_pixel_size_is_shown(self):
        """Two button presses produce a number; the user should read it, not infer it."""
        self.assertIn("size-selector-readout", self.code)

    def test_base_and_ratio_are_independent_controls(self):
        self.assertIn("size-base-row", self.code)
        self.assertIn("size-ratio-row", self.code)

    def test_it_adopts_a_size_set_from_outside(self):
        """A hydrated request or a model switch changes width/height behind the
        control. Without an effect the buttons would show a stale selection while
        the dimensions underneath had changed."""
        self.assertRegex(
            self.code,
            r"useEffect\(\(\)\s*=>\s*\{[\s\S]{0,400}?derive\(width, height\)",
            "the selector must re-derive its state when width/height change externally",
        )

    def test_the_old_fused_dropdown_is_gone(self):
        params = Path(__file__).resolve().parents[1] / "frontend" / "src" / "components" / "GenerationParams.jsx"
        code = re.sub(r"//[^\n]*", "", params.read_text(encoding="utf-8"))
        self.assertNotIn("STANDARD_SIZES", code, "the fused pixel dropdown is back")
        self.assertNotIn("size-row", code, "the fused pixel dropdown is back")


if __name__ == "__main__":
    unittest.main()

class CustomModeTests(unittest.TestCase):
    """Custom mode must survive the user typing in it.

    The reported symptom was "custom size is bugged". The cause was the sync effect:
    it re-derives base+ratio from width/height on every change, and a size the user
    typed that happened to land on the grid was read as an EXTERNAL change, which
    switched the control back to standard mode mid-keystroke. The user was fighting
    the control.
    """

    @classmethod
    def setUpClass(cls):
        raw = COMPONENT.read_text(encoding="utf-8")
        cls.code = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
        cls.code = re.sub(r"//[^\n]*", "", cls.code)

    def test_our_own_writes_are_not_mistaken_for_external_changes(self):
        self.assertIn("selfWriteRef", self.code)
        self.assertRegex(
            self.code,
            r"if \(selfWriteRef\.current === key\)",
            "the sync effect must skip dimensions this component just wrote",
        )

    def test_every_write_path_is_marked(self):
        """Both the ratio/base buttons and the Custom inputs must mark their write."""
        self.assertGreaterEqual(
            self.code.count("selfWriteRef.current ="),
            2,
            "both apply() and the custom-field setter must mark their own write",
        )

    def test_custom_mode_has_a_keep_ratio_control(self):
        self.assertIn("lockRatio", self.code)
        self.assertIn("params.size.lockRatio", self.code)

    def test_keep_ratio_derives_the_other_side(self):
        self.assertRegex(
            self.code,
            r"if \(lockRatio\)",
            "with the ratio locked, editing one side must derive the other",
        )

    def test_the_budget_warning_ignores_an_absent_cap(self):
        """Number(null) is 0, which is finite and below every real size, so a model
        with no pixel cap rendered a permanent red "Over 0 px" warning."""
        self.assertRegex(
            self.code,
            r"maxPixels == null",
            "an absent pixel cap must not be treated as a budget of zero",
        )
        self.assertIn("hasBudget", self.code)

    def test_the_four_new_ratios_are_present(self):
        for ratio in ("4:3", "3:4", "21:9", "9:21"):
            with self.subTest(ratio=ratio):
                self.assertRegex(self.code, rf'id: "{re.escape(ratio)}"')


class StructureRegressionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "frontend" / "src"
        cls.form = re.sub(r"//[^\n]*", "", (root / "components" / "GenerateForm.jsx").read_text())
        cls.params = re.sub(r"//[^\n]*", "", (root / "components" / "ParametersTab.jsx").read_text())
        cls.app = re.sub(r"//[^\n]*", "", (root / "App.jsx").read_text())

    def test_lora_panel_is_its_own_disclosure(self):
        self.assertIn("lora-disclosure", self.form)
        self.assertIn("<summary", self.form)

    def test_the_lora_disclosure_opens_when_a_lora_is_active(self):
        self.assertRegex(self.form, r"open=\{loras\.length > 0\}")

    def test_licences_is_a_parameters_subtab_not_a_top_level_tab(self):
        self.assertIn("subtab === \"licences\"", cls := self.params)
        self.assertNotIn('tab === "licences"', self.app)
        self.assertNotIn("LicencesTab", self.app)

    def test_preset_buttons_are_still_gone(self):
        self.assertNotIn("preset-btn", self.form)
        self.assertNotIn("PRESETS", self.form)
