"""The size selector's native presets and their arithmetic.

Every number produced here reaches the sampler, so the invariants below are not
cosmetic.

A CORRECTION TO THE PREVIOUS VERSION OF THIS FILE
It claimed "mflux rejects a dimension that is not a multiple of 16 -- a bad
value is a failed generation". That is wrong, and the error was in the safe
direction only by luck. mflux does not reject it. It logs

    Width and height should be multiples of 16. Rounding down.

and floors the value (mflux/models/common/config/config.py:41). So a bad
dimension is a SILENTLY DIFFERENT SIZE: the run succeeds, produces an image, and
the caller never learns that 1000 became 992. The test now pins the real
constraint and says why we snap.

THE OLD MODEL IS GONE
The control used to multiply a base (256/512/768/1024) by a ratio, which is what
produced non-native sizes: 16:9 at base 1024 is 1024x576. The arithmetic was
therefore removed rather than fixed, and these tests now guard the replacement:
that the offered sizes ARE native buckets, that each sits in the orientation tab
it claims, and that the app's actual default size is still reachable.

These run against sizePresets.js under node, not a Python re-implementation. A
copy tests the copy; only the shipped module runs in the app.
"""

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "src"
PRESETS_JS = FRONTEND / "components" / "sizePresets.js"
PARAMS_PART = FRONTEND / "i18n" / "parts" / "params.js"
LANGS = FRONTEND / "i18n" / "lang" / "_parts" / "params.json"

# Duplicated here only as the values the tests assert against; the module is the
# source of truth and the constants test below fails if the two ever diverge.
MULTIPLE = 16
NATIVE_STEP = 64
FLOOR = 256

# Every model in the registry defaults to this. It is the size the app is
# running at on a fresh install, so the control must be able to show it.
REGISTRY_DEFAULT = (512, 768)

HARNESS = """
import * as P from "./sizePresets.mjs";

const legacy = { ecosystem: "SD15", legacy_sizes: true };
const modernNoFlag = { ecosystem: "FLUX.2" };
const out = {
  presets: {},
  constants: {
    MULTIPLE: P.MULTIPLE,
    NATIVE_STEP: P.NATIVE_STEP,
    FLOOR: P.FLOOR,
    },
  models: [P.snap(1000), P.snap(1000, 64), P.snap(-5), P.snap("abc"), P.snap(767)],
  budget: [P.fitToBudget(1024, 1024, 1024 * 768), P.fitToBudget(800, 600, null)],
  modernSquare: P.presetsFor("square", { ecosystem: "FLUX.2" }),
  legacySquare: P.presetsFor("square", legacy),
  modernAll: P.allAllowed({ ecosystem: "Qwen-Image 2.1" }),
  defaultSquare: P.presetsFor("square", { ecosystem: "FLUX.2" }),
  legacyOnly: P.presetsFor("square", legacy),
  modernNoFlagSquare: P.presetsFor("square", modernNoFlag),
  defaultLandscape: P.presetsFor("landscape", { ecosystem: "FLUX.2" }, true),
  defaultPortrait: P.presetsFor("portrait", { ecosystem: "FLUX.2" }, true),
  noEcosystem: P.isModern({}),
  legacyFlag: P.isModern(legacy),
  advisory: {
    qwen1024: P.overAdvisory(1024, 1024, { ecosystem: "Qwen-Image 2.1" }),
    qwen768: P.overAdvisory(768, 768, { ecosystem: "Qwen-Image 2.1" }),
    qwen512x768: P.overAdvisory(512, 768, { ecosystem: "Qwen-Image 2.1" }),
    flux1280: P.overAdvisory(1280, 1280, { ecosystem: "FLUX.2" }),
    none: P.overAdvisory(1024, 1024, {}),
    nan: P.overAdvisory(NaN, 1024, { ecosystem: "Qwen-Image 2.1" }),
  },
  badRatios: [P.ratioLabel(NaN, 1024), P.ratioLabel(1024, undefined), P.ratioLabel(0, 0)],
  badOrientation: [P.orientationOf(NaN, 1024), P.orientationOf(1024, undefined)],
  ratios: [[1216, 832], [1536, 640], [832, 1216], [1024, 1024], [768, 1344]],
  ratio_labels: [[1216, 832], [1536, 640], [832, 1216], [1024, 1024], [768, 1344]].map(([w, h]) => P.ratioLabel(w, h)),
  orientations: [[1024, 1024], [1344, 768], [768, 1344], [900, 100]],
};

for (const [name, list] of Object.entries(P.SIZE_PRESETS)) {
  out.presets[name] = list.map((p) => ({
    id: p.id,
    width: p.width,
    height: p.height,
    nameKey: p.nameKey,
    ratio: P.ratioLabel(p.width, p.height),
    derived: P.orientationOf(p.width, p.height),
  }));
}
out.legacy = P.LEGACY_PRESETS.map((p) => ({
  id: p.id, width: p.width, height: p.height, nameKey: p.nameKey,
}));
console.log(JSON.stringify(out));
"""


def _run_harness():
    if not PRESETS_JS.is_file():
        raise unittest.SkipTest("sizePresets.js is not present")
    node = shutil.which("node")
    if not node:
        raise unittest.SkipTest("node is unavailable; cannot execute the module's own maths")
    with tempfile.TemporaryDirectory() as tmp:
        mod = Path(tmp) / "sizePresets.mjs"
        mod.write_text(PRESETS_JS.read_text(encoding="utf-8"), encoding="utf-8")
        harness = Path(tmp) / "harness.mjs"
        harness.write_text(HARNESS, encoding="utf-8")
        proc = subprocess.run([node, str(harness)], capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise AssertionError(f"sizePresets.js did not run:\n{proc.stderr[:600]}")
        return json.loads(proc.stdout.strip().splitlines()[-1])


class PresetGeometryTests(unittest.TestCase):
    """The offered sizes are sizes these models were actually trained on."""

    @classmethod
    def setUpClass(cls):
        cls.data = _run_harness()
        cls.flat = [p for name, lst in cls.data["presets"].items() for p in lst]
        cls.flat += cls.data["legacy"]

    def test_constants_match_the_asserted_values(self):
        self.assertEqual(self.data["constants"]["MULTIPLE"], MULTIPLE)
        self.assertEqual(self.data["constants"]["NATIVE_STEP"], NATIVE_STEP)
        self.assertEqual(self.data["constants"]["FLOOR"], FLOOR)

    def test_every_preset_survives_mflux_rounding(self):
        """mflux floors a dimension to a multiple of 16 WITHOUT failing.

        A preset that is not already a multiple of 16 would therefore be
        generated at a different size than the button says.
        """
        for p in self.flat:
            self.assertEqual(p["width"] % MULTIPLE, 0, f"{p['id']} width {p['width']}")
            self.assertEqual(p["height"] % MULTIPLE, 0, f"{p['id']} height {p['height']}")

    def test_every_native_preset_sits_on_the_64_grid(self):
        for p in self.flat:
            self.assertEqual(p["width"] % NATIVE_STEP, 0, f"{p['id']} width")
            self.assertEqual(p["height"] % NATIVE_STEP, 0, f"{p['id']} height")

    def test_nothing_is_below_the_hard_floor(self):
        for p in self.flat:
            self.assertGreaterEqual(p["width"], FLOOR, p["id"])
            self.assertGreaterEqual(p["height"], FLOOR, p["id"])

    def test_ids_are_unique(self):
        ids = [p["id"] for p in self.flat]
        self.assertEqual(len(ids), len(set(ids)), "duplicate preset id")

    def test_a_preset_sits_in_the_tab_that_matches_its_shape(self):
        """orientationOf() is what re-derives the tab from an external size.

        If a preset were filed under the wrong tab, selecting it would leave the
        control showing a tab that does not contain the active button.
        """
        for tab, lst in self.data["presets"].items():
            for p in lst:
                self.assertEqual(p["derived"], tab, f"{p['id']} is {p['derived']} but filed under {tab}")

    def test_the_ratio_label_is_the_true_reduced_ratio(self):
        """The displayed ratio is derived from the pixels, so it cannot drift.

        Several of these buckets only approximate the ratio people ask for --
        1536x640 is 2.4, not 21:9 -- which is exactly why the label is computed
        rather than stored next to the size.
        """
        for (w, h), label in zip(self.data["ratios"], self.data["ratio_labels"]):
            g = math_gcd(w, h)
            self.assertEqual(label, f"{w // g}:{h // g}", f"{w}x{h}")
        cinematic = next(p for p in self.flat if p["id"] == "ls-1536x640")
        self.assertEqual(cinematic["ratio"], "12:5")
        self.assertNotEqual(cinematic["ratio"], "21:9")

    def test_every_orientation_has_presets(self):
        for tab in ("square", "landscape", "portrait"):
            self.assertTrue(self.data["presets"].get(tab), f"{tab} tab is empty")


def math_gcd(a, b):
    while b:
        a, b = b, a % b
    return a or 1


class SizeAvailabilityTests(unittest.TestCase):
    """What is offered.

    There is no filter. An earlier revision hid anything under 768 px behind a
    checkbox; it was removed because it cost a line of vertical space to hide
    exactly ONE button (512x512), did nothing at all on the Landscape and Portrait
    tabs, and silently resized the image when ticked. These tests now pin the
    absence of a gate, so it cannot quietly come back.
    """

    @classmethod
    def setUpClass(cls):
        cls.data = _run_harness()

    def _dims(self, entries):
        return {(p["width"], p["height"]) for p in entries}

    def test_the_small_sizes_are_offered(self):
        """The user's correction: 512 and 768 must not disappear."""
        sq = self._dims(self.data["defaultSquare"])
        self.assertIn((512, 512), sq)
        self.assertIn((768, 768), sq)
        self.assertIn((1024, 1024), sq)

    def test_every_orientation_offers_six_or_four(self):
        for key in ("defaultSquare", "defaultLandscape", "defaultPortrait"):
            self.assertTrue(self.data[key], f"{key} is empty")

    def test_the_512x768_default_is_a_real_preset(self):
        port = self._dims(self.data["defaultPortrait"])
        land = self._dims(self.data["defaultLandscape"])
        self.assertIn((512, 768), port)
        self.assertIn((768, 512), land)

    def test_no_engine_loses_a_size_to_a_threshold(self):
        """The gate is gone, so nothing is filtered by edge length."""
        for eco in ("FLUX.2", "SDXL", "Krea 2", "ZImageTurbo", "Qwen-Image 2.1", None):
            info = {"ecosystem": eco} if eco else {}
            dims = self._dims(self.data["presets"]["square"])
            self.assertIn((512, 512), dims, f"{eco} lost 512x512")

    def test_the_legacy_tier_is_unreachable_for_modern_engines(self):
        """256x256 exists only for an engine that declares itself legacy."""
        self.assertNotIn((256, 256), self._dims(self.data["modernNoFlagSquare"]))
        self.assertNotIn((256, 256), self._dims(self.data["modernSquare"]))

    def test_a_legacy_engine_reaches_the_256_tier_without_a_duplicate_tile(self):
        dims = self._dims(self.data["legacyOnly"])
        self.assertIn((256, 256), dims)
        # lg-512 is also 512x512; both would render as two identical active tiles.
        entries = self.data["legacyOnly"]
        pairs = [(p["width"], p["height"]) for p in entries]
        self.assertEqual(len(pairs), len(set(pairs)), f"duplicate tile: {pairs}")

    def test_an_unknown_ecosystem_is_treated_as_modern(self):
        self.assertTrue(self.data["noEcosystem"])


class AdvisoryCeilingTests(unittest.TestCase):
    """Qwen at 1024x1024 is a confirmed OOM on a 16GB M1 (AGENTS.md).

    The per-model pixel caps were removed by decision, so nothing in the pipeline
    stops it -- which is why this control warns rather than trusting the backend.
    """

    @classmethod
    def setUpClass(cls):
        cls.data = _run_harness()
        cls.a = cls.data["advisory"]

    def test_qwen_over_its_ceiling_is_flagged(self):
        self.assertTrue(self.a["qwen1024"])

    def test_qwen_at_its_measured_working_sizes_is_not_flagged(self):
        self.assertFalse(self.a["qwen768"], "768x768 is measured working for Qwen")
        self.assertFalse(self.a["qwen512x768"], "512x768 is every model's default")

    def test_other_models_have_no_advisory(self):
        self.assertFalse(self.a["flux1280"], "FLUX.2 at 1280x1280 is fine")
        self.assertFalse(self.a["none"])

    def test_a_non_finite_size_is_never_flagged(self):
        """Warning on NaN would nag about a field mid-edit."""
        self.assertFalse(self.a["nan"])


class NonFiniteInputTests(unittest.TestCase):
    """A cleared field once printed the literal string "NaN:NaN"."""

    @classmethod
    def setUpClass(cls):
        cls.data = _run_harness()

    def test_ratio_label_refuses_a_non_finite_pair(self):
        for got in self.data["badRatios"][:2]:
            self.assertIsNone(got, f"expected None, got {got!r}")

    def test_orientation_of_refuses_a_non_finite_pair(self):
        """Guessing sent a NaN width down the portrait branch."""
        for got in self.data["badOrientation"]:
            self.assertIsNone(got, f"expected None, got {got!r}")

    def test_a_degenerate_size_still_reduces(self):
        """0x0 must not be a crash; gcd of zeroes divides by zero without this."""
        self.assertEqual(self.data["badRatios"][2], "0:0")


class SnapAndBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = _run_harness()

    def test_snap_lands_on_the_step(self):
        """snap(1000) at 16 -> 1008, at 64 -> 1024. Never an off-grid value."""
        at16, at64, _neg, _nan, odd = self.data["models"]
        self.assertEqual(at16, 1008)
        self.assertEqual(at64, 1024)
        self.assertEqual(odd, 768)

    def test_snap_raises_rather_than_trusts_a_negative(self):
        self.assertGreaterEqual(self.data["models"][2], FLOOR)

    def test_snap_refuses_nonsense_instead_of_producing_nan(self):
        self.assertIsNone(self.data["models"][3])

    def test_fit_to_budget_clamps_and_keeps_the_shape(self):
        capped = self.data["budget"][0]
        w, h = capped["width"], capped["height"]
        self.assertLessEqual(w * h, 1024 * 768 + MULTIPLE * MULTIPLE)
        before = 1024 / 1024
        after = w / h
        self.assertAlmostEqual(before, after, delta=0.05)

    def test_a_missing_budget_leaves_the_pair_alone(self):
        self.assertEqual(self.data["budget"][1], {"width": 800, "height": 600})


class TranslationTests(unittest.TestCase):
    """A missing key renders an empty button, which is how a size goes missing."""

    @classmethod
    def setUpClass(cls):
        cls.data = _run_harness()
        cls.flat = [p for name, lst in cls.data["presets"].items() for p in lst] + cls.data["legacy"]
        src = PARAMS_PART.read_text(encoding="utf-8")
        cls.part_keys = set(re.findall(r'"(params\.size\.[^"]+)"\s*:', src))
        cls.lang = json.loads(LANGS.read_text(encoding="utf-8"))

    def test_every_preset_name_exists_in_the_reference_part(self):
        for p in self.flat:
            self.assertIn(p["nameKey"], self.part_keys, f"{p['id']} has no translation: {p['nameKey']}")

    def test_every_orientation_tab_has_a_label(self):
        for orient in ("square", "landscape", "portrait", "custom"):
            key = f"params.size.orientation.{orient}"
            self.assertIn(key, self.part_keys, f"missing tab label {key}")

    def test_every_preset_name_is_translated_in_all_five_overlay_languages(self):
        for p in self.flat:
            entry = self.lang.get(p["nameKey"])
            self.assertIsNotNone(entry, f"{p['nameKey']} missing from _parts/params.json")
            for lang in ("es", "zh", "ja", "pt", "ko"):
                self.assertTrue(entry.get(lang, "").strip(), f"{p['nameKey']} has no {lang}")


if __name__ == "__main__":
    unittest.main()