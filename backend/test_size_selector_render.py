"""SizeSelector and ModelInstaller must render without throwing.

THE REASON THIS FILE EXISTS
In v0.3.4 a temporal dead zone shipped to the public release and blanked the app
on launch: `canStart` read `showBar` three lines above `showBar`'s own
declaration, so every render of this component threw a ReferenceError, and the
interface recovery boundary turned that into a dead app. It passed oxlint, all
504 backend tests and a green CI run, because none of them evaluate a React
component. A human opening the installed build was the only thing that found it.

That is the whole argument for rendering the component in CI. A static TDZ
scanner was tried first and rejected: it reported 63 false positives, because it
cannot distinguish a function parameter from a declaration in a different scope.
Rendering has no such blind spot.

WHAT IS ASSERTED
That each case does not throw, plus two content assertions that catch the
mistakes a throw-free render would still hide: a ratio label that renders
"NaN:NaN", and an advisory that silently fails to appear.

The i18n provider is replaced with a stub that echoes the key it was asked for,
so the markup also records which keys the component requests. A mistyped
labelKey therefore shows up as a suspicious «params.size.nmae.standard» rather
than as a silently blank button.
"""

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
RUNNER = FRONTEND / "tools" / "run-size-selector-render.mjs"
ENTRY = FRONTEND / "tools" / "size-selector-render.mjs"


class SizeSelectorRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RUNNER.is_file() or not ENTRY.is_file():
            raise unittest.SkipTest("the render harness is not present")
        if not shutil.which("node"):
            raise unittest.SkipTest("node is unavailable")
        if not (FRONTEND / "node_modules" / "vite").is_dir():
            raise unittest.SkipTest("frontend dependencies are not installed")

        cls.proc = subprocess.run(
            ["node", str(RUNNER), str(FRONTEND)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=FRONTEND,
        )
        if cls.proc.returncode != 0 and not cls.proc.stdout.strip().startswith("["):
            raise AssertionError(
                "the render harness did not produce output:\n"
                f"{cls.proc.stderr[-1500:]}"
            )
        # The bundle prints the case list as JSON on its first line, then a
        # human-readable summary. Parse the JSON, ignore the summary.
        first = cls.proc.stdout.strip().splitlines()[0]
        cls.cases = {c["name"]: c for c in json.loads(first)}
        cls.html = {n: c.get("html", "") for n, c in cls.cases.items()}

    def test_every_case_rendered_without_throwing(self):
        broken = {n: c.get("error") for n, c in self.cases.items() if not c["ok"]}
        self.assertEqual(broken, {}, f"SizeSelector threw on render: {broken}")

    def test_all_expected_cases_ran(self):
        """A silently empty case list must not read as a pass."""
        self.assertEqual(len(self.cases), 9, f"expected 9 cases, got {sorted(self.cases)}")

    def test_the_model_installer_renders(self):
        """The component whose temporal dead zone shipped in v0.3.4.

        Kept in this harness deliberately: it is the regression this file was
        written for, and the fix was a one-line reordering that any future
        refactor could reintroduce. It was verified against the pre-fix revision
        from git, which fails here with
        "Cannot access 'showBar' before initialization".
        """
        for name in ("ModelInstaller idle", "ModelInstaller with a cancelled task"):
            self.assertIn(name, self.cases)
            self.assertTrue(self.cases[name]["ok"], f"{name}: {self.cases[name].get('error')}")
        # The stub provider returns keys, so assert on the key the button asks for.
        self.assertIn("«installer.downloadBtn»", self.html["ModelInstaller idle"])

    def test_the_square_preset_shows_its_dimensions(self):
        h = self.html["square preset selected"]
        self.assertIn("1024×1024", h)
        self.assertIn("1280×1280", h)
        self.assertIn("params.size.orientation.square", h)

    def test_the_registry_default_lands_on_a_real_preset(self):
        """512x768 is every model's default, so it must be selectable.

        An earlier revision had no preset for it and the control opened on Custom
        -- correct as far as it went, but it made the app's own default look like
        a hand-typed exception. It is offered in both orientations now.
        """
        h = self.html["registry default 512x768"]
        active = re.search(r'aria-selected="true" class="size-selector-tab is-active">'
                           r"«params\.size\.([\w.]+)»", h)
        self.assertIsNotNone(active, "no tab is marked active")
        self.assertEqual(active.group(1), "orientation.portrait")
        self.assertIn("512×768", h)
        self.assertIn("is-active", h)
        self.assertNotIn("size-selector-custom", h, "should not fall through to Custom")

    def test_the_small_sizes_are_visible(self):
        """The correction that mattered: 512 and 768 are simply offered."""
        h = self.html["square preset selected"]
        for dims in ("512×512", "768×768", "1024×1024", "1280×1280"):
            self.assertIn(dims, h, f"{dims} must be offered by default")
        self.assertNotIn("size-tier-toggle", h, "the 768 px filter was removed")
        self.assertNotIn("type=\"checkbox\"", h, "the picker has no filters left")

    def test_the_portrait_tab_offers_its_compact_size(self):
        h = self.html["portrait preset"]
        self.assertIn("512×768", h, "portrait compact is 512x768")
        self.assertIn("896×1152", h, "portrait natives are still offered")

    def test_the_qwen_advisory_appears_and_states_both_numbers(self):
        h = self.html["qwen 1280x1280 over its comfortable budget"]
        self.assertIn("size-advisory", h, "Qwen over its ceiling must say so")
        self.assertIn("1,638,400", h)
        self.assertIn("589,824", h)

    def test_a_model_with_no_advisory_shows_no_advisory(self):
        self.assertNotIn("size-advisory", self.html["square preset selected"])

    def test_non_numeric_dimensions_never_render_nan(self):
        """This was a real defect: the Custom panel printed the literal "NaN:NaN".

        It appeared whenever width or height was not a finite number -- a field
        cleared mid-edit, or a size arriving undefined -- which rendered a ratio
        beside two empty inputs.
        """
        h = self.html["non-numeric dimensions"]
        self.assertNotIn("NaN", h)
        self.assertNotIn("size-selector-hint", h)

    def test_no_requested_translation_key_is_empty_or_malformed(self):
        """A wrong labelKey renders «params.size.nmae.standard» -- invisible to lint."""
        keys = set()
        for h in self.html.values():
            keys |= set(re.findall(r"«([^»]+)»", h))
        self.assertTrue(keys, "expected the stub provider to record some keys")
        for k in keys:
            self.assertTrue(k.startswith(("params.size.", "installer.")), f"unexpected key {k}")
            self.assertNotIn("  ", k, f"double space in {k}")
        # The keys the component asks for must be real catalogue keys, not typos.
        self.assertIn("params.size.orientation.square", keys)
        self.assertIn("params.size.name.standard", keys)
        self.assertIn("params.size.widthLabel", keys)
        self.assertIn("params.size.heightLabel", keys)


if __name__ == "__main__":
    unittest.main()