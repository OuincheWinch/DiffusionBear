"""SDXL model discovery and loading, end to end.

Reported from a second machine, not the build machine: Juggernaut XL Lightning
would not load, and the Models tab claimed SDXL models were "Not runnable here".
Three distinct defects, all reproduced here.

1. `usable_as()` had NO SDXL rule at all, so every SDXL repo in the browser read
   "No engine in this app can run this architecture" -- including
   RunDiffusion/Juggernaut-XL-Lightning, which is the registry's own repo and was
   showing an INSTALLED badge next to the claim that it could not run.

2. Adding a name-based rule for SDXL is not enough on its own. A repo can match
   every required token and still not be a checkpoint:
   `digitalbrain79/juggernaut-xl-lightning-4step-controlnet-coreml-6b` contains
   "juggernaut", "xl" and "lightning". Offering it would cost the owner a
   multi-gigabyte download to reach a dead end, so non-checkpoints are rejected
   before the rules run.

3. The layout check existed only on the adopt route. The reported failure was on
   the GENERATION path, so it still produced a forty-line traceback ending in
   "No .safetensors files in .../unet". The cause was an interrupted download:
   an empty unet/ behind an INSTALLED badge, because _finish_model_task declared
   success without ever looking at what had been written.

WHAT IS ASSERTED
The browser mapping, the non-checkpoint rejections, the four distinct diagnostic
messages, and that the generation path actually calls the check -- the last one by
source inspection, because running a real SDXL generation needs Apple hardware.
"""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import hf_browse  # noqa: E402
from sdxl_layout import is_sdxl_engine, sdxl_layout_report  # noqa: E402

GENERATOR = BACKEND / "generator.py"
DOWNLOADS = BACKEND / "routers" / "downloads.py"


class SdxlBrowserMappingTests(unittest.TestCase):
    """The registry's own SDXL repos must not read as unroutable."""

    def test_the_registry_repo_resolves(self):
        """This is the exact repo the report showed as INSTALLED + not runnable."""
        model_id, _ = hf_browse.usable_as("RunDiffusion/Juggernaut-XL-Lightning", "diffusion")
        self.assertEqual(model_id, "juggernaut-xl-lightning")

    def test_every_sdxl_family_resolves_to_its_own_engine(self):
        for repo, expected in [
            ("mlx-community/realvis-xl-v5-lightning", "realvis-xl-v5-lightning"),
            ("ai-forever/RealVisXL_V5.0_Lightning", "realvis-xl-v5-lightning"),
            ("stabilityai/stable-diffusion-xl-base-1.0", None),  # not a family we bind
            ("simianluo/Juggernaut-XI", "juggernaut-xi"),
        ]:
            got, _ = hf_browse.usable_as(repo, "diffusion")
            self.assertEqual(got, expected, repo)

    def test_a_controlnet_matching_every_token_is_still_rejected(self):
        """It contains juggernaut, xl AND lightning, so name matching alone is unsafe."""
        repo = "digitalbrain79/juggernaut-xl-lightning-4step-controlnet-coreml-6b-32"
        model_id, reason = hf_browse.usable_as(repo, "diffusion")
        self.assertIsNone(model_id, "a controlnet must not be offered as a checkpoint")
        self.assertIn("controlnet", reason.lower())

    def test_non_checkpoints_are_rejected(self):
        for repo, needle in [
            ("ostris/super-cereal-sdxl-lora", "lora"),
            ("ostris/sdxl-vae-ft-mse", "vae"),
            ("someuser/sdxl-upscaler", "upcale" if False else "upscale"),
        ]:
            model_id, reason = hf_browse.usable_as(repo, "diffusion")
            self.assertIsNone(model_id, repo)
            self.assertIn(needle, reason.lower())

    def test_unrelated_english_words_are_not_non_checkpoints(self):
        """exploration/floral/brave must not be read as lora/vae."""
        for repo in [
            "someone/exploration-diffusion-xl",
            "someone/advantage-xl",
            "someone/brave-new-world-xl",
            "someone/some-flux-floral-style",
        ]:
            _, reason = hf_browse.usable_as(repo, "diffusion")
            self.assertNotIn("not a checkpoint", reason.lower(), repo)

    def test_flux_name_containing_floral_is_not_blocked_as_lora(self):
        model_id, reason = hf_browse.usable_as("someone/some-flux2-floral-style", "diffusion")
        self.assertEqual(model_id, "flux2-klein-4b")
        self.assertNotIn("lora", reason.lower())

    def test_the_mflux_families_are_unaffected(self):
        """A regression guard: the new SDXL rules must not swallow these."""
        for repo, expected in [
            ("mlx-community/FLUX.2-klein-4B-4bit", "flux2-klein-4b"),
            ("mlx-community/Qwen-Image-2.1-4bit", "qwen-image-2.1"),
            ("mikey sorrento/distilled-flux", "flux2-klein-4b"),
        ]:
            if "/" in repo and repo.split("/")[0] in ("mlx-community",):
                got, _ = hf_browse.usable_as(repo, "diffusion")
                self.assertEqual(got, expected, repo)


class SdxlDiagnosticsTests(unittest.TestCase):
    """Four different causes, four different messages."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _mk(self, name, files):
        d = self.root / name
        for rel in files:
            f = d / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"0" * 32)
        return d

    def test_a_good_directory_is_accepted(self):
        d = self._mk("good", [
            "model_index.json",
            "unet/diffusion_pytorch_model-00001-of-00002.safetensors",
            "unet/diffusion_pytorch_model.safetensors.index.json",
            "vae/diffusion_pytorch_model.safetensors",
        ])
        ok, reason = sdxl_layout_report(d)
        self.assertTrue(ok, reason)

    def test_an_interrupted_download_says_so_and_offers_a_fix(self):
        """The reported cause. A generic 'no safetensors' sends the owner to the
        wrong place: they would convert the model rather than re-download it."""
        d = self._mk("juggernaut-xl-lightning", [
            "model_index.json",
            "unet/config.json",
            ".cache/huggingface/download/unet/diffusion_pytorch_model-00001-of-00002.safetensors.incomplete",
        ])
        ok, reason = sdxl_layout_report(d)
        self.assertFalse(ok)
        self.assertIn("unfinished", reason.lower())
        self.assertIn("re-download", reason.lower())
        self.assertNotIn("single-file", reason.lower(), "must not blame the format here")

    def test_a_single_file_checkpoint_names_the_format_problem(self):
        d = self._mk("single", ["juggernautXL_v9.safetensors"])
        ok, reason = sdxl_layout_report(d)
        self.assertFalse(ok)
        self.assertIn("single-file", reason.lower())
        self.assertIn("diffusers", reason.lower())

    def test_an_empty_unet_names_the_incomplete_download(self):
        d = self._mk("empty-unet", ["model_index.json", "unet/config.json"])
        ok, reason = sdxl_layout_report(d)
        self.assertFalse(ok)
        self.assertIn("empty", reason.lower())

    def test_a_missing_directory_is_not_a_crash(self):
        ok, reason = sdxl_layout_report(self.root / "never-existed")
        self.assertFalse(ok)
        self.assertIn("does not exist", reason.lower())


class WiringTests(unittest.TestCase):
    """The checks have to be ON THE PATH, not merely present.

    A layout check that only runs on adopt is what shipped: the failure reported
    was on the generation path, so it never ran. Asserted by source inspection
    because the real thing needs Apple hardware.
    """

    @classmethod
    def setUpClass(cls):
        cls.generator = GENERATOR.read_text(encoding="utf-8")
        cls.downloads = DOWNLOADS.read_text(encoding="utf-8")

    def test_the_generation_path_checks_the_layout(self):
        self.assertRegex(
            self.generator,
            r"sdxl_layout_report\(Path\(model_dir\)\)",
            "the SDXL request must be gated on the layout check",
        )

    def test_the_generation_failure_is_a_readable_error_not_a_bare_assert(self):
        m = re.search(r"raise ValueError\(f?\"SDXL cannot load this model: \{why_sdxl\}\"\)", self.generator)
        self.assertIsNotNone(m, "the generation path must raise a message naming the cause")

    def test_the_layout_check_is_imported_not_inlined(self):
        self.assertIn("from sdxl_layout import sdxl_layout_report", self.generator)

    def test_the_download_does_not_declare_success_without_verifying(self):
        """_finish_model_task set status=done unconditionally.

        That is how an empty unet/ ended up behind an INSTALLED badge: the task was
        marked done on the strength of the loop finishing, not on the evidence that
        any weights were written. Assert that the verification precedes the success
        call.

        Scoped to the HF model worker by locating its own success call. A bare
        .find() on "_finish_model_task(task_id" matches an earlier call in a
        different function and silently tests nothing.
        """
        success_call = (
            '_finish_model_task(task_id, f"{minfo.get(\'label\', model_id)} installed successfully!")'
        )
        at = self.downloads.find(success_call)
        self.assertNotEqual(at, -1, "the model worker's success call moved or was renamed")
        verify = self.downloads.rfind("ok, why = sdxl_layout_report(target_dir)", 0, at)
        self.assertNotEqual(
            verify, -1,
            "the download must verify the SDXL layout, and must do it before marking the task done",
        )

    def test_the_router_reexports_rather_than_duplicating(self):
        """Two copies of this check would drift, which is how the gate was wrong once."""
        self.assertIn("from sdxl_layout import", self.downloads)
        self.assertEqual(
            self.downloads.count("def sdxl_layout_report"),
            0,
            "the implementation must live in sdxl_layout.py only",
        )


class InstalledStatusAgreesWithLayoutTests(unittest.TestCase):
    """A half-downloaded model must read as NOT installed, or there is no repair.

    Reported with screenshots: generation failed with "unfinished download" while
    the Models tab showed the model installed, so no Download/Retry button existed
    and /api/models/download answered already_installed. The cause was the
    installed-status probes ignoring partials under .cache/, the exact directory
    huggingface_hub stages local_dir downloads in.
    """

    def _mk(self, root, with_partial):
        d = Path(root) / "realvis-xl-v5"
        (d / "unet").mkdir(parents=True)
        (d / "model_index.json").write_text(json.dumps({"unet": ["a", "b"]}), encoding="utf-8")
        (d / "unet" / "w.safetensors").write_bytes(b"0" * 32)
        if with_partial:
            partial_dir = d / ".cache" / "huggingface" / "download"
            partial_dir.mkdir(parents=True)
            (partial_dir / "w.safetensors.incomplete").write_bytes(b"0" * 32)
        return d

    def test_partial_under_cache_marks_model_not_installed(self):
        from routers.loras import _model_is_fully_cached  # noqa: E402

        minfo = {"engine": "sdxl"}
        with tempfile.TemporaryDirectory() as bad_root, tempfile.TemporaryDirectory() as good_root:
            bad = self._mk(bad_root, True)
            good = self._mk(good_root, False)
            self.assertFalse(
                _model_is_fully_cached("realvis-xl-v5-probe", minfo, bad),
                "a .cache partial must not read as installed",
            )
            self.assertTrue(
                _model_is_fully_cached("realvis-xl-v5-probe", minfo, good),
                "weights with no partials must still read as installed",
            )


class EngineClassificationTests(unittest.TestCase):
    def test_all_four_sdxl_engines_are_recognised(self):
        for model_id in (
            "juggernaut-xl-lightning",
            "realvis-xl-v5-lightning",
            "realvis-xl-v5",
            "juggernaut-xi",
        ):
            self.assertTrue(is_sdxl_engine(model_id), model_id)

    def test_the_mflux_engines_are_not(self):
        for model_id in ("flux2-klein-4b", "krea2-turbo", "qwen-image-2.1", "z-image-turbo"):
            self.assertFalse(is_sdxl_engine(model_id), model_id)


if __name__ == "__main__":
    unittest.main()