"""Three defects found by looking at the Models tab rather than at the tests.

Each shipped green. They are grouped here because all three are the same mistake in
different clothes: a plausible-looking value that nobody checked against reality.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient

import hf_browse
import main


class NonCheckpointNameTests(unittest.TestCase):
    """A VAE and a language model are not runnable image models.

    `taesdxl` tokenises as one word, so the "vae" entry in the deny-list never matched
    it -- and then it satisfied the SDXL executable rule on its "sd" + "xl" tokens and
    was offered as a runnable diffusion model. It is a VAE approximation.

    `qwen2.5-0.5b-instruct-4bit` is this app's own prompt enhancer. "instruct" was not
    on the list, and an inspected architecture hint then bound it to FLUX.2-klein.

    Both were visible on the Models tab as a "Use for ..." button on a row that had no
    business offering one.
    """

    def test_a_taesd_vae_is_not_offered_as_a_model(self):
        model_id, reason = hf_browse.usable_as("taesdxl", "diffusion")
        self.assertIsNone(model_id)
        self.assertIn("not be loaded as a model", reason)

    def test_an_instruction_tuned_language_model_is_not_offered(self):
        model_id, _ = hf_browse.usable_as("qwen2.5-0.5b-instruct-4bit", "diffusion")
        self.assertIsNone(model_id)

    def test_the_real_models_still_resolve(self):
        """The deny-list must not be so broad that it eats genuine checkpoints."""
        for repo, expected in (
            ("juggernaut-xl-lightning", "juggernaut-xl-lightning"),
            ("flux2-klein-4b-4bit", "flux2-klein-4b"),
            ("RunDiffusion/Juggernaut-XL-Lightning", "juggernaut-xl-lightning"),
            ("SG161222/RealVisXL_V5.0_Lightning", "realvis-xl-v5-lightning"),
        ):
            with self.subTest(repo=repo):
                model_id, _ = hf_browse.usable_as(repo, "diffusion")
                self.assertEqual(model_id, expected)

    def test_a_sdxl_vae_is_still_refused(self):
        self.assertIsNone(hf_browse.usable_as("madebyollin/sdxl-vae-fp16-fix", "diffusion")[0])


class ModelLicenceRefreshTests(unittest.TestCase):
    """The button on the Models table called the LoRA endpoint.

    So pressing "Refresh licences" on a table full of models refreshed a different
    registry, found nothing to change, and reported it -- which reads as a button that
    does nothing. The LoRA endpoint was correct; the wiring was wrong, and no test
    could see it because both routes return 200 with a similar body.
    """

    def setUp(self):
        self.client = TestClient(main.app)

    def test_the_model_endpoint_exists(self):
        with mock.patch("state.refresh_model_licenses",
                        return_value={"updated": 0, "skipped": 0, "failed": 0}) as fn:
            r = self.client.post("/api/models/refresh-licenses")
        self.assertEqual(r.status_code, 200)
        fn.assert_called_once()

    def test_the_two_endpoints_are_not_the_same_route(self):
        """The regression itself, stated as an assertion."""
        with mock.patch("state.refresh_licenses",
                        return_value={"updated": 0, "skipped": 0, "failed": 0}) as lora:
            with mock.patch("state.refresh_model_licenses",
                            return_value={"updated": 0, "skipped": 0, "failed": 0}) as model:
                self.client.post("/api/models/refresh-licenses")
                lora.assert_not_called()
                model.assert_called_once()

    def test_a_cached_licence_reaches_the_inventory(self):
        import app_settings

        cached = {"identifier": "apache-2.0", "url": None,
                  "source_url": "https://huggingface.co/x/y", "permissions": None,
                  "fetched_at": 1.0}
        with mock.patch.object(app_settings, "get_setting",
                               side_effect=lambda k, d=None: (
                                   {"flux2-klein-4b": cached} if k == "model_licenses" else d)):
            items = self.client.get("/api/models/inventory").json()["items"]
        row = next(i for i in items if i["name"] == "flux2-klein-4b")
        self.assertEqual(row["license"]["identifier"], "apache-2.0")

    def test_an_absent_licence_is_none_not_guessed(self):
        import app_settings

        with mock.patch.object(app_settings, "get_setting",
                               side_effect=lambda k, d=None: (
                                   {} if k == "model_licenses" else d)):
            items = self.client.get("/api/models/inventory").json()["items"]
        for row in items:
            if row["origin"] == "registry":
                self.assertIsNone(row.get("license"),
                                  "a licence was invented for a model with no cached block")


if __name__ == "__main__":
    unittest.main()


class ArchitectureHintPrecedenceTests(unittest.TestCase):
    """A name beats a hint. A hint only helps when the name says nothing.

    The detected-models scan passes the INSPECTED architecture as a hint, and the hint
    used to REPLACE the name entirely. So a coarse inspection result could overrule a
    folder that named its own architecture:

        Qwen-Image-2.1-mflux-q4   by name -> qwen-image-2.1      correct
                                 hint "FLUX" -> flux2-klein-4b  wrong

    That is how a Qwen-Image model was offered as "Use for FLUX.2-Klein 4B". The hint
    exists for folders named something opaque, not to outvote one that is explicit.
    """

    def test_a_specific_name_is_not_overruled_by_a_coarse_hint(self):
        model_id, _ = hf_browse.usable_as("Qwen-Image-2.1-mflux-q4", "diffusion",
                                          arch_hint="FLUX")
        self.assertEqual(model_id, "qwen-image-2.1")

    def test_a_non_checkpoint_stays_refused_despite_a_hint(self):
        for name, hint in (("taesdxl", "FLUX"), ("qwen2.5-0.5b-instruct-4bit", "FLUX")):
            with self.subTest(name=name):
                self.assertIsNone(
                    hf_browse.usable_as(name, "diffusion", arch_hint=hint)[0],
                    "the hint must not be able to resurrect a denied folder")

    def test_the_hint_still_helps_when_the_name_says_nothing(self):
        """Otherwise opaque download folders become unregistrable, which is worse."""
        model_id, _ = hf_browse.usable_as("some-opaque-download-folder", "diffusion",
                                          arch_hint="SDXL")
        self.assertEqual(model_id, "juggernaut-xl-lightning")

    def test_no_hint_behaviour_is_unchanged(self):
        for repo, expected in (("juggernaut-xl-lightning", "juggernaut-xl-lightning"),
                               ("flux2-klein-4b-4bit", "flux2-klein-4b"),
                               ("SG161222/RealVisXL_V5.0_Lightning", "realvis-xl-v5-lightning")):
            with self.subTest(repo=repo):
                self.assertEqual(hf_browse.usable_as(repo, "diffusion")[0], expected)
