"""Tests for backend/licensing.py.

Both source functions are exercised through pure `*_from_payload` helpers against
recorded real payloads, so the shape rules are tested without a network. The network
functions are thin wrappers over these and are deliberately not unit tested here -- a
test that asserts "requests.get was called" is a test of the mock.

The fixtures are real responses captured from the live APIs with the app's own token:

  civitai_model_2890476.json  GET /api/v1/models/2890476
  hf_flux1_dev.json           GET https://huggingface.co/api/models/black-forest-labs/FLUX.1-dev

The HF one is the case that matters: FLUX.1-dev reports `cardData.license == "other"`,
because its real terms live in the README card rather than in that field. Nothing here is
allowed to replace it with a name.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import licensing  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


class CivitaiLicenseTests(unittest.TestCase):
    PAYLOAD = fixture("civitai_model_2890476.json")

    def test_permissions_are_lifted_from_the_model_payload(self):
        info = licensing.fetch_civitai_license_from_payload(self.PAYLOAD)
        self.assertEqual(info.permissions, {
            "no_credit": True,
            "commercial_use": ["Image", "RentCivit", "Rent", "Sell", "SellMerge"],
            "derivatives": True,
            "different_license": True,
        })

    def test_no_licence_name_is_invented(self):
        # Global Constraint 1: Civitai has no licence field anywhere, so identifier
        # stays None rather than being filled from the permission set.
        info = licensing.fetch_civitai_license_from_payload(self.PAYLOAD)
        self.assertIsNone(info.identifier)

    def test_source_url_points_at_the_model_page(self):
        info = licensing.fetch_civitai_license_from_payload(self.PAYLOAD)
        self.assertEqual(info.source_url,
                         f"https://civitai.com/models/{self.PAYLOAD['id']}")

    def test_a_payload_with_no_permission_fields_gives_none_not_false(self):
        info = licensing.fetch_civitai_license_from_payload({"id": 7})
        self.assertIsNone(info.permissions)

    def test_a_missing_model_id_is_refused(self):
        with self.assertRaises(ValueError):
            licensing.fetch_civitai_license_from_payload({})


class HfLicenseTests(unittest.TestCase):
    PAYLOAD = fixture("hf_flux1_dev.json")

    def test_the_other_identifier_is_carried_through_not_replaced(self):
        self.assertEqual(
            licensing.fetch_hf_license_from_payload(self.PAYLOAD).identifier, "other")

    def test_hf_has_no_permissions_so_they_are_none(self):
        self.assertIsNone(
            licensing.fetch_hf_license_from_payload(self.PAYLOAD).permissions)

    def test_a_license_tag_is_used_only_when_carddata_is_absent(self):
        payload = {"id": "x/y", "tags": ["license:mit"]}
        self.assertEqual(
            licensing.fetch_hf_license_from_payload(payload).identifier, "mit")

    def test_carddata_wins_over_a_tag_when_both_are_present(self):
        payload = {"id": "x/y", "tags": ["license:mit"],
                   "cardData": {"license": "other"}}
        self.assertEqual(
            licensing.fetch_hf_license_from_payload(payload).identifier, "other")

    def test_no_carddata_and_no_tag_gives_none_not_a_guess(self):
        self.assertIsNone(
            licensing.fetch_hf_license_from_payload({"id": "x/y"}).identifier)

    def test_source_url_points_at_the_model_page(self):
        self.assertEqual(
            licensing.fetch_hf_license_from_payload(self.PAYLOAD).source_url,
            "https://huggingface.co/black-forest-labs/FLUX.1-dev")

    def test_a_payload_with_no_id_is_refused(self):
        with self.assertRaises(ValueError):
            licensing.fetch_hf_license_from_payload({"cardData": {}})


class NetworkWrapperTests(unittest.TestCase):
    """The wrappers delegate; they must not reshape the data on the way through."""

    def test_both_wrappers_are_callable_with_an_explicit_token(self):
        self.assertTrue(callable(licensing.fetch_civitai_license))
        self.assertTrue(callable(licensing.fetch_hf_license))

    def test_a_failed_fetch_returns_none_rather_than_raising(self):
        # Global Constraint 5 / Review Focus 4: an unreachable source must not
        # propagate an exception into a list render.
        import urllib.error
        err = urllib.error.URLError("unreachable")
        self.assertIsNone(licensing._safe(lambda: (_ for _ in ()).throw(err)))
        self.assertIsNone(licensing._safe(lambda: (_ for _ in ()).throw(
            TimeoutError("slow"))))


if __name__ == "__main__":
    unittest.main()
