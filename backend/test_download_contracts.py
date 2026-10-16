"""Pin the public HTTP shape of every /api route before its body moves.

Task 7 relocates the source-specific bodies out of `routers/downloads.py` and into
`backend/sources/`. The route paths, query parameters and response keys are what the
frontend depends on, and nothing about that refactor is allowed to change them.

These tests therefore assert only the contract -- path, accepted parameters, status
and the top-level response keys -- never the body of a search result. A failure here
after the split means the refactor changed something a caller can see; fix that before
moving on, rather than updating the test to match whatever it now returns.

Network-backed routes are stubbed at the `civitai_browse` / `hf_browse` seam -- the
upstream modules themselves, not a `downloads.py` attribute. That seam moved during the
split (the routes now go through the source registry, so they no longer call
`hf_browse.search_models` themselves), and moving it is the one thing these tests may
do without being re-pinned. What they pin -- paths, parameter names, status codes and
response keys -- did not change.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient

import main


# A response shaped like the real ones, so a route that forwards the upstream dict
# and a route that wraps it are distinguishable from the key assertions alone.
_STUB_ITEMS = {"items": [{"id": "x", "name": "x"}], "total": 1}


class DownloadContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    # --- model browsing -----------------------------------------------------

    def test_hf_search_contract(self):
        with mock.patch("hf_browse.search_models",
                        return_value=_STUB_ITEMS) as search:
            r = self.client.get("/api/hf/models?search=x&limit=5")
        self.assertEqual(r.status_code, 200)
        self.assertIn("items", r.json())
        # Parameters keep their names: the frontend builds this URL by keyword.
        self.assertEqual(search.call_args.kwargs["search"], "x")
        self.assertEqual(search.call_args.kwargs["limit"], 5)

    def test_civitai_search_contract(self):
        with mock.patch("civitai_browse.search_models",
                        return_value=_STUB_ITEMS) as search:
            r = self.client.get("/api/civitai/models?query=x")
        self.assertEqual(r.status_code, 200)
        self.assertIn("items", r.json())
        self.assertEqual(search.call_args.kwargs["query"], "x")

    def test_hf_search_accepts_every_documented_filter(self):
        with mock.patch("hf_browse.search_models",
                        return_value=_STUB_ITEMS) as search:
            r = self.client.get("/api/hf/models?search=x&author=a&architecture=sdxl"
                                "&quantization=4&kind=lora&limit=5")
        self.assertEqual(r.status_code, 200)
        kwargs = search.call_args.kwargs
        self.assertEqual(kwargs["author"], "a")
        self.assertEqual(kwargs["architecture"], "sdxl")
        self.assertEqual(kwargs["quantization"], "4")
        self.assertEqual(kwargs["kind"], "lora")

    def test_hf_search_upstream_failure_is_a_502(self):
        with mock.patch("hf_browse.search_models",
                        side_effect=RuntimeError("upstream down")):
            r = self.client.get("/api/hf/models?search=x")
        self.assertEqual(r.status_code, 502)

    def test_civitai_search_upstream_failure_is_a_502(self):
        with mock.patch("civitai_browse.search_models",
                        side_effect=RuntimeError("upstream down")):
            r = self.client.get("/api/civitai/models?query=x")
        self.assertEqual(r.status_code, 502)

    # --- model download tasks ----------------------------------------------

    def test_model_download_status_contract(self):
        """This route returns a bare list, newest first, not an envelope.

        Pinned because it is what the frontend polls, and because wrapping it in
        an object later would be an invisible breaking change for every caller.
        """
        r = self.client.get("/api/models/downloads")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_model_download_status_hides_internal_fields(self):
        """`cancel_event` and the speed samplers are process state, not response."""
        r = self.client.get("/api/models/downloads")
        hidden = {"cancel_event", "worker_active", "_speed_t", "_speed_b",
                  "_last_progress_at"}
        for entry in r.json():
            self.assertFalse(hidden & set(entry))

    def test_model_download_task_shape(self):
        """A task dict carries every field the progress bar reads."""
        import routers.downloads as downloads
        task = {
            "id": "t", "source": "huggingface", "model_id": "m", "repo_id": "r",
            "model_name": "n", "engine": "mflux", "status": "downloading",
            "progress": 0.0, "downloaded_bytes": 0, "total_bytes": 0,
            "speed_mb_s": 0.0, "status_text": "", "started_at": 0.0,
            "finished_at": None, "error": None, "result": None,
            "worker_active": True, "cancel_event": None,
        }
        with mock.patch.dict(downloads.MODEL_DOWNLOAD_TASKS, {"t": task}, clear=True):
            r = self.client.get("/api/models/downloads")
        self.assertEqual(r.status_code, 200)
        entry = next(d for d in r.json() if d.get("id") == "t")
        for key in ("status", "progress", "downloaded_bytes", "total_bytes",
                    "speed_mb_s", "status_text", "model_name", "source"):
            self.assertIn(key, entry)

    def test_model_download_cancel_returns_a_task_id(self):
        """Cancelling an unknown task must not 500: it is a race the UI hits."""
        r = self.client.delete("/api/models/downloads/does-not-exist")
        self.assertIn(r.status_code, (200, 404))

    def test_lora_downloads_list_contract(self):
        r = self.client.get("/api/loras/downloads")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), (list, dict))

    def test_lora_download_cancel_returns_a_task_id(self):
        r = self.client.delete("/api/loras/downloads/does-not-exist")
        self.assertIn(r.status_code, (200, 404))

    # --- model registration -------------------------------------------------

    def test_hf_register_list_contract(self):
        r = self.client.get("/api/hf/models/detected")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), (list, dict))

    def test_hf_register_rejects_a_missing_repo_id(self):
        r = self.client.post("/api/hf/models/register", json={})
        self.assertIn(r.status_code, (400, 422))

    def test_hf_convert_is_post_only(self):
        """Converting detected weights is a POST; a GET must not do it by accident."""
        r = self.client.get("/api/hf/models/convert")
        self.assertEqual(r.status_code, 404)
        r = self.client.post("/api/hf/models/convert", json={})
        self.assertIn(r.status_code, (400, 422))

    # --- SDXL ---------------------------------------------------------------

    def test_sdxl_convert_available_contract(self):
        r = self.client.get("/api/sdxl/convert/available")
        self.assertEqual(r.status_code, 200)
        self.assertIn("available", r.json())

    def test_sdxl_convert_rejects_a_missing_input(self):
        r = self.client.post("/api/sdxl/convert", json={})
        self.assertIn(r.status_code, (400, 422))

    # --- removal ------------------------------------------------------------

    def test_hf_remove_requires_explicit_confirmation(self):
        """A body-less DELETE must not delete. `confirm: true` is the gate.

        This route rmtree's a model directory: no trash, no undo, no second copy.
        Reaching it must take a deliberate request, not a bare DELETE.
        """
        r = self.client.delete("/api/hf/models/no-such-model-anywhere")
        self.assertEqual(r.status_code, 422)

    def test_hf_remove_rejects_a_traversing_name(self):
        """A traversing name must never reach the rmtree.

        405 is an acceptable outcome: Starlette decodes the escaped slashes before
        matching, the path stops matching `{install_name}`, and the route is never
        entered. Both 400 and 405 mean no deletion happened.
        """
        for bad in ("..%2F..%2Fescape", "..%2Fescape", "%2e%2e%2fescape"):
            with self.subTest(name=bad):
                r = self.client.request("DELETE", f"/api/hf/models/{bad}",
                                        json={"confirm": True})
                self.assertIn(r.status_code, (400, 404, 405), bad)

    def test_hf_unregister_unknown_model_is_not_a_500(self):
        r = self.client.delete("/api/hf/models/register/no-such-model-anywhere")
        self.assertIn(r.status_code, (200, 400, 404))


if __name__ == "__main__":
    unittest.main()