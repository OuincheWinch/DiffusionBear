"""Tests for the /api/images/{id}/fill route and /api/fill/engines.

The route's job is to be a trustworthy boundary: reject what it cannot do with a
status the UI can act on, and never leak an internal error string to the client.

`test_repo_security.py` audits outbound hosts; `test_i18n.py` and
`test_contrast.py` guard the frontend. This file covers the fill surface, and it
asserts on the HTTP contract rather than on implementation detail, so a refactor
that keeps the contract passes.
"""

import base64
import io
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import fill
import routers.gallery as gallery_router


def mask_data_url(size=(32, 32), box=(8, 8, 24, 24)):
    m = Image.new("L", size, 0)
    for x in range(box[0], box[2]):
        for y in range(box[1], box[3]):
            m.putpixel((x, y), 255)
    buf = io.BytesIO()
    m.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class FillRouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._saved_gen = fill.GENERATED_DIR
        self._saved_data = fill.DATA_DIR
        fill.GENERATED_DIR = self.root
        fill.DATA_DIR = self.root
        # The route validates the source id through state._load_image_meta, which
        # reads state.GENERATED_DIR -- a different module-level binding from
        # fill.GENERATED_DIR. Patching only the latter left the route looking in
        # the real store, which is why every request 404'd. Both, plus the router's
        # own imported name, since gallery.py did `from state import GENERATED_DIR`.
        import state
        self._saved_state_gen = state.GENERATED_DIR
        state.GENERATED_DIR = self.root
        self._saved_router_gen = gallery_router.GENERATED_DIR
        gallery_router.GENERATED_DIR = self.root

        self.image_id = "c" * 32
        Image.new("RGB", (32, 32), (200, 30, 40)).save(self.root / f"{self.image_id}.png")
        (self.root / f"{self.image_id}.json").write_text(
            '{"id": "%s", "format": "png", "width": 32, "height": 32}' % self.image_id,
            encoding="utf-8",
        )

        # Inject a fake engine so no model is loaded.
        self.calls = []

        def fake_generate(**kwargs):
            self.calls.append(kwargs)
            new_id = "d" * 32
            Image.new("RGB", (32, 32), (10, 220, 90)).save(self.root / f"{new_id}.png")
            (self.root / f"{new_id}.json").write_text('{"id": "%s"}' % new_id, encoding="utf-8")
            return {"id": new_id, "format": "png", "seed": 7, "steps": 4}

        self._saved_fill = fill.fill_image
        import generator
        self._saved_generate = generator.generate
        generator.generate = fake_generate

        # Build a client over just this router so the test does not need the whole
        # app (and its model registry / disk scanning) to come up.
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(gallery_router.router)
        self.client = TestClient(app)

    def tearDown(self):
        import generator
        generator.generate = self._saved_generate
        import state
        state.GENERATED_DIR = self._saved_state_gen
        gallery_router.GENERATED_DIR = self._saved_router_gen
        fill.GENERATED_DIR = self._saved_gen
        fill.DATA_DIR = self._saved_data
        self.tmp.cleanup()

    # --- happy path ------------------------------------------------------

    def test_fill_returns_the_new_image_metadata(self):
        r = self.client.post(
            f"/api/images/{self.image_id}/fill",
            json={"mask": mask_data_url(), "prompt": "add a red hat"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("id", body)
        self.assertEqual(body["filled_from"], self.image_id)
        self.assertTrue((self.root / f"{body['id']}.png").exists())

    def test_fill_defaults_to_the_default_engine(self):
        r = self.client.post(
            f"/api/images/{self.image_id}/fill",
            json={"mask": mask_data_url(), "prompt": "add a hat"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.calls[0]["model"], fill.DEFAULT_FILL_ENGINE)

    def test_fill_honours_an_explicit_engine(self):
        r = self.client.post(
            f"/api/images/{self.image_id}/fill",
            json={"mask": mask_data_url(), "prompt": "add a hat", "model": "krea2-turbo"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.calls[0]["model"], "krea2-turbo")

    def test_the_source_image_is_the_reference(self):
        self.client.post(
            f"/api/images/{self.image_id}/fill",
            json={"mask": mask_data_url(), "prompt": "add a hat"},
        )
        refs = self.calls[0]["reference_images"]
        self.assertEqual(len(refs), 1)
        self.assertTrue(refs[0].endswith(f"{self.image_id}.png"))

    # --- rejections, with the right status --------------------------------

    def test_an_engine_without_ref_support_is_422_not_500(self):
        """Rejected by the validator before anything is loaded."""
        r = self.client.post(
            f"/api/images/{self.image_id}/fill",
            json={"mask": mask_data_url(), "prompt": "x", "model": "juggernaut-xl-lightning"},
        )
        self.assertEqual(r.status_code, 422, r.text)
        self.assertEqual(self.calls, [], "the engine must not have been called")

    def test_an_unknown_engine_is_422(self):
        r = self.client.post(
            f"/api/images/{self.image_id}/fill",
            json={"mask": mask_data_url(), "prompt": "x", "model": "not-a-model"},
        )
        self.assertEqual(r.status_code, 422)

    def test_a_blank_prompt_is_422(self):
        r = self.client.post(
            f"/api/images/{self.image_id}/fill",
            json={"mask": mask_data_url(), "prompt": "   "},
        )
        self.assertEqual(r.status_code, 422)
        self.assertEqual(self.calls, [])

    def test_a_missing_prompt_is_422(self):
        r = self.client.post(
            f"/api/images/{self.image_id}/fill",
            json={"mask": mask_data_url()},
        )
        self.assertEqual(r.status_code, 422)

    def test_a_mismatched_mask_is_400_with_the_reason(self):
        """The UI has to be able to say what is actually wrong."""
        r = self.client.post(
            f"/api/images/{self.image_id}/fill",
            json={"mask": mask_data_url(size=(64, 64)), "prompt": "add a hat"},
        )
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("64x64", r.json()["detail"])
        self.assertIn("32x32", r.json()["detail"])

    def test_an_empty_mask_is_400_and_costs_no_render(self):
        empty = Image.new("L", (32, 32), 0)
        buf = io.BytesIO()
        empty.save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        r = self.client.post(
            f"/api/images/{self.image_id}/fill",
            json={"mask": url, "prompt": "add a hat"},
        )
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(self.calls, [], "an empty mask must not cost a render")

    def test_garbage_mask_is_400(self):
        r = self.client.post(
            f"/api/images/{self.image_id}/fill",
            json={"mask": "data:image/png;base64,bm90IGFuIGltYWdl", "prompt": "x"},
        )
        self.assertEqual(r.status_code, 400)

    def test_a_missing_image_is_404(self):
        r = self.client.post(
            f"/api/images/{'e' * 32}/fill",
            json={"mask": mask_data_url(), "prompt": "add a hat"},
        )
        self.assertIn(r.status_code, (404, 400), r.text)

    def test_a_malformed_image_id_is_rejected(self):
        """Path traversal must not reach the filesystem."""
        r = self.client.post(
            "/api/images/..%2F..%2Fetc%2Fpasswd/fill",
            json={"mask": mask_data_url(), "prompt": "x"},
        )
        self.assertIn(r.status_code, (400, 404, 422), r.text)

    def test_an_internal_error_is_500_without_leaking_detail(self):
        import generator

        def boom(**kwargs):
            raise RuntimeError("sensitive internal detail /Users/admin/secret")

        generator.generate = boom
        r = self.client.post(
            f"/api/images/{self.image_id}/fill",
            json={"mask": mask_data_url(), "prompt": "add a hat"},
        )
        self.assertEqual(r.status_code, 500, r.text)
        self.assertNotIn("secret", r.text)
        self.assertNotIn("Users/admin", r.text)

    def test_fill_image_is_the_single_enforcement_point(self):
        """The allowlist lives in fill_image(), so it holds for every caller.

        The route does not re-check the engine: the request model rejects a
        disallowed one with 422 before the handler runs, and the image lookup runs
        first regardless. An earlier version duplicated the check in the handler and
        it was untestable dead code. This asserts the check that is real.
        """
        import generator
        original = generator.generate
        generator.generate = lambda **kw: self._raise_if_called()
        try:
            with self.assertRaises(fill.FillError) as ctx:
                fill.fill_image(
                    self.image_id, mask_data_url(), "add a hat",
                    model="juggernaut-xl-lightning", generate_fn=generator.generate,
                )
        finally:
            generator.generate = original
        self.assertIn("cannot be used for fill", str(ctx.exception))

    def _raise_if_called(self, **kwargs):
        raise AssertionError("the engine must not be reached for a disallowed model")

    def test_unknown_fields_are_forbidden(self):
        r = self.client.post(
            f"/api/images/{self.image_id}/fill",
            json={"mask": mask_data_url(), "prompt": "x", "nonsense": 1},
        )
        self.assertEqual(r.status_code, 422)


class FillEnginesRouteTests(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(gallery_router.router)
        self.client = TestClient(app)

    def test_lists_only_engines_that_can_fill(self):
        r = self.client.get("/api/fill/engines")
        self.assertEqual(r.status_code, 200, r.text)
        ids = {e["id"] for e in r.json()["engines"]}
        self.assertEqual(ids, fill.FILL_ENGINES)

    def test_exactly_one_default_and_it_is_allowed(self):
        body = self.client.get("/api/fill/engines").json()
        defaults = [e["id"] for e in body["engines"] if e["is_default"]]
        self.assertEqual(defaults, [body["default"]])
        self.assertIn(body["default"], fill.FILL_ENGINES)

    def test_every_engine_carries_a_label(self):
        for engine in self.client.get("/api/fill/engines").json()["engines"]:
            self.assertTrue(engine["label"], f"{engine['id']} has no label")


if __name__ == "__main__":
    unittest.main()
