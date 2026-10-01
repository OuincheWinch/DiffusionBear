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

import pathlib

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
            # The reference is a real file, so exercise that too rather than trusting
            # it: a path that does not exist would 500 later in production only.
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
            json={"mask": mask_data_url(), "prompt": "add a hat", "model": "flux2-klein-9b"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.calls[0]["model"], "flux2-klein-9b")

    def test_the_engine_gets_a_burnt_reference_not_the_original(self):
        """Route-level check on the fix: the reference is a scratch file with the
        painted area greyed out, never the untouched gallery image.

        The file is read DURING the call, from inside the fake engine. Reading it
        afterwards fails with FileNotFoundError, because the handler deletes the
        burnt reference in a `finally` the moment generate() returns -- which is the
        behaviour we want, and also why this test has to look at it mid-flight.
        """
        import generator
        captured = {}

        original_gen = generator.generate

        def spy(**kwargs):
            ref = kwargs["reference_images"][0]
            captured["path"] = ref
            with Image.open(ref) as im:
                captured["ref"] = im.convert("RGB").copy()
            captured["px"] = captured["ref"].load()
            return original_gen(**kwargs)

        generator.generate = spy
        try:
            self.client.post(
                f"/api/images/{self.image_id}/fill",
                json={"mask": mask_data_url(), "prompt": "add a hat"},
            )
        finally:
            generator.generate = original_gen

        self.assertEqual(len(self.calls[0]["reference_images"]), 1)
        self.assertNotEqual(captured["path"], str(self.root / f"{self.image_id}.png"))
        w, h = captured["ref"].size
        inside = [
            captured["px"][x, y]
            for y in range(h // 5, (h * 4) // 5)
            for x in range(w // 5, (w * 4) // 5)
        ]
        self.assertGreater(
            len({c for pix in inside for c in pix}), 12,
            "the hole must not be flat, or the engine reproduces it verbatim",
        )
        self.assertEqual(captured["px"][2, 2], (200, 30, 40), "outside must be untouched")
        self.assertFalse(
            pathlib.Path(captured["path"]).exists(),
            "the burnt reference must be deleted once generation returns",
        )

    # --- rejections, with the right status --------------------------------

    def test_a_ref_capable_but_non_fill_capable_engine_is_refused(self):
        """krea2 accepts references but cannot fill: no spatial correspondence.

        Rejected by the request validator before the engine is loaded.
        """
        r = self.client.post(
            f"/api/images/{self.image_id}/fill",
            json={"mask": mask_data_url(), "prompt": "x", "model": "krea2-turbo"},
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


class FillProgressRouteTests(unittest.TestCase):
    """The progress and cancel endpoints.

    Separate from FillRouteTests because these need a *running* fill, and the fake
    engine there returns instantly. Here it blocks on an event, so a second request can
    be made while the first is still in flight -- which is the entire point of the
    feature and the thing that is easy to get wrong.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._saved_gen = fill.GENERATED_DIR
        self._saved_data = fill.DATA_DIR
        fill.GENERATED_DIR = self.root
        fill.DATA_DIR = self.root
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

        import threading

        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancel_event = None

        def blocking_generate(**kwargs):
            self.cancel_event = kwargs.get("cancel_event")
            self.entered.set()
            # Respect a cancel the way the real engine does, so the test proves the
            # endpoint's signal actually reaches the render rather than just flipping
            # a flag nobody reads.
            for _ in range(200):
                if self.cancel_event is not None and self.cancel_event.is_set():
                    import generator
                    raise generator.GenerationCancelled()
                if self.release.wait(0.05):
                    break
            new_id = "d" * 32
            Image.new("RGB", (32, 32), (10, 220, 90)).save(self.root / f"{new_id}.png")
            (self.root / f"{new_id}.json").write_text('{"id": "%s"}' % new_id, encoding="utf-8")
            return {"id": new_id, "format": "png", "seed": 7, "steps": 4}

        import generator
        self._saved_generate = generator.generate
        generator.generate = blocking_generate

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(gallery_router.router)
        self.client = TestClient(app)

        import threading as _t

        self.result = {}

        def run():
            try:
                self.result["response"] = self.client.post(
                    f"/api/images/{self.image_id}/fill",
                    json={"mask": mask_data_url(), "prompt": "add a hat",
                          "token": "test-token-123"},
                )
            except Exception as exc:  # pragma: no cover - surfaced via self.result
                self.result["error"] = exc

        self.thread = _t.Thread(target=run, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.release.set()
        self.thread.join(timeout=10)
        import generator
        generator.generate = self._saved_generate
        import state
        state.GENERATED_DIR = self._saved_state_gen
        gallery_router.GENERATED_DIR = self._saved_router_gen
        fill.GENERATED_DIR = self._saved_gen
        fill.DATA_DIR = self._saved_data
        self.tmp.cleanup()

    def test_a_running_fill_reports_progress(self):
        self.assertTrue(self.entered.wait(10), "the fake engine never ran")
        r = self.client.get("/api/fill/test-token-123/progress")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["token"], "test-token-123")
        self.assertEqual(body["image_id"], self.image_id)
        self.assertIn("phase", body)
        self.assertTrue(body.get("phase_detail"), "progress needs something to display")

    def test_progress_for_an_unknown_token_is_404(self):
        self.assertEqual(self.client.get("/api/fill/nope-nope-nope/progress").status_code, 404)

    def test_cancelling_ends_the_request_with_409(self):
        self.assertTrue(self.entered.wait(10), "the fake engine never ran")
        r = self.client.post("/api/fill/test-token-123/cancel")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["status"], "cancelling")
        self.thread.join(timeout=15)
        self.assertIn("response", self.result, self.result.get("error"))
        response = self.result["response"]
        self.assertEqual(response.status_code, 409, response.text)
        # Not 400: a cancel is not a malformed request.
        self.assertNotEqual(response.status_code, 400)

    def test_a_cancelled_fill_writes_no_image(self):
        self.assertTrue(self.entered.wait(10), "the fake engine never ran")
        self.client.post("/api/fill/test-token-123/cancel")
        self.thread.join(timeout=15)
        filled = []
        for png in self.root.glob("*.png"):
            meta = self.root / png.with_suffix(".json")
            if meta.exists():
                import json
                try:
                    if json.loads(meta.read_text()).get("filled_from"):
                        filled.append(png.name)
                except Exception:
                    pass
        self.assertEqual(filled, [], "cancelled fill left an image behind")

    def test_cancelling_an_unknown_token_is_not_an_error(self):
        r = self.client.post("/api/fill/nope-nope-nope/cancel")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["status"], "not_running")

    def test_progress_is_cleaned_up_when_the_fill_ends(self):
        self.assertTrue(self.entered.wait(10), "the fake engine never ran")
        self.release.set()
        self.thread.join(timeout=15)
        self.assertEqual(
            self.client.get("/api/fill/test-token-123/progress").status_code, 404,
            "a finished fill must not leave a progress record behind",
        )

    def test_a_malformed_token_is_rejected(self):
        self.assertEqual(self.client.get("/api/fill/bad!token/progress").status_code, 422)

    def test_a_token_with_bad_characters_is_rejected_on_submit(self):
        r = self.client.post(
            f"/api/images/{self.image_id}/fill",
            json={"mask": mask_data_url(), "prompt": "x", "token": "bad token!"},
        )
        self.assertEqual(r.status_code, 422, r.text)


class FillProgressShapeTests(unittest.TestCase):
    """The progress payload must carry everything the UI needs to render a bar.

    Found by running it for real: the readout said "Denoising step 2/?" and the ETA was
    absent, because the fill's request omits `steps` (the engine's own default applies)
    and the route divided by req.steps -- which is None.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._saved = (fill.GENERATED_DIR, fill.DATA_DIR)
        fill.GENERATED_DIR = self.root
        fill.DATA_DIR = self.root
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

    def tearDown(self):
        import state
        state.GENERATED_DIR = self._saved_state_gen
        gallery_router.GENERATED_DIR = self._saved_router_gen
        fill.GENERATED_DIR, fill.DATA_DIR = self._saved
        self.tmp.cleanup()

    def test_effective_steps_falls_back_to_the_model_default(self):
        # omitted by the caller, as the UI does
        self.assertEqual(fill.effective_steps("flux2-klein-4b", None), 4)
        # explicit wins
        self.assertEqual(fill.effective_steps("flux2-klein-4b", 12), 12)
        # never returns None: the ETA divides by this
        self.assertIsNotNone(fill.effective_steps("no-such-model", None))

    def test_progress_reports_a_real_step_total_and_eta(self):
        """Drives the route's own on_step, so this fails if req.steps is used again."""
        seen = []

        def gen(**kwargs):
            cb = kwargs.get("progress_cb")
            if cb:
                cb(0)
                cb(1)
                cb(2)
                cb(3)
            new_id = "d" * 32
            Image.new("RGB", (32, 32), (10, 220, 90)).save(self.root / f"{new_id}.png")
            (self.root / f"{new_id}.json").write_text('{"id": "%s"}' % new_id, encoding="utf-8")
            return {"id": new_id, "format": "png", "seed": 7, "steps": 4}

        import generator

        def run():
            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(gallery_router.router)
            client = TestClient(app)
            r = client.post(
                f"/api/images/{self.image_id}/fill",
                json={"mask": mask_data_url(), "prompt": "add a hat",
                      "token": "shapetoken1234"},
            )
            self.assertEqual(r.status_code, 200, r.text)

        import threading

        original = generator.generate
        generator.generate = gen
        self.addCleanup(setattr, generator, "generate", original)
        self.addCleanup(gallery_router._FILL_PROGRESS.clear)

        # Capture what on_step published by wrapping the dict the route writes to.
        published = []
        original_put = gallery_router._fill_progress_put

        def spy(token, value):
            published.append(dict(value))
            return original_put(token, value)

        gallery_router._fill_progress_put = spy
        self.addCleanup(setattr, gallery_router, "_fill_progress_put", original_put)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=20)

        steps = [p for p in published if "step" in p]
        self.assertTrue(steps, "on_step never published a step update")
        for record in steps:
            self.assertIsInstance(record.get("steps"), int,
                                  "steps must be a number for the progress bar")
            self.assertNotIn("?", record.get("phase_detail", ""),
                             "phase_detail must not show an unknown total")
        self.assertEqual(steps[-1]["step"], 4)
        self.assertEqual(steps[-1]["steps"], 4)
        self.assertIsNotNone(steps[-1].get("eta_seconds"))
        self.assertEqual(steps[-1]["eta_seconds"], 0.0)


class FileUrlRouteTests(unittest.TestCase):
    """GET /api/images/{id}/file-url -- the drag-out-to-Finder contract.

    Found by the user: dragging a gallery card onto the Desktop produced a link to
    http://127.0.0.1:8001/... instead of the image. macOS decides what a drag is from the
    URL scheme, and it reads an http DownloadURL as a web link, so the drop wrote a link
    stub pointing back at the backend -- a stub that also dies with the app.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # A directory name WITH A SPACE, because the real data dir lives on a volume
        # whose name has one and an unencoded space truncates the path.
        self.root = Path(self.tmp.name) / "My Data dir"
        self.root.mkdir()
        self._saved = (fill.GENERATED_DIR, fill.DATA_DIR)
        fill.GENERATED_DIR = self.root
        fill.DATA_DIR = self.root
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

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(gallery_router.router)
        self.client = TestClient(app)

    def tearDown(self):
        import state
        state.GENERATED_DIR = self._saved_state_gen
        gallery_router.GENERATED_DIR = self._saved_router_gen
        fill.GENERATED_DIR, fill.DATA_DIR = self._saved
        self.tmp.cleanup()

    def test_it_returns_a_file_scheme_url(self):
        r = self.client.get(f"/api/images/{self.image_id}/file-url")
        self.assertEqual(r.status_code, 200, r.text)
        url = r.json()["file_url"]
        self.assertTrue(
            url.startswith("file://"),
            f"macOS needs a file:// DownloadURL or the drop writes a link stub; got {url!r}",
        )

    def test_the_path_survives_a_space_in_the_directory_name(self):
        """The regression this route exists beside: /Volumes/Externe/IA/MLX-DIFFUSION
        OpenCode. An unencoded space truncates the path at the first one, and Finder
        would silently receive a path that does not exist."""
        r = self.client.get(f"/api/images/{self.image_id}/file-url")
        url = r.json()["file_url"]
        self.assertNotIn(" ", url, f"the file URL must be percent-encoded: {url!r}")
        self.assertIn("%20", url, "the space must be encoded, not dropped")

    def test_the_url_actually_resolves_to_the_file_on_disk(self):
        from urllib.parse import unquote, urlparse

        r = self.client.get(f"/api/images/{self.image_id}/file-url")
        url = r.json()["file_url"]
        path = Path(unquote(urlparse(url).path))
        self.assertTrue(path.is_file(), f"{path} does not exist")
        self.assertEqual(path.name, f"{self.image_id}.png")

    def test_it_reports_the_real_filename(self):
        body = self.client.get(f"/api/images/{self.image_id}/file-url").json()
        self.assertEqual(body["filename"], f"{self.image_id}.png")

    def test_a_missing_image_is_404(self):
        self.assertEqual(self.client.get(f"/api/images/{'f' * 32}/file-url").status_code, 404)

    def test_a_malformed_id_is_rejected(self):
        self.assertIn(
            self.client.get("/api/images/..%2Fetc/file-url").status_code,
            (400, 404, 422),
        )

    def test_it_never_points_outside_the_gallery(self):
        """Discloses nothing the SPA could not already fetch by id."""
        from urllib.parse import unquote, urlparse

        for name in ("../../etc/passwd", "..", "/etc/hosts"):
            r = self.client.get(f"/api/images/{name}/file-url")
            if r.status_code == 200:
                path = Path(unquote(urlparse(r.json()["file_url"]).path)).resolve()
                self.assertTrue(
                    path.is_relative_to(self.root.resolve()),
                    f"{name} escaped the gallery: {path}",
                )


class GalleryFileUrlTests(unittest.TestCase):
    """The gallery listing carries each image's on-disk location.

    Needed by the native drag: a drag begins in mouseDown, and the shell cannot call back
    into JavaScript in time (evaluateJavaScript is async), so it needs the file URL before
    the gesture starts. Deriving it from a hover-time fetch meant the FIRST drag of a
    session had nothing to use.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "Data Dir With Space"
        self.root.mkdir()
        self._saved = (fill.GENERATED_DIR, fill.DATA_DIR)
        fill.GENERATED_DIR = self.root
        fill.DATA_DIR = self.root
        import state
        self._saved_state_gen = state.GENERATED_DIR
        state.GENERATED_DIR = self.root
        self._saved_router_gen = gallery_router.GENERATED_DIR
        gallery_router.GENERATED_DIR = self.root

        for name in ("a" * 32, "b" * 32):
            Image.new("RGB", (32, 32), (10, 20, 30)).save(self.root / f"{name}.png")
            (self.root / f"{name}.json").write_text(
                '{"id": "%s", "format": "png", "width": 32, "height": 32, "created_at": 1}'
                % name, encoding="utf-8",
            )

        from fastapi import FastAPI
        import state as state_mod

        app = FastAPI()
        app.include_router(gallery_router.router)
        self.client = TestClient(app)
        state_mod.GALLERY_INDEX.clear()
        self._saved_index = dict(state_mod.GALLERY_INDEX)
        # ensure_gallery_index() rate-limits its empty-index retry against this stamp, and
        # a previous test leaves it set, so the rebuild is silently suppressed and the
        # fixture looks empty. Zeroing it is what makes this fixture independent of order.
        self._saved_built_at = state_mod._GALLERY_INDEX_BUILT_AT
        state_mod._GALLERY_INDEX_BUILT_AT = 0.0

    def tearDown(self):
        import state as state_mod
        state_mod.GALLERY_INDEX.clear()
        state_mod.GALLERY_INDEX.update(self._saved_index)
        state_mod._GALLERY_INDEX_BUILT_AT = self._saved_built_at
        state_mod.GENERATED_DIR = self._saved_state_gen
        gallery_router.GENERATED_DIR = self._saved_router_gen
        fill.GENERATED_DIR, fill.DATA_DIR = self._saved
        self.tmp.cleanup()

    def test_every_listed_item_carries_a_file_url(self):
        import state as state_mod

        state_mod.ensure_gallery_index()
        body = self.client.get("/api/gallery?limit=10").json()
        self.assertTrue(body["items"], "fixture produced no items")
        for item in body["items"]:
            with self.subTest(id=item["id"]):
                self.assertIn("file_url", item, "a listed image must expose its path")
                self.assertTrue(item["file_url"].startswith("file://"))

    def test_the_url_is_percent_encoded(self):
        import state as state_mod

        state_mod.ensure_gallery_index()
        items = self.client.get("/api/gallery?limit=10").json()["items"]
        urls = [i["file_url"] for i in items]
        self.assertTrue(any("%20" in u for u in urls),
                        f"the space in the data dir must be encoded, got {urls}")

    def test_the_url_resolves_to_a_real_file(self):
        from urllib.parse import unquote, urlparse

        import state as state_mod

        state_mod.ensure_gallery_index()
        for item in self.client.get("/api/gallery?limit=10").json()["items"]:
            path = Path(unquote(urlparse(item["file_url"]).path))
            with self.subTest(id=item["id"]):
                self.assertTrue(path.is_file(), f"{path} does not exist")

    def test_a_deleted_image_gets_no_file_url(self):
        import state as state_mod

        state_mod.ensure_gallery_index()
        items = self.client.get("/api/gallery?limit=10").json()["items"]
        victim = items[0]
        (self.root / f"{victim['id']}.png").unlink()
        refreshed = self.client.get("/api/gallery?limit=10").json()["items"]
        entry = next(i for i in refreshed if i["id"] == victim["id"])
        self.assertNotIn("file_url", entry,
                         "a file that is gone must not advertise a path to it")
