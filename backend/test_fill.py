"""Tests for the generative-fill composite.

The guarantee under test is narrow and absolute: a fill must not change a single
pixel outside the mask. Everything else in this module is presentation, and this is
the one property that, if broken, silently damages a user's image. So the central
test compares raw bytes outside the mask rather than asserting on an average or a
tolerance.

The seam tests exist because a visible cut is the whole failure mode of
mask-compositing: a hard edge reads as a mistake, and the feather is what hides it.
"""

import base64
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

import fill
from fill import FillError, composite_fill, mask_bounding_box, _decode_mask


def solid(size, colour):
    return Image.new("RGB", size, colour)


def mask_image(size, box, colour=255):
    """A mask with a single filled rectangle."""
    m = Image.new("L", size, 0)
    for x in range(box[0], box[2]):
        for y in range(box[1], box[3]):
            m.putpixel((x, y), colour)
    return m


def b64_png(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class CompositeTests(unittest.TestCase):
    def test_pixels_outside_the_mask_are_untouched(self):
        """The guarantee. Byte-exact, not approximate."""
        original = solid((64, 64), (200, 30, 40))
        generated = solid((64, 64), (10, 220, 90))
        mask = mask_image((64, 64), (20, 20, 44, 44))

        out = composite_fill(original, generated, mask, feather_px=0)

        self.assertEqual(out.size, original.size)
        # A pixel well outside the box must be exactly the original colour.
        for xy in [(0, 0), (63, 0), (0, 63), (63, 63), (10, 32), (50, 32)]:
            self.assertEqual(
                out.getpixel(xy),
                (200, 30, 40),
                f"pixel {xy} outside the mask was modified",
            )

    def test_pixels_inside_the_mask_come_from_the_generation(self):
        original = solid((64, 64), (200, 30, 40))
        generated = solid((64, 64), (10, 220, 90))
        # A box large enough that its centre is far from the feather, which is only
        # relevant when feather_px > 0; at 0 the whole box is a hard take.
        mask = mask_image((64, 64), (16, 16, 48, 48))

        out = composite_fill(original, generated, mask, feather_px=0)

        self.assertEqual(out.getpixel((32, 32)), (10, 220, 90))
        self.assertEqual(out.getpixel((20, 20)), (10, 220, 90))

    def test_full_white_mask_is_a_full_rerender(self):
        """A legitimate request, not an error: regenerate the whole image."""
        original = solid((32, 32), (1, 2, 3))
        generated = solid((32, 32), (9, 8, 7))
        mask = Image.new("L", (32, 32), 255)

        out = composite_fill(original, generated, mask, feather_px=0)
        self.assertEqual(out.getpixel((16, 16)), (9, 8, 7))

    def test_feather_softens_the_boundary(self):
        """The seam must be a ramp, not a step.

        Without a feather, the pixel just inside the edge is 100% generated and the
        one just outside is 100% original -- a one-pixel step from one flat colour
        to another, which is the visible cut. With a feather, the transition spans
        several pixels, so each step is a small colour delta instead.
        """
        original = solid((64, 64), (0, 0, 0))
        generated = solid((64, 64), (255, 255, 255))
        mask = mask_image((64, 64), (32, 0, 64, 64))

        hard = composite_fill(original, generated, mask, feather_px=0)
        soft = composite_fill(original, generated, mask, feather_px=12)

        def max_step(img):
            """Largest single-pixel horizontal jump in the red channel."""
            px = img.load()
            return max(
                abs(px[x + 1, y][0] - px[x, y][0])
                for y in range(img.size[1])
                for x in range(img.size[0] - 1)
            )

        self.assertEqual(max_step(hard), 255, "a hard mask must produce a one-pixel step")
        self.assertLess(
            max_step(soft),
            255,
            "the feather did not soften the boundary at all",
        )

    def test_generated_size_mismatch_is_rejected(self):
        original = solid((32, 32), (0, 0, 0))
        generated = solid((16, 16), (255, 255, 255))
        mask = Image.new("L", (32, 32), 255)
        with self.assertRaises(FillError):
            composite_fill(original, generated, mask)

    def test_mask_size_mismatch_is_rejected(self):
        original = solid((32, 32), (0, 0, 0))
        generated = solid((32, 32), (255, 255, 255))
        mask = Image.new("L", (16, 16), 255)
        with self.assertRaises(FillError):
            composite_fill(original, generated, mask)

    def test_empty_mask_is_rejected(self):
        """Better to fail fast than burn a two-minute render for a no-op."""
        original = solid((32, 32), (0, 0, 0))
        generated = solid((32, 32), (255, 255, 255))
        mask = Image.new("L", (32, 32), 0)
        with self.assertRaises(FillError):
            composite_fill(original, generated, mask, feather_px=0)


class MaskDecodeTests(unittest.TestCase):
    def test_decodes_a_matching_mask(self):
        mask = mask_image((40, 30), (5, 5, 20, 20))
        out = _decode_mask(b64_png(mask), (40, 30))
        self.assertEqual(out.size, (40, 30))
        self.assertEqual(out.mode, "L")

    def test_accepts_a_data_url(self):
        mask = mask_image((16, 16), (2, 2, 8, 8))
        payload = "data:image/png;base64," + b64_png(mask)
        self.assertEqual(_decode_mask(payload, (16, 16)).size, (16, 16))

    def test_dimension_mismatch_is_rejected_not_resized(self):
        """A fractional rescale of a mask edge is the hard seam this avoids."""
        mask = mask_image((16, 16), (2, 2, 8, 8))
        with self.assertRaises(FillError) as ctx:
            _decode_mask(b64_png(mask), (32, 32))
        self.assertIn("16x16", str(ctx.exception))

    def test_garbage_is_rejected(self):
        with self.assertRaises(FillError):
            _decode_mask("not base64 at all !!", (16, 16))

    def test_empty_payload_is_rejected(self):
        with self.assertRaises(FillError):
            _decode_mask("", (16, 16))

    def test_a_non_image_payload_is_rejected(self):
        with self.assertRaises(FillError):
            _decode_mask(base64.b64encode(b"hello world").decode(), (16, 16))


class TranslucentPaintTests(unittest.TestCase):
    """The brush paints translucent white so the overlay is a wash, not a bar.

    It still has to mask. The backend reads the mask via convert("L"), which
    reads luma, so a translucent white must still yield 255. Guessing that is how
    the overlay ends up visible but non-functional, so it is checked here against
    the real decoder.
    """

    def test_translucent_white_still_masks(self):
        for alpha in (107, 160, 255):
            with self.subTest(alpha=alpha):
                m = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
                m.paste((255, 255, 255, alpha), (16, 16, 48, 48))
                decoded = _decode_mask(b64_png(m), (64, 64))
                self.assertEqual(decoded.getpixel((32, 32)), 255)
                self.assertEqual(decoded.getpixel((2, 2)), 0)

    def test_translucent_paint_produces_a_bounding_box(self):
        m = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        m.paste((255, 255, 255, 107), (10, 10, 50, 50))
        decoded = _decode_mask(b64_png(m), (64, 64))
        self.assertIsNotNone(mask_bounding_box(decoded))

    def test_painted_strokes_are_opaque_so_erase_works(self):
        """Documents why the canvas must hold opaque white, not a translucent wash.

        Painting translucent white masks fine, but ERASING it does not. The canvas
        `destination-out` reduces the stroke's ALPHA to 0 while leaving its RGB
        white, and PIL's convert("L") reads luma, not alpha -- so the "erased"
        region still reads as 255 and keeps masking. The eraser silently did
        nothing.

        Verified with the exact arithmetic the browser performs:
            dst_alpha *= (1 - src_alpha)
        The RGB channels are untouched by that operation, which is the whole
        problem. Asserted here so a future change to the brush that reintroduces
        translucent paint is caught by a test rather than by a user.
        """
        w = h = 64
        painted = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(painted).rectangle([10, 10, 50, 50], fill=(255, 255, 255, 255))
        self.assertEqual(painted.getpixel((30, 30)), (255, 255, 255, 255))

        # destination-out: alpha is multiplied down, RGB is left alone.
        erased = painted.copy()
        put = Image.new("L", (w, h), 0)
        ImageDraw.Draw(put).rectangle([20, 20, 40, 40], fill=255)
        # destination-out on a solid stroke drives its alpha to 0 in the erased box.
        px = erased.load()
        for y in range(20, 41):
            for x in range(20, 41):
                r, g, b, _a = px[x, y]
                px[x, y] = (r, g, b, 0)

        self.assertEqual(
            erased.getpixel((30, 30)),
            (255, 255, 255, 0),
            "erase lowers alpha but leaves RGB white -- this is why a translucent "
            "canvas would make the eraser a no-op",
        )
        self.assertEqual(
            _decode_mask(b64_png(erased), (w, h)).getpixel((30, 30)),
            255,
            "and convert('L') therefore still masks the erased region",
        )


class BoundingBoxTests(unittest.TestCase):
    def test_finds_the_marked_region(self):
        mask = mask_image((100, 100), (10, 20, 40, 60))
        self.assertEqual(mask_bounding_box(mask), (10, 20, 40, 60))

    def test_returns_none_when_nothing_is_marked(self):
        self.assertIsNone(mask_bounding_box(Image.new("L", (50, 50), 0)))

    def test_ignores_faint_noise(self):
        """A stray antialiased pixel must not be treated as a selection."""
        mask = Image.new("L", (50, 50), 0)
        mask.putpixel((5, 5), 3)          # below the threshold
        mask_bounding_box(mask, threshold=16)
        self.assertIsNone(mask_bounding_box(mask, threshold=16))


class EnginePolicyTests(unittest.TestCase):
    """Scope is the point: not every engine can serve a fill."""

    def test_only_ref_capable_engines_are_allowed(self):
        import generator
        for engine in fill.FILL_ENGINES:
            with self.subTest(engine=engine):
                self.assertTrue(
                    generator.MODELS[engine].get("supports_ref"),
                    f"{engine} is in FILL_ENGINES but supports_ref is False, so a "
                    f"fill would have no reference to condition on",
                )

    def test_every_allowed_engine_is_ref_capable(self):
        """Necessary condition: an allowed engine must at least accept a reference."""
        import generator
        for engine in fill.FILL_ENGINES:
            with self.subTest(engine=engine):
                self.assertTrue(generator.MODELS[engine].get("supports_ref"))

    def test_only_in_context_engines_are_allowed(self):
        """The condition that actually matters, and it is stricter than supports_ref.

        Measured on the bear image: with z-image-turbo, "pink sunglasses" painted
        across the eyes returned altered eyes and no sunglasses, and a grey reference
        came back as a grey rectangle. generator.py:2338 passes a non-FLUX2 engine a
        single `image_path`, which mflux uses for style conditioning with no spatial
        correspondence -- so there is no region to fill. Only the FLUX2 `edit` variant
        (generator.py:2327, `image_paths=[...]`) attends to reference AND prompt
        together.

        If this test fails after adding an engine, that engine must be checked for real
        spatial conditioning first, not just supports_ref.
        """
        import generator
        for engine in fill.FILL_ENGINES:
            with self.subTest(engine=engine):
                self.assertTrue(
                    engine.startswith("flux2"),
                    f"{engine} is offered for fill but is not an FLUX2 in-context "
                    f"engine; it will copy the painted region through instead of "
                    f"filling it",
                )

    def test_default_engine_can_actually_fill(self):
        self.assertIn(fill.DEFAULT_FILL_ENGINE, fill.FILL_CAPABLE_ENGINES)

    def test_z_image_is_refused_because_it_cannot_fill(self):
        """The exact failure the user hit. Pinned so it cannot come back quietly."""
        with self.assertRaises(fill.FillError) as ctx:
            fill.fill_image("x" * 32, "unused", "pink sunglasses", model="z-image-turbo")
        self.assertIn("spatial", str(ctx.exception).lower())


class FillImageTests(unittest.TestCase):
    """End-to-end with the engine injected, so no model is loaded."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._saved = (fill.GENERATED_DIR, fill.DATA_DIR)
        fill.GENERATED_DIR = self.root
        fill.DATA_DIR = self.root

    def tearDown(self):
        fill.GENERATED_DIR, fill.DATA_DIR = self._saved
        self.tmp.cleanup()

    def _seed_source(self, size=(48, 48), colour=(200, 30, 40)):
        self.src_id = "a" * 32
        Image.new("RGB", size, colour).save(self.root / f"{self.src_id}.png")
        (self.root / f"{self.src_id}.json").write_text(
            '{"id": "%s", "format": "png", "tags": ["kept"]}' % self.src_id,
            encoding="utf-8",
        )
        return self.src_id

    def _fake_generate(self, generated_colour=(10, 220, 90)):
        """Stands in for generator.generate, writing a render where it is expected."""
        calls = {}

        def gen(**kwargs):
            calls.update(kwargs)
            new_id = "b" * 32
            Image.new("RGB", (48, 48), generated_colour).save(self.root / f"{new_id}.png")
            (self.root / f"{new_id}.json").write_text(
                '{"id": "%s"}' % new_id, encoding="utf-8"
            )
            return {"id": new_id, "format": "png", "seed": 42, "steps": 4}

        gen.calls = calls
        return gen

    def test_fill_writes_a_new_image_and_sidecar(self):
        self._seed_source()
        gen = self._fake_generate()
        mask = mask_image((48, 48), (12, 12, 36, 36))

        meta = fill.fill_image(
            self.src_id, b64_png(mask), "add a hat", generate_fn=gen
        )

        self.assertTrue((self.root / f"{meta['id']}.png").exists())
        self.assertTrue((self.root / f"{meta['id']}.json").exists())
        self.assertEqual(meta["filled_from"], self.src_id)
        self.assertTrue(meta["fill_method"].startswith("masked-composite"))
        self.assertIn("fill", meta["tags"])
        self.assertIn("kept", meta["tags"], "parent tags must survive")

    def test_the_engine_receives_a_mask_burnt_reference(self):
        """Not the original. This is the whole trick.

        Handing the engine the untouched source plus "pink sunglasses" gives it no
        idea where the user painted, so it regenerates the whole image and the
        composite keeps whatever happened to land in the box -- altered eyes, no
        sunglasses. The reference must have the painted area burnt out so the engine
        has both the intent and the location.
        """
        self._seed_source(colour=(200, 30, 40))
        gen = self._fake_generate()
        mask = mask_image((48, 48), (12, 12, 36, 36))

        captured = {}

        def spy(**kwargs):
            from PIL import Image as _I
            ref = kwargs["reference_images"][0]
            with _I.open(ref) as im:
                captured["pixels"] = im.convert("RGB").copy()   # .copy(): the file is closed after
            return gen(**kwargs)

        fill.fill_image(self.src_id, b64_png(mask), "pink sunglasses", generate_fn=spy)

        ref_path = gen.calls["reference_images"][0]
        self.assertNotEqual(
            ref_path, str(self.root / f"{self.src_id}.png"),
            "the untouched source must not be handed to the engine",
        )
        ref = captured["pixels"]
        px = ref.load()          # .load() gives the pixel accessor; the Image is not subscriptable
        # The reference is rendered at the region's render size, NOT the source size,
        # so the mask box has to be mapped into reference coordinates. Hardcoding 14..34
        # silently sampled the top-left corner of a 128x128 reference -- unmasked flat
        # red -- and reported a healthy hole as "flat".
        sx, sy = ref.width / mask.width, ref.height / mask.height
        mb = mask.getbbox()
        inset = 2
        x0, y0 = int(mb[0] * sx) + inset, int(mb[1] * sy) + inset
        x1, y1 = int(mb[2] * sx) - inset, int(mb[3] * sy) - inset
        inside = [px[x, y] for y in range(y0, y1) for x in range(x0, x1)]
        tones = {c for pix in inside for c in pix}
        self.assertGreater(
            len(tones), 12,
            "the hole is flat: a uniform region is reproduced faithfully by the "
            "engine, which is what produced a grey box instead of a fill",
        )
        self.assertEqual(px[2, 2], (200, 30, 40), "unmasked area was altered")

    def test_the_burnt_reference_is_deleted_afterwards(self):
        """A scratch artifact must never end up in the user's library."""
        self._seed_source()
        gen = self._fake_generate()
        mask = mask_image((48, 48), (12, 12, 36, 36))

        fill.fill_image(self.src_id, b64_png(mask), "add a hat", generate_fn=gen)

        leftover = [p.name for p in (self.root / "uploads").glob(".fill-ref-*.png")]
        self.assertEqual(leftover, [], f"burnt reference left behind: {leftover}")

    def test_the_burnt_reference_is_removed_even_when_generation_fails(self):
        self._seed_source()
        mask = mask_image((48, 48), (12, 12, 36, 36))

        def boom(**kwargs):
            raise RuntimeError("engine died")

        with self.assertRaises(RuntimeError):
            fill.fill_image(self.src_id, b64_png(mask), "add a hat", generate_fn=boom)
        self.assertEqual(list((self.root / "uploads").glob(".fill-ref-*.png")), [])

    def test_the_mask_is_not_sent_as_a_second_reference(self):
        """The region reaches the engine burnt into the image, not as a separate
        reference. Passing it as its own reference would ask the model to DRAW the
        mask rather than fill the region."""
        self._seed_source()
        gen = self._fake_generate()
        mask = mask_image((48, 48), (12, 12, 36, 36))

        fill.fill_image(self.src_id, b64_png(mask), "add a hat", generate_fn=gen)

        self.assertEqual(len(gen.calls["reference_images"]), 1)

    def test_the_unmasked_intermediate_is_removed(self):
        """Otherwise the gallery shows a full render the user never asked for."""
        self._seed_source()
        gen = self._fake_generate()
        mask = mask_image((48, 48), (12, 12, 36, 36))

        meta = fill.fill_image(self.src_id, b64_png(mask), "add a hat", generate_fn=gen)

        self.assertFalse((self.root / f"{'b' * 32}.png").exists())
        self.assertFalse((self.root / f"{'b' * 32}.json").exists())
        self.assertTrue((self.root / f"{meta['id']}.png").exists())

    def test_the_source_image_is_never_overwritten(self):
        self._seed_source()
        gen = self._fake_generate()
        mask = mask_image((48, 48), (12, 12, 36, 36))

        fill.fill_image(self.src_id, b64_png(mask), "add a hat", generate_fn=gen)

        with Image.open(self.root / f"{self.src_id}.png") as img:
            self.assertEqual(img.getpixel((0, 0)), (200, 30, 40))

    def test_a_disallowed_engine_is_refused_before_loading_anything(self):
        self._seed_source()
        gen = self._fake_generate()
        mask = mask_image((48, 48), (12, 12, 36, 36))

        with self.assertRaises(FillError) as ctx:
            fill.fill_image(
                self.src_id, b64_png(mask), "add a hat",
                model="juggernaut-xl-lightning", generate_fn=gen,
            )
        self.assertIn("cannot be used for fill", str(ctx.exception))
        self.assertEqual(gen.calls, {}, "the engine must not have been called")

    def test_an_empty_prompt_is_refused(self):
        self._seed_source()
        gen = self._fake_generate()
        mask = mask_image((48, 48), (12, 12, 36, 36))
        with self.assertRaises(FillError):
            fill.fill_image(self.src_id, b64_png(mask), "   ", generate_fn=gen)

    def test_an_empty_mask_is_refused_before_rendering(self):
        self._seed_source()
        gen = self._fake_generate()
        empty = Image.new("L", (48, 48), 0)
        with self.assertRaises(FillError):
            fill.fill_image(self.src_id, b64_png(empty), "add a hat", generate_fn=gen)
        self.assertEqual(gen.calls, {}, "an empty mask must not cost a render")

    def test_a_missing_source_is_refused(self):
        gen = self._fake_generate()
        mask = mask_image((48, 48), (12, 12, 36, 36))
        with self.assertRaises(FillError):
            fill.fill_image("f" * 32, b64_png(mask), "add a hat", generate_fn=gen)
        self.assertEqual(gen.calls, {})

    def test_a_corrupt_source_sidecar_does_not_block_the_fill(self):
        """The image is the source of truth for pixels; the sidecar is metadata."""
        self._seed_source()
        (self.root / f"{self.src_id}.json").write_text("{not json", encoding="utf-8")
        gen = self._fake_generate()
        mask = mask_image((48, 48), (12, 12, 36, 36))

        meta = fill.fill_image(self.src_id, b64_png(mask), "add a hat", generate_fn=gen)
        self.assertTrue(meta["id"])


if __name__ == "__main__":
    unittest.main()


class RenderPlanTests(unittest.TestCase):
    """The render budget, which is what stops a large source from hanging.

    Measured on flux2-klein-4b 4 steps: 590k px costs ~197s, and cost is linear in
    pixel count. So a 1536x1536 source rendered whole would be ~789s and a 3072x3072
    one ~3156s. These tests pin the arithmetic that bounds it, without loading a model.
    """

    @staticmethod
    def _mask(size, frac):
        w, h = size
        m = Image.new("L", (w, h), 0)
        side = max(2, int(min(w, h) * (frac ** 0.5)))
        cx, cy = w // 2, h // 2
        for x in range(cx - side // 2, cx + side // 2):
            for y in range(cy - side // 2, cy + side // 2):
                m.putpixel((x, y), 255)
        return m

    def test_a_small_edit_to_a_huge_image_renders_small(self):
        """The whole point: cost tracks what was painted, not the source size."""
        mask = self._mask((3072, 3072), 0.05)
        plan = fill._render_plan(mask, fill.mask_bounding_box(mask))
        rw, rh = plan["render"]
        self.assertLessEqual(rw * rh, fill.FILL_RENDER_MAX_PIXELS)

    def test_the_budget_is_never_exceeded(self):
        for size in [(512, 512), (768, 768), (1536, 1536), (3072, 3072), (4096, 4096)]:
            for frac in (0.05, 0.25, 0.5, 1.0):
                with self.subTest(size=size, frac=frac):
                    mask = self._mask(size, frac)
                    plan = fill._render_plan(mask, fill.mask_bounding_box(mask))
                    rw, rh = plan["render"]
                    self.assertLessEqual(
                        rw * rh,
                        fill.FILL_RENDER_MAX_PIXELS,
                        f"{size} at {frac:.0%} renders {rw}x{rh}, over budget",
                    )

    def test_the_crop_always_contains_the_painted_region(self):
        """Cropping must never cut into the mask -- that would drop part of the fill."""
        for size in [(512, 512), (1536, 1536), (3072, 3072)]:
            for frac in (0.05, 0.4, 1.0):
                with self.subTest(size=size, frac=frac):
                    mask = self._mask(size, frac)
                    box = fill.mask_bounding_box(mask)
                    crop = fill._render_plan(mask, box)["crop"]
                    self.assertLessEqual(crop[0], box[0])
                    self.assertLessEqual(crop[1], box[1])
                    self.assertGreaterEqual(crop[2], box[2])
                    self.assertGreaterEqual(crop[3], box[3])

    def test_the_crop_stays_inside_the_image(self):
        for size in [(512, 512), (1536, 1536)]:
            for frac in (0.05, 0.9, 1.0):
                with self.subTest(size=size, frac=frac):
                    mask = self._mask(size, frac)
                    crop = fill._render_plan(mask, fill.mask_bounding_box(mask))["crop"]
                    self.assertGreaterEqual(crop[0], 0)
                    self.assertGreaterEqual(crop[1], 0)
                    self.assertLessEqual(crop[2], size[0])
                    self.assertLessEqual(crop[3], size[1])

    def test_a_small_image_is_not_upscaled(self):
        """Never render larger than the source: that costs time and invents detail."""
        mask = self._mask((256, 256), 0.5)
        plan = fill._render_plan(mask, fill.mask_bounding_box(mask))
        self.assertLessEqual(plan["render"][0], 256)
        self.assertLessEqual(plan["render"][1], 256)
        self.assertEqual(plan["scale"], 1.0)

    def test_a_full_frame_mask_gets_context_but_stays_bounded(self):
        """A full-frame fill is legitimate and expensive; it must still fit the budget."""
        w, h = 1536, 1536
        mask = Image.new("L", (w, h), 255)
        plan = fill._render_plan(mask, (0, 0, w, h))
        rw, rh = plan["render"]
        self.assertLessEqual(rw * rh, fill.FILL_RENDER_MAX_PIXELS)
        self.assertGreater(plan["scale"], 0.0)


class RenderedSizeTests(unittest.TestCase):
    """A downscaled render must still composite to full source resolution."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._gen, self._data = fill.GENERATED_DIR, fill.DATA_DIR
        fill.GENERATED_DIR = self.root
        fill.DATA_DIR = self.root

    def tearDown(self):
        fill.GENERATED_DIR, fill.DATA_DIR = self._gen, self._data
        self.tmp.cleanup()

    def test_the_output_is_always_the_full_source_size(self):
        """Whatever the render was scaled to, the result is source-sized."""
        size = (1024, 1024)
        image_id = "e" * 32
        Image.new("RGB", size, (200, 30, 40)).save(self.root / f"{image_id}.png")
        (self.root / f"{image_id}.json").write_text(
            '{"id": "%s", "format": "png"}' % image_id, encoding="utf-8"
        )

        m = Image.new("L", size, 0)
        d = ImageDraw.Draw(m)
        d.rectangle([300, 420, 700, 600], fill=255)
        buf = io.BytesIO(); m.save(buf, format="PNG")
        payload = base64.b64encode(buf.getvalue()).decode("ascii")

        def gen(**kwargs):
            # Answer with whatever size was asked for, then a different tone, so the
            # composite is visibly the generated one.
            w, h = kwargs["width"], kwargs["height"]
            new_id = "f" * 32
            Image.new("RGB", (w, h), (10, 220, 90)).save(self.root / f"{new_id}.png")
            (self.root / f"{new_id}.json").write_text('{"id": "%s"}' % new_id, encoding="utf-8")
            return {"id": new_id, "format": "png", "seed": 1, "steps": 4}

        meta = fill.fill_image(image_id, payload, "pink sunglasses", generate_fn=gen)
        with Image.open(self.root / f"{meta['id']}.png") as out:
            self.assertEqual(out.size, size, "the fill must be source-sized")
        self.assertLess(meta["fill_render_size"][0] * meta["fill_render_size"][1], size[0] * size[1])
        out = Image.open(self.root / f"{meta['id']}.png").convert("RGB")
        px = out.load()
        self.assertEqual(px[2, 2], (200, 30, 40), "outside the mask must be untouched")
        # Inside the mask the result must differ from the source. Poisson blending
        # shifts colour to match the destination gradient, so the interior is NOT the
        # raw generated RGB -- asserting that exact value would fail on a correct
        # implementation. What matters is that the region changed at all.
        self.assertNotEqual(
            px[500, 510], (200, 30, 40),
            "inside the mask must not be the untouched source",
        )


class HaloMustNotReachTheOutputTests(unittest.TestCase):
    """The brush's dark halo is visualisation, not mask.

    The brush strokes a dark ring around each dab so a selection is visible on a
    white image. It lives in the same canvas as the mask, so without hard
    thresholding it arrived as a partial mask value and was composited into the
    result as a dark border around every fill. Users reported exactly that.
    """

    HALO = 60      # a plausible halo luma
    CORE = 255

    def _stroke_with_halo(self, size=(96, 96), core=30, halo=45):
        m = Image.new("L", size, 0)
        cx = cy = size[0] // 2
        # core disc
        for x in range(cx - core, cx + core):
            for y in range(cy - core, cy + core):
                if (x - cx) ** 2 + (y - cy) ** 2 <= core ** 2:
                    m.putpixel((x, y), self.CORE)
        # halo ring
        for x in range(cx - halo, cx + halo):
            for y in range(cy - halo, cy + halo):
                d2 = (x - cx) ** 2 + (y - cy) ** 2
                if core ** 2 < d2 <= halo ** 2:
                    m.putpixel((x, y), self.HALO)
        return m

    def test_the_halo_is_stripped_by_the_decoder(self):
        decoded = _decode_mask(b64_png(self._stroke_with_halo()), (96, 96))
        self.assertEqual(decoded.getpixel((48, 48)), 255, "core must still select")
        # A point in the ring but well outside the core.
        ring = decoded.getpixel((48 + 38, 48))
        self.assertEqual(ring, 0, "the dark halo must not survive as a mask value")

    def test_the_bounding_box_ignores_the_halo(self):
        decoded = _decode_mask(b64_png(self._stroke_with_halo()), (96, 96))
        box = mask_bounding_box(decoded)
        self.assertIsNotNone(box)
        cx = cy = 48
        # The box should hug the core, not the halo.
        self.assertLessEqual(box[2] - box[0], 2 * 31)

    def test_a_halo_only_stroke_does_not_mask_anything(self):
        """The failure mode if the threshold were ever lost: the whole ring becomes
        part of the mask and the fill silently regenerates a ring-shaped region."""
        m = Image.new("L", (64, 64), 0)
        for x in range(8, 56):
            for y in range(8, 56):
                if not (20 < x < 44 and 20 < y < 44):
                    m.putpixel((x, y), self.HALO)
        decoded = _decode_mask(b64_png(m), (64, 64))
        self.assertIsNone(
            mask_bounding_box(decoded),
            "a halo with no white core must not produce a fill region",
        )


class SeamTests(unittest.TestCase):
    """A fill must not leave a visible rectangle where the region was replaced."""

    def _scene(self, size=(160, 160)):
        """A smooth background, so any seam shows up as a step in the gradient."""
        img = Image.new("RGB", (size[0], size[1]))
        px = img.load()
        for y in range(size[1]):
            for x in range(size[0]):
                v = 60 + (x * 120) // size[0]
                px[x, y] = (v, v, v)
        return img

    def _square_mask(self, size=(160, 160), box=(48, 48, 112, 112)):
        return mask_image(size, box)

    def test_seamless_composite_is_used(self):
        original = self._scene()
        generated = self._scene()
        gp = generated.load()
        for y in range(48, 112):                 # a distinctly different interior
            for x in range(48, 112):
                gp[x, y] = (240, 30, 30)
        mask = self._square_mask()
        out = fill._seamless_composite(original, generated, mask)
        self.assertEqual(out.size, original.size)

    def test_nothing_outside_the_mask_changes(self):
        original = self._scene()
        generated = self._scene()
        gp = generated.load()
        for y in range(48, 112):
            for x in range(48, 112):
                gp[x, y] = (240, 30, 30)
        mask = self._square_mask()
        out = fill._seamless_composite(original, generated, mask)
        op, xp = original.load(), out.load()
        for y in (0, 20, 47, 113, 140, 159):
            for x in range(160):
                self.assertEqual(xp[x, y], op[x, y], f"row {y} outside the mask changed")
        for x in (0, 20, 47, 113, 140, 159):
            for y in range(160):
                self.assertEqual(xp[x, y], op[x, y], f"col {x} outside the mask changed")

    def test_the_interior_actually_changed(self):
        original = self._scene()
        generated = self._scene()
        gp = generated.load()
        for y in range(48, 112):
            for x in range(48, 112):
                gp[x, y] = (240, 30, 30)
        out = fill._seamless_composite(original, generated, self._square_mask())
        self.assertNotEqual(out.load()[80, 80], original.load()[80, 80])

    def test_poisson_blending_beats_a_hard_alpha_edge(self):
        """The reason seamlessClone is used: measure the step at the boundary.

        A feathered alpha composite leaves a colour step where the generated content
        meets the preserved pixels. Poisson blending solves for a smooth transition,
        so the largest single-pixel jump across the boundary should be much smaller.
        """
        original = self._scene()
        generated = self._scene()
        gp = generated.load()
        for y in range(48, 112):
            for x in range(48, 112):
                gp[x, y] = (240, 30, 30)
        mask = self._square_mask()

        def max_step_horizontal(img):
            px = img.load()
            return max(
                sum(abs(px[x + 1, y][c] - px[x, y][c]) for c in range(3))
                for y in range(img.size[1])
                for x in range(img.size[0] - 1)
            )

        alpha = fill.composite_fill(original, generated, mask, feather_px=12)
        poisson = fill._seamless_composite(original, generated, mask)
        self.assertLess(
            max_step_horizontal(poisson),
            max_step_horizontal(alpha),
            "Poisson blending should show a smaller step at the boundary than an "
            "alpha composite",
        )


class RenderFloorTests(unittest.TestCase):
    """The engine rejects any dimension under 128 (generator._validate_dimensions).

    A thin selection is the common case -- sunglasses are a band -- so this is not a
    theoretical edge. Measured: a 54px-tall band over a 768px image rendered 507x81
    and the whole fill failed with ValueError.
    """

    SHAPES = [
        ("thin band", (768, 768), (215, 307, 553, 361)),
        ("wide band", (512, 512), (120, 180, 392, 240)),
        ("small blob", (768, 768), (360, 360, 400, 392)),
        ("corner blob", (768, 768), (0, 0, 40, 32)),
        ("portrait band", (512, 768), (100, 300, 400, 320)),
        ("huge source", (3072, 3072), (1400, 1500, 1600, 1520)),
    ]

    @staticmethod
    def _mask(size, box):
        m = Image.new("L", size, 0)
        ImageDraw.Draw(m).rounded_rectangle(list(box), radius=20, fill=255)
        return m

    def test_every_shape_clears_the_engine_floor(self):
        for name, size, box in self.SHAPES:
            with self.subTest(shape=name):
                m = self._mask(size, box)
                rw, rh = fill._render_plan(m, fill.mask_bounding_box(m))["render"]
                self.assertGreaterEqual(rw, fill.MIN_RENDER_SIDE, f"{name}: width {rw}")
                self.assertGreaterEqual(rh, fill.MIN_RENDER_SIDE, f"{name}: height {rh}")

    def test_every_shape_stays_within_the_pixel_budget(self):
        for name, size, box in self.SHAPES:
            with self.subTest(shape=name):
                m = self._mask(size, box)
                rw, rh = fill._render_plan(m, fill.mask_bounding_box(m))["render"]
                self.assertLessEqual(rw * rh, fill.FILL_RENDER_MAX_PIXELS, name)

    def test_the_crop_never_cuts_the_mask(self):
        for name, size, box in self.SHAPES:
            with self.subTest(shape=name):
                m = self._mask(size, box)
                b = fill.mask_bounding_box(m)
                c = fill._render_plan(m, b)["crop"]
                self.assertLessEqual(c[0], b[0], f"{name}: crop cuts the mask")
                self.assertLessEqual(c[1], b[1], f"{name}: crop cuts the mask")
                self.assertGreaterEqual(c[2], b[2], f"{name}: crop cuts the mask")
                self.assertGreaterEqual(c[3], b[3], f"{name}: crop cuts the mask")

    def test_a_full_frame_fill_is_still_bounded(self):
        w, h = 1536, 1536
        m = Image.new("L", (w, h), 255)
        rw, rh = fill._render_plan(m, (0, 0, w, h))["render"]
        self.assertGreaterEqual(rw, fill.MIN_RENDER_SIDE)
        self.assertGreaterEqual(rh, fill.MIN_RENDER_SIDE)
        self.assertLessEqual(rw * rh, fill.FILL_RENDER_MAX_PIXELS)


class ContextPaddingTests(unittest.TestCase):
    """Padding is the adherence/context trade-off, and 0.6 leaked content.

    At 0.6 the model was shown the selection plus 60% extra on every side, so a pair
    of sunglasses painted across the eyes was drawn considerably wider than the
    selection and the composite clipped it at the mask edge -- a hat ending in a
    straight line, lenses cut off. Tighter crop = better adherence; Poisson blending
    now compensates for the missing context at the seam.
    """

    def test_padding_is_tight_enough_for_the_model_to_obey_the_selection(self):
        self.assertLessEqual(
            fill.FILL_CONTEXT_PAD_RATIO, 0.35,
            "a wide pad gives the model room to draw outside the selection, which the "
            "composite then clips into a visible hard edge",
        )

    def test_padding_is_not_zero(self):
        """Some context is still needed or the model has nothing to blend against."""
        self.assertGreater(fill.FILL_CONTEXT_PAD_RATIO, 0.0)
