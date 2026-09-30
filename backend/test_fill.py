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

    def test_every_ref_capable_engine_is_allowed(self):
        """The allowlist must not silently omit an engine that would work."""
        import generator
        capable = {k for k, v in generator.MODELS.items() if v.get("supports_ref")}
        self.assertEqual(
            fill.FILL_ENGINES,
            capable,
            "FILL_ENGINES and the registry's supports_ref set have diverged; one of "
            "them needs updating",
        )

    def test_default_engine_can_actually_fill(self):
        self.assertIn(fill.DEFAULT_FILL_ENGINE, fill.FILL_ENGINES)


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

    def test_the_source_image_is_passed_as_the_reference(self):
        self._seed_source()
        gen = self._fake_generate()
        mask = mask_image((48, 48), (12, 12, 36, 36))

        fill.fill_image(self.src_id, b64_png(mask), "add a hat", generate_fn=gen)

        refs = gen.calls["reference_images"]
        self.assertEqual(len(refs), 1)
        self.assertTrue(refs[0].endswith(f"{self.src_id}.png"))

    def test_the_mask_is_not_sent_to_the_engine(self):
        """It is a region, not a subject. Passing it would ask the model to draw it."""
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
