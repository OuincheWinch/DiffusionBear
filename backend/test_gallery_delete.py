"""Tests for the image deletion endpoint (/api/images/{image_id})."""

import shutil
import tempfile
import unittest
from pathlib import Path
from fastapi import HTTPException

from routers import gallery
from routers.gallery import delete_image, _IMAGE_MUTATION_LOCK, _image_is_deleted


class TestGalleryDelete(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.orig_generated_dir = gallery.GENERATED_DIR
        gallery.GENERATED_DIR = self.tmp
        with gallery._gallery_lock:
            self.orig_gallery_index = dict(gallery.GALLERY_INDEX)

    def tearDown(self):
        gallery.GENERATED_DIR = self.orig_generated_dir
        with gallery._gallery_lock:
            gallery.GALLERY_INDEX = self.orig_gallery_index
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_delete_removes_png_json_and_thumbnail(self):
        img_id = "test_img_abcdef123456"
        png_file = self.tmp / f"{img_id}.png"
        json_file = self.tmp / f"{img_id}.json"
        thumb_file = self.tmp / f"{img_id}_thumb.png"

        png_file.write_bytes(b"dummy png content")
        json_file.write_text('{"id": "test_img_abcdef123456", "prompt": "a test"}')
        thumb_file.write_bytes(b"dummy thumb content")

        with gallery._gallery_lock:
            gallery.GALLERY_INDEX[img_id] = {"id": img_id, "prompt": "a test"}

        self.assertTrue(png_file.exists())
        self.assertTrue(json_file.exists())
        self.assertTrue(thumb_file.exists())

        res = delete_image(img_id)
        self.assertEqual(res, {"deleted": img_id})

        # Check all files unlinked
        self.assertFalse(png_file.exists())
        self.assertFalse(json_file.exists())
        self.assertFalse(thumb_file.exists())

        # Check in-memory index cleared
        with gallery._gallery_lock:
            self.assertNotIn(img_id, gallery.GALLERY_INDEX)

        # Check marked as deleted
        self.assertTrue(_image_is_deleted(img_id))

    def test_delete_jpeg_image(self):
        img_id = "test_jpeg_abcdef123456"
        jpeg_file = self.tmp / f"{img_id}.jpeg"
        json_file = self.tmp / f"{img_id}.json"

        jpeg_file.write_bytes(b"dummy jpeg content")
        json_file.write_text('{"id": "test_jpeg_abcdef123456"}')

        res = delete_image(img_id)
        self.assertEqual(res, {"deleted": img_id})
        self.assertFalse(jpeg_file.exists())
        self.assertFalse(json_file.exists())

    def test_delete_non_existent_raises_404(self):
        with self.assertRaises(HTTPException) as ctx:
            delete_image("non_existent_image_id")
        self.assertEqual(ctx.exception.status_code, 404)

    def test_delete_invalid_id_raises_400(self):
        with self.assertRaises(HTTPException) as ctx:
            delete_image("../traversal_attempt")
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
