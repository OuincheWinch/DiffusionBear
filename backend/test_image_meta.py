import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from image_meta import atomic_write_json, build_generation_metadata_text, build_pnginfo, extract_image_metadata


class ImageMetaTests(unittest.TestCase):
    def test_atomic_write_replaces_complete_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            path.write_text('{"old":true}', encoding="utf-8")
            atomic_write_json(path, {"new": [1, 2]})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"new": [1, 2]})
            self.assertEqual(list(Path(tmp).glob(".metadata.json.tmp")), [])

    def test_atomic_write_preserves_old_json_when_serialization_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            path.write_text('{"old":true}', encoding="utf-8")
            with self.assertRaises(TypeError):
                atomic_write_json(path, {"bad": object()})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"old": True})
            self.assertEqual(list(Path(tmp).glob(".metadata.json.tmp")), [])

    def test_embedded_metadata_redacts_paths_and_private_fields(self):
        meta = {
            "id": "private-id",
            "model": "/Users/alice/private/flux-model",
            "prompt": "a lighthouse",
            "negative_prompt": "",
            "artist": "Test Artist",
            "file": "private-id.png",
            "tags": ["private-tag"],
            "reference_images": ["private-reference.png"],
            "stealth": False,
            "loras": [
                {
                    "name": "Public LoRA Name",
                    "path": "/Users/alice/private/private-lora.safetensors",
                    "scale": 0.8,
                    "modelVersionId": 12345,
                    "sha256": "1234567890abcdef",
                }
            ],
        }
        fake_generator = SimpleNamespace(get_model_info=lambda _name: {})
        with patch.dict(sys.modules, {"generator": fake_generator}):
            text = build_generation_metadata_text(meta)
        self.assertNotIn("/Users/alice", text)
        self.assertNotIn("private-reference.png", text)
        self.assertNotIn("private-tag", text)
        self.assertIn('"modelVersionId":12345', text)
        self.assertIn("<lora:Public_LoRA_Name:0.8>", text)

        pnginfo = build_pnginfo(meta, text)
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2), "white").save(buffer, format="PNG", pnginfo=pnginfo)
        buffer.seek(0)
        with Image.open(buffer) as image:
            chunks = image.text
        self.assertNotIn("loras", chunks)
        self.assertNotIn("reference_images", chunks)
        self.assertNotIn("stealth", chunks)
        self.assertNotIn("file", chunks)
        self.assertNotIn("tags", chunks)
        self.assertNotIn("/Users/alice", "\n".join(chunks.values()))
        self.assertIn("parameters", chunks)
        buffer.seek(0)
        extracted = extract_image_metadata(buffer)
        self.assertEqual(extracted["loras"][0]["modelVersionId"], 12345)
        self.assertEqual(extracted["civitai_metadata"]["resources"][0]["modelVersionId"], 12345)


if __name__ == "__main__":
    unittest.main()
