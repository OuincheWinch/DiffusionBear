"""Tests for SDXL CLIP Skip parameter support in schema, generator dispatch, metadata, and engine prompt caching."""

import json
import unittest
from pathlib import Path
from unittest import mock

import image_meta
import state
from state import GenerateRequest


class TestSDXLClipSkipSchema(unittest.TestCase):
    def test_default_clip_skip_is_two(self):
        req = GenerateRequest(prompt="a lovely cat")
        self.assertEqual(req.clip_skip, 2)

    def test_custom_clip_skip_valid(self):
        for val in (1, 2, 3, 4, 11, 12):
            req = GenerateRequest(prompt="a lovely cat", clip_skip=val)
            self.assertEqual(req.clip_skip, val)

    def test_clip_skip_out_of_bounds_rejected(self):
        with self.assertRaises(Exception):
            GenerateRequest(prompt="a lovely cat", clip_skip=0)
        with self.assertRaises(Exception):
            GenerateRequest(prompt="a lovely cat", clip_skip=13)
        with self.assertRaises(Exception):
            GenerateRequest(prompt="a lovely cat", clip_skip=-1)


class TestSDXLClipSkipMetadata(unittest.TestCase):
    def test_image_meta_uses_custom_clip_skip(self):
        meta = {
            "prompt": "portrait of an astronaut",
            "model": "juggernaut-xl-lightning",
            "engine": "sdxl",
            "clip_skip": 1,
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "seed": 42,
        }
        params_text = image_meta.build_generation_metadata_text(meta)
        self.assertIn("Clip skip: 1", params_text)

    def test_image_meta_defaults_sdxl_to_two_when_unspecified(self):
        meta = {
            "prompt": "portrait of an astronaut",
            "model": "juggernaut-xl-lightning",
            "engine": "sdxl",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "seed": 42,
        }
        params_text = image_meta.build_generation_metadata_text(meta)
        self.assertIn("Clip skip: 2", params_text)


class TestSDXLEnginePromptCache(unittest.TestCase):
    def test_sdxl_prompt_cache_differentiates_by_clip_skip(self):
        import sdxl_engine

        sdxl_engine._prompt_cache.clear()
        
        class DummyPipe:
            def __init__(self):
                self._clip_skip = 2
            def encode_prompt(self, prompt):
                return ("fake_embeds_cs2", "fake_pooled")

        fake_pipe = DummyPipe()
        sdxl_engine._patch_encode_prompt(fake_pipe)

        # Mock mx.eval so synthetic string embeddings do not raise
        with mock.patch("sdxl_engine.mx.eval"):
            # Call with default clip_skip=2
            fake_pipe._clip_skip = 2
            pe1, pooled1 = fake_pipe.encode_prompt("hello world")
            self.assertIn(("hello world", 2), sdxl_engine._prompt_cache)

            # Mock _encode_prompt_mlx for non-default clip_skip=1
            with mock.patch("sdxl_engine._encode_prompt_mlx", return_value=("fake_embeds_cs1", "fake_pooled")):
                fake_pipe._clip_skip = 1
                pe2, pooled2 = fake_pipe.encode_prompt("hello world")
                self.assertIn(("hello world", 1), sdxl_engine._prompt_cache)
                self.assertEqual(pe2, "fake_embeds_cs1")

            self.assertNotEqual(pe1, pe2)


class TestSDXLGeneratorClipSkip(unittest.TestCase):
    def test_generate_sdxl_signature_and_defaults(self):
        import inspect
        import generator

        sig_gen = inspect.signature(generator.generate)
        self.assertIn("clip_skip", sig_gen.parameters)
        self.assertEqual(sig_gen.parameters["clip_skip"].default, 2)

        sig_sdxl = inspect.signature(generator._generate_sdxl)
        self.assertIn("clip_skip", sig_sdxl.parameters)
        self.assertEqual(sig_sdxl.parameters["clip_skip"].default, 2)


if __name__ == "__main__":
    unittest.main()
