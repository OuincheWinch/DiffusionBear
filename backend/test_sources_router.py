"""Tests for the /api/sources router.

Three of these are Review Focus items:
  RF1  civitai-match never attaches anything by itself -- it returns candidates.
  RF3  a conversion's skipped_tensors is reported, never hidden.
  RF5  a conversion refuses to overwrite an existing output.

The Draw Things folder is read-only and a traversal in `filename` must be refused
(Global Constraint 3 / Review Focus 5).
"""

import contextlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient

import generator
import main
import routers.sources as sources_router
from lora_convert import DRAW_THINGS_MODEL_DIR


# Minimal valid tensor-name sets. The converter needs real Draw Things names and
# real (if tiny) payloads; these exercise the endpoint's routing, validation and
# skip accounting without a 300 MB file.
def _mk_lora(prefix, layer=0):
    return [f"{prefix}[t-{layer}-{p}]__{w}__"
            for w in ("up", "down") for p in ("0",)]


_LORA_TENSORS = {"a.ckpt": _mk_lora("__unet__", 25)}
_MODEL_TENSORS = {"m.ckpt": ["__unet__[t-2-0]", "__unet__[t-2-1]"]}
# A FLUX LoRA: two x_q/x_v single-block-style slots plus the three norm2 chunks
# that have no target and are therefore skipped.
_FLUX_TENSORS = {"f.ckpt": [
    "__dit__[t-x_q-0-0]__up__", "__dit__[t-x_q-0-0]__down__",
    "__dit__[t-x_v-0-0]__up__", "__dit__[t-x_v-0-0]__down__",
    "__dit__[t-x_ada_ln_3-0-0]__up__", "__dit__[t-x_ada_ln_3-0-0]__down__",
    "__dit__[t-x_ada_ln_4-0-0]__up__", "__dit__[t-x_ada_ln_4-0-0]__down__",
    "__dit__[t-x_ada_ln_5-0-0]__up__", "__dit__[t-x_ada_ln_5-0-0]__down__",
]}


@contextlib.contextmanager
def _temp_asset_dir():
    """Point the converter's output at a fresh temp asset directory."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "lora_files").mkdir(parents=True)
    try:
        with mock.patch.object(generator, "ASSET_DIR", tmp), \
             mock.patch.object(sources_router.generator, "ASSET_DIR", tmp):
            yield tmp / "lora_files"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def _with_dt_dir(self, tensor_map):
        """Point the router at a temp Draw Things folder of synthetic .ckpt files.

        Every test here runs against a fixture, never the user's real folder.
        `test_model_download_path` permanently patches `Path.home()` to a temp dir,
        so any test that reads the real folder mid-suite sees nothing there and
        fails for reasons that have nothing to do with what it is testing.
        """
        import sqlite3
        root = Path(tempfile.mkdtemp())
        for name, tensors in tensor_map.items():
            db = sqlite3.connect(str(root / name))
            db.execute("CREATE TABLE tensors (name TEXT)")
            db.executemany("INSERT INTO tensors VALUES (?)", [(t,) for t in tensors])
            db.commit()
            db.close()
        patcher = mock.patch.object(sources_router, "_dt_dir", lambda: root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, root, True)
        return root

    def test_convert_rejects_a_traversal_filename(self):
        # Review Focus 5 / Global Constraint 3
        for bad in ("../../etc/passwd", "/etc/passwd", "sub/evil.ckpt"):
            with self.subTest(filename=bad):
                r = self.client.post("/api/sources/drawthings/convert",
                                     json={"filename": bad, "base_model": "sdxl"})
                self.assertIn(r.status_code, (400, 404), bad)

    def test_convert_rejects_an_unknown_base_model(self):
        r = self.client.post("/api/sources/drawthings/convert",
                             json={"filename": "a.ckpt", "base_model": "wan"})
        self.assertEqual(r.status_code, 422)

    def test_convert_refuses_to_overwrite_an_existing_output(self):
        # Review Focus 5
        self._with_dt_dir(_LORA_TENSORS)
        with _temp_asset_dir() as out_dir:
            target = out_dir / "a.safetensors"
            target.write_bytes(b"existing")
            r = self.client.post("/api/sources/drawthings/convert",
                                 json={"filename": "a.ckpt", "base_model": "sdxl"})
            self.assertEqual(r.status_code, 409)
            self.assertEqual(target.read_bytes(), b"existing")

    def test_convert_refuses_a_mismatched_base_model(self):
        # Asking for flux on an SDXL LoRA must be refused, not forced through.
        self._with_dt_dir(_LORA_TENSORS)
        with _temp_asset_dir():
            r = self.client.post("/api/sources/drawthings/convert",
                                 json={"filename": "a.ckpt", "base_model": "flux"})
            self.assertEqual(r.status_code, 400)
            self.assertIn("sdxl", r.json()["detail"])

    def test_convert_passes_the_output_dir_and_reports_the_summary(self):
        # The endpoint owns routing and skip accounting; the converter itself is
        # exercised against real files by Task 13. Patch it so this test asserts
        # what the endpoint adds, not that a 300 MB file converts.
        self._with_dt_dir(_LORA_TENSORS)
        seen = {}
        def fake_convert(src, out, base, alpha=None, allow_root=None):
            seen.update(src=src, out=out, base=base, allow_root=allow_root)
            Path(out).write_bytes(b"x")
            return {"pairs": 1, "tensors": 2, "bytes": 1, "skipped_tensors": 3}
        with _temp_asset_dir(), \
             mock.patch.object(sources_router.lora_convert, "convert", fake_convert):
            r = self.client.post("/api/sources/drawthings/convert",
                                 json={"filename": "a.ckpt", "base_model": "sdxl"})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["skipped_tensors"], 3)   # Review Focus 3
        self.assertEqual(body["base_model"], "sdxl")
        self.assertTrue(body["output_path"].endswith("a.safetensors"))
        self.assertTrue(str(seen["out"]).endswith("a.safetensors"))

    def test_convert_turns_a_conversion_error_into_a_400(self):
        self._with_dt_dir(_LORA_TENSORS)
        def boom(*a, **k):
            raise sources_router.lora_convert.ConversionError("no destination for x")
        with _temp_asset_dir(), \
             mock.patch.object(sources_router.lora_convert, "convert", boom):
            r = self.client.post("/api/sources/drawthings/convert",
                                 json={"filename": "a.ckpt", "base_model": "sdxl"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("no destination", r.json()["detail"])

    def test_files_endpoint_lists_only_ckpt_and_never_writes(self):
        import hashlib
        root = self._with_dt_dir({**_LORA_TENSORS, **_FLUX_TENSORS, "notes.txt": ["x"]})
        def fp():
            h = hashlib.sha256()
            for p in sorted(root.glob("*")):
                st = p.stat()
                h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}".encode())
            return h.hexdigest()
        before = fp()
        r = self.client.get("/api/sources/drawthings/files")
        self.assertEqual(r.status_code, 200)
        self.assertEqual({i["name"] for i in r.json()["items"]}, {"a.ckpt", "f.ckpt"})
        self.assertEqual(before, fp())

    def test_files_endpoint_separates_loras_from_models(self):
        self._with_dt_dir({**_LORA_TENSORS, **_MODEL_TENSORS})
        items = {i["name"]: i for i in self.client.get(
            "/api/sources/drawthings/files").json()["items"]}
        self.assertEqual(items["a.ckpt"]["kind"], "lora")
        self.assertEqual(items["m.ckpt"]["kind"], "model")

    def test_civitai_match_returns_candidates_and_attaches_nothing(self):
        # Review Focus 1: a match is never auto-attached.
        r = self.client.post("/api/sources/drawthings/civitai-match",
                             json={"query": "add_detail_xl"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("candidates", body)
        self.assertFalse(body["attached"])
        for candidate in body["candidates"]:
            self.assertIn("source_url", candidate)
            self.assertIn("license", candidate)

    def test_civitai_match_accepts_a_pasted_version_url(self):
        r = self.client.post("/api/sources/drawthings/civitai-match",
                             json={"version_url":
                                   "https://civitai.com/models/2890476?modelVersionId=3306914"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("candidates", r.json())

    def test_civitai_match_requires_a_query_or_url(self):
        r = self.client.post("/api/sources/drawthings/civitai-match", json={})
        self.assertEqual(r.status_code, 400)


class DiffusionBeeRouterTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def _with_db_dir(self, files_dict):
        from test_sources_diffusionbee import create_synthetic_tdict
        root = Path(tempfile.mkdtemp())
        for name, keys in files_dict.items():
            create_synthetic_tdict(root / name, keys)
        patcher = mock.patch.object(sources_router, "_db_dirs", lambda: [root])
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, root, True)
        return root

    def test_diffusionbee_files_endpoint(self):
        self._with_db_dir({
            "a_lora.tdict": {"lora_unet.weight_lora_up": {"shape": [4, 4], "dtype": "float32"}},
            "m_model.tdict": {"model.diffusion_model.input_blocks.0.0.weight": {"shape": [4, 4], "dtype": "float32"}},
        })
        res = self.client.get("/api/sources/diffusionbee/files")
        self.assertEqual(res.status_code, 200)
        items = {i["name"]: i for i in res.json()["items"]}
        self.assertEqual(items["a_lora.tdict"]["kind"], "lora")
        self.assertEqual(items["m_model.tdict"]["kind"], "model")

    def test_diffusionbee_convert_rejects_traversal(self):
        for bad in ("../../etc/passwd", "/etc/passwd", "sub/evil.tdict"):
            with self.subTest(filename=bad):
                r = self.client.post("/api/sources/diffusionbee/convert",
                                     json={"filename": bad, "base_model": "sdxl"})
                self.assertIn(r.status_code, (400, 404), bad)

    def test_diffusionbee_convert_refuses_to_overwrite_existing(self):
        self._with_db_dir({
            "lora.tdict": {
                "lora_unet_down.weight_lora_up": {"shape": [4, 4], "dtype": "float32"},
                "lora_unet_down.weight_lora_down": {"shape": [4, 4], "dtype": "float32"},
            }
        })
        with _temp_asset_dir() as out_dir:
            target = out_dir / "lora.safetensors"
            target.write_bytes(b"existing")
            r = self.client.post("/api/sources/diffusionbee/convert",
                                 json={"filename": "lora.tdict", "base_model": "sdxl"})
            self.assertEqual(r.status_code, 409)
            self.assertEqual(target.read_bytes(), b"existing")

    def test_diffusionbee_convert_refuses_model(self):
        self._with_db_dir({
            "model.tdict": {
                "model.diffusion_model.input_blocks.0.0.weight": {"shape": [4, 4], "dtype": "float32"},
            }
        })
        with _temp_asset_dir():
            r = self.client.post("/api/sources/diffusionbee/convert",
                                 json={"filename": "model.tdict", "base_model": "sdxl"})
            self.assertEqual(r.status_code, 400)
            self.assertIn("not a LoRA", r.json()["detail"])

    def test_diffusionbee_convert_refuses_mismatched_base_model(self):
        self._with_db_dir({
            "sdxl_lora.tdict": {
                "lora_te2_text_model_encoder_layers_0.weight_lora_up": {"shape": [4, 4], "dtype": "float32"},
            }
        })
        with _temp_asset_dir():
            r = self.client.post("/api/sources/diffusionbee/convert",
                                 json={"filename": "sdxl_lora.tdict", "base_model": "flux"})
            self.assertEqual(r.status_code, 400)
            self.assertIn("sdxl", r.json()["detail"])

    def test_diffusionbee_convert_succeeds_and_reports_summary(self):
        self._with_db_dir({
            "sample_lora.tdict": {
                "lora_unet_down.weight_lora_up": {"shape": [4, 4], "dtype": "float32"},
                "lora_unet_down.weight_lora_down": {"shape": [4, 4], "dtype": "float32"},
                "unmapped": {"shape": [4], "dtype": "float32"},
            }
        })
        with _temp_asset_dir():
            r = self.client.post("/api/sources/diffusionbee/convert",
                                 json={"filename": "sample_lora.tdict", "base_model": "sdxl"})
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body["skipped_tensors"], 1)
            self.assertEqual(body["base_model"], "sdxl")
            self.assertTrue(body["output_path"].endswith("sample_lora.safetensors"))

    def test_diffusionbee_civitai_match(self):
        r = self.client.post("/api/sources/diffusionbee/civitai-match",
                             json={"query": "test"})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["attached"])


if __name__ == "__main__":
    unittest.main()

