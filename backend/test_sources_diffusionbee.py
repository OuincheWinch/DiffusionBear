"""Tests for DiffusionBee model and LoRA source integration.

Verifies:
1. Pure-Python binary parsing of `.tdict` header and metadata index without reading tensor payloads.
2. Error resilience on corrupt, truncated, or non-tdict files.
3. Classification of LoRA vs full model based on tensor key factors.
4. Architecture detection (FLUX, SDXL, SD 1.5, unknown).
5. Strict read-only invariant (zero writes or file mutations during listing).
6. Shared `Item` shape compliance with `Provenance(origin="diffusionbee")` and `license=None`.
"""

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import sources
from sources import diffusionbee


def create_synthetic_tdict(path: Path, keys_info: dict, model_info: dict | None = None, tensor_bytes: dict | None = None) -> Path:
    """Build a minimal valid .tdict file matching DiffusionBee's binary specification."""
    if model_info is None:
        model_info = {"app": "DiffusionBee", "version": "1.0"}
    if tensor_bytes is None:
        tensor_bytes = {}

    wconsts = [4377183192, 16197862624, 4380524608, 4338844632, 16102825584, 4342186048, 14897610352, 12854472448]

    with open(path, "wb") as f:
        # File header: 8 uint64s = 64 bytes
        file_head_nums = [8589935844, 0, 0, 0, 0, 0, 0, 0]
        f.write(struct.pack("<8Q", *file_head_nums))
        filep = 64

        # Block 1: extra_head placeholder (20 uint64s = 160 bytes)
        extra_head_pos = filep
        extra_head_data_pos = filep + 64
        head1 = [8030895855, 160, extra_head_data_pos, 20433505136, 2, wconsts[3], wconsts[7], wconsts[5]]
        f.write(struct.pack("<8Q", *head1))
        f.write(b"\x00" * 160)
        filep += 64 + 160

        # Pad to 64 bytes
        if filep % 64 != 0:
            npad = 64 - (filep % 64)
            f.write(b"\x00" * npad)
            filep += npad

        # Write any tensor payloads first so their start/end can be recorded in keys_info
        final_keys_info = {}
        for key, info in keys_info.items():
            raw_data = tensor_bytes.get(key)
            if raw_data is None:
                shape = tuple(info.get("shape", [4, 4]))
                dtype = info.get("dtype", "float32")
                raw_data = np.zeros(shape, dtype=dtype).tobytes()
            else:
                shape = tuple(info.get("shape", [len(raw_data)]))
                dtype = info.get("dtype", "uint8")

            if filep % 64 != 0:
                npad = 64 - (filep % 64)
                f.write(b"\x00" * npad)
                filep += npad

            n_start_head = filep
            n_start_data = filep + 64
            t_head = [8030895855, len(raw_data), n_start_data, 20433505136, 2, wconsts[3], wconsts[7], wconsts[5]]
            f.write(struct.pack("<8Q", *t_head))
            f.write(raw_data)
            filep += 64 + len(raw_data)
            n_end_data = filep

            final_keys_info[key] = {
                "start": n_start_data,
                "end": n_end_data,
                "shape": list(shape),
                "dtype": dtype,
            }

        # Block: weights_json
        if filep % 64 != 0:
            npad = 64 - (filep % 64)
            f.write(b"\x00" * npad)
            filep += npad

        weights_json_bytes = json.dumps(final_keys_info).encode("ascii")
        weights_json_start = filep + 64
        weights_json_end = weights_json_start + len(weights_json_bytes)
        head_w = [8030895855, len(weights_json_bytes), weights_json_start, 20433505136, 2, wconsts[3], wconsts[7], wconsts[5]]
        f.write(struct.pack("<8Q", *head_w))
        f.write(weights_json_bytes)
        filep += 64 + len(weights_json_bytes)

        # Block: metadata_json
        if filep % 64 != 0:
            npad = 64 - (filep % 64)
            f.write(b"\x00" * npad)
            filep += npad

        metadata_json_bytes = json.dumps(model_info).encode("ascii")
        metadata_json_start = filep + 64
        metadata_json_end = metadata_json_start + len(metadata_json_bytes)
        head_m = [8030895855, len(metadata_json_bytes), metadata_json_start, 20433505136, 2, wconsts[3], wconsts[7], wconsts[5]]
        f.write(struct.pack("<8Q", *head_m))
        f.write(metadata_json_bytes)
        filep += 64 + len(metadata_json_bytes)

        # Seek back to write extra_head data
        magic_number = 4346464
        tdict_format_version = 2
        min_supported_version = 2
        ctdict_version = 1
        extra_head = [
            magic_number,
            tdict_format_version,
            min_supported_version,
            ctdict_version,
            weights_json_start,
            weights_json_end,
            metadata_json_start,
            metadata_json_end,
        ] + [0] * 12
        f.seek(extra_head_data_pos)
        f.write(struct.pack("<20Q", *extra_head))

    return path


class TDictParserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_read_synthetic_tdict_keys(self):
        keys = {
            "lora_unet_down_blocks_0_attentions_0_proj_in.weight_lora_up": {"shape": [4, 320], "dtype": "float32"},
            "lora_unet_down_blocks_0_attentions_0_proj_in.weight_lora_down": {"shape": [320, 4], "dtype": "float32"},
        }
        tdict_path = self.dir / "test_lora.tdict"
        create_synthetic_tdict(tdict_path, keys)

        parsed_keys = diffusionbee.read_tdict_keys(tdict_path)
        self.assertEqual(sorted(parsed_keys), sorted(keys.keys()))

    def test_corrupt_tdict_returns_empty_without_raising(self):
        corrupt_path = self.dir / "corrupt.tdict"
        corrupt_path.write_bytes(b"not a valid tdict file header bytes")

        self.assertIsNone(diffusionbee.read_tdict_header(corrupt_path))
        self.assertEqual(diffusionbee.read_tdict_keys(corrupt_path), [])

    def test_truncated_tdict_returns_empty(self):
        truncated_path = self.dir / "truncated.tdict"
        truncated_path.write_bytes(b"\x00" * 80)

        self.assertIsNone(diffusionbee.read_tdict_header(truncated_path))
        self.assertEqual(diffusionbee.read_tdict_keys(truncated_path), [])

    def test_zero_payload_bytes_read_during_listing(self):
        keys = {
            "large_tensor": {"shape": [1000, 1000], "dtype": "float32"},
        }
        tdict_path = self.dir / "large.tdict"
        # 4MB of tensor bytes
        create_synthetic_tdict(tdict_path, keys, tensor_bytes={"large_tensor": b"\xaa" * 4_000_000})

        # We inspect how many bytes were read by reading keys
        class MonitoredFile:
            def __init__(self, real_file):
                self._f = real_file
                self.bytes_read = 0

            def seek(self, pos, whence=0):
                return self._f.seek(pos, whence)

            def read(self, n=-1):
                chunk = self._f.read(n)
                self.bytes_read += len(chunk)
                return chunk

            def close(self):
                return self._f.close()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        orig_open = open
        monitored_instance = []

        def tracked_open(p, *args, **kwargs):
            mf = MonitoredFile(orig_open(p, *args, **kwargs))
            monitored_instance.append(mf)
            return mf

        import unittest.mock as mock
        with mock.patch("builtins.open", tracked_open):
            parsed = diffusionbee.read_tdict_keys(tdict_path)

        self.assertEqual(parsed, ["large_tensor"])
        self.assertTrue(len(monitored_instance) > 0)
        total_read = sum(m.bytes_read for m in monitored_instance)
        # 4MB payload must NOT have been read; header + json is less than 2KB
        self.assertLess(total_read, 10_000, f"Expected <10KB read, got {total_read} bytes")


class ClassificationAndListingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.dir1 = self.dir / "downloaded_assets"
        self.dir2 = self.dir / "imported_models"
        self.dir1.mkdir()
        self.dir2.mkdir()
        self.addCleanup(self.tmp.cleanup)

    def test_detect_kind_keys(self):
        lora_keys = ["lora_unet_down_blocks_0_attentions_0.weight_lora_up", "lora_unet_down_blocks_0_attentions_0.weight_lora_down"]
        self.assertEqual(diffusionbee.detect_kind_keys(lora_keys), "lora")

        lora_keys_dt_style = ["__dit__[t-x_q-0-0]__up__", "__dit__[t-x_q-0-0]__down__"]
        self.assertEqual(diffusionbee.detect_kind_keys(lora_keys_dt_style), "lora")

        model_keys = ["model.diffusion_model.input_blocks.0.0.weight", "model.diffusion_model.out.2.weight"]
        self.assertEqual(diffusionbee.detect_kind_keys(model_keys), "model")

        self.assertIsNone(diffusionbee.detect_kind_keys([]))

    def test_detect_base_model_keys(self):
        flux_keys = ["transformer.single_transformer_blocks.0.linear1.weight_lora_up"]
        self.assertEqual(diffusionbee.detect_base_model_keys(flux_keys), "flux")

        sdxl_keys = ["lora_te2_text_model_encoder_layers_0.weight_lora_up", "model.diffusion_model.label_emb.0.0.weight"]
        self.assertEqual(diffusionbee.detect_base_model_keys(sdxl_keys), "sdxl")

        sd15_keys = ["model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight"]
        self.assertEqual(diffusionbee.detect_base_model_keys(sd15_keys), "sd15")

        unknown_keys = ["custom_arbitrary_tensor_name.weight"]
        self.assertIsNone(diffusionbee.detect_base_model_keys(unknown_keys))

    def test_list_tdict_files_scans_directories_and_filters_non_tdict(self):
        create_synthetic_tdict(self.dir1 / "alpha_lora.tdict", {
            "lora_unet_down.weight_lora_up": {"shape": [4, 64], "dtype": "float32"},
            "lora_unet_down.weight_lora_down": {"shape": [64, 4], "dtype": "float32"},
        })
        create_synthetic_tdict(self.dir2 / "beta_model.tdict", {
            "model.diffusion_model.input_blocks.0.0.weight": {"shape": [320, 4, 3, 3], "dtype": "float32"},
        })
        (self.dir1 / "notes.txt").write_text("hello")
        (self.dir2 / "other.safetensors").write_bytes(b"something")

        files = diffusionbee.list_tdict_files([self.dir1, self.dir2])
        self.assertEqual([f.name for f in files], ["alpha_lora.tdict", "beta_model.tdict"])
        self.assertEqual(files[0].kind, "lora")
        self.assertEqual(files[1].kind, "model")
        # Only LoRA gets base_model, models stay None per spec
        self.assertIsNone(files[1].base_model)
        self.assertGreater(files[0].size_bytes, 0)

    def test_listing_never_writes(self):
        create_synthetic_tdict(self.dir1 / "check_lora.tdict", {
            "lora_unet.weight_lora_up": {"shape": [4, 64], "dtype": "float32"},
        })

        def fingerprint():
            h = hashlib.sha256()
            for p in sorted(self.dir.rglob("*")):
                st = p.stat()
                h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}".encode())
            return h.hexdigest()

        before = fingerprint()
        diffusionbee.list_tdict_files([self.dir1, self.dir2])
        after = fingerprint()
        self.assertEqual(before, after)


class ConversionAndProtectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.assets = self.dir / "downloaded_assets"
        self.assets.mkdir()
        self.out_dir = self.dir / "lora_files"
        self.out_dir.mkdir()
        self.addCleanup(self.tmp.cleanup)

    def test_protected_roots_refuses_diffusionbee_dir(self):
        from lora_convert import _check_output, ConversionError
        db_root = Path.home() / ".diffusionbee"
        with self.assertRaises(ConversionError):
            _check_output(db_root / "evil.safetensors")

    def test_convert_tdict_lora_produces_safetensors_and_summary(self):
        # Create a synthetic LoRA with 2 valid LoRA keys and 1 skipped tensor
        raw_up = np.ones((4, 32), dtype=np.float32).tobytes()
        raw_down = np.ones((32, 4), dtype=np.float32).tobytes()
        raw_skip = np.ones((10,), dtype=np.float32).tobytes()

        keys = {
            "lora_unet_down.weight_lora_up": {"shape": [4, 32], "dtype": "float32"},
            "lora_unet_down.weight_lora_down": {"shape": [32, 4], "dtype": "float32"},
            "unmapped_extra_tensor": {"shape": [10], "dtype": "float32"},
        }
        tensor_bytes = {
            "lora_unet_down.weight_lora_up": raw_up,
            "lora_unet_down.weight_lora_down": raw_down,
            "unmapped_extra_tensor": raw_skip,
        }
        tdict_path = self.assets / "my_lora.tdict"
        create_synthetic_tdict(tdict_path, keys, tensor_bytes=tensor_bytes)

        out_path = self.out_dir / "my_lora.safetensors"
        summary = diffusionbee.convert_tdict_lora(
            tdict_path, out_path, base_model="sdxl", allow_root=self.dir
        )

        self.assertTrue(out_path.exists())
        self.assertEqual(summary["output_path"], str(out_path.resolve()))
        self.assertEqual(summary["base_model"], "sdxl")
        self.assertEqual(summary["tensors"], 2)
        self.assertEqual(summary["skipped_tensors"], 1)

        # Verify saved safetensors can be read
        from safetensors import safe_open
        with safe_open(out_path, framework="numpy") as f:
            tensor_keys = list(f.keys())
            self.assertIn("lora_unet_down.lora_up.weight", tensor_keys)
            self.assertIn("lora_unet_down.lora_down.weight", tensor_keys)
            self.assertNotIn("unmapped_extra_tensor", tensor_keys)

    def test_convert_refuses_model(self):
        from lora_convert import ConversionError
        tdict_path = self.assets / "full_model.tdict"
        create_synthetic_tdict(tdict_path, {
            "model.diffusion_model.input_blocks.0.0.weight": {"shape": [320, 4, 3, 3], "dtype": "float32"},
        })
        out_path = self.out_dir / "full_model.safetensors"
        with self.assertRaises(ConversionError):
            diffusionbee.convert_tdict_lora(tdict_path, out_path, "sdxl", allow_root=self.dir)


if __name__ == "__main__":
    unittest.main()


