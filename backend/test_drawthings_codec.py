"""Tests for the Draw Things tensor codec.

Fixtures here are synthetic: every `.ckpt` is written into a tempdir with the
real schema. No test may read from the real Draw Things model directory --
verification against those files is manual (Task 8), not part of the suite.
"""

import contextlib
import sqlite3
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import drawthings_codec as codec  # noqa: E402
from drawthings_codec import CodecError  # noqa: E402

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS tensors (name TEXT, type INTEGER, format INTEGER, "
    "datatype INTEGER, dim BLOB, data BLOB, PRIMARY KEY (name))"
)


def _dim_blob(shape):
    """Pack a shape into the 48-byte, 12 x int32 little-endian `dim` blob."""
    padded = list(shape) + [0] * (12 - len(shape))
    return struct.pack("<12i", *padded)


def pack_msb(values, bits):
    """Pack `values` as one MSB-first big-endian bitstream of `bits` width.

    Mirrors what the Draw Things writer does for q5p/q6p index payloads, and
    is the inverse of `codec.unpack_msb`.
    """
    stream = 0
    for v in values:
        stream = (stream << bits) | int(v)
    total = bits * len(values)
    stream <<= (-total) % 8  # pad to a whole number of bytes
    return stream.to_bytes((total + 7) // 8, "big")


@contextlib.contextmanager
def write_ckpt(tmpdir, tensors, external=False, codec_id=0, block_size=1024,
               fmt=1, nulls=()):
    """Write a minimal Draw Things `.ckpt` and yield its path.

    `tensors` maps name -> (shape, payload bytes). With `external=True` the
    payload moves into a sibling `-tensordata` sidecar and the row's `data`
    becomes the 16-byte (offset, length) pointer, matching the real writer.
    `codec_id` selects a real codec so routing can be exercised end to end.
    `fmt` sets the format column; `nulls` names columns to write as NULL.
    """
    path = Path(tmpdir) / "model.ckpt"
    sidecar = Path(str(path) + "-tensordata")

    blob = b""
    rows = []
    for name, (shape, payload) in tensors.items():
        full_id = codec_id | codec.CODEC_EXTERNAL if external else codec_id
        if external:
            offset = len(blob)
            blob += payload
            if codec_id:
                # Quantized sidecars use a 24-byte pointer that also carries
                # the block size: [u32 block][u32 magic][u64 offset][u64 length].
                data = struct.pack("<IIQQ", block_size, 0xDEADBEEF, offset, len(payload))
            else:
                data = struct.pack("<QQ", offset, len(payload))
        else:
            data = payload
        cols = {
            "name": name, "type": (full_id << 32) | 1, "format": fmt,
            "datatype": 131072, "dim": _dim_blob(shape), "data": data,
        }
        for col in nulls:
            cols[col] = None
        rows.append(tuple(cols[c] for c in
                          ("name", "type", "format", "datatype", "dim", "data")))

    with sqlite3.connect(path) as db:
        db.execute(SCHEMA)
        db.executemany("INSERT OR REPLACE INTO tensors VALUES (?,?,?,?,?,?)", rows)
    if external:
        sidecar.write_bytes(blob)

    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)


class StoreReaderTests(unittest.TestCase):
    """Pins the container schema: dim parsing, dtype mapping, sidecar location."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_read_shape_takes_leading_nonzero_int32s(self):
        blob = struct.pack("<12i", 768, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.assertEqual(codec.read_shape(blob), (768, 16))

    def test_read_shape_preserves_all_twelve_axes(self):
        # CCV_NNC_MAX_DIM_ALLOC is 12. A reader assuming <=4 axes drops the tail.
        dims = tuple(range(1, 13))
        self.assertEqual(codec.read_shape(struct.pack("<12i", *dims)), dims)

    def test_read_shape_rejects_wrong_sized_blob(self):
        with self.assertRaises(CodecError):
            codec.read_shape(struct.pack("<4i", 2, 2, 0, 0))

    def test_dtype_code_maps_f16_and_f32(self):
        self.assertEqual(codec._dtype_code(131072), 2)  # CCV_16F
        self.assertEqual(codec._dtype_code(262144), 4)  # CCV_32F

    def test_dtype_code_rejects_unknown(self):
        with self.assertRaises(CodecError):
            codec._dtype_code(999)

    def test_read_returns_f16_array_for_raw_tensor(self):
        raw = np.arange(32, dtype=np.float16).tobytes()
        with write_ckpt(self.temp.name, {"t": ([4, 8], raw)}) as p:
            with codec.Store.open(p) as s:
                shape, arr = s.read("t")
        self.assertEqual(shape, (4, 8))
        self.assertEqual(arr.dtype, np.float16)
        np.testing.assert_array_equal(arr, np.arange(32, dtype=np.float16).reshape(4, 8))

    def test_read_resolves_external_sidecar(self):
        raw = np.arange(32, dtype=np.float16).tobytes()
        with write_ckpt(self.temp.name, {"t": ([4, 8], raw)}, external=True) as p:
            with codec.Store.open(p) as s:
                shape, arr = s.read("t")
        self.assertEqual(shape, (4, 8))
        np.testing.assert_array_equal(arr, np.arange(32, dtype=np.float16).reshape(4, 8))

    def test_missing_sidecar_raises_codec_error(self):
        """A row may promise an external blob the file cannot satisfy."""
        raw = np.arange(32, dtype=np.float16).tobytes()
        with write_ckpt(self.temp.name, {"t": ([4, 8], raw)}, external=True) as p:
            Path(str(p) + "-tensordata").unlink()
            with codec.Store.open(p) as s:
                with self.assertRaises(CodecError) as cm:
                    s.read("t")
        self.assertIn("tensordata", str(cm.exception))

    def test_unknown_tensor_name_raises_codec_error(self):
        with write_ckpt(self.temp.name, {"t": ([2, 2], b"\x00" * 8)}) as p:
            with codec.Store.open(p) as s:
                with self.assertRaises(CodecError):
                    s.read("absent")

    def test_names_lists_stored_tensors(self):
        with write_ckpt(self.temp.name, {"a": ([2, 2], b"\x00" * 8)}) as p:
            with codec.Store.open(p) as s:
                self.assertEqual(s.names(), ["a"])

    def test_open_is_read_only(self):
        """mode=ro is what enforces the read-only constraint on real model files."""
        with write_ckpt(self.temp.name, {"t": ([2, 2], b"\x00" * 8)}) as p:
            s = codec.Store.open(p)
            with self.assertRaises(sqlite3.OperationalError):
                s._db.execute("DELETE FROM tensors")
            s.close()


class UnpackTests(unittest.TestCase):
    """Pins the contiguous MSB-first bitstream used by q5p and q6p indices."""

    def test_six_bit_packing_matches_swift_formula(self):
        # Swift: b0=(i0<<2)|(i1>>4); b1=(i1<<4)|(i2>>2); b2=(i2<<6)|i3
        data = bytes([
            (0 << 2) | (63 >> 4),
            ((63 << 4) & 0xFF) | (31 >> 2),
            ((31 << 6) & 0xFF) | 17,
        ])
        np.testing.assert_array_equal(codec.unpack_msb(data, 6, 4), [0, 63, 31, 17])

    def test_five_bit_packing_eight_per_five_bytes(self):
        # 8 x 5 bits = 40 bits = exactly 5 bytes.
        vals = [0, 1, 7, 8, 15, 16, 30, 31]
        packed = pack_msb(vals, 5)
        self.assertEqual(len(packed), 5)
        np.testing.assert_array_equal(codec.unpack_msb(packed, 5, 8), vals)

    def test_pack_unpack_round_trips_every_width(self):
        rng = np.random.default_rng(7)
        for bits in (1, 5, 6, 8):
            limit = 1 << bits
            vals = rng.integers(0, limit, 200, dtype=np.uint8)
            got = codec.unpack_msb(pack_msb(vals, bits), bits, len(vals))
            np.testing.assert_array_equal(got, vals, err_msg=f"width {bits}")

    def test_eight_bit_is_identity(self):
        data = bytes(range(32))
        np.testing.assert_array_equal(codec.unpack_msb(data, 8, 32), list(range(32)))

    def test_single_bit_stream(self):
        vals = [1, 0, 1, 1, 0, 0, 1, 1]
        packed = bytes([0b10110011])
        np.testing.assert_array_equal(codec.unpack_msb(packed, 1, 8), vals)

    def test_short_buffer_raises_codec_error(self):
        # Two bytes cannot hold 4 six-bit values (needs 3).
        with self.assertRaises(CodecError):
            codec.unpack_msb(b"\x00\x00", 6, 4)

    def test_invalid_bit_width_raises_codec_error(self):
        for bad in (0, 9, -1):
            with self.assertRaises(CodecError):
                codec.unpack_msb(b"\x00" * 8, bad, 1)

    def test_excess_buffer_is_tolerated(self):
        # Readers hand over the whole block; trailing bits are not our concern.
        np.testing.assert_array_equal(codec.unpack_msb(b"\xff\xff\xff\xff", 6, 4), [63, 63, 63, 63])


def q6p_block(centroids, indices):
    """One headerless q6p block: 64 centroids then 6-bit packed indices."""
    assert centroids.size == 64
    return centroids.tobytes() + pack_msb(indices, 6)


class Q6PDecodeTests(unittest.TestCase):
    """Pins the q6p layout: u32 block size, 64 centroids, 6-bit indices."""

    def test_decodes_two_full_blocks(self):
        centroids = np.arange(64, dtype=np.float16)
        indices = np.tile(np.arange(64, dtype=np.uint8), 4096 // 64)
        payload = q6p_block( centroids, indices) * 2
        out = codec.decode_q6p(payload, 8192, 2, block_size=4096)
        self.assertEqual(out.shape, (8192,))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, np.tile(np.arange(64, dtype=np.float32), 128))

    def test_decodes_partial_final_block(self):
        centroids = np.arange(64, dtype=np.float16)
        indices = np.arange(32, dtype=np.uint8)
        payload = q6p_block( centroids, indices)
        out = codec.decode_q6p(payload, 32, 2, block_size=4096)
        np.testing.assert_allclose(out, np.arange(32, dtype=np.float32))

    def test_every_value_is_a_centroid(self):
        rng = np.random.default_rng(0)
        c = rng.standard_normal(64).astype(np.float16)
        idx = rng.integers(0, 64, 4096, dtype=np.uint8)
        out = codec.decode_q6p(q6p_block(c, idx), 4096, 2, block_size=4096)
        self.assertTrue(np.all(np.isin(out, c.astype(np.float32))))

    def test_reads_block_size_from_blob(self):
        # A non-default block size must be honoured, not assumed.
        centroids = np.arange(64, dtype=np.float16)
        indices = np.arange(64, dtype=np.uint8)
        payload = q6p_block( centroids, indices)
        self.assertEqual(codec.decode_q6p(payload, 64, 2, block_size=4096).shape, (64,))

    def test_truncated_block_raises_codec_error(self):
        """Review Focus #1: a payload cut short must refuse, not return short."""
        centroids = np.arange(64, dtype=np.float16)
        indices = np.arange(64, dtype=np.uint8)
        payload = q6p_block( centroids, indices)
        with self.assertRaises(CodecError):
            codec.decode_q6p(payload[:-10], numel=4096, elem_size=2)

    def test_max_index_is_accepted(self):
        # A 6-bit index cannot exceed 63, so a 64-entry codebook can never be
        # out of range by construction. The decoder's range guard is
        # defence-in-depth against a garbled payload, not a reachable branch.
        centroids = np.arange(64, dtype=np.float16)
        indices = np.full(64, 63, dtype=np.uint8)
        out = codec.decode_q6p(q6p_block(centroids, indices), 64, 2, block_size=4096)
        np.testing.assert_allclose(out, np.full(64, 63.0, dtype=np.float32))

    def test_zero_block_size_raises_codec_error(self):
        with self.assertRaises(CodecError):
            codec.decode_q6p(struct.pack("<I", 0), numel=8, elem_size=2)

    def test_truncated_centroid_table_raises_codec_error(self):
        # Not even room for the 64-entry table.
        with self.assertRaises(CodecError):
            codec.decode_q6p(struct.pack("<I", 4096) + b"\x00" * 16, numel=8, elem_size=2)


def q8p_block(centroids, indices):
    """One headerless q8p block: 256 centroids then one raw uint8 per element."""
    assert centroids.size == 256
    return centroids.tobytes() + np.asarray(indices, dtype=np.uint8).tobytes()


class Q8PDecodeTests(unittest.TestCase):
    """Pins q8p: 256 centroids, one raw byte per element, no bit packing."""

    def test_decodes_known_codebook(self):
        c = np.arange(256, dtype=np.float16)
        payload = q8p_block( c, np.arange(256, dtype=np.uint8))
        out = codec.decode_q8p(payload, 256, 2, block_size=16384)
        np.testing.assert_allclose(out, np.arange(256, dtype=np.float32))

    def test_reads_block_size_from_blob(self):
        # A non-default block size must be honoured, not assumed.
        c = np.arange(256, dtype=np.float16)
        payload = q8p_block( c, np.arange(256, dtype=np.uint8))
        self.assertEqual(codec.decode_q8p(payload, 256, 2, block_size=16384).shape, (256,))

    def test_decodes_two_full_blocks(self):
        c = np.arange(256, dtype=np.float16)
        idx = np.tile(np.arange(256, dtype=np.uint8), 64)
        payload = q8p_block( c, idx) * 2
        out = codec.decode_q8p(payload, 32768, 2, block_size=16384)
        np.testing.assert_allclose(out, np.tile(np.arange(256, dtype=np.float32), 128))

    def test_max_index_is_accepted(self):
        # An 8-bit index also cannot exceed a 256-entry table, so the decoder's
        # range guard is defence-in-depth here too.
        c = np.arange(256, dtype=np.float16)
        payload = q8p_block( c, np.full(16, 255, dtype=np.uint8))
        self.assertEqual(codec.decode_q8p(payload, 16, 2, block_size=16384).size, 16)

    def test_truncated_indices_raise_codec_error(self):
        c = np.arange(256, dtype=np.float16)
        payload = q8p_block( c, np.arange(4, dtype=np.uint8))
        with self.assertRaises(CodecError):
            codec.decode_q8p(payload, 4096, 2, block_size=16384)

    def test_truncated_centroid_table_raises_codec_error(self):
        with self.assertRaises(CodecError):
            codec.decode_q8p(struct.pack("<I", 16384) + b"\x00" * 16, numel=8, elem_size=2)

    def test_zero_block_size_raises_codec_error(self):
        with self.assertRaises(CodecError):
            codec.decode_q8p(struct.pack("<I", 0), numel=8, elem_size=2)


def q5p_block(centroids, indices):
    """One headerless q5p block: 32 centroids then 5-bit packed indices."""
    assert centroids.size == 32
    return centroids.tobytes() + pack_msb(indices, 5)


def inline(payload, block_size):
    """Prefix a headerless payload with the block size, for the inline form."""
    return struct.pack("<I", block_size) + payload


class Q5PDecodeTests(unittest.TestCase):
    """Pins q5p: 32 centroids, 5-bit packed, block size read from the blob."""

    def test_decodes_known_codebook(self):
        c = np.arange(32, dtype=np.float16)
        payload = q5p_block( c, np.arange(32, dtype=np.uint8))
        np.testing.assert_allclose(codec.decode_q5p(payload, 32, 2, block_size=2048), np.arange(32, dtype=np.float32))

    def test_decodes_two_full_blocks(self):
        c = np.arange(32, dtype=np.float16)
        idx = np.tile(np.arange(32, dtype=np.uint8), 2048 // 32)
        payload = q5p_block( c, idx) * 2
        out = codec.decode_q5p(payload, 4096, 2, block_size=2048)
        np.testing.assert_allclose(out, np.tile(np.arange(32, dtype=np.float32), 128))

    def test_decodes_partial_final_block(self):
        c = np.arange(32, dtype=np.float16)
        payload = q5p_block( c, np.arange(5, dtype=np.uint8))
        out = codec.decode_q5p(payload, 5, 2, block_size=2048)
        np.testing.assert_allclose(out, np.arange(5, dtype=np.float32))

    def test_every_value_is_a_centroid(self):
        rng = np.random.default_rng(3)
        c = rng.standard_normal(32).astype(np.float16)
        idx = rng.integers(0, 32, 2048, dtype=np.uint8)
        out = codec.decode_q5p(q5p_block(c, idx), 2048, 2, block_size=2048)
        self.assertTrue(np.all(np.isin(out, c.astype(np.float32))))

    def test_reads_block_size_from_blob(self):
        c = np.arange(32, dtype=np.float16)
        payload = q5p_block( c, np.arange(32, dtype=np.uint8))
        self.assertEqual(codec.decode_q5p(payload, 32, 2, block_size=2048).shape, (32,))

    def test_absurd_block_size_raises_codec_error(self):
        """Review Focus #3: block size comes from the blob and must be validated.

        The payload is otherwise complete, so the guard is what fires -- a
        short payload would raise for an unrelated reason and prove nothing.
        """
        c = np.arange(32, dtype=np.float16)
        idx = np.zeros(64, dtype=np.uint8)
        for bad in (0, 1):
            payload = inline(q5p_block(c, idx), bad)
            with self.assertRaises(CodecError):
                codec.decode_q5p(payload, 64, 2)

    def test_block_size_larger_than_numel_is_one_partial_block(self):
        # Legitimate: the final block is short, so block_size may exceed numel.
        c = np.arange(32, dtype=np.float16)
        payload = q5p_block( c, np.arange(5, dtype=np.uint8))
        self.assertEqual(codec.decode_q5p(payload, 5, 2, block_size=2048).shape, (5,))

    def test_truncated_block_raises_codec_error(self):
        c = np.arange(32, dtype=np.float16)
        payload = q5p_block( c, np.arange(32, dtype=np.uint8))
        with self.assertRaises(CodecError):
            codec.decode_q5p(payload[:-6], 2048, 2, block_size=2048)


def encode_ezm7_reference(values):
    """Mirror the Draw Things ezm7 encoder for tests.

    Transcribed from s4nnc/nnc/Store.swift `ezm7Encode` at revision
    0ecfd9b5: exponents are deflated with raw `inflateInit2_(-MAX_WBITS)`,
    and each mantissa is truncated to 7 bits with round-half-to-even, falling
    back to truncation when rounding would overflow into bit 7.
    """
    raw = np.asarray(values, dtype=np.float16).view(np.uint16)
    exponents = (raw >> 10) & 0x1F
    signs = raw >> 15
    mantissas = raw & 0x3FF

    truncated = (mantissas >> 3).astype(np.uint16)
    discard = mantissas & 7
    round_up = (discard > 4) | ((discard == 4) & ((truncated & 1) == 1))
    rounded = truncated + round_up.astype(np.uint16)
    truncated = np.where(rounded & 0x80, truncated, rounded).astype(np.uint8)

    floats_without_exp = ((signs << 7) | truncated).astype(np.uint8)
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    zipped = compressor.compress(exponents.astype(np.uint8).tobytes())
    zipped += compressor.flush()

    out = struct.pack("<i", len(zipped)) + zipped + floats_without_exp.tobytes()
    return out


class Ezm7DecodeTests(unittest.TestCase):
    """Pins ezm7: deflated 5-bit exponents plus 7-bit mantissas and a sign."""

    def test_round_trips_values_within_tolerance(self):
        src = np.array([1.0, -2.0, 0.5, 3.25, -0.125, 7.5], dtype=np.float16)
        out = codec.decode_ezm7(encode_ezm7_reference(src), numel=src.size)
        self.assertEqual(out.dtype, np.float16)
        self.assertEqual(out.shape, src.shape)
        np.testing.assert_allclose(out, src, rtol=0.02)

    def test_sign_bit_is_preserved_exactly(self):
        src = np.array([1.0, -1.0, 4.0, -4.0], dtype=np.float16)
        out = codec.decode_ezm7(encode_ezm7_reference(src), numel=src.size)
        np.testing.assert_array_equal(np.signbit(out), np.signbit(src))

    def test_exponent_is_exact_but_mantissa_is_lossy(self):
        # ezm7 keeps the 5-bit exponent and sign, drops 3 mantissa bits, so the
        # decoded value must share the exponent field exactly.
        src = np.array([1.0, 1.5, 1.75, 1.875], dtype=np.float16)
        out = codec.decode_ezm7(encode_ezm7_reference(src), numel=src.size)
        np.testing.assert_array_equal(
            (out.view(np.uint16) >> 10) & 0x1F, (src.view(np.uint16) >> 10) & 0x1F
        )

    def test_truncated_payload_raises_codec_error(self):
        with self.assertRaises(CodecError):
            codec.decode_ezm7(struct.pack("<i", 4) + b"\x00\x00", numel=8)

    def test_corrupt_deflate_stream_raises_codec_error(self):
        payload = struct.pack("<i", 4) + b"garbage" + b"\x00" * 8
        with self.assertRaises(CodecError):
            codec.decode_ezm7(payload, numel=8)

    def test_declared_size_overrunning_payload_raises_codec_error(self):
        with self.assertRaises(CodecError):
            codec.decode_ezm7(struct.pack("<i", 1 << 20) + b"\x00" * 8, numel=4)

    def test_negative_exponent_size_raises_codec_error(self):
        with self.assertRaises(CodecError):
            codec.decode_ezm7(struct.pack("<i", -4) + b"\x00" * 8, numel=4)


class DispatchTests(unittest.TestCase):
    """Pins codec-id routing and the refusal to guess."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_routes_q6p_id_to_the_q6p_decoder(self):
        c = np.arange(64, dtype=np.float16)
        payload = q6p_block( c, np.arange(64, dtype=np.uint8))
        out = codec.decode_tensor(payload, codec.CODEC_Q6P, (64,), 2, block_size=4096)
        np.testing.assert_allclose(out, np.arange(64, dtype=np.float32))

    def test_routes_q5p_and_q8p_ids(self):
        c5 = np.arange(32, dtype=np.float16)
        out5 = codec.decode_tensor(q5p_block(c5, np.arange(32, dtype=np.uint8)),
                                   codec.CODEC_Q5P, (32,), 2, block_size=2048)
        np.testing.assert_allclose(out5, np.arange(32, dtype=np.float32))
        c8 = np.arange(256, dtype=np.float16)
        out8 = codec.decode_tensor(q8p_block(c8, np.arange(256, dtype=np.uint8)),
                                   codec.CODEC_Q8P, (256,), 2, block_size=16384)
        np.testing.assert_allclose(out8, np.arange(256, dtype=np.float32))

    def test_zero_codec_id_is_raw(self):
        raw = np.arange(8, dtype=np.float16)
        out = codec.decode_tensor(raw.tobytes(), 0, (2, 4), 2)
        self.assertEqual(out.dtype, np.float16)
        np.testing.assert_array_equal(out, raw.reshape(2, 4))

    def test_external_bit_is_masked_before_dispatch(self):
        """Pins the real bit position, observed in flux_1_schnell_q5p.ckpt.

        The external flag is bit 28 (0x1000_0000), OR'd onto the real codec:
        0x10000000 alone for a raw tensor, 0x108a1e5b for q5p, 0x108a1e8b for
        q8p. A test that only used the symbolic constant would not catch a
        wrong bit, which is exactly what happened once.
        """
        self.assertEqual(codec.CODEC_EXTERNAL, 0x10000000)
        self.assertEqual(codec.CODEC_EXTERNAL & codec.CODEC_Q6P, 0)
        for observed, expected in ((0x10000000, 0), (0x108A1E5B, codec.CODEC_Q5P),
                                   (0x108A1E8B, codec.CODEC_Q8P),
                                   (0x108A1E6B, codec.CODEC_Q6P)):
            masked = observed & ~codec.CODEC_EXTERNAL
            self.assertEqual(masked, expected, f"0x{observed:x}")
            self.assertTrue(observed & codec.CODEC_EXTERNAL, f"0x{observed:x}")

    def test_unknown_codec_raises_codec_error_naming_the_id(self):
        """Review Focus #4: q4p/i8x/fpzip must refuse, never guess."""
        for bad in (codec.CODEC_Q4P, codec.CODEC_I8X, codec.CODEC_FPZIP):
            with self.assertRaises(CodecError) as cm:
                codec.decode_tensor(b"\x00" * 64, bad, (8,), 2)
            self.assertIn(f"{bad:x}", str(cm.exception).lower())

    def test_store_read_routes_a_quantized_tensor(self):
        """Store.read must return decoded values, not the raw blob."""
        c = np.arange(64, dtype=np.float16)
        payload = inline(q6p_block(np.arange(64, dtype=np.float16),
                                    np.arange(64, dtype=np.uint8)), 4096)
        with write_ckpt(self.temp.name, {"t": ([64], payload)},
                        codec_id=codec.CODEC_Q6P) as p:
            with codec.Store.open(p) as s:
                shape, arr = s.read("t")
        self.assertEqual(shape, (64,))
        self.assertEqual(arr.dtype, np.float32)
        np.testing.assert_allclose(arr, np.arange(64, dtype=np.float32))

    def test_store_read_routes_an_external_quantized_tensor(self):
        c = np.arange(64, dtype=np.float16)
        payload = q6p_block( c, np.arange(64, dtype=np.uint8))
        with write_ckpt(self.temp.name, {"t": ([64], payload)},
                        external=True, codec_id=codec.CODEC_Q6P) as p:
            with codec.Store.open(p) as s:
                _shape, arr = s.read("t")
        np.testing.assert_allclose(arr, np.arange(64, dtype=np.float32))

    def test_store_read_raises_for_unknown_codec(self):
        with write_ckpt(self.temp.name, {"t": ([8], b"\x00" * 16)},
                        codec_id=codec.CODEC_Q4P) as p:
            with codec.Store.open(p) as s:
                with self.assertRaises(CodecError):
                    s.read("t")


class ExternalQuantisedPointerTests(unittest.TestCase):
    """Pins the 24-byte sidecar pointer found in real Draw Things model files.

    Task 8 gate finding: a quantised sidecar tensor uses a 24-byte pointer
    carrying `[u32 block_size][u32 magic][u64 offset][u64 length]`, and the
    payload it points at starts directly with the codebook -- the block size is
    NOT repeated at the head of the payload. Verified against
    flux_1_schnell_q5p.ckpt (block_size=1024 for q5p, 16384 for q8p).
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def _payload(self, centroids, indices, block_size):
        """Headerless block payload: centroids immediately, then packed indices."""
        return centroids.tobytes() + pack_msb(indices, 5)

    def test_block_size_comes_from_the_pointer_not_the_payload(self):
        c = np.arange(32, dtype=np.float16)
        idx = np.tile(np.arange(32, dtype=np.uint8), 1024 // 32)
        payload = self._payload(c, idx, 1024)
        with write_ckpt(self.temp.name, {"t": ([1024], payload)},
                        external=True, codec_id=codec.CODEC_Q5P, block_size=1024) as p:
            with codec.Store.open(p) as s:
                _shape, arr = s.read("t")
        np.testing.assert_allclose(arr, np.tile(np.arange(32, dtype=np.float32), 32))

    def test_two_blocks_decode_with_a_shared_codebook_each(self):
        c = np.arange(32, dtype=np.float16)
        idx = np.tile(np.arange(32, dtype=np.uint8), 1024 // 32)
        payload = self._payload(c, idx, 1024) * 2
        with write_ckpt(self.temp.name, {"t": ([2048], payload)},
                        external=True, codec_id=codec.CODEC_Q5P, block_size=1024) as p:
            with codec.Store.open(p) as s:
                _shape, arr = s.read("t")
        self.assertEqual(arr.size, 2048)
        np.testing.assert_allclose(arr, np.tile(np.arange(32, dtype=np.float32), 64))

    def test_q8p_pointer_round_trip(self):
        c = np.arange(256, dtype=np.float16)
        idx = np.tile(np.arange(256, dtype=np.uint8), 16384 // 256)
        payload = c.tobytes() + idx.tobytes()
        with write_ckpt(self.temp.name, {"t": ([16384], payload)},
                        external=True, codec_id=codec.CODEC_Q8P, block_size=16384) as p:
            with codec.Store.open(p) as s:
                _shape, arr = s.read("t")
        np.testing.assert_allclose(arr, np.tile(np.arange(256, dtype=np.float32), 64))

    def test_raw_external_still_uses_a_16_byte_pointer(self):
        raw = np.arange(8, dtype=np.float16)
        with write_ckpt(self.temp.name, {"t": ([2, 4], raw.tobytes())}, external=True) as p:
            with codec.Store.open(p) as s:
                _shape, arr = s.read("t")
        np.testing.assert_array_equal(arr, raw.reshape(2, 4))

    def test_malformed_pointer_length_raises(self):
        c = np.arange(32, dtype=np.float16)
        idx = np.tile(np.arange(32, dtype=np.uint8), 1024 // 32)
        payload = self._payload(c, idx, 1024)
        with write_ckpt(self.temp.name, {"t": ([1024], payload)},
                        external=True, codec_id=codec.CODEC_Q5P) as p:
            with codec.Store.open(p) as s:
                row = s._db.execute("SELECT data FROM tensors WHERE name='t'").fetchone()
                s._db.close()
                with self.assertRaises(CodecError):
                    codec.Store._read_external(s, row[0][:20])


class ReviewFixTests(unittest.TestCase):
    """Regressions for the final-review findings.

    Each pins one defect a reviewer demonstrated against the pre-fix code.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    # Critical 1: a '#' or '?' in the path used to move `mode=ro` into the URI
    # fragment, so SQLite opened the file read-write and could create one.
    def test_path_with_uri_metacharacters_is_still_read_only(self):
        for ch in ("#", "?", "%"):
            with self.subTest(ch=ch):
                p = Path(self.temp.name) / f"we{ch}ird.ckpt"
                with write_ckpt(self.temp.name, {"t": ([2, 2], b"\x00" * 8)}) as _:
                    Path(p).write_bytes(b"")
                s = codec.Store.open(p)
                try:
                    with self.assertRaises(sqlite3.OperationalError):
                        s._db.execute("CREATE TABLE probe (x)")
                finally:
                    s.close()
                self.assertFalse(
                    (Path(self.temp.name) / "we").exists(),
                    "SQLite created a stray database, so the URI was not read-only",
                )

    # Critical 2: a 16-byte sidecar pointer leaves block_size None, and the
    # decoder then read four centroid bytes as the block size, returning
    # plausible-but-wrong weights with no error.
    def test_external_kmeans_without_a_block_size_raises(self):
        c = np.arange(32, dtype=np.float16)
        payload = q5p_block(c, np.tile(np.arange(32, dtype=np.uint8), 32))
        with write_ckpt(self.temp.name, {"t": ([1024], payload)},
                        external=True, codec_id=codec.CODEC_Q5P) as p:
            # Force the 16-byte pointer even though the codec is quantised.
            import sqlite3 as sq
            db = sq.connect(p)
            db.execute("UPDATE tensors SET data=? WHERE name='t'",
                       (struct.pack("<QQ", 0, len(payload)),))
            db.commit()
            db.close()
            with codec.Store.open(p) as s:
                with self.assertRaises(CodecError):
                    s.read("t")

    # Critical 3: an exponent byte above 31 wrapped the uint16 shift and
    # produced a finite but wrong value, which raw DEFLATE cannot detect.
    def test_ezm7_rejects_an_out_of_range_exponent(self):
        for bad in (32, 64, 255):
            with self.subTest(exponent=bad):
                comp = zlib.compressobj(9, zlib.DEFLATED, -15)
                exp = comp.compress(bytes([bad])) + comp.flush()
                payload = (struct.pack("<i", len(exp)) + exp + b"\x00" * 1)
                with self.assertRaises(CodecError):
                    codec.decode_ezm7(payload, numel=1)

    # Important: a dim blob with a non-zero entry after a zero is corrupt.
    def test_read_shape_rejects_nonzero_after_a_zero(self):
        with self.assertRaises(CodecError):
            codec.read_shape(struct.pack("<12i", 4, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    def test_null_data_or_dim_raises_codec_error(self):
        for col in ("data", "dim"):
            with self.subTest(col=col):
                with write_ckpt(self.temp.name, {"t": ([2, 2], b"\x00" * 8)},
                                nulls=(col,)) as p:
                    with codec.Store.open(p) as s:
                        with self.assertRaises(CodecError):
                            s.read("t")

    def test_absurd_sidecar_length_raises_codec_error(self):
        with write_ckpt(self.temp.name, {"t": ([2, 2], b"\x00" * 8)},
                        external=True) as p:
            import sqlite3 as sq
            db = sq.connect(p)
            db.execute("UPDATE tensors SET data=? WHERE name='t'",
                       (struct.pack("<QQ", 0, 1 << 63),))
            db.commit()
            db.close()
            with codec.Store.open(p) as s:
                with self.assertRaises(CodecError):
                    s.read("t")

    def test_unknown_format_raises_codec_error(self):
        with write_ckpt(self.temp.name, {"t": ([2, 2], b"\x00" * 8)}, fmt=7) as p:
            with codec.Store.open(p) as s:
                with self.assertRaises(CodecError):
                    s.read("t")

    def test_decode_tensor_rejects_unknown_elem_size(self):
        with self.assertRaises(CodecError):
            codec.decode_tensor(b"\x00" * 12, 0, (4,), 3)

    # Important: the inline-header success path had no positive test, so
    # deleting `offset = 4` left the suite green. This is what hid Critical 2.
    def test_inline_header_form_decodes(self):
        c = np.arange(64, dtype=np.float16)
        idx = np.arange(64, dtype=np.uint8)
        payload = inline(q6p_block(c, idx), 4096)
        out = codec.decode_q6p(payload, 64, 2)
        np.testing.assert_allclose(out, np.arange(64, dtype=np.float32))

    # Minor promoted to Important: Plan 2 gets (shape, array) from every codec,
    # so a flat array from the k-means paths is a live trap.
    def test_read_returns_a_shaped_array_for_every_codec(self):
        c5 = np.arange(32, dtype=np.float16)
        cases = {
            "q5p": (codec.CODEC_Q5P, 1024,
                    q5p_block(c5, np.tile(np.arange(32, dtype=np.uint8), 32))),
            "q8p": (codec.CODEC_Q8P, 16384,
                    q8p_block(np.arange(256, dtype=np.float16),
                              np.tile(np.arange(256, dtype=np.uint8), 64))),
            "ezm7": (codec.CODEC_EZM7, None,
                     encode_ezm7_reference(np.arange(32, dtype=np.float16))),
        }
        for label, (cid, block, payload) in cases.items():
            with self.subTest(codec=label):
                shape = (32, 2) if label == "q5p" else ((32, 256) if label == "q8p" else (32,))
                with write_ckpt(self.temp.name, {"t": (list(shape), payload)},
                                codec_id=cid, block_size=block) as p:
                    with codec.Store.open(p) as s:
                        got_shape, arr = s.read("t")
                self.assertEqual(got_shape, tuple(shape))
                self.assertEqual(arr.shape, tuple(shape), f"{label} returned a flat array")


if __name__ == "__main__":
    unittest.main()