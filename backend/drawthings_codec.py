"""Decode tensors from Draw Things `.ckpt` files.

Draw Things stores model weights in a SQLite database with an optional external
sidecar (`<name>.ckpt-tensordata`) for large tensors. Each row records the
tensor's shape, dtype and -- packed into the high 32 bits of `type` -- which
codec its payload uses.

This module reads that container with nothing but the standard library and
numpy. No torch, no MLX, no subprocess: decoding is pure byte layout work.

The Draw Things model directory is treated as strictly read-only. SQLite is
always opened with a `mode=ro` URI, so a stray write fails loudly rather than
touching a real model file.
"""

import contextlib
import os
import sqlite3
import struct
import sys
from urllib.parse import quote
import zlib

import numpy as np

# Codec identifiers, carried in the high 32 bits of the `type` column.
# Source: s4nnc/nnc/Store.swift `Codec` OptionSet.
CODEC_FPZIP = 0xF7217
CODEC_EZM7 = 0x511
CODEC_Q4P = 0x8A1E4B
CODEC_Q5P = 0x8A1E5B
CODEC_Q6P = 0x8A1E6B
CODEC_Q7P = 0x8A1E7B
CODEC_Q8P = 0x8A1E8B
CODEC_I8X = 0x8A1E9B
# Marks a tensor whose payload lives in the `<name>.ckpt-tensordata` sidecar,
# OR'd onto whichever codec the tensor actually uses. This is bit 28
# (0x1000_0000) -- confirmed against flux_1_schnell_q5p.ckpt, where a raw
# tensor reads 0x10000000 and a quantised one 0x108a1e5b / 0x108a1e8b.
CODEC_EXTERNAL = 0x10000000

# `datatype` encodes the element type in its low bits, shifted left by 12.
DTYPE_F16 = 131072  # CCV_16F
DTYPE_F32 = 262144  # CCV_32F

FORMAT_NHWC = 1  # CCV_TENSOR_FORMAT_NHWC

_DIM_BYTES = 48
_MAX_DIMS = 12
_TYPE_COLUMN_MASK = 0xFFFFFFFF

_ELEMENT_SIZES = {DTYPE_F16: 2, DTYPE_F32: 4}
_NUMPY_DTYPES = {DTYPE_F16: np.float16, DTYPE_F32: np.float32}
_ELEM_NP = {2: np.float16, 4: np.float32}

_SELECT = "SELECT data, type, format, datatype, dim FROM tensors WHERE name = ?"
_LIST = "SELECT name FROM tensors"


class CodecError(RuntimeError):
    """Raised when a tensor cannot be decoded.

    Every failure mode raises this rather than returning a short or
    mis-shaped array: a wrong decode that still produces plausible values is
    far more expensive to debug than a refusal.
    """


def read_shape(dim_blob):
    """Return the tensor shape from the 48-byte `dim` blob.

    The blob is 12 little-endian int32 values; the shape is the leading
    non-zero entries, and CCV_NNC_MAX_DIM_ALLOC is 12, so all twelve are
    honoured rather than assuming a small fixed rank.
    """
    if len(dim_blob) != _DIM_BYTES:
        raise CodecError(f"dim blob is {len(dim_blob)} bytes, expected {_DIM_BYTES}")
    dims = struct.unpack_from("<12i", bytes(dim_blob))
    shape = []
    for d in dims:
        if d == 0:
            break
        if d < 0:
            raise CodecError(f"negative dimension {d} in dim blob")
        shape.append(d)
    if not shape:
        raise CodecError("dim blob describes a 0-dimensional tensor")
    if any(dims[i] != 0 for i in range(len(shape), _MAX_DIMS)):
        raise CodecError(
            "dim blob has a zero followed by a non-zero entry: "
            f"{list(dims[:4])}..."
        )
    return tuple(shape)



def _dtype_code(datatype):
    """Return the element size in bytes for a `datatype` column value."""
    try:
        return _ELEMENT_SIZES[datatype]
    except KeyError:
        raise CodecError(f"unsupported datatype {datatype}") from None


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _decode_blocks(payload, numel, elem_size, codebook, bits, block_size=None):
    """Shared body of the k-means codecs.

    Each block is `[codebook of `codebook` entries at `elem_size`][packed
    indices]`. The three codecs differ only in codebook size and index width,
    so they share this walk.

    `block_size` normally comes from the tensor's sidecar pointer, because a
    real quantised payload starts straight at the codebook and does not repeat
    it. When it is `None` the leading `uint32` is read from the payload itself,
    which is the inline form.

    Indices wider than the codebook, a short read, and a degenerate block size
    all raise: a decode that returns plausible-but-wrong values is worse than
    one that refuses.
    """
    if elem_size not in (2, 4):
        raise CodecError(f"unsupported element size {elem_size}")
    centroid_dt = _ELEM_NP[elem_size]
    table_bytes = codebook * elem_size

    offset = 0
    if block_size is None:
        if len(payload) < 4:
            raise CodecError("payload ended before the block size header")
        block_size = struct.unpack_from("<I", payload, offset)[0]
        offset = 4

    out = []
    remaining = numel

    while remaining > 0:
        if block_size < 2:
            # Zero makes no progress; one forces a byte-at-a-time walk over a
            # multi-million element tensor. Neither occurs in a real file.
            raise CodecError(f"implausible block size {block_size}")

        if offset + table_bytes > len(payload):
            raise CodecError("payload ended inside a codebook table")
        table = np.frombuffer(payload, dtype=centroid_dt, count=codebook, offset=offset)
        offset += table_bytes

        n = min(block_size, remaining)
        if bits == 8:
            if offset + n > len(payload):
                raise CodecError("payload ended inside an index block")
            indices = np.frombuffer(payload, dtype=np.uint8, count=n, offset=offset)
            offset += n
        else:
            index_bytes = (n * bits + 7) // 8
            if offset + index_bytes > len(payload):
                raise CodecError("payload ended inside an index block")
            indices = unpack_msb(payload[offset:offset + index_bytes], bits, n)
            offset += index_bytes

        if indices.size and int(indices.max()) >= codebook:
            raise CodecError(
                f"index {int(indices.max())} exceeds the {codebook}-entry codebook"
            )
        out.append(table[indices].astype(np.float32))
        remaining -= n

    return np.concatenate(out)[:numel]


def decode_ezm7(payload, numel):
    """Decode an ezm7 payload (codec id 0x511).

    Layout, from s4nnc/nnc/Store.swift `ezm7Decode` at revision 0ecfd9b5:

        |-- i32 zippedExponentsSize --|-- deflated exponents --|-- mantissa+sign --|

    The "zipped" exponents are raw DEFLATE (`inflateInit2_` with -MAX_WBITS),
    not a bit-packed stream -- they are one byte per element, holding the 5-bit
    f16 exponent. Each trailing byte is `sign << 7 | mantissa7`.

    This is a *lossy* re-encoding, not a quantisation scheme: the exponent and
    sign survive intact, three mantissa bits do not. It is therefore the only
    decoder here that returns float16 rather than float32.
    """
    if len(payload) < 4:
        raise CodecError("ezm7 payload is too short to hold its header")
    zip_size = struct.unpack_from("<i", payload, 0)[0]
    if zip_size < 0:
        raise CodecError(f"negative ezm7 exponent size {zip_size}")
    if len(payload) < 4 + zip_size + numel:
        raise CodecError(
            f"ezm7 payload is {len(payload)} bytes, need {4 + zip_size + numel}"
        )

    zipped = payload[4:4 + zip_size]
    # Bounded decompression: the output is exactly `numel` bytes, and the
    # length check above already proves numel <= len(payload), so capping the
    # inflate at numel makes a decompression bomb impossible. An unbounded
    # zlib.decompress here would expand attacker-controlled bytes without limit.
    try:
        inflate = zlib.decompressobj(-15)
        exponents = inflate.decompress(bytes(zipped), numel)
        if inflate.unconsumed_tail:
            raise CodecError("ezm7 exponent stream expands beyond the tensor size")
    except zlib.error as exc:
        raise CodecError(f"ezm7 exponent stream is not valid raw deflate: {exc}") from None
    if len(exponents) < numel:
        raise CodecError(
            f"ezm7 exponent stream holds {len(exponents)} bytes, need {numel}"
        )

    floats_without_exp = np.frombuffer(payload, dtype=np.uint8, count=numel,
                                       offset=4 + zip_size).astype(np.uint16)
    exponent = np.frombuffer(exponents, dtype=np.uint8, count=numel)

    # The encoder only ever writes `(bits >> 10) & 0x1F`. Raw DEFLATE carries
    # no checksum, so a corrupt stream can hand us any byte at all, and a
    # value above 31 would wrap the uint16 shift into a finite, wrong float
    # with nothing to catch it downstream.
    if exponent.size and int(exponent.max()) > 31:
        raise CodecError(
            f"ezm7 exponent {int(exponent.max())} exceeds the 5-bit f16 range"
        )
    exponent = exponent.astype(np.uint16)
    sign = floats_without_exp >> 7
    mantissa = (floats_without_exp & 0x7F) << 3

    bits = (sign << 15) | (exponent << 10) | mantissa
    return bits.astype(np.uint16).view(np.float16)


def decode_q5p(payload, numel, elem_size, block_size=None):
    """Decode a 5-bit k-means payload (32 centroids).

    Unlike q6p and q8p the block size was never confirmed from the Draw Things
    source, so it is read from the leading `uint32` and validated rather than
    assumed: a block size of zero would stall, and one larger than the tensor
    would over-read the blob.
    """
    return _decode_blocks(payload, numel, elem_size, 32, 5, block_size)


def decode_q8p(payload, numel, elem_size, block_size=None):
    """Decode an 8-bit k-means payload (block size 16384, 256 centroids).

    Indices are one raw byte per element, so no bit unpacking is involved.
    """
    return _decode_blocks(payload, numel, elem_size, 256, 8, block_size)


def decode_q6p(payload, numel, elem_size, block_size=None):
    """Decode a 6-bit k-means payload (block size 4096, 64 centroids)."""
    return _decode_blocks(payload, numel, elem_size, 64, 6, block_size)


def unpack_msb(data, bits, count):
    """Unpack `count` MSB-first bit-packed values of `bits` width from `data`.

    q5p (5 bits) and q6p (6 bits) write their codebook indices as one
    contiguous big-endian bitstream rather than per-byte fields -- 8 five-bit
    values in 5 bytes, 4 six-bit values in 3 bytes. Reading it that way avoids
    hand-rolled per-byte shift arithmetic, which is where an off-by-one-bit
    error would otherwise hide.

    Excess trailing bytes are ignored: callers hand over a whole block.
    """
    if not 1 <= bits <= 8:
        raise CodecError(f"unsupported bit width {bits}")
    needed = (count * bits + 7) // 8
    if len(data) < needed:
        raise CodecError(
            f"packed payload is {len(data)} bytes, need {needed} for "
            f"{count} values of {bits} bits"
        )
    total_bits = count * bits
    flat = np.unpackbits(
        np.frombuffer(bytes(data), dtype=np.uint8), bitorder="big"
    )[:total_bits]
    weights = (1 << np.arange(bits - 1, -1, -1)).astype(np.uint16)
    return (flat.reshape(count, bits) * weights).sum(axis=1).astype(np.uint8)


_DECODERS = {
    CODEC_Q5P: decode_q5p,
    CODEC_Q6P: decode_q6p,
    CODEC_Q8P: decode_q8p,
}


def decode_tensor(payload, codec_id, shape, elem_size, block_size=None):
    """Decode one tensor's raw payload according to its codec id.

    `CODEC_EXTERNAL` is stripped first: it is OR'd onto another codec id to
    mark a sidecar-resident tensor, not a codec in its own right. `Store.read`
    has already located the payload by then, so this only interprets it.

    An unrecognised id raises rather than falling back to a guess -- a decode
    that silently produces plausible wrong values is the failure mode this
    whole module exists to prevent.
    """
    codec_id &= ~CODEC_EXTERNAL
    if elem_size not in _ELEM_NP:
        raise CodecError(f"unsupported element size {elem_size}")

    if codec_id == 0:
        expected = _numel(shape) * elem_size
        if len(payload) != expected:
            raise CodecError(
                f"raw payload is {len(payload)} bytes, expected {expected}"
            )
        return np.frombuffer(payload, dtype=_ELEM_NP[elem_size]).reshape(shape)

    # Every path returns an array shaped like `shape`, so a caller never has to
    # remember which codecs hand back a flat buffer.
    decoder = _DECODERS.get(codec_id)
    if decoder is not None:
        return decoder(payload, _numel(shape), elem_size, block_size).reshape(shape)

    if codec_id == CODEC_EZM7:
        return decode_ezm7(payload, _numel(shape)).reshape(shape)

    raise CodecError(f"unsupported codec id 0x{codec_id:x}")



class Store:
    """Read-only accessor for one Draw Things `.ckpt` database."""

    def __init__(self, db, path):
        self._db = db
        self._path = path

    @classmethod
    def open(cls, path):
        """Open `path` read-only.

        The path is percent-escaped because SQLite reads everything after a
        `#` as a URI fragment and everything after a `?` as the query: a model
        filename containing either would push `mode=ro` out of the URI and
        silently downgrade the handle to read-write. `PRAGMA query_only` is
        then set as a second, non-URI guarantee, so the read-only contract
        holds even if the URI is ever assembled differently.
        """
        uri = "file:" + quote(str(path), safe="/") + "?mode=ro"
        db = sqlite3.connect(uri, uri=True)
        db.execute("PRAGMA query_only=1")
        return cls(db, str(path))

    @property
    def sidecar_path(self):
        """Path of the external `-tensordata` blob, whether or not it exists."""
        return self._path + "-tensordata"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self._db.close()

    def names(self):
        return [r[0] for r in self._db.execute(_LIST)]

    def read(self, name):
        """Return `(shape, array)` for one tensor.

        Payload *location* is resolved here; codec *interpretation* is added
        by `decode_tensor`, which Task 7 wires in.
        """
        row = self._db.execute(_SELECT, (name,)).fetchone()
        if row is None:
            raise CodecError(f"no tensor named {name!r}")
        data, type_value, fmt, datatype, dim = row

        # A NULL here means the blob was too large for one SQLite value and
        # was split into the `tensor_splits` table, which this reader does not
        # implement. Say so rather than failing with a TypeError on None.
        if data is None or dim is None:
            raise CodecError(
                f"tensor {name!r} has a NULL column; large blobs live in the "
                "unsupported tensor_splits table"
            )
        if fmt != FORMAT_NHWC:
            raise CodecError(f"tensor {name!r}: unsupported format {fmt}")

        codec_id = (type_value >> 32) & _TYPE_COLUMN_MASK
        block_size = None
        if codec_id & CODEC_EXTERNAL:
            payload, block_size = self._read_external(data)
            codec_id &= ~CODEC_EXTERNAL
            if block_size is None and codec_id in _DECODERS:
                # Only the 24-byte pointer carries a block size. Without one the
                # decoder would read four codebook bytes as the block size and
                # return plausible-but-wrong weights, so refuse instead.
                raise CodecError(
                    f"tensor {name!r}: 16-byte sidecar pointer carries no block "
                    f"size for codec 0x{codec_id:x}"
                )
        else:
            payload = bytes(data)

        shape = read_shape(dim)
        elem_size = _dtype_code(datatype)
        return shape, decode_tensor(payload, codec_id, shape, elem_size, block_size)

    def _read_external(self, data):
        """Resolve a sidecar pointer, returning `(payload, block_size)`.

        Two pointer widths exist, confirmed against flux_1_schnell_q5p.ckpt:

        - 16 bytes `[u64 offset][u64 length]` for an uncompressed tensor, whose
          payload is raw f16 and carries no block size.
        - 24 bytes `[u32 block_size][u32 magic][u64 offset][u64 length]` for a
          quantised one. The block size lives here rather than at the head of
          the payload, which starts straight at the codebook.
        """
        if len(data) == 16:
            offset, length = struct.unpack_from("<QQ", data)
            block_size = None
        elif len(data) == 24:
            block_size, _magic, offset, length = struct.unpack_from("<IIQQ", data)
        else:
            raise CodecError(
                f"external pointer is {len(data)} bytes, expected 16 or 24"
            )

        sidecar = self.sidecar_path
        if not os.path.exists(sidecar):
            raise CodecError(f"missing external data file {sidecar}")
        # A corrupt u64 length must not reach read(), which raises OverflowError
        # for anything above sys.maxsize.
        if length > sys.maxsize or offset > sys.maxsize:
            raise CodecError(f"implausible external pointer offset/length: {offset}, {length}")
        try:
            with open(sidecar, "rb") as fh:
                fh.seek(offset)
                payload = fh.read(length)
        except OSError as exc:
            raise CodecError(f"cannot read external data file {sidecar}: {exc}") from None
        if len(payload) != length:
            raise CodecError(
                f"external read short: got {len(payload)} bytes, expected {length}"
            )
        return payload, block_size


@contextlib.contextmanager
def open_store(path):
    """Context manager form of `Store.open`."""
    store = Store.open(path)
    try:
        yield store
    finally:
        store.close()