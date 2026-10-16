"""Tests for pairing, validation and reassembly of Draw Things LoRA tensors.

Three reassembly shapes occur, and they are not interchangeable. Each is pinned
here against the real tensors that produced it, because getting one wrong
yields a right-shaped wrong-valued array -- Review Focus #1.

  sum   the pieces share the output rows and split the input columns, so the
        products add. FLUX `single_transformer_blocks.*.proj_out`, whose DT
        sources `x_o` (3072-wide) and `x_out_proj` (12288-wide) are literally
        the two column blocks of mflux's Linear(15360, 3072).

  block the pieces split the output rows and share the input columns, so the
        result is block diagonal with the rank axis zero padded. FLUX AdaLN
        chunks; and SDXL `attn.in_proj_weight`, whose three slots carry three
        *different* down matrices and so are three independent LoRAs.

  shared one weight split along axis 0 with the down matrix duplicated across
        the slots. SDXL `ff.net.0.proj` and its relatives: measured, every
        slot's down is byte-identical to the others.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lora_convert as convert  # noqa: E402


def down(rank, width):
    return np.arange(rank * width, dtype=np.float32).reshape(rank, width)


def up(rows, rank):
    return np.arange(rows * rank, dtype=np.float32).reshape(rows, rank) + 0.5


class SumMergeTests(unittest.TestCase):
    """Pieces share output rows; their products add."""

    def test_product_is_the_sum_of_the_piece_products(self):
        # The merged product is the concatenation of the two piece products:
        # each occupies its own column block of the summed input width.
        a, b = up(3072, 4), up(3072, 4)
        da, db = down(4, 3072), down(4, 12288)
        merged = convert.merge_pieces([(a, da), (b, db)], convert.MERGE_SUM)
        self.assertEqual(merged.up.shape, (3072, 8))
        self.assertEqual(merged.down.shape, (8, 15360))
        delta = merged.up @ merged.down
        np.testing.assert_allclose(delta[:, :3072], a @ da, atol=1e-3)
        np.testing.assert_allclose(delta[:, 3072:], b @ db, atol=1e-3)

    def test_pieces_must_share_output_width(self):
        with self.assertRaises(convert.ConversionError):
            convert.merge_pieces([(up(3072, 4), down(4, 3072)),
                                  (up(2048, 4), down(4, 12288))],
                                 convert.MERGE_SUM)

    def test_pieces_must_share_rank(self):
        with self.assertRaises(convert.ConversionError):
            convert.merge_pieces([(up(3072, 4), down(4, 3072)),
                                  (up(3072, 8), down(8, 12288))],
                                 convert.MERGE_SUM)


class BlockMergeTests(unittest.TestCase):
    """Pieces split the output rows; the result is block diagonal."""

    def test_shape_is_rows_times_pieces_by_summed_rank(self):
        pieces = [(up(1280, 16), down(16, 1280))] * 3
        merged = convert.merge_pieces(pieces, convert.MERGE_BLOCK)
        self.assertEqual(merged.up.shape, (3840, 48))
        self.assertEqual(merged.down.shape, (48, 1280))

    def test_off_diagonal_blocks_are_zero(self):
        a, b, c = up(4, 2), up(4, 2), up(4, 2)
        merged = convert.merge_pieces([(a, down(2, 5)), (b, down(2, 5)),
                                       (c, down(2, 5))], convert.MERGE_BLOCK)
        # Row block 0 must not see column blocks 1 and 2.
        delta = merged.up @ merged.down
        np.testing.assert_allclose(delta[:4, :], a @ down(2, 5), atol=1e-3)
        np.testing.assert_allclose(delta[4:8, :], b @ down(2, 5), atol=1e-3)
        np.testing.assert_allclose(delta[8:12, :], c @ down(2, 5), atol=1e-3)

    def test_pieces_must_share_input_width(self):
        with self.assertRaises(convert.ConversionError):
            convert.merge_pieces([(up(1280, 8), down(8, 1280)),
                                  (up(1280, 8), down(8, 640))],
                                 convert.MERGE_BLOCK)


class SharedDownMergeTests(unittest.TestCase):
    """One weight split on axis 0 with the down duplicated across slots."""

    def test_down_is_taken_once_and_up_stacked(self):
        a, b = up(2560, 16), up(2560, 16)
        d = down(16, 640)
        merged = convert.merge_pieces([(a, d), (b, d)], convert.MERGE_SHARED)
        self.assertEqual(merged.up.shape, (5120, 16))
        self.assertEqual(merged.down.shape, (16, 640))

    def test_identical_downs_are_taken_from_any_slot(self):
        a, b = up(8, 3), up(8, 3)
        d = down(3, 6)
        merged = convert.merge_pieces([(a, d), (b, d.copy())],
                                      convert.MERGE_SHARED)
        np.testing.assert_allclose(merged.down, d)

    def test_differing_downs_are_refused(self):
        # This is the whole point: silently picking one of two different down
        # matrices would emit a LoRA that loads and generates nothing.
        with self.assertRaises(convert.ConversionError) as cm:
            convert.merge_pieces([(up(8, 3), down(3, 6)), (up(8, 3), down(3, 6) + 1)],
                                 convert.MERGE_SHARED)
        self.assertIn("down", str(cm.exception))

    def test_pieces_must_share_rank_and_input_width(self):
        d = down(3, 6)
        with self.assertRaises(convert.ConversionError):
            convert.merge_pieces([(up(8, 3), d), (up(8, 4), down(4, 6))],
                                 convert.MERGE_SHARED)


class DetectMergeTests(unittest.TestCase):
    """The SDXL merge mode is decided by the data, not by the key spelling."""

    def test_identical_downs_select_shared(self):
        pieces = [(up(8, 4), down(4, 6)) for _ in range(2)]
        self.assertEqual(convert.detect_merge(pieces), convert.MERGE_SHARED)

    def test_differing_downs_select_block(self):
        pieces = [(up(8, 4), down(4, 6)), (up(8, 4), down(4, 6) + 3)]
        self.assertEqual(convert.detect_merge(pieces), convert.MERGE_BLOCK)

    def test_a_single_piece_needs_no_merge(self):
        self.assertEqual(convert.detect_merge([(up(8, 4), down(4, 6))]),
                         convert.MERGE_SINGLE)

    def test_mixed_downs_are_refused_as_ambiguous(self):
        pieces = [(up(8, 4), down(4, 6)), (up(8, 4), down(4, 6)),
                  (up(8, 4), down(4, 6) + 2)]
        with self.assertRaises(convert.ConversionError) as cm:
            convert.detect_merge(pieces)
        self.assertIn("ambiguous", str(cm.exception))


class RankTests(unittest.TestCase):
    def test_rank_comes_from_down(self):
        pair = convert.Pair("k", np.zeros((16, 8)), np.zeros((32, 16)))
        self.assertEqual(convert.rank_of(pair), 16)

    def test_mismatched_rank_is_refused(self):
        with self.assertRaises(convert.ConversionError) as cm:
            convert._check_pair("base", down(16, 8), up(32, 8))
        self.assertIn("base", str(cm.exception))

    def test_down_must_be_two_dimensional(self):
        with self.assertRaises(convert.ConversionError):
            convert._check_pair("base", np.zeros(16), np.zeros((16, 16)))


class FakeStore:
    """Minimal stand-in for drawthings_codec.Store.

    Takes `{tensor_name: array}` and serves `names()` and `read()`, which is the
    whole surface `build_pairs` uses.
    """

    def __init__(self, tensors):
        self._tensors = dict(tensors)

    def names(self):
        return list(self._tensors)

    def read(self, name):
        array = self._tensors[name]
        return array.shape, array


def sdxl_pair(prefix, layer, param=0):
    return (f"{prefix}[t-{layer}-{param}]__down__",
            f"{prefix}[t-{layer}-{param}]__up__")


class BuildPairsSdxlTests(unittest.TestCase):
    """Pins the SDXL key spelling and the refusal behaviour."""

    def test_unet_pair_gets_the_diffusers_key_spelling(self):
        store = FakeStore({
            "__unet__[t-25-0]__down__": down(16, 3072),
            "__unet__[t-25-0]__up__": up(3072, 16),
        })
        pairs = convert.build_pairs(store, "sdxl")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(
            pairs[0].base,
            "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q",
        )

    def test_text_encoder_pairs_get_their_own_prefix(self):
        # `load_multilora` walks the path below `text_encoder.` literally against the
        # real pipeline, where BOTH encoders are CLIPTextModel rooted at `text_model`.
        # Measured on juggernaut-xl-lightning; see ComponentRootTests.
        store = FakeStore({
            "__text_model__[t-9-0]__down__": down(16, 768),
            "__text_model__[t-9-0]__up__": up(768, 16),
        })
        pairs = convert.build_pairs(store, "sdxl")
        self.assertEqual(pairs[0].base,
                         "text_encoder.text_model.encoder.layers.0.mlp.fc2")

    def test_second_text_encoder_prefix(self):
        store = FakeStore({
            "__te2__text_model__[t-8-0]__down__": down(16, 3072),
            "__te2__text_model__[t-8-0]__up__": up(3072, 16),
        })
        pairs = convert.build_pairs(store, "sdxl")
        self.assertEqual(pairs[0].base,
                         "text_encoder_2.text_model.encoder.layers.0.mlp.fc1")

    def test_incomplete_pair_is_refused(self):
        store = FakeStore({"__unet__[t-25-0]__down__": down(16, 3072)})
        with self.assertRaises(convert.ConversionError) as cm:
            convert.build_pairs(store, "sdxl")
        self.assertIn("t-25-0", str(cm.exception))

    def test_mismatched_rank_is_refused(self):
        store = FakeStore({
            "__unet__[t-25-0]__down__": down(16, 3072),
            "__unet__[t-25-0]__up__": up(3072, 8),
        })
        with self.assertRaises(convert.ConversionError):
            convert.build_pairs(store, "sdxl")

    def test_unmapped_tensor_is_refused_naming_it(self):
        store = FakeStore({
            "__unet__[t-999999-0]__down__": down(16, 4),
            "__unet__[t-999999-0]__up__": up(4, 16),
        })
        with self.assertRaises(convert.ConversionError) as cm:
            convert.build_pairs(store, "sdxl")
        self.assertIn("t-999999-0", str(cm.exception))

    def test_unknown_base_model_is_refused(self):
        with self.assertRaises(convert.ConversionError):
            convert.build_pairs(FakeStore({}), "wan")


class BuildPairsFluxTests(unittest.TestCase):
    """Pins the FLUX key spelling and the merged projections."""

    def test_single_block_pair_gets_the_mflux_key(self):
        store = FakeStore({
            "__dit__[t-x_q-19-0]__down__": down(8, 3072),
            "__dit__[t-x_q-19-0]__up__": up(3072, 8),
        })
        pairs = convert.build_pairs(store, "flux")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].base,
                         "single_transformer_blocks.0.attn.to_q")

    def test_single_block_proj_out_merges_two_scopes(self):
        store = FakeStore({
            "__dit__[t-x_o-19-0]__down__": down(8, 3072),
            "__dit__[t-x_o-19-0]__up__": up(3072, 8),
            "__dit__[t-x_out_proj-19-0]__down__": down(8, 12288),
            "__dit__[t-x_out_proj-19-0]__up__": up(3072, 8),
        })
        pairs = convert.build_pairs(store, "flux")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(
            pairs[0].base,
            "single_transformer_blocks.0.proj_out")
        # One LoRA of double rank, not two LoRAs.
        self.assertEqual(pairs[0].up.shape, (3072, 16))
        self.assertEqual(pairs[0].down.shape, (16, 15360))

    def test_single_block_norm_chunks_merge_block_diagonally(self):
        tensors = {}
        for i in range(3):
            tensors[f"__dit__[t-x_ada_ln_{i}-19-0]__down__"] = down(8, 3072)
            tensors[f"__dit__[t-x_ada_ln_{i}-19-0]__up__"] = up(3072, 8)
        pairs = convert.build_pairs(FakeStore(tensors), "flux")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(
            pairs[0].base,
            "single_transformer_blocks.0.norm.linear")
        self.assertEqual(pairs[0].up.shape, (9216, 24))
        self.assertEqual(pairs[0].down.shape, (24, 3072))

    def test_dual_norm1_zero_fills_the_norm2_half(self):
        # mflux's norm1.linear is Linear(3072, 18432); DT only supplies the
        # three norm1 chunks, so the other three are zeros.
        tensors = {}
        for i in range(3):
            tensors[f"__dit__[t-x_ada_ln_{i}-0-0]__down__"] = down(8, 3072)
            tensors[f"__dit__[t-x_ada_ln_{i}-0-0]__up__"] = up(3072, 8)
        pairs = convert.build_pairs(FakeStore(tensors), "flux")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].base,
                         "transformer_blocks.0.norm1.linear")
        self.assertEqual(pairs[0].up.shape, (18432, 48))
        self.assertEqual(pairs[0].down.shape, (48, 3072))
        delta = pairs[0].up @ pairs[0].down
        np.testing.assert_allclose(delta[9216:, :], 0.0, atol=0.0)

    def test_unaddressable_scope_is_skipped_and_reported(self):
        # `x_ada_ln_3/4/5` are the dual block's norm2 modulation and mflux has
        # no module for them. Every real FLUX.1 LoRA carries them, so refusing
        # the whole file would make FLUX conversion useless -- but dropping them
        # silently is the failure this project treats as worst. They are skipped
        # and counted.
        store = FakeStore({
            "__dit__[t-x_ada_ln_3-0-0]__down__": down(8, 3072),
            "__dit__[t-x_ada_ln_3-0-0]__up__": up(3072, 8),
            "__dit__[t-x_q-0-0]__down__": down(8, 3072),
            "__dit__[t-x_q-0-0]__up__": up(3072, 8),
        })
        report = {}
        pairs = convert.build_pairs(store, "flux", report=report)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(report["skipped_tensors"], 2)
        self.assertTrue(all("x_ada_ln_3" in n for n in report["skipped"]))

    def test_an_unexpected_scope_is_still_refused(self):
        # Only the known-unaddressable scopes may be skipped.
        store = FakeStore({
            "__dit__[t-nonsense-0-0]__down__": down(8, 3072),
            "__dit__[t-nonsense-0-0]__up__": up(3072, 8),
        })
        with self.assertRaises(convert.ConversionError) as cm:
            convert.build_pairs(store, "flux")
        self.assertIn("nonsense", str(cm.exception))

    def test_flux_keys_carry_no_extra_root(self):
        # mflux matches weight keys by exact string equality against
        # FluxLoRAMapping's patterns, which are already rooted at
        # `transformer_blocks.{b}...` / `single_transformer_blocks.{b}...`.
        store = FakeStore({
            "__dit__[t-x_q-0-0]__down__": down(8, 3072),
            "__dit__[t-x_q-0-0]__up__": up(3072, 8),
        })
        pairs = convert.build_pairs(store, "flux")
        keys = convert._tensors_for(pairs)
        self.assertEqual(
            sorted(keys),
            ["transformer_blocks.0.attn.to_q.lora_down.weight",
             "transformer_blocks.0.attn.to_q.lora_up.weight"],
        )


class Te2ConversionTests(unittest.TestCase):
    """The te2 rewrite, exercised through `build_pairs`.

    `attn.in_proj_weight` must not be merged: its three Draw Things slots are
    three independent LoRAs, one per projection, so the converted file has three
    single-slot pairs rather than one triple-rank pair.
    """

    LAYER = 12

    def _tid(self, offset):
        return f"t-{self.LAYER * 8 + offset}-0"

    def test_fused_qkv_becomes_three_pairs(self):
        tensors = {}
        for offset in (3, 4, 5):
            d = f"__te2__text_model__[{self._tid(offset)}]__down__"
            u = f"__te2__text_model__[{self._tid(offset)}]__up__"
            tensors[d] = down(16, 1280)
            tensors[u] = up(1280, 16)
        pairs = convert.build_pairs(FakeStore(tensors), "sdxl")
        self.assertEqual(len(pairs), 3)
        self.assertEqual(
            sorted(p.base for p in pairs),
            ["text_encoder_2.text_model.encoder.layers.12.self_attn.k_proj",
             "text_encoder_2.text_model.encoder.layers.12.self_attn.q_proj",
             "text_encoder_2.text_model.encoder.layers.12.self_attn.v_proj"],
        )
        # One rank each, not a merged triple-rank tensor.
        for pair in pairs:
            self.assertEqual(pair.down.shape, (16, 1280))
            self.assertEqual(pair.up.shape, (1280, 16))

    def test_feed_forward_renames(self):
        tensors = {}
        for offset, (width, rows) in ((8, (1280, 5120)), (9, (5120, 1280))):
            d = f"__te2__text_model__[{self._tid(offset)}]__down__"
            u = f"__te2__text_model__[{self._tid(offset)}]__up__"
            tensors[d] = down(16, width)
            tensors[u] = up(rows, 16)
        pairs = convert.build_pairs(FakeStore(tensors), "sdxl")
        self.assertEqual(
            sorted(p.base for p in pairs),
            ["text_encoder_2.text_model.encoder.layers.12.mlp.fc1",
             "text_encoder_2.text_model.encoder.layers.12.mlp.fc2"],
        )

    def test_no_te2_key_mentions_open_clip_spelling(self):
        tensors = {}
        for offset in (3, 4, 5, 6, 8, 9):
            d = f"__te2__text_model__[{self._tid(offset)}]__down__"
            u = f"__te2__text_model__[{self._tid(offset)}]__up__"
            tensors[d] = down(16, 1280)
            tensors[u] = up(1280, 16)
        for pair in convert.build_pairs(FakeStore(tensors), "sdxl"):
            self.assertNotIn("resblocks", pair.base)
            self.assertNotIn("c_fc", pair.base)
            self.assertNotIn("c_proj", pair.base)
            self.assertNotIn("in_proj", pair.base)

    def test_unet_fusion_still_merges(self):
        # The unet rewrite must not regress: `ff.net.0.proj` is a genuinely
        # fused SDXL weight and still merges two slots into one pair.
        store = FakeStore({
            "__unet__[t-33-0]__down__": down(16, 640),
            "__unet__[t-33-0]__up__": up(2560, 16),
            "__unet__[t-34-0]__down__": down(16, 640),
            "__unet__[t-34-0]__up__": up(2560, 16),
        })
        pairs = convert.build_pairs(store, "sdxl")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].down.shape, (16, 640))
        self.assertEqual(pairs[0].up.shape, (5120, 16))


if __name__ == "__main__":
    unittest.main()