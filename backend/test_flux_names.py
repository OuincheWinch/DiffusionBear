"""Tests for the FLUX scope rules in drawthings_names.resolve_flux.

Everything here is derived from two sources and nothing else:

- Draw Things' own `Flux1.swift` at revision 060f99d -- the `LoRAFlux1` /
  `LoRASingleTransformerBlock` / `LoRADualTransformerBlock` mappers, which say
  what each `x_*` / `c_*` scope actually computes.
- mflux's own module tree (`models/flux/model/flux_transformer/*.py`) and its
  `FluxLoRAMapping`, which say what the converted file has to be called.

No scope meaning is inferred from its spelling. `x_o` and `x_out_proj` are the
two cases where the obvious reading is wrong and the arithmetic is what settles
it -- see `test_single_block_projections_merge_into_proj_out`.
"""

import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import drawthings_names as names  # noqa: E402


class FluxScopeTests(unittest.TestCase):
    """Pins each scope to the mflux module path it must produce."""

    def _path(self, scope, layer, param=0):
        target = names.resolve_flux(scope, layer, param)
        self.assertIsNotNone(target, f"{scope} @ {layer} did not resolve")
        return target.module_path

    def test_dual_attention_projections(self):
        self.assertEqual(self._path("x_q", 9), "transformer_blocks.9.attn.to_q")
        self.assertEqual(self._path("x_k", 9), "transformer_blocks.9.attn.to_k")
        self.assertEqual(self._path("x_v", 9), "transformer_blocks.9.attn.to_v")
        self.assertEqual(self._path("x_o", 9), "transformer_blocks.9.attn.to_out.0")

    def test_dual_feed_forward(self):
        self.assertEqual(self._path("x_linear1", 3), "transformer_blocks.3.ff.linear1")
        self.assertEqual(self._path("x_out_proj", 3), "transformer_blocks.3.ff.linear2")

    def test_context_stream_maps_to_the_added_projections(self):
        # `c_*` is the text/context stream, which in diffusers is the "add"
        # attention on the dual blocks.
        self.assertEqual(self._path("c_q", 3), "transformer_blocks.3.attn.add_q_proj")
        self.assertEqual(self._path("c_k", 3), "transformer_blocks.3.attn.add_k_proj")
        self.assertEqual(self._path("c_v", 3), "transformer_blocks.3.attn.add_v_proj")
        self.assertEqual(self._path("c_o", 3), "transformer_blocks.3.attn.to_add_out")
        self.assertEqual(self._path("c_linear1", 3), "transformer_blocks.3.ff_context.linear1")
        self.assertEqual(self._path("c_out_proj", 3), "transformer_blocks.3.ff_context.linear2")

    def test_single_block_attention_projections(self):
        self.assertEqual(
            self._path("x_q", 19), "single_transformer_blocks.0.attn.to_q")
        self.assertEqual(
            self._path("x_v", 56), "single_transformer_blocks.37.attn.to_v")

    def test_single_block_feed_forward_first_layer(self):
        self.assertEqual(
            self._path("x_linear1", 19), "single_transformer_blocks.0.proj_mlp")

    def test_context_stream_has_no_single_block_counterpart(self):
        # Single blocks attend to the image sequence only; there is no context
        # stream to add. Must refuse rather than fall back to the dual path.
        for scope in ("c_q", "c_k", "c_v", "c_o", "c_linear1", "c_out_proj",
                      "context_ada_ln_0"):
            with self.subTest(scope=scope):
                self.assertIsNone(names.resolve_flux(scope, 30, 0))

    def test_single_block_projections_merge_into_proj_out(self):
        # The one place the obvious reading is wrong.
        #
        # mflux's SingleTransformerBlock does
        #     proj_out(concat([attn_output, mlp_hidden]))   # Linear(15360, 3072)
        # DT does
        #     out = x_o(attn_out) + xFF(xOut)               # 3072->3072 and 12288->3072
        # so mflux's proj_out weight is literally [ x_o | x_out_proj ]. They are
        # two LoRAs that must become one, not two separate targets -- mflux has
        # no single-block to_out at all.
        attn = names.resolve_flux("x_o", 19, 0)
        mlp = names.resolve_flux("x_out_proj", 19, 0)
        self.assertEqual(attn.module_path, "single_transformer_blocks.0.proj_out")
        self.assertEqual(mlp.module_path, "single_transformer_blocks.0.proj_out")
        self.assertTrue(attn.merged)
        self.assertTrue(mlp.merged)
        # Both chunks share the same output rows, so the merge is a sum: `up`
        # concatenates on its rank axis and `down` on its rank axis too.
        self.assertEqual(attn.merge_axis, 1)
        self.assertEqual(mlp.merge_axis, 1)
        self.assertEqual((attn.slice_index, attn.slice_count), (0, 2))
        self.assertEqual((mlp.slice_index, mlp.slice_count), (1, 2))

    def test_dual_x_o_and_x_out_proj_stay_separate(self):
        # On the dual blocks they really are two different Linears.
        attn = names.resolve_flux("x_o", 9, 0)
        mlp = names.resolve_flux("x_out_proj", 9, 0)
        self.assertFalse(attn.merged)
        self.assertFalse(mlp.merged)
        self.assertNotEqual(attn.module_path, mlp.module_path)


class FluxAdaLNormTests(unittest.TestCase):
    """Pins the AdaLN chunk merge.

    DT's dual blocks split their modulation into `(0..<6)` chunks and its single
    blocks into `(0..<3)` (Flux1.swift lines 789 and 612). mflux's
    `AdaLayerNormZero.linear` is `Linear(3072, 18432)` = 6 x 3072 and
    `AdaLayerNormZeroSingle.linear` is `Linear(dim, 3 * dim)` = 3 x 3072. The
    chunk counts match exactly, so the chunks are one weight split along its
    output rows.
    """

    def test_dual_has_six_norm1_slices_of_which_three_are_addressable(self):
        # mflux's norm1.linear is Linear(3072, 18432): six 3072-wide chunks.
        # DT splits its norm1 modulation three ways and its norm2 three ways,
        # so only three of the six have data. The other three are zeros, which
        # is what a LoRA that did not train them means -- not a guess.
        paths = {names.resolve_flux(f"x_ada_ln_{i}", 9, 0).module_path for i in range(3)}
        self.assertEqual(paths, {"transformer_blocks.9.norm1.linear"})
        target = names.resolve_flux("x_ada_ln_0", 9, 0)
        self.assertTrue(target.merged)
        # Chunks stack along the weight's output rows.
        self.assertEqual(target.merge_axis, 0)
        self.assertEqual(target.slice_count, 6)
        for i in range(3):
            with self.subTest(chunk=i):
                self.assertEqual(names.resolve_flux(f"x_ada_ln_{i}", 9, 0).slice_index, i)

    def test_single_has_three_norm_chunks(self):
        for i in range(3):
            with self.subTest(chunk=i):
                target = names.resolve_flux(f"x_ada_ln_{i}", 19, 0)
                self.assertEqual(target.module_path,
                                 "single_transformer_blocks.0.norm.linear")
                self.assertEqual((target.slice_index, target.slice_count), (i, 3))
                self.assertEqual(target.merge_axis, 0)

    def test_dual_norm2_chunks_are_refused(self):
        # DT's x_ada_ln_3/4/5 feed norm2, the feed-forward modulation. mflux's
        # joint block has norm2 as a non-affine LayerNorm with no Linear, so
        # there is no module to attach a LoRA to. Must refuse, not guess.
        for i in (3, 4, 5):
            with self.subTest(chunk=i):
                self.assertIsNone(names.resolve_flux(f"x_ada_ln_{i}", 9, 0))

    def test_dual_context_ada_ln_chunks(self):
        for i in range(6):
            with self.subTest(chunk=i):
                target = names.resolve_flux(f"context_ada_ln_{i}", 9, 0)
                self.assertEqual(target.module_path,
                                 "transformer_blocks.9.norm1_context.linear")
                self.assertEqual((target.slice_index, target.slice_count), (i, 6))


class FluxRefusalTests(unittest.TestCase):
    """The resolver must refuse rather than guess."""

    def test_unknown_scope_returns_none(self):
        self.assertIsNone(names.resolve_flux("nonsense", 1, 0))
        self.assertIsNone(names.resolve_flux("x_w1", 1, 0))  # a rank-32 variant scope
        self.assertIsNone(names.resolve_flux("t_embedder_0", 0, 0))

    def test_param_one_is_refused(self):
        # mflux's LoRA targets have no bias member, so a bias tensor has nowhere
        # to go. Every tensor measured in a real FLUX.1 LoRA is param == 0.
        self.assertIsNone(names.resolve_flux("x_q", 9, 1))

    def test_layer_outside_the_model_returns_none(self):
        self.assertIsNone(names.resolve_flux("x_q", -1, 0))
        self.assertIsNone(names.resolve_flux("x_q", 57, 0))
        self.assertIsNone(names.resolve_flux("x_q", 999, 0))

    def test_no_fuzzy_fallback(self):
        # `x_attn_to_q` is not `x_q`. Matching must be exact.
        self.assertIsNone(names.resolve_flux("x_attn_to_q", 9, 0))
        self.assertIsNone(names.resolve_flux("X_Q", 9, 0))
        self.assertIsNone(names.resolve_flux(" x_q", 9, 0))


class FluxCoverageTests(unittest.TestCase):
    """Every scope in a real FLUX.1 LoRA must resolve, and to a real shape.

    Measured from `bitcoin_lora_flux1d_f16_lora_f16.ckpt`: 24 scopes, 798
    up/down pairs, layers 0..56. This is the standing rule -- a row count, not
    a spot check.
    """

    OBSERVED_SCOPES = {
        "x_q", "x_k", "x_v", "x_o", "x_linear1", "x_out_proj",
        "x_ada_ln_0", "x_ada_ln_1", "x_ada_ln_2",
        "x_ada_ln_3", "x_ada_ln_4", "x_ada_ln_5",
        "c_q", "c_k", "c_v", "c_o", "c_linear1", "c_out_proj",
        "context_ada_ln_0", "context_ada_ln_1", "context_ada_ln_2",
        "context_ada_ln_3", "context_ada_ln_4", "context_ada_ln_5",
    }
    # x_ada_ln_3/4/5 exist only because DT has a norm2 the target has not.
    UNADDRESSABLE = {"x_ada_ln_3", "x_ada_ln_4", "x_ada_ln_5"}

    DUAL_LAYERS = tuple(range(0, 19))
    SINGLE_LAYERS = tuple(range(19, 57))
    ALL_LAYERS = DUAL_LAYERS + SINGLE_LAYERS

    def test_the_observed_scope_set_is_exactly_what_the_table_covers(self):
        self.assertEqual(set(names.FLUX_SCOPES) | self.UNADDRESSABLE,
                         self.OBSERVED_SCOPES)
        self.assertEqual(set(names.FLUX_SCOPES) & self.UNADDRESSABLE, set())

    def _layers_for(self, scope):
        """Layers at which `scope` actually appears in the reference file."""
        if scope in self.UNADDRESSABLE:
            return self.DUAL_LAYERS
        if scope.startswith("c"):
            return self.DUAL_LAYERS
        return self.ALL_LAYERS

    def test_every_scope_resolves_at_the_layers_it_occurs_at(self):
        unmapped = []
        for scope in sorted(self.OBSERVED_SCOPES - self.UNADDRESSABLE):
            for layer in self._layers_for(scope):
                if names.resolve_flux(scope, layer, 0) is None:
                    unmapped.append((scope, layer))
        self.assertEqual(unmapped, [], f"{len(unmapped)} scope/layer pairs unmapped")

    def test_pair_count_covered(self):
        # 798 pairs in the reference file. Everything except the dual-only
        # norm2 chunks must have a destination.
        pairs = sum(len(self._layers_for(s)) for s in self.OBSERVED_SCOPES)
        self.assertEqual(pairs, 798)

    def test_addressable_pair_count(self):
        blocked = sum(len(self._layers_for(s))
                      for s in self.OBSERVED_SCOPES if s in self.UNADDRESSABLE)
        self.assertEqual(blocked, 57)

    def test_every_dual_target_is_a_real_mflux_module_path(self):
        expected = {
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
            "attn.to_add_out", "ff.linear1", "ff.linear2",
            "ff_context.linear1", "ff_context.linear2",
            "norm1.linear", "norm1_context.linear",
        }
        seen = set()
        for scope in sorted(self.OBSERVED_SCOPES):
            target = names.resolve_flux(scope, 0, 0)
            if target is not None:
                seen.add(target.module_path.split(".", 2)[2])
        self.assertTrue(seen <= expected, f"unexpected dual targets: {seen - expected}")
        self.assertEqual(seen, expected)


class FluxSingleTargetTests(unittest.TestCase):
    """Every single-block target must be a real mflux single-block module."""

    def test_single_targets(self):
        expected = {
            "attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp", "proj_out",
            "norm.linear",
        }
        seen = set()
        for scope in sorted(FluxCoverageTests.OBSERVED_SCOPES):
            target = names.resolve_flux(scope, 19, 0)
            if target is not None:
                seen.add(target.module_path.split(".", 2)[2])
        self.assertEqual(seen, expected)


if __name__ == "__main__":
    unittest.main()