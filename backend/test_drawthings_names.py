"""Tests for Draw Things tensor-name parsing and mapping."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import drawthings_names as names  # noqa: E402


class ParseNameTests(unittest.TestCase):
    """Pins the DT internal name grammar and its refusals."""

    def test_parses_an_up_tensor(self):
        t = names.parse_name("__unet__[t-25-0]__up__")
        self.assertEqual(
            (t.prefix, t.scope, t.layer, t.param, t.part),
            ("__unet__", "", 25, 0, "up"),
        )

    def test_parses_a_text_encoder_two_tensor(self):
        t = names.parse_name("__te2__text_model__[t-8-0]__down__")
        self.assertEqual(
            (t.prefix, t.layer, t.part), ("__te2__text_model__", 8, "down")
        )

    def test_parses_a_flux_scoped_tensor(self):
        t = names.parse_name("__dit__[t-x_v-9-1]")
        self.assertEqual(
            (t.prefix, t.scope, t.layer, t.param, t.part),
            ("__dit__", "x_v", 9, 1, "weight"),
        )

    def test_bare_model_tensor_has_no_lora_part(self):
        t = names.parse_name("__unet__[t-25-0]")
        self.assertEqual(t.part, "weight")
        self.assertFalse(t.is_lora)

    def test_param_one_means_bias(self):
        self.assertEqual(names.parse_name("__unet__[t-28-1]__up__").param, 1)

    def test_down_is_parsed_as_a_part(self):
        self.assertEqual(names.parse_name("__unet__[t-25-0]__down__").part, "down")

    def test_rejects_unknown_prefix(self):
        # __te3__ does not exist in Draw Things.
        with self.assertRaises(names.NameMappingError):
            names.parse_name("__te3__[t-1-0]__up__")

    def test_rejects_loha_parts(self):
        for part in ("w1_a", "w1_b", "w2_a", "w2_b"):
            with self.subTest(part=part):
                with self.assertRaises(names.NameMappingError):
                    names.parse_name(f"__unet__[t-25-0]__{part}__")

    def test_rejects_malformed_names(self):
        for bad in ("", "t-25-0__up__", "__unet__[t-x]__up__", "__unet__[t-25]__up__"):
            with self.subTest(bad=bad):
                with self.assertRaises(names.NameMappingError):
                    names.parse_name(bad)

    def test_refusal_message_names_the_tensor(self):
        # The refusal message is the only debugging tool available here.
        with self.assertRaises(names.NameMappingError) as cm:
            names.parse_name("__nope__[t-1-0]__up__")
        self.assertIn("__nope__[t-1-0]__up__", str(cm.exception))

    def test_accepts_every_known_prefix(self):
        for prefix in names.KNOWN_PREFIXES:
            with self.subTest(prefix=prefix):
                self.assertIsNotNone(
                    names.parse_name(f"{prefix}[t-1-0]__up__")
                )


class HarvestedMapTests(unittest.TestCase):
    """Pins rows transcribed from ModelMapping.swift at revision 060f99d."""

    @classmethod
    def setUpClass(cls):
        cls.tables = names.load_name_map()

    def test_unetxlbase_row(self):
        row = self.tables["UNetXLBase"][
            "model.diffusion_model.input_blocks.4.1."
            "transformer_blocks.0.attn1.to_q.weight"
        ]
        self.assertEqual(row, ["t-25-0"])

    def test_fused_ff_net_0_proj_has_two_entries(self):
        row = self.tables["UNetXLBase"][
            "model.diffusion_model.input_blocks.4.1."
            "transformer_blocks.0.ff.net.0.proj.weight"
        ]
        self.assertEqual(row, ["t-34-0", "t-33-0"])

    def test_clip_text_model_layer_zero_fc2(self):
        row = self.tables["CLIPTextModel"][
            "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc2.weight"
        ]
        self.assertEqual(row, ["t-9-0"])

    def test_open_clip_g_resblock_zero_c_fc(self):
        row = self.tables["OpenCLIPTextModelG"][
            "cond_stage_model.model.transformer.resblocks.0.mlp.c_fc.weight"
        ]
        self.assertEqual(row, ["t-8-0"])

    def test_required_tables_are_present(self):
        for table in ("UNetXLBase", "UNetXLBaseFixed", "UNet",
                      "UNetXLRefiner", "UNetXLRefinerFixed",
                      "CLIPTextModel", "OpenCLIPTextModel", "OpenCLIPTextModelG"):
            with self.subTest(table=table):
                self.assertIn(table, self.tables)
                self.assertGreater(len(self.tables[table]), 0)

    def test_every_value_is_a_scoped_name(self):
        for table, rows in self.tables.items():
            for key, ids in rows.items():
                self.assertTrue(ids, f"{table}/{key} has no ids")
                for i in ids:
                    self.assertRegex(i, r"^t-(?:[^-]+-)?\d+-\d+$", f"{table}/{key} -> {i}")

    def test_no_duplicate_id_within_a_table(self):
        # A duplicate would make the reverse map ambiguous.
        for table, rows in self.tables.items():
            seen = {}
            for key, ids in rows.items():
                for i in ids:
                    self.assertNotIn(
                        i, seen, f"{table}: {i} claimed by both "
                                 f"{seen.get(i)} and {key}"
                    )
                    seen[i] = key


class HarvestRowCountTests(unittest.TestCase):
    """Row counts are the pin. A spot check cannot see a dropped row.

    Every count here was derived by hand from revision 060f99d, not read back
    from the harvester's own output -- a pin written from the generator proves
    nothing. The text tables are the ones that regressed: a regex capturing one
    id per assignment lost the 64 fused `attn.in_proj_{weight,bias}` rows of
    `OpenCLIPTextModelG`, and a regex reading only the dictionary-literal form
    lost the 4 rows per text table that sit outside the `for` loop.

    Loop rows: layers x assignments per layer, plus the 4 rows outside the loop
    (position_embedding, token_embedding, final_layer_norm weight and bias).
    """

    ROW_COUNTS = {
        "CLIPTextModel": (12, 16),
        "OpenCLIPTextModel": (23, 16),
        "OpenCLIPTextModelG": (32, 12),
        "OpenCLIPTextModelGTransformers": (32, 16),
    }

    # The four rows outside each table's loop, spelled as the table spells them:
    # OpenCLIPTextModelG is the open_clip layout and uses positional_embedding /
    # ln_final where the transformers-layout tables use position_embedding /
    # final_layer_norm.
    OUTSIDE_LOOP = {
        "CLIPTextModel": ("embeddings.position_embedding.weight",
                          "embeddings.token_embedding.weight",
                          "final_layer_norm.weight", "final_layer_norm.bias"),
        "OpenCLIPTextModel": ("embeddings.position_embedding.weight",
                              "embeddings.token_embedding.weight",
                              "final_layer_norm.weight", "final_layer_norm.bias"),
        "OpenCLIPTextModelG": ("positional_embedding", "token_embedding.weight",
                               "ln_final.weight", "ln_final.bias"),
        "OpenCLIPTextModelGTransformers": ("embeddings.position_embedding.weight",
                                          "embeddings.token_embedding.weight",
                                          "final_layer_norm.weight",
                                          "final_layer_norm.bias"),
    }

    EXACT_ROWS = {
        "UNet": 686,
        "UNetXLBase": 1540,
        "UNetXLBaseFixed": 140,
        "UNetXLRefiner": 1132,
        "UNetXLRefinerFixed": 88,
    }

    @classmethod
    def setUpClass(cls):
        cls.tables = names.load_name_map()

    def test_text_table_row_counts(self):
        for table, (layers, per_layer) in self.ROW_COUNTS.items():
            with self.subTest(table=table):
                self.assertEqual(len(self.tables[table]), layers * per_layer + 4)

    def test_unet_table_row_counts(self):
        for table, expected in self.EXACT_ROWS.items():
            with self.subTest(table=table):
                self.assertEqual(len(self.tables[table]), expected)

    def test_text_tables_carry_their_four_rows_outside_the_loop(self):
        for table, suffixes in self.OUTSIDE_LOOP.items():
            rows = self.tables[table]
            for suffix in suffixes:
                with self.subTest(table=table, suffix=suffix):
                    hits = [k for k in rows if k.endswith(suffix)]
                    self.assertEqual(len(hits), 1, f"{table}: {suffix} -> {hits}")

    def test_open_clip_g_carries_every_fused_attn_row(self):
        # 32 resblocks x (in_proj_weight, in_proj_bias), three ids each.
        rows = self.tables["OpenCLIPTextModelG"]
        for layer in range(32):
            for leaf in ("attn.in_proj_weight", "attn.in_proj_bias"):
                key = f"cond_stage_model.model.transformer.resblocks.{layer}.{leaf}"
                with self.subTest(layer=layer, leaf=leaf):
                    self.assertEqual(
                        rows[key],
                        [f"t-{layer * 8 + 3}-{rows[key][0][-1]}",
                         f"t-{layer * 8 + 4}-{rows[key][0][-1]}",
                         f"t-{layer * 8 + 5}-{rows[key][0][-1]}"],
                    )

    def test_the_fused_ids_that_broke_the_real_lora_resolve(self):
        # The 192 unresolved tensors in add_detail_xl_lora_f16.ckpt were all
        # __te2__ ids t-99/t-100/t-101, resblock 12's fused QKV.
        for layer, want in ((99, "q"), (100, "k"), (101, "v")):
            with self.subTest(layer=layer):
                self.assertEqual(
                    names.resolve_sd("__te2__text_model__", layer, 0),
                    f"text_model.encoder.layers.12.self_attn.{want}_proj")

    def test_the_fused_bias_ids_are_refused(self):
        # Same three slots at param 1. `load_multilora` only attaches a LoRA to
        # a Linear, and a bias has no separate module to target there, so this
        # is refused rather than pointed at the weight. Every tensor in a real
        # SDXL LoRA is param 0, so this costs nothing.
        for layer in (99, 100, 101):
            with self.subTest(layer=layer):
                self.assertIsNone(
                    names.resolve_sd("__te2__text_model__", layer, 1))


class Te2LayoutTests(unittest.TestCase):
    """`__te2__text_model__` targets HF CLIPTextModelWithProjection, not open_clip.

    Draw Things' `OpenCLIPTextModelG` names open_clip's `CLIPTransformer`:
    `transformer.resblocks.N.attn.in_proj_weight`, `mlp.c_fc`, `mlp.c_proj`. The
    SDXL engine's `text_encoder_2` is a HuggingFace
    `CLIPTextModelWithProjection`, which has none of those -- it has
    `text_model.encoder.layers.N.self_attn.{q,k,v,out}_proj` and
    `mlp.fc{1,2}`. Emitting DT's spelling produces a file the engine cannot
    load at all, so the rename is not cosmetic.

    The fused `in_proj_weight` is the interesting case: one DT weight of
    (3840, 1280) against three separate (1280, 1280) Linears. The three Draw
    Things slots are three *independent* LoRAs -- measured, their down matrices
    all differ -- so each slot is its own target and nothing has to be split
    after the fact.
    """

    LAYER = 12

    def _id(self, offset):
        return self.LAYER * 8 + offset

    def test_fused_qkv_slots_map_to_the_three_separate_projections(self):
        self.assertEqual(
            names.resolve_sd("__te2__text_model__", self._id(3), 0),
            "text_model.encoder.layers.12.self_attn.q_proj")
        self.assertEqual(
            names.resolve_sd("__te2__text_model__", self._id(4), 0),
            "text_model.encoder.layers.12.self_attn.k_proj")
        self.assertEqual(
            names.resolve_sd("__te2__text_model__", self._id(5), 0),
            "text_model.encoder.layers.12.self_attn.v_proj")

    def test_attention_output_projection(self):
        self.assertEqual(
            names.resolve_sd("__te2__text_model__", self._id(6), 0),
            "text_model.encoder.layers.12.self_attn.out_proj")

    def test_feed_forward_uses_hf_names(self):
        # open_clip calls these c_fc / c_proj; HF calls them fc1 / fc2.
        self.assertEqual(
            names.resolve_sd("__te2__text_model__", self._id(8), 0),
            "text_model.encoder.layers.12.mlp.fc1")
        self.assertEqual(
            names.resolve_sd("__te2__text_model__", self._id(9), 0),
            "text_model.encoder.layers.12.mlp.fc2")

    def test_every_te2_linear_target_uses_the_hf_layout(self):
        for layer in range(32):
            for offset in (3, 4, 5, 6, 8, 9):
                path = names.resolve_sd("__te2__text_model__", layer * 8 + offset, 0)
                with self.subTest(layer=layer, offset=offset):
                    self.assertTrue(
                        path.startswith(f"text_model.encoder.layers.{layer}."),
                        f"t-{layer * 8 + offset}-0 -> {path}")
                    self.assertNotIn("resblocks", path)
                    self.assertNotIn("transformer.", path)

    def test_te2_layer_norms_are_refused(self):
        # ln_1 / ln_2 are LayerNorms. The engine only attaches a LoRA to a
        # Linear, so there is no module to target. Refuse rather than emit a
        # key that fails at load time.
        for offset in (2, 7):
            with self.subTest(offset=offset):
                self.assertIsNone(
                    names.resolve_sd("__te2__text_model__", self._id(offset), 0))

    def test_te2_out_of_range_layer_is_refused(self):
        self.assertIsNone(names.resolve_sd("__te2__text_model__", 32 * 8 + 3, 0))

    def test_text_encoder_one_keeps_the_hf_layout(self):
        # `__text_model__` resolves through plain `kohya_to_diffusers`, whose paths are
        # relative to `text_model` -- the standard HF CLIP root. The prefix is added
        # here, exactly as it is for te2. See ComponentRootTests.
        self.assertEqual(
            names.resolve_sd("__text_model__", 9, 0),
            "text_model.encoder.layers.0.mlp.fc2")


class ComponentRootTests(unittest.TestCase):
    """Both text encoders are rooted at `text_model`, and that was once wrong.

    `mlx_diffuser`'s SDXL pipeline loads BOTH encoders as `CLIPTextModel`, and both
    root at `text_model.encoder.layers.N...`. Measured against the real
    `juggernaut-xl-lightning` pipeline:

        pipe.text_encoder.named_modules()     -> text_model.encoder.layers.0.mlp.fc1
        pipe.text_encoder_2.named_modules()   -> text_model.encoder.layers.0.mlp.fc1

    This class previously asserted the opposite -- that `text_encoder` has no
    `text_model.` prefix and `text_encoder_2` does -- on the premise that
    `CLIPTextModel` "has no `text_model` attribute". That premise is false. `text_model`
    is the standard HF CLIP layout; what makes `CLIPTextModelWithProjection` different
    is its `text_projection` weight. In this pipeline the projection is on
    `text_encoder` and *absent* from `text_encoder_2`, which is the opposite of what the
    old comment claimed in every particular.

    The wrong prefix survived a unit test AND a stub-pipe gate, because the stub's module
    tree was built from the very keys the converter emitted: a wrong key agreed with
    itself. It surfaced only in the real pipeline, as
    `RuntimeError: Failed loading LoRA: LoRA target text_encoder:encoder.layers.0.mlp.fc1
    is unavailable`. These tests now assert the shared root, and the class is named for
    what is true rather than for what was believed.
    """

    def test_text_encoder_one_is_rooted_at_text_model(self):
        for layer in range(12):
            path = names.resolve_sd("__text_model__", layer * 8 + 9, 0)
            with self.subTest(layer=layer):
                self.assertTrue(path.startswith(f"text_model.encoder.layers.{layer}."), path)

    def test_text_encoder_two_is_rooted_at_text_model(self):
        for layer in range(32):
            path = names.resolve_sd("__te2__text_model__", layer * 8 + 8, 0)
            with self.subTest(layer=layer):
                self.assertTrue(
                    path.startswith(f"text_model.encoder.layers.{layer}."), path)

    def test_both_encoders_share_one_root(self):
        """They are the same class with the same root; a difference here is the bug."""
        one = names.resolve_sd("__text_model__", 8, 0)
        two = names.resolve_sd("__te2__text_model__", 8, 0)
        self.assertEqual(one.split("encoder", 1)[0], two.split("encoder", 1)[0])
        self.assertEqual(one.split("encoder", 1)[0], "text_model.")


class GroupFusedTests(unittest.TestCase):
    """Pins the split direction: Draw Things' Format.O splits axis 0.

    A fused row carries several internal ids for one diffusers weight. The
    position inside the row's id list is what decides concatenation order
    downstream, so it has to survive the trip through a dict -- JSON is written
    with sorted keys, so table order cannot be what carries the sequence.
    """

    @classmethod
    def setUpClass(cls):
        cls.tables = names.load_name_map()
        cls.groups = names.group_fused(cls.tables)

    def _base(self, key, table="UNetXLBase"):
        return self.groups[table][key]

    def test_single_tensor_row_has_one_entry(self):
        key = ("model.diffusion_model.input_blocks.4.1."
               "transformer_blocks.0.attn1.to_q.weight")
        self.assertEqual(self._base(key), [(0, "t-25-0")])

    def test_fused_row_keeps_table_order(self):
        key = ("model.diffusion_model.input_blocks.4.1."
               "transformer_blocks.0.ff.net.0.proj.weight")
        self.assertEqual(self._base(key), [(0, "t-34-0"), (1, "t-33-0")])

    def test_positions_are_dense_and_ordered(self):
        # Every group, in the whole harvested map.
        for table, rows in self.groups.items():
            for key, entries in rows.items():
                with self.subTest(table=table, key=key):
                    positions = [p for p, _ in entries]
                    self.assertEqual(positions, list(range(len(entries))))

    def test_weight_and_bias_of_one_linear_never_merge(self):
        # `attn1.to_out.0` has both a weight and a bias row, each with its own
        # id. Grouping by the normalised module path would fuse them into one
        # group and hand the caller a bias where it expected weight.
        weight = ("model.diffusion_model.input_blocks.4.1."
                  "transformer_blocks.0.attn1.to_out.0.weight")
        bias = ("model.diffusion_model.input_blocks.4.1."
                "transformer_blocks.0.attn1.to_out.0.bias")
        self.assertEqual(self._base(weight), [(0, "t-28-0")])
        self.assertEqual(self._base(bias), [(0, "t-28-1")])

    def test_every_row_has_a_group(self):
        for table, rows in self.tables.items():
            for key, ids in rows.items():
                with self.subTest(table=table, key=key):
                    self.assertEqual(
                        [(i, tid) for i, tid in enumerate(ids)],
                        self.groups[table][key],
                    )

    def test_the_same_key_in_two_tables_keeps_both_ids(self):
        # CLIPTextModel and OpenCLIPTextModel both spell final_layer_norm.bias
        # and give it different ids. A flat map would have to drop one.
        key = "cond_stage_model.transformer.text_model.final_layer_norm.bias"
        self.assertEqual(self._base(key, "CLIPTextModel"), [(0, "t-98-1")])
        self.assertEqual(self._base(key, "OpenCLIPTextModel"), [(0, "t-186-1")])

    def test_fused_te2_attn_rows_carry_three_positions(self):
        # The rows that were dropped outright before the harvest fix. Each is
        # q, k and v concatenated along axis 0.
        for layer in range(32):
            for leaf in ("attn.in_proj_weight", "attn.in_proj_bias"):
                key = (f"cond_stage_model.model.transformer.resblocks."
                       f"{layer}.{leaf}")
                base = layer * 8 + 3
                param = 0 if leaf.endswith("weight") else 1
                with self.subTest(layer=layer, leaf=leaf):
                    self.assertEqual(
                        self._base(key, "OpenCLIPTextModelG"),
                        [(0, f"t-{base}-{param}"),
                         (1, f"t-{base + 1}-{param}"),
                         (2, f"t-{base + 2}-{param}")],
                    )

    def test_an_empty_table_yields_no_groups(self):
        self.assertEqual(names.group_fused({"UNetXLBase": {}}), {"UNetXLBase": {}})


class ResolveSDTests(unittest.TestCase):
    """Pins SD/SDXL resolution and the reverse-map collision guard."""

    def test_resolves_attn1_to_q_to_a_diffusers_path(self):
        self.assertEqual(
            names.resolve_sd("__unet__", 25, 0),
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q",
        )

    def test_resolves_clip_mlp_fc2_layer_zero(self):
        # Rooted at `text_model`, like every other HF CLIP component and like te2.
        self.assertEqual(
            names.resolve_sd("__text_model__", 9, 0),
            "text_model.encoder.layers.0.mlp.fc2",
        )

    def test_resolves_te2_feed_forward_to_the_hf_layout(self):
        # open_clip's `mlp.c_fc` is HF's `mlp.fc1`. Emitting Draw Things' own
        # spelling here produced a file the SDXL engine could not load.
        self.assertEqual(
            names.resolve_sd("__te2__text_model__", 8, 0),
            "text_model.encoder.layers.0.mlp.fc1",
        )

    def test_every_attention_bearing_unet_key_resolves(self):
        """Every LoRA-targetable UNet key must map to a real diffusers path.

        kohya resnet leaves (in_layers.0, out_layers.0, conv_shortcut) have no
        diffusers counterpart -- diffusers names them conv1/conv2/conv_shortcut
        -- and are Conv2d, which a LoRA cannot target. input_blocks.0.0 is the
        stem conv. Those are excluded; anything with an attention, ff, norm or
        time_emb_proj leaf must resolve.
        """
        targets = ("attn", "transformer_blocks", "ff.net", "proj_in", "proj_out",
                   "time_emb_proj", "norm")
        unresolved = []
        for key in names.load_name_map()["UNetXLBase"]:
            path = names.normalise_key(key)
            leaf = key.rsplit(".", 1)[0]
            if path.endswith("input_blocks.0.0"):
                continue
            if not any(t in leaf for t in targets):
                continue
            if names.kohya_to_diffusers(path) is None:
                unresolved.append(path)
        self.assertEqual(unresolved, [], f"{len(unresolved)} LoRA targets unmapped")

    def test_every_linear_bearing_unet_id_resolves(self):
        """Every UNetXLBase id whose leaf is a Linear must resolve.

        And conversely: an id that does not resolve must have a leaf with no
        Linear behind it. Asserted on the leaf rather than on a count of ids,
        because the ids are not the invariant -- `norm1` at 140 ids and a bare
        ModuleList index at 356 are both "no Linear here", and a new Conv2d row
        in a future Draw Things revision must fail this too without anyone
        editing a magic number.
        """
        table = names.load_name_map()["UNetXLBase"]
        reverse = names.build_reverse_map({"UNetXLBase": table})
        wrongly_refused = []
        wrongly_resolved = []
        for entry, normalised in reverse.items():
            _table, layer, param = entry.split("::", 1)[1].split("-")
            path = names.resolve_sd("__unet__", int(layer), int(param))
            mapped = names.kohya_to_diffusers(normalised)
            is_linear_leaf = mapped is not None and names._has_linear(mapped)
            if is_linear_leaf and path is None:
                wrongly_refused.append((entry, mapped))
            if not is_linear_leaf and path is not None:
                wrongly_resolved.append((entry, normalised))
        self.assertEqual(wrongly_refused[:8], [],
                         f"{len(wrongly_refused)} Linear ids refused")
        self.assertEqual(wrongly_resolved[:8], [],
                         f"{len(wrongly_resolved)} non-Linear ids resolved")

    def test_non_linear_unet_rows_are_refused(self):
        """LayerNorms, Conv2d indices and skip_connection have no Linear.

        Checked against real keys rather than a list of leaf names, because the
        leaf cannot decide: `attn1.to_out.0` is a Linear and `in_layers.0` is a
        Conv2d, and both normalise to the leaf `0`.
        """
        refused = ("down_blocks.0.resnets.0.norm1",
                   "down_blocks.0.resnets.0.norm2",
                   "down_blocks.0.resnets.0.conv_shortcut",
                   "down_blocks.0.resnets.0.skip_connection",
                   "down_blocks.0.resnets.0.conv1")
        for path in refused:
            with self.subTest(path=path):
                self.assertFalse(names._has_linear(path))
        linear = ("down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0",
                  "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q",
                  "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj",
                  "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2",
                  "down_blocks.1.attentions.0.proj_out",
                  "text_model.encoder.layers.0.mlp.fc2")
        for path in linear:
            with self.subTest(path=path):
                self.assertTrue(names._has_linear(path))

    def test_the_stem_conv_still_has_no_target(self):
        # `input_blocks.0.0` is the SDXL stem Conv2d. It sits at `down_blocks.0`
        # as a whole with no resnet/attention sub-slot, so no derived block entry
        # can name it, and the allowlist refuses it for the same reason.
        self.assertIsNone(names.resolve_sd("__unet__", 4, 0))
        self.assertIsNone(names.resolve_sd("__unet__", 4, 1))

    def test_text_encoder_non_linear_rows_are_refused(self):
        for offset in (2, 7):          # layer_norm1, layer_norm2
            with self.subTest(offset=offset):
                self.assertIsNone(names.resolve_sd("__text_model__", offset, 0))
        # final_layer_norm, position_embedding, token_embedding
        self.assertIsNone(names.resolve_sd("__text_model__", 98, 0))
        self.assertIsNone(names.resolve_sd("__text_model__", 0, 0))
        self.assertIsNone(names.resolve_sd("__text_model__", 1, 0))

    def test_middle_block_maps_to_attentions(self):
        # Draw Things writes middle_block; diffusers writes mid_block.
        self.assertEqual(
            names.kohya_to_diffusers("middle_block.1.attn1.to_q.weight"),
            "mid_block.attentions.0.attn1.to_q.weight",
        )

    def test_unknown_index_returns_none(self):
        self.assertIsNone(names.resolve_sd("__unet__", 999999, 0))
        self.assertIsNone(names.kohya_to_diffusers("input_blocks.99.1"))

    def test_passes_through_a_path_with_no_block_prefix(self):
        self.assertEqual(
            names.kohya_to_diffusers("time_embed.0"),
            "time_embed.0",
        )

    def test_collision_in_reverse_map_raises(self):
        # Review Focus #2: two diffusers keys claiming one t-N-M is a data bug
        # in the harvest, never a last-wins overwrite.
        tables = {"T": {"a.x.weight": ["t-1-0"], "b.x.weight": ["t-1-0"]}}
        with self.assertRaises(names.NameMappingError):
            names.build_reverse_map(tables)

    def test_aliases_normalising_to_one_path_do_not_collide(self):
        # .weight and .bias normalise to the same module path, so they are the
        # same target, not an ambiguity.
        tables = {"T": {"a.x.weight": ["t-1-0"], "a.x.bias": ["t-1-0"]}}
        rev = names.build_reverse_map(tables)
        self.assertEqual(rev["T::t-1-0"], "a.x")

    def test_reverse_map_keys_are_qualified_by_table(self):
        rev = names.build_reverse_map(names.load_name_map())
        self.assertIn("UNetXLBase::t-25-0", rev)
        self.assertIn("CLIPTextModel::t-9-0", rev)

    def test_real_tables_have_no_ambiguous_ids(self):
        names.build_reverse_map(names.load_name_map())


if __name__ == "__main__":
    unittest.main()