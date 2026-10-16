"""A LoRA that does not fit the model must say so BEFORE the generation runs.

What this pins down, from a real failure on 2026-10-13:

    44 of the 144 layers named by Logo.Redmond__Logo_Lora__FLUX.2.Klein... could not
    be applied (single_transformer_blocks.23.attn.to_out, ... and 39 more)

That is the FLUX.2 Klein **9B** LoRA loaded into the **4B** model. Both sizes are
FLUX.2 Klein and both use `single_transformer_blocks`, so nothing about the block NAMES
gives it away -- which is exactly why the first version of this check was wrong. It
compared block families, found zero overlap between the adapter's spelling and the
model's, and called the LoRA 0% compatible on the 4B *and* the 9B. Shipping that would
have refused a perfectly good LoRA on the one model it actually works with.

What separates them is block COUNT, and only the real matcher sees that: against the
4B, 100 of 144 layers resolve; against the 9B, all 144 do. `test_block_count_decides`
is that distinction, kept as a test because it is the one a future rewrite throws away.

The guard against re-introducing a hand-rolled comparison is
`test_resolution_goes_through_the_loader`: a stub vocabulary agrees with a stub
adapter by construction, which is how a check can pass while being wrong. Only mflux's
own matcher cannot.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import generator
import lora_compat
import state
from routers import loras as loras_router

# Spellings taken from mflux's Flux2LoRAMapping, not invented: each pair below is a
# real (adapter key -> transformer module) mapping, verified against the live matcher.
#
#   double_blocks.N.img_attn.qkv     -> transformer_blocks.N.attn.to_{q,k,v}
#   double_blocks.N.img_attn.proj    -> transformer_blocks.N.attn.to_out
#   double_blocks.N.img_mlp.0        -> transformer_blocks.N.ff.linear_in
#   single_blocks.N.linear1          -> single_transformer_blocks.N.attn.to_qkv_mlp_proj
#   single_blocks.N.linear2          -> single_transformer_blocks.N.attn.to_out
KEY_QKV = "diffusion_model.double_blocks.{b}.img_attn.qkv"
KEY_PROJ = "diffusion_model.double_blocks.{b}.img_attn.proj"
KEY_MLP = "diffusion_model.double_blocks.{b}.img_mlp.0"
KEY_LIN1 = "diffusion_model.single_blocks.{b}.linear1"
KEY_LIN2 = "diffusion_model.single_blocks.{b}.linear2"

MODEL_PATHS_ONE_BLOCK = {
    "transformer_blocks.0.attn.to_q",
    "transformer_blocks.0.attn.to_k",
    "transformer_blocks.0.attn.to_v",
    "transformer_blocks.0.attn.to_out",
    "transformer_blocks.0.ff.linear_in",
    "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
    "single_transformer_blocks.0.attn.to_out",
}


def module_paths(double_blocks: int, single_blocks: int) -> set[str]:
    paths = set()
    for b in range(double_blocks):
        paths |= {
            f"transformer_blocks.{b}.attn.to_q",
            f"transformer_blocks.{b}.attn.to_k",
            f"transformer_blocks.{b}.attn.to_v",
            f"transformer_blocks.{b}.attn.to_out",
            f"transformer_blocks.{b}.ff.linear_in",
        }
    for b in range(single_blocks):
        paths |= {
            f"single_transformer_blocks.{b}.attn.to_qkv_mlp_proj",
            f"single_transformer_blocks.{b}.attn.to_out",
        }
    return paths


def write_lora(directory: Path, templates, blocks) -> Path:
    """A safetensors file carrying lora_A/lora_B for each (template, block)."""
    from safetensors.torch import save_file
    import torch

    tensors = {}
    for template in templates:
        for block in blocks:
            base = template.format(b=block)
            tensors[f"{base}.lora_A.weight"] = torch.zeros(1, 1)
            tensors[f"{base}.lora_B.weight"] = torch.zeros(1, 1)
    path = Path(directory) / "adapter.safetensors"
    save_file(tensors, str(path))
    return path


class TestCompatibilityVerdict(unittest.TestCase):
    def check_with(self, templates, adapter_blocks, model_double, model_single):
        with tempfile.TemporaryDirectory() as d:
            lora = write_lora(Path(d), templates, adapter_blocks)
            paths = module_paths(model_double, model_single)
            with mock.patch.object(lora_compat, "model_module_paths", return_value=paths):
                return lora_compat.check(lora, "flux2-klein-4b")

    def test_adapter_covering_the_model_is_compatible(self):
        result = self.check_with(
            [KEY_QKV, KEY_PROJ, KEY_MLP, KEY_LIN1, KEY_LIN2],
            adapter_blocks=[0], model_double=1, model_single=1)
        self.assertTrue(result["compatible"], result["message"])
        self.assertEqual(result["reason"], "ok")
        self.assertEqual(result["missing_families"], {})

    def test_block_count_decides(self):
        """The case a family comparison cannot see, and the reason this file exists.

        An adapter trained on a DEEPER model names block indices the shallower model
        does not have, and every one of those is a target mflux tries and fails to
        apply. This is the 9B-LoRA-on-4B case: identical block families, identical
        spelling, different depth, opposite verdicts.

        The direction matters and is not symmetric. An adapter covering FEWER blocks
        than the model has is normal -- it simply adapts less -- and mflux applies it
        without complaint. Only a too-deep adapter fails. Asserting the shallow case
        would have invented a restriction the loader does not have.
        """
        deep_adapter = [KEY_LIN1, KEY_LIN2]

        fits = self.check_with(deep_adapter, adapter_blocks=[0, 1, 2, 3, 4, 5],
                               model_double=0, model_single=6)
        self.assertTrue(fits["compatible"], fits["message"])

        too_deep = self.check_with(deep_adapter, adapter_blocks=[0, 1, 2, 3, 4, 5],
                                   model_double=0, model_single=2)
        self.assertFalse(too_deep["compatible"])
        self.assertEqual(too_deep["reason"], "architecture_mismatch")
        # Blocks 2-5 of both modules: 4 blocks x 2 modules.
        self.assertEqual(too_deep["missing_families"], {"single_transformer_blocks": 8})
        # The message names the model, not the block anatomy: the counts stay
        # in missing_families for debugging, out of the user's face.
        self.assertEqual(too_deep["fits_models"], [])
        self.assertIn("4B", too_deep["message"])
        self.assertNotIn("single_transformer_blocks", too_deep["message"])
        # The old message's shape: a bare total nobody can act on.
        self.assertNotIn(" of 16 layers named by", too_deep["message"])

    def test_shallower_adapter_is_not_rejected(self):
        """Not a restriction mflux has: adapting fewer blocks than exist is fine."""
        result = self.check_with([KEY_LIN1, KEY_LIN2], adapter_blocks=[0, 1],
                                 model_double=0, model_single=6)
        self.assertTrue(result["compatible"], result["message"])

    def test_wrong_family_is_reported_by_name(self):
        """An SDXL-shaped adapter offered to a FLUX.2 model."""
        result = self.check_with(
            [KEY_MLP], adapter_blocks=[0], model_double=0, model_single=4)
        self.assertFalse(result["compatible"])
        self.assertEqual(result["reason"], "architecture_mismatch")
        self.assertEqual(result["missing_families"], {"transformer_blocks": 1})

    def test_completely_unrecognised_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            from safetensors.torch import save_file
            import torch
            lora = Path(d) / "a.safetensors"
            save_file({"totally.unrelated.key": torch.zeros(1, 1)}, str(lora))
            with mock.patch.object(lora_compat, "model_module_paths",
                                   return_value=module_paths(1, 1)):
                result = lora_compat.check(lora, "flux2-klein-4b")
        self.assertFalse(result["compatible"])
        self.assertEqual(result["reason"], "no_recognised_keys")

    def test_unverifiable_is_never_reported_compatible(self):
        """A missing index must not become a silent pass."""
        with tempfile.TemporaryDirectory() as d:
            lora = write_lora(Path(d), [KEY_LIN1], [0])
            with mock.patch.object(lora_compat, "model_module_paths", return_value=set()):
                result = lora_compat.check(lora, "flux2-klein-4b")
        self.assertIsNone(result["compatible"])
        self.assertTrue(result["unknown"])
        self.assertEqual(result["reason"], "weights_missing")

    def test_unreadable_adapter_is_not_compatible(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "not-a-safetensors.safetensors"
            bad.write_bytes(b"this is not a safetensors file")
            with mock.patch.object(lora_compat, "model_module_paths",
                                   return_value=module_paths(1, 1)):
                result = lora_compat.check(bad, "flux2-klein-4b")
        self.assertIsNone(result["compatible"])
        self.assertEqual(result["reason"], "unreadable")

    def test_sdxl_model_is_reported_as_unchecked_not_compatible(self):
        """mflux is not what loads SDXL adapters, so a verdict here would be a lie."""
        with mock.patch.dict(generator.MODELS,
                             {"juggernaut-xl-lightning": {
                                 "id": "juggernaut-xl-lightning", "repo": "x",
                                 "supports_loras": True}}, clear=False):
            result = lora_compat.check("/nonexistent.safetensors",
                                       "juggernaut-xl-lightning")
        self.assertIsNone(result["compatible"])
        self.assertEqual(result["reason"], "unsupported")

    def test_model_without_lora_support(self):
        entry = {"id": "no-lora", "repo": "x", "supports_loras": False}
        with mock.patch.dict(generator.MODELS, {"no-lora": entry}, clear=False):
            with mock.patch.object(lora_compat, "mapping_for", return_value=object()):
                result = lora_compat.check("/nonexistent.safetensors", "no-lora")
        self.assertIsNone(result["compatible"])
        self.assertEqual(result["reason"], "no_lora_support")


class TestMessageShape(unittest.TestCase):
    def test_missing_paths_group_by_family_with_counts(self):
        grouped = lora_compat.group_by_family({
            "single_transformer_blocks.0.attn.to_out",
            "single_transformer_blocks.1.attn.to_out",
            "single_transformer_blocks.2.attn.to_out",
            "transformer_blocks.0.attn.to_q",
        })
        self.assertEqual(grouped, {"single_transformer_blocks": 3, "transformer_blocks": 1})

    def test_path_without_a_block_family_is_labelled_other(self):
        self.assertEqual(lora_compat.group_by_family({"guidance_embed.linear1"}),
                         {"other": 1})


class TestResolutionGoesThroughTheLoader(unittest.TestCase):
    """Guard against this drifting back to a hand-rolled comparison.

    A stub vocabulary agrees with a stub adapter by construction, which is how a check
    can pass while being wrong. Only mflux's real matcher cannot agree with itself.
    """

    def test_uses_the_loaders_own_matcher(self):
        source = Path(lora_compat.__file__).read_text(encoding="utf-8")
        self.assertIn("LoRALoader._build_pattern_mappings", source)
        self.assertIn("LoRALoader._match_pattern", source)

    def test_every_declared_prefix_resolves_to_a_real_mapping(self):
        """A typo'd import path would silently downgrade every verdict to unknown."""
        for prefix in lora_compat._MAPPING_CLASSES:
            mapping = lora_compat.mapping_for(f"{prefix}-anything")
            self.assertIsNotNone(mapping, f"no LoRA mapping resolved for {prefix}")
            self.assertTrue(mapping.get_mapping())


class TestEndpoint(unittest.TestCase):
    """The route the UI actually calls."""

    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routers import loras as loras_router

        app = FastAPI()
        app.include_router(loras_router.router)
        return TestClient(app)

    def test_route_is_registered_at_the_path_the_frontend_calls(self):
        # This FastAPI version does not flatten include_router into app.routes, so
        # iterating directly finds 2 routes and reports every call as broken -- the
        # exact cry-wolf failure documented in test_frontend_backend_wiring._walk.
        from fastapi.routing import APIRoute
        import main

        def walk(routes):
            out = []
            for route in routes:
                if isinstance(route, APIRoute):
                    out.append(route)
                original = getattr(route, "original_router", None)
                if original is not None and hasattr(original, "routes"):
                    out.extend(walk(original.routes))
            return out

        paths = {r.path for r in walk(main.app.routes)}
        self.assertGreater(len(paths), 50, "route walker missed nested routers")
        self.assertIn("/api/loras/{name}/compatibility", paths)

    def test_missing_lora_is_404_not_a_verdict(self):
        """A 404 must not read as compatible. It is the case a UI is most tempted to
        treat as 'no problem found'."""
        with mock.patch.object(loras_router, "_read_loras", return_value=[]):
            response = self.client().get(
                "/api/loras/nope/compatibility?model=flux2-klein-4b")
        self.assertEqual(response.status_code, 404)

    def test_entry_whose_file_vanished_is_400(self):
        entries = [{"name": "ghost", "path": "/nonexistent/gone.safetensors"}]
        with mock.patch.object(loras_router, "_read_loras", return_value=entries):
            response = self.client().get(
                "/api/loras/ghost/compatibility?model=flux2-klein-4b")
        self.assertEqual(response.status_code, 400)

    def test_incompatible_adapter_is_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            lora = write_lora(Path(d), [KEY_LIN1, KEY_LIN2], [0, 1, 2, 3])
            entries = [{"name": "deep", "path": str(lora)}]
            with mock.patch.object(loras_router, "_read_loras", return_value=entries):
                with mock.patch.object(lora_compat, "model_module_paths",
                                       return_value=frozenset(module_paths(0, 2))):
                    response = self.client().get(
                        "/api/loras/deep/compatibility?model=flux2-klein-4b")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["compatible"])
        self.assertEqual(body["reason"], "architecture_mismatch")
        self.assertEqual(body["fits_models"], [])
        self.assertNotIn("single_transformer_blocks", body["message"])


class TestCaching(unittest.TestCase):
    """Both inputs are cached, so polling the picker does not re-read a 165 MB header."""

    def test_module_paths_are_cached(self):
        lora_compat.model_module_paths.cache_clear()
        with mock.patch.object(lora_compat, "_find_weight_index",
                               return_value=None) as finder:
            lora_compat.model_module_paths("flux2-klein-4b")
            lora_compat.model_module_paths("flux2-klein-4b")
            lora_compat.model_module_paths("flux2-klein-4b")
        self.assertEqual(finder.call_count, 1)

    def test_cached_paths_cannot_be_mutated_by_a_caller(self):
        """lru_cache hands out the SAME object. A mutable set returned to a caller that
        then adds to it would corrupt every later answer for the session."""
        lora_compat.model_module_paths.cache_clear()
        with mock.patch.object(lora_compat, "_find_weight_index", return_value=None):
            first = lora_compat.model_module_paths("flux2-klein-9b")
        self.assertIsInstance(first, frozenset)
        with self.assertRaises(AttributeError):
            first.add("single_transformer_blocks.0")


class TestSizeMismatch(unittest.TestCase):
    """When the adapter fits another size of the same family, name it.

    The banner used to lecture about block families and layer counts: "It
    targets block types this model does not have: transformer_blocks
    (36 layers)... 100 of 144 layers resolve". The ask is one sentence
    naming the mismatch instead: this LoRA fits the 9B, not the 4B.

    Fixtures reuse this file's real key spellings against the real Flux2
    matcher; only the model indexes are stood in, per model id, so the
    sibling probe has two different depths to compare.
    """

    def check_against(self, templates, adapter_blocks, paths_by_model):
        with tempfile.TemporaryDirectory() as d:
            lora = write_lora(Path(d), templates, adapter_blocks)
            with mock.patch.object(
                lora_compat, "model_module_paths",
                side_effect=lambda mid: paths_by_model.get(mid, frozenset()),
            ):
                return lora_compat.check(lora, "flux2-klein-4b")

    def test_names_the_fitting_sibling_size(self):
        result = self.check_against(
            [KEY_QKV, KEY_PROJ, KEY_MLP], [0, 1],
            {"flux2-klein-4b": module_paths(1, 0),
             "flux2-klein-9b": module_paths(2, 0)})
        self.assertFalse(result["compatible"])
        self.assertEqual(result["reason"], "size_mismatch")
        self.assertEqual(result["fits_models"], ["flux2-klein-9b"])
        self.assertIn("9B", result["message"])
        self.assertIn("4B", result["message"])
        self.assertNotIn("transformer_blocks", result["message"])

    def test_no_fitting_sibling_gives_a_short_generic_message(self):
        result = self.check_against(
            [KEY_QKV], [5],
            {"flux2-klein-4b": module_paths(1, 0),
             "flux2-klein-9b": module_paths(2, 0)})
        self.assertFalse(result["compatible"])
        self.assertEqual(result["reason"], "architecture_mismatch")
        self.assertEqual(result["fits_models"], [])
        self.assertLess(len(result["message"]), 120)
        self.assertNotIn("transformer_blocks", result["message"])


class TestRealAdapterAgainstRealWeights(unittest.TestCase):
    """The reported case end to end, against the user's actual files.

    Skipped when either the model weights or the adapter are absent, so it never
    invents a pass: absent files leave the test skipped, never green.
    """

    LORA = str(Path(state.ASSET_DIR) / "lora_files" /
            "Logo.Redmond_-_Logo_Lora__FLUX.2.Klein_Logo_Redmond.safetensors")

    def setUp(self):
        # The caching tests above mock _find_weight_index and leave an EMPTY result in
        # the module cache for these same model ids. Without this the real-file test
        # sees an empty set, concludes the weights are absent, and SKIPS -- which is a
        # green suite reporting that nothing was checked. Same class of leak as the
        # Path.home patch recorded in AGENTS.md: restore global state, and clear the
        # cache before AND after mocking it.
        lora_compat.model_module_paths.cache_clear()
        self.addCleanup(lora_compat.model_module_paths.cache_clear)
        try:
            with open(self.LORA, "rb") as f:
                f.read(8)
        except (OSError, PermissionError):
            self.skipTest("adapter not present or readable on this machine")
        if not lora_compat.model_module_paths("flux2-klein-4b"):
            self.skipTest("FLUX.2 Klein weights not in the local HF cache")

    def test_9b_adapter_is_rejected_by_4b_and_accepted_by_9b(self):
        four_b = lora_compat.check(self.LORA, "flux2-klein-4b")
        nine_b = lora_compat.check(self.LORA, "flux2-klein-9b")
        self.assertFalse(four_b["compatible"], four_b["message"])
        self.assertTrue(nine_b["compatible"], nine_b["message"])
        # The loader reported exactly 44 failures out of 144 on the 4B.
        self.assertEqual(four_b["total_layers"] - four_b["resolved_layers"], 44)


if __name__ == "__main__":
    unittest.main()