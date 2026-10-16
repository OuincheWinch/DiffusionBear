"""Task 6: family dispatch for custom ids.

Hermetic: synthetic registry entries (temp-file registry, Task 2 precedent)
plus fake weight dirs holding one empty .safetensors. Builders stubbed at
the constructor boundary via sys.modules fakes that record kwargs — no
weights, no processes, no network.

(builder, config, path) triples:
- custom flux2 4b -> Flux2Klein, no model_config, entry dir
- custom flux2 9b -> Flux2Klein, 9B ModelConfig, entry dir
- custom flux2 edit variant -> Flux2KleinEdit, entry dir
- custom zit -> ZImageTurbo, entry dir
- custom krea -> Krea2 (quantize 4), entry dir
- custom qwen base/turbo -> subprocess model path is the entry dir,
  variant selects the canonical defaults (25 vs 8 steps)
- custom sdxl -> daemon directory is the entry dir

Vanished directories fail with the existing friendly validation error,
never a traceback: flux via validate_local_model_dir ("path does not
exist"), sdxl via sdxl_layout_report ("SDXL cannot load this model"),
qwen before any subprocess is spawned.
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import custom_models
import generator
import state

CALLS = []


def _recorder(name):
    def __init__(self, *args, **kwargs):
        CALLS.append((name, dict(kwargs)))

    return type(f"Fake{name}", (), {"__init__": __init__})


def _fake_module(name, **attrs):
    module = types.ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _FakeModelConfig:
    @staticmethod
    def flux2_klein_9b():
        return "CFG9-SENTINEL"


FAKE_MODULES = {
    "mflux": _fake_module("mflux"),
    "mflux.models": _fake_module("mflux.models"),
    "mflux.models.flux2": _fake_module("mflux.models.flux2", Flux2Klein=_recorder("Flux2Klein")),
    "mflux.models.flux2.variants": _fake_module("mflux.models.flux2.variants"),
    "mflux.models.flux2.variants.edit": _fake_module("mflux.models.flux2.variants.edit"),
    "mflux.models.flux2.variants.edit.flux2_klein_edit": _fake_module(
        "mflux.models.flux2.variants.edit.flux2_klein_edit",
        Flux2KleinEdit=_recorder("Flux2KleinEdit"),
    ),
    "mflux.models.z_image": _fake_module(
        "mflux.models.z_image", ZImageTurbo=_recorder("ZImageTurbo")
    ),
    "mflux.models.krea2": _fake_module("mflux.models.krea2", Krea2=_recorder("Krea2")),
    "mflux.models.common": _fake_module("mflux.models.common"),
    "mflux.models.common.config": _fake_module("mflux.models.common.config"),
    "mflux.models.common.config.model_config": _fake_module(
        "mflux.models.common.config.model_config", ModelConfig=_FakeModelConfig
    ),
}


def _entry(model_id, family, directory, variant=None):
    entry = {
        "id": model_id,
        "label": f"Test {model_id}",
        "family": family,
        "source": "local",
        "local_path": str(directory),
    }
    if variant is not None:
        entry["variant"] = variant
    return entry


class CustomDispatchTests(unittest.TestCase):
    def setUp(self):
        self.saved_file = custom_models.CUSTOM_MODELS_FILE
        self.saved_cache = custom_models._custom_models_cache
        self.temp = tempfile.TemporaryDirectory(dir=state.DATA_DIR)
        custom_models._custom_models_cache = {}
        custom_models.CUSTOM_MODELS_FILE = Path(self.temp.name) / "custom_models.json"
        self.dirs = {}
        for name in ("flux4", "flux9", "fluxedit", "zit", "krea", "qwenbase", "qwenturbo", "sdxl"):
            directory = Path(self.temp.name) / f"weights-{name}"
            directory.mkdir(parents=True)
            (directory / "weights.safetensors").write_bytes(b"")
            self.dirs[name] = directory
        custom_models.write_custom_models(
            [
                _entry("t-flux4", "flux2", self.dirs["flux4"], variant="4b"),
                _entry("t-flux9", "flux2", self.dirs["flux9"], variant="9b"),
                _entry("t-fluxedit", "flux2", self.dirs["fluxedit"], variant="4b"),
                _entry("t-zit", "zit", self.dirs["zit"]),
                _entry("t-krea", "krea", self.dirs["krea"]),
                _entry("t-qwenbase", "qwen", self.dirs["qwenbase"], variant="base"),
                _entry("t-qwenturbo", "qwen", self.dirs["qwenturbo"], variant="turbo"),
                _entry("t-sdxl", "sdxl", self.dirs["sdxl"]),
            ]
        )
        self.modules_patch = mock.patch.dict(sys.modules, FAKE_MODULES)
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        for helper in (
            "_install_zit_dynamic_padding",
            "_install_krea_sampler_scalar_cache",
            "_apply_krea_vae_tiling",
        ):
            patcher = mock.patch.object(generator, helper, lambda *a, **k: None)
            patcher.start()
            self.addCleanup(patcher.stop)
        CALLS.clear()
        self.saved_pipeline = generator._pipeline
        self.saved_key = generator._current_pipeline_key
        self.saved_model = generator._current_pipeline_model
        generator._pipeline = None
        generator._current_pipeline_key = None
        generator._current_pipeline_model = None

    def tearDown(self):
        generator._pipeline = self.saved_pipeline
        generator._current_pipeline_key = self.saved_key
        generator._current_pipeline_model = self.saved_model
        custom_models.CUSTOM_MODELS_FILE = self.saved_file
        custom_models._custom_models_cache = self.saved_cache
        self.temp.cleanup()
        CALLS.clear()

    def _recorded(self, name):
        matches = [kwargs for builder, kwargs in CALLS if builder == name]
        self.assertEqual(len(matches), 1, f"expected one {name} build, got {CALLS}")
        return matches[0]

    def test_flux4_uses_default_config_and_entry_path(self):
        generator._get_pipeline("t-flux4", 4, [])
        kwargs = self._recorded("Flux2Klein")
        self.assertNotIn("model_config", kwargs)
        self.assertEqual(kwargs["model_path"], str(self.dirs["flux4"]))

    def test_flux9_uses_9b_config_and_entry_path(self):
        generator._get_pipeline("t-flux9", 4, [])
        kwargs = self._recorded("Flux2Klein")
        self.assertEqual(kwargs["model_config"], "CFG9-SENTINEL")
        self.assertEqual(kwargs["model_path"], str(self.dirs["flux9"]))

    def test_flux_edit_variant_uses_edit_builder(self):
        generator._get_pipeline("t-fluxedit", 4, [], variant="edit")
        kwargs = self._recorded("Flux2KleinEdit")
        self.assertEqual(kwargs["model_path"], str(self.dirs["fluxedit"]))

    def test_zit_uses_entry_path(self):
        generator._get_pipeline("t-zit", 4, [])
        kwargs = self._recorded("ZImageTurbo")
        self.assertEqual(kwargs["model_path"], str(self.dirs["zit"]))

    def test_krea_uses_entry_path_at_4bit(self):
        generator._get_pipeline("t-krea", 4, [])
        kwargs = self._recorded("Krea2")
        self.assertEqual(kwargs["model_path"], str(self.dirs["krea"]))
        self.assertEqual(kwargs["quantize"], 4)

    def test_qwen_entry_dir_selects_subprocess_path_and_variant_defaults(self):
        base = generator.get_model_info("t-qwenbase")
        turbo = generator.get_model_info("t-qwenturbo")
        self.assertEqual(generator._qwen_local_model(base), str(self.dirs["qwenbase"]))
        self.assertEqual(generator._qwen_local_model(turbo), str(self.dirs["qwenturbo"]))
        self.assertEqual(base["default_steps"], 25)
        self.assertEqual(turbo["default_steps"], 8)

    def test_sdxl_daemon_dir_is_entry_dir(self):
        info = generator.get_model_info("t-sdxl")
        self.assertEqual(
            generator._resolve_sdxl_model_dir("t-sdxl", info), str(self.dirs["sdxl"])
        )

    def test_vanished_flux_dir_fails_friendly(self):
        target = self.dirs["flux4"]
        for child in target.iterdir():
            child.unlink()
        target.rmdir()
        with self.assertRaises(ValueError) as ctx:
            generator._get_pipeline("t-flux4", 4, [])
        self.assertIn("does not exist", str(ctx.exception))
        self.assertEqual(CALLS, [])

    def test_vanished_sdxl_dir_fails_friendly_without_spawn(self):
        target = self.dirs["sdxl"]
        for child in target.iterdir():
            child.unlink()
        target.rmdir()
        with mock.patch.object(
            generator, "_get_sdxl_daemon", side_effect=AssertionError("must not spawn")
        ):
            with self.assertRaises(ValueError) as ctx:
                generator._generate_sdxl(
                    prompt="a test",
                    width=512,
                    height=512,
                    steps=4,
                    guidance=1.0,
                    seed=1,
                    loras=[],
                    progress_cb=None,
                    model="t-sdxl",
                    cancel_event=None,
                    negative_prompt="",
                    sampler=None,
                )
        self.assertIn("SDXL cannot load this model", str(ctx.exception))

    def test_vanished_qwen_dir_fails_friendly_without_spawn(self):
        target = self.dirs["qwenturbo"]
        for child in target.iterdir():
            child.unlink()
        target.rmdir()
        with mock.patch.object(
            generator.subprocess,
            "Popen",
            side_effect=AssertionError("must not spawn"),
        ):
            with self.assertRaises(ValueError) as ctx:
                generator._generate_qwen_subprocess(
                    prompt="a test",
                    width=512,
                    height=512,
                    steps=4,
                    guidance=1.0,
                    seed=1,
                    model="t-qwenturbo",
                )
        self.assertIn("does not exist", str(ctx.exception))


    def test_custom_model_resolves_local_model_path(self):
        self.assertEqual(generator.resolve_local_model_path("t-sdxl"), self.dirs["sdxl"])
        self.assertEqual(generator.resolve_local_model_path("t-flux4"), self.dirs["flux4"])

    def test_custom_model_is_sdxl_engine(self):
        from sdxl_layout import is_sdxl_engine

        self.assertTrue(is_sdxl_engine("t-sdxl"))
        self.assertFalse(is_sdxl_engine("t-flux4"))

    def test_custom_model_prepare_request_in_jobs(self):
        from routers.jobs import _prepare_request

        req = _prepare_request({"prompt": "a test", "model": "t-sdxl"})
        self.assertEqual(req.model, "t-sdxl")

    def test_custom_model_lora_compatibility_not_unknown_model(self):
        import lora_compat

        res = lora_compat.check(self.dirs["sdxl"] / "weights.safetensors", "t-sdxl")
        self.assertNotEqual(res.get("reason"), "unknown_model")
        self.assertEqual(res.get("reason"), "unsupported")

    def test_custom_model_list_local_sources_succeeds(self):
        from routers.models import list_local_sources

        with mock.patch("generator.list_local_model_sources", return_value=[]):
            res = list_local_sources("t-sdxl")
            self.assertEqual(res, {"sources": []})

    def test_custom_sdxl_not_distilled_allows_50_steps(self):
        info = generator.get_model_info("t-sdxl")
        self.assertFalse(info["is_distilled"])
        self.assertIsNone(info.get("max_steps"))
        self.assertEqual(generator._effective_steps(info, 30), 30)
        self.assertEqual(generator._effective_steps(info, 50), 50)

    def test_custom_sdxl_explicit_distilled_honored(self):
        distilled_entry = _entry("t-sdxl-dist", "sdxl", self.dirs["sdxl"])
        distilled_entry["is_distilled"] = True
        custom_models.write_custom_models(custom_models.read_custom_models() + [distilled_entry])
        info = generator.get_model_info("t-sdxl-dist")
        self.assertTrue(info["is_distilled"])
        self.assertEqual(generator._effective_steps(info, 30), generator._DISTILLED_MAX_STEPS)


if __name__ == "__main__":
    unittest.main()

