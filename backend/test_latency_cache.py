import json
import os
import re
import sys
import time
import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import generator
from fastapi import HTTPException

import routers.gallery as gallery
import state

# CI runs on ubuntu without MLX, and the local venv has it, so anything that
# actually calls into MLX has to be skipped rather than imported eagerly. Same
# guard as test_upscale.py.
try:
    import mlx.core  # noqa: F401

    _MLX_AVAILABLE = True
except ModuleNotFoundError:
    _MLX_AVAILABLE = False

# Mirrors the auto-injection predicate in generator._generate_sdxl: a 4-step
# Lightning LoRA is silently appended to any SDXL run that is short enough or
# uses a trailing sampler. Applied to a non-distilled model's *default* config
# it means a 4-step distill adapter running at 25 steps.
def _injects_lightning_lora(minfo: dict, steps: int, sampler: str) -> bool:
    if minfo.get("is_distilled"):
        return False
    return steps <= 4 or sampler in ("euler_trailing", "trailing")


class DefaultSamplerTests(unittest.TestCase):
    def test_default_sampler_is_supported_for_every_model(self):
        for model_id, minfo in generator.MODELS.items():
            sampler = generator._validate_sampler(None, minfo)
            self.assertIn(sampler, minfo.get("samplers") or [sampler], f"{model_id}")

    def test_juggernaut_xi_default_sampler_matches_its_photo_preset(self):
        minfo = generator.MODELS["juggernaut-xi"]
        photo = next(p for p in minfo["presets"] if p["id"] == "photo")
        self.assertEqual(generator._validate_sampler(None, minfo), photo["sampler"])

    def test_default_config_never_injects_lightning_lora(self):
        # Regression guard: juggernaut-xi used to default to euler_trailing with
        # 25 steps, which made generator._generate_sdxl append
        # sdxl_lightning_4step_lora to a 25-step / guidance-4.0 run.
        for model_id, minfo in generator.MODELS.items():
            if minfo.get("engine") != "sdxl":
                continue
            steps = minfo["default_steps"]
            sampler = generator._validate_sampler(None, minfo)
            self.assertFalse(
                _injects_lightning_lora(minfo, steps, sampler),
                f"{model_id} default ({steps} steps, {sampler}) triggers Lightning LoRA injection",
            )

    def test_explicit_short_sampler_still_injects(self):
        # The injection itself must keep working: a user asking for 4 steps on a
        # base model is exactly the case it exists for.
        minfo = generator.MODELS["juggernaut-xi"]
        self.assertTrue(_injects_lightning_lora(minfo, 4, "dpmpp_2m_karras"))
        self.assertTrue(_injects_lightning_lora(minfo, 25, "euler_trailing"))


class DistilledGuidanceTests(unittest.TestCase):
    """A guidance-distilled model must not default to guidance > 1.0.

    mlx_diffuser/pipelines/sdxl.py:164 turns on CFG for any guidance above 1.0
    and :212-216 then concatenates the batch to 2 on every step, so guidance
    1.5 costs twice the UNet evaluations per step for no benefit on a model
    distilled at guidance 1.0.

    Measured on realvis-xl-v5-lightning, 512x768, 6 steps, q4, TAESD, warm
    daemon, min of 5 interleaved repeats: 24.34s at g1.5 vs 12.58s at g1.0.
    Side-by-side contact sheets showed no visible quality difference.
    """

    def test_distilled_models_default_to_guidance_one(self):
        offenders = {
            model_id: minfo["default_guidance"]
            for model_id, minfo in generator.MODELS.items()
            if minfo.get("is_distilled")
            and minfo.get("supports_guidance")
            and float(minfo.get("default_guidance") or 1.0) > 1.0
        }
        self.assertEqual(offenders, {}, f"distilled models paying for CFG by default: {offenders}")

    def test_no_preset_reenables_cfg_on_a_distilled_model(self):
        offenders = [
            f"{model_id}/{preset['id']}={preset.get('guidance')}"
            for model_id, minfo in generator.MODELS.items()
            if minfo.get("is_distilled")
            for preset in minfo.get("presets") or []
            if float(preset.get("guidance", 1.0) or 1.0) > 1.0
        ]
        self.assertEqual(offenders, [], f"presets re-enabling CFG on distilled models: {offenders}")

    def test_realvis_lightning_keeps_six_steps(self):
        # 4 steps is ~1.5x faster again but visibly softer: rust speckle and
        # skin micro-texture go mushy. Held at 6 deliberately.
        minfo = generator.MODELS["realvis-xl-v5-lightning"]
        self.assertEqual(minfo["default_steps"], 6)
        self.assertEqual(minfo["default_guidance"], 1.0)


class ImageFileCacheHeaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.previous_dir = gallery.GENERATED_DIR
        self.previous_index = dict(state.GALLERY_INDEX)
        gallery.GENERATED_DIR = self.root
        state.GALLERY_INDEX.clear()
        state._IMAGE_TOMBSTONES.clear()
        self.image_id = "a" * 32
        (self.root / f"{self.image_id}.png").write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-png")

    def tearDown(self):
        gallery.GENERATED_DIR = self.previous_dir
        state.GALLERY_INDEX.clear()
        state.GALLERY_INDEX.update(self.previous_index)
        state._IMAGE_TOMBSTONES.clear()
        self.temp.cleanup()

    def test_original_carries_cache_control(self):
        response = gallery.image_file(self.image_id)
        self.assertIn("Cache-Control", response.headers)
        self.assertIn("max-age=3600", response.headers["Cache-Control"])
        self.assertIn("stale-while-revalidate", response.headers["Cache-Control"])

    def test_cache_does_not_swallow_content_disposition(self):
        response = gallery.image_file(self.image_id)
        self.assertIn("Content-Disposition", response.headers)
        self.assertIn(self.image_id, response.headers["Content-Disposition"])
        self.assertIn("Access-Control-Expose-Headers", response.headers)

    def test_missing_thumb_falls_back_to_original_with_headers(self):
        with patch.object(generator, "thumbnail_path", return_value=self.root / "nope.png"):
            response = gallery.image_file(self.image_id, thumb=True)
        self.assertIn("Cache-Control", response.headers)
        self.assertIn("Content-Disposition", response.headers)

    def test_deleted_image_is_404(self):
        state._mark_image_deleted(self.image_id)
        with self.assertRaises(HTTPException) as ctx:
            gallery.image_file(self.image_id)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_deleted_image_is_404_on_the_thumb_path_too(self):
        state._mark_image_deleted(self.image_id)
        with self.assertRaises(HTTPException) as ctx:
            gallery.image_file(self.image_id, thumb=True)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_unknown_image_is_404(self):
        with self.assertRaises(HTTPException) as ctx:
            gallery.image_file("b" * 32)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_path_traversal_is_rejected(self):
        with self.assertRaises(HTTPException):
            gallery.image_file("../../../etc/passwd")


class StorageStatsCacheTests(unittest.TestCase):
    def setUp(self):
        self.previous_dir = generator.GENERATED_DIR
        self.temp = tempfile.TemporaryDirectory()
        generator.GENERATED_DIR = Path(self.temp.name)
        generator._storage_stats_cache = None

    def tearDown(self):
        generator.GENERATED_DIR = self.previous_dir
        generator._storage_stats_cache = None
        self.temp.cleanup()

    def test_counts_only_image_suffixes(self):
        # Pre-existing behaviour, asserted rather than changed: the scan filters
        # on suffix only, so a generated {id}_thumb.png is counted as an image.
        # The displayed number is a disk gauge, not a count of originals.
        root = Path(self.temp.name)
        (root / "one.png").write_bytes(b"x")
        (root / "two.jpeg").write_bytes(b"yy")
        (root / "three.json").write_text("{}")
        (root / "one_thumb.png").write_bytes(b"z" * 10)
        stats = generator._storage_stats()
        self.assertEqual(stats["image_count"], 3)
        self.assertEqual(stats["image_bytes"], 13)
        self.assertNotIn("error", stats)

    def test_second_call_is_served_from_cache(self):
        (Path(self.temp.name) / "one.png").write_bytes(b"x")
        first = generator._storage_stats()
        (Path(self.temp.name) / "two.png").write_bytes(b"y")
        self.assertEqual(generator._storage_stats(), first)

    def test_cache_expires(self):
        (Path(self.temp.name) / "one.png").write_bytes(b"x")
        self.assertEqual(generator._storage_stats()["image_count"], 1)
        (Path(self.temp.name) / "two.png").write_bytes(b"y")
        generator._storage_stats_cache = (generator._storage_stats_cache[0] - generator._STORAGE_STATS_TTL_S - 1, generator._storage_stats_cache[1])
        self.assertEqual(generator._storage_stats()["image_count"], 2)

    def test_returns_a_copy(self):
        stats = generator._storage_stats()
        stats["image_count"] = -1
        self.assertNotEqual(generator._storage_stats().get("image_count"), -1)


@unittest.skipUnless(_MLX_AVAILABLE, "MLX runtime is unavailable")
class KreaTextEncoderCacheTests(unittest.TestCase):
    """The quantized Krea 2 text encoder is reused across pipeline rebuilds.

    Measured on this M1 16GB with the krea2 4-step distill LoRA attached:
    cold build 95.4s / warm build 41.7s, i.e. 53.7s saved per rebuild. Output is
    bit-identical between the two (0 differing pixels of 1,179,648 at 512x768,
    seed 777) because it is the same quantized module, not a recomputation.
    """

    def setUp(self):
        self.previous = dict(generator._krea_te_cache)
        generator._krea_te_cache = {}
        self.previous_env = os.environ.pop("MLX_DISABLE_KREA_TE_CACHE", None)

    def tearDown(self):
        generator._krea_te_cache = self.previous
        if self.previous_env is not None:
            os.environ["MLX_DISABLE_KREA_TE_CACHE"] = self.previous_env
        else:
            os.environ.pop("MLX_DISABLE_KREA_TE_CACHE", None)

    def _fake_pipe(self, te):
        pipe = type("P", (), {})()
        pipe.text_encoder = te
        return pipe

    @staticmethod
    def _te(name):
        # nn.quantize() walks leaf_modules() and calls update_modules(); identity
        # is the only property these tests care about.
        return type(name, (), {
            "leaf_modules": lambda self: iter(()),
            "update_modules": lambda self, leaves: None,
            "parameters": lambda self: iter(()),
        })()

    def test_second_build_adopts_the_cached_encoder(self):
        first = self._te("TE")
        pipe_a = self._fake_pipe(first)
        generator._krea_quantized_text_encoder(pipe_a, "/models/krea2", bits=4, group_size=64)
        self.assertIs(pipe_a.text_encoder, first)
        self.assertIsNotNone(generator._krea_te_cache.get("module"))

        cached = generator._krea_te_cache["module"]
        fresh = self._te("TE")
        pipe_b = self._fake_pipe(fresh)
        returned = generator._krea_quantized_text_encoder(pipe_b, "/models/krea2", bits=4, group_size=64)
        self.assertIs(returned, cached)
        self.assertIs(pipe_b.text_encoder, cached)
        self.assertIsNot(pipe_b.text_encoder, fresh)

    def test_different_model_path_misses_the_cache(self):
        generator._krea_quantized_text_encoder(self._fake_pipe(self._te("TE")), "/models/a", bits=4, group_size=64)
        cached = generator._krea_te_cache["module"]
        other = self._te("TE")
        pipe = self._fake_pipe(other)
        returned = generator._krea_quantized_text_encoder(pipe, "/models/b", bits=4, group_size=64)
        self.assertIs(returned, other)
        self.assertIsNot(returned, cached)

    def test_different_bits_misses_the_cache(self):
        generator._krea_quantized_text_encoder(self._fake_pipe(self._te("TE")), "/models/a", bits=4, group_size=64)
        cached = generator._krea_te_cache["module"]
        other = self._te("TE")
        returned = generator._krea_quantized_text_encoder(self._fake_pipe(other), "/models/a", bits=8, group_size=64)
        self.assertIs(returned, other)
        self.assertIsNot(returned, cached)

    def test_kill_switch_disables_reuse(self):
        os.environ["MLX_DISABLE_KREA_TE_CACHE"] = "1"
        first = self._te("First")
        returned = generator._krea_quantized_text_encoder(self._fake_pipe(first), "/models/a", bits=4, group_size=64)
        self.assertIs(returned, first)
        self.assertEqual(generator._krea_te_cache, {}, "kill switch must not populate the cache")

        second = self._te("Second")
        returned = generator._krea_quantized_text_encoder(self._fake_pipe(second), "/models/a", bits=4, group_size=64)
        self.assertIs(returned, second, "kill switch must not reuse a previous encoder")
        self.assertIsNot(returned, first)

    def test_clear_releases_the_cache(self):
        generator._krea_quantized_text_encoder(self._fake_pipe(self._te("TE")), "/models/a", bits=4, group_size=64)
        self.assertNotEqual(generator._krea_te_cache, {})
        generator._clear_krea_te_cache()
        self.assertEqual(generator._krea_te_cache, {})

    def test_pipeline_without_text_encoder_is_tolerated(self):
        pipe = type("P", (), {})()
        pipe.text_encoder = None
        self.assertIsNone(generator._krea_quantized_text_encoder(pipe, "/models/a", bits=4, group_size=64))


class QwenDaemonLifecycleTests(unittest.TestCase):
    """The Qwen engine is a resident daemon, not a per-job subprocess.

    Measured: load_time per job went [2.14, 0.0, 0.0] with an unchanged PID,
    and interpreter boot + mflux import (1.6-1.8s, previously paid on every
    single job) is now paid once per daemon lifetime.
    """

    def setUp(self):
        self.saved = (
            generator._qwen_process, generator._qwen_reader_messages,
            generator._qwen_reader_done, generator._qwen_watchdog,
        )
        generator._qwen_process = None
        generator._qwen_reader_messages = None
        generator._qwen_reader_done = None
        generator._qwen_watchdog = None

    def tearDown(self):
        (generator._qwen_process, generator._qwen_reader_messages,
         generator._qwen_reader_done, generator._qwen_watchdog) = self.saved

    def test_no_process_is_not_healthy(self):
        self.assertFalse(generator._qwen_process_is_healthy())

    def test_dead_process_is_not_healthy(self):
        class Dead:
            stdin = object()
            def poll(self):
                return 0
        generator._qwen_process = Dead()
        self.assertFalse(generator._qwen_process_is_healthy())

    def test_live_process_without_readers_is_not_healthy(self):
        class Live:
            stdin = object()
            def poll(self):
                return None
        generator._qwen_process = Live()
        generator._qwen_reader_messages = object()
        generator._qwen_reader_done = threading.Event()  # not set
        self.assertTrue(generator._qwen_process_is_healthy())

        generator._qwen_reader_done.set()
        self.assertFalse(generator._qwen_process_is_healthy(), "a finished reader means EOF")

    def test_watchdog_is_idempotent_and_released_on_kill(self):
        class FakeTimer:
            def __init__(self, *a, **k):
                self.cancelled = False
            def start(self):
                pass
            def is_alive(self):
                return not self.cancelled
            def cancel(self):
                self.cancelled = True

        class Live:
            stdin = object()
            def poll(self):
                return None

        with patch.object(generator, "_qwen_idle_kill_s", return_value=0):
            generator._arm_qwen_watchdog()
            self.assertIsNone(generator._qwen_watchdog, "idle_kill_s=0 must not arm")

        with patch.object(generator, "_qwen_idle_kill_s", return_value=300), \
             patch.object(generator.threading, "Timer", FakeTimer):
            generator._arm_qwen_watchdog()
            self.assertIsNone(generator._qwen_watchdog, "no live process means nothing to arm")

            generator._qwen_process = Live()
            generator._arm_qwen_watchdog()
            self.assertIsNotNone(generator._qwen_watchdog)
            first = generator._qwen_watchdog
            generator._arm_qwen_watchdog()
            self.assertIsNot(generator._qwen_watchdog, first, "re-arm must replace the timer")
            self.assertTrue(first.cancelled)
            generator._cancel_qwen_watchdog()
            self.assertIsNone(generator._qwen_watchdog)
            self.assertIsNone(generator._qwen_idle_since)

    def test_drop_mflux_pipeline_does_not_kill_the_qwen_daemon(self):
        # Regression guard: _generate_qwen_subprocess() calls _drop_mflux_pipeline()
        # on entry, so having the teardown inside it made every Qwen job kill its
        # own daemon and reload the weights. load_time was 2.1-2.9s forever.
        import inspect
        src = inspect.getsource(generator._drop_mflux_pipeline)
        self.assertNotIn("_kill_qwen_process", src)

    def test_idle_setting_is_validated(self):
        import app_settings
        validator = app_settings._VALIDATORS["idle_kill_s_qwen"]
        self.assertTrue(validator(None))
        self.assertTrue(validator(0))
        self.assertTrue(validator(300))
        self.assertFalse(validator(-1))
        self.assertFalse(validator(86401))
        self.assertFalse(validator("300"))


class TransientStatTests(unittest.TestCase):
    """A failed filesystem probe must not be read as a fact about the disk.

    The model store is on the external volume /Volumes/Externe, which returned a
    transient ENOENT from stat() during a benchmark. Each consumer used to treat
    that as truth, at three very different prices.
    """

    def setUp(self):
        generator._lora_sig_cache = {}
        generator._model_cached_cache = None
        self.saved_delay = generator._STAT_RETRY_DELAY_S
        generator._STAT_RETRY_DELAY_S = 0

    def tearDown(self):
        generator._lora_sig_cache = {}
        generator._model_cached_cache = None
        generator._STAT_RETRY_DELAY_S = self.saved_delay

    def test_probe_retries_then_succeeds(self):
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise OSError(2, "No such file or directory")
            return "ok"

        ok, value = generator._probe_fs(flaky)
        self.assertTrue(ok)
        self.assertEqual(value, "ok")
        self.assertEqual(state["n"], 3, "must actually retry")

    def test_probe_gives_up_after_the_configured_attempts(self):
        calls = []

        def always_fails():
            calls.append(1)
            raise OSError(5, "I/O error")

        ok, err = generator._probe_fs(always_fails)
        self.assertFalse(ok)
        self.assertIsInstance(err, OSError)
        self.assertEqual(len(calls), generator._STAT_RETRY_ATTEMPTS)

    def test_probe_swallows_value_error(self):
        # candidate.relative_to(path) raises ValueError on a path that escapes.
        def bad():
            raise ValueError("not a subpath")

        ok, _ = generator._probe_fs(bad)
        self.assertFalse(ok)

    def test_lora_signature_keeps_the_key_stable_across_a_blip(self):
        # A path we have never seen still reports None, which is what a genuinely
        # missing file must look like.
        with tempfile.TemporaryDirectory() as tmp:
            missing = generator._lora_signature(str(Path(tmp) / "nope.safetensors"))
            self.assertEqual(missing[1], None)

        good = type("S", (), {"st_size": 10, "st_mtime_ns": 5})
        with patch("generator._normalize_lora_path", return_value="/models/lora.safetensors"), \
             patch("pathlib.Path.stat", return_value=good()):
            first = generator._lora_signature("/models/lora.safetensors")
        self.assertIsNotNone(first[1])

        with patch("generator._normalize_lora_path", return_value="/models/lora.safetensors"), \
             patch("pathlib.Path.stat", side_effect=OSError(2, "No such file or directory")):
            during = generator._lora_signature("/models/lora.safetensors")
        self.assertEqual(during, first, "a transient failure must not change the pipeline key")

    def test_pipeline_key_survives_a_stat_blip(self):
        lora = {"path": "/models/lora.safetensors", "scale": 1.0}
        with patch("generator._normalize_lora_path", return_value="/models/lora.safetensors"), \
             patch("pathlib.Path.stat", return_value=type("S", (), {"st_size": 10, "st_mtime_ns": 5})()):
            before = generator._make_pipeline_key("flux2-klein-4b", 4, [lora])
        with patch("generator._normalize_lora_path", return_value="/models/lora.safetensors"), \
             patch("pathlib.Path.stat", side_effect=OSError(2, "No such file or directory")):
            after = generator._make_pipeline_key("flux2-klein-4b", 4, [lora])
        self.assertEqual(before, after, "a blip must not force a 20-40s pipeline reload")

    def test_incomplete_probe_does_not_claim_incomplete_when_unreadable(self):
        # Returning True here is what made the UI offer to re-download 13GB.
        with patch("pathlib.Path.rglob", side_effect=OSError(5, "I/O error")):
            self.assertFalse(generator._model_path_has_incomplete(Path("/models/whatever")))

    def test_incomplete_probe_still_detects_real_partials(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "model.safetensors.incomplete").write_bytes(b"x")
            self.assertTrue(generator._model_path_has_incomplete(Path(tmp)))
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "model.safetensors").write_bytes(b"x")
            self.assertFalse(generator._model_path_has_incomplete(Path(tmp)))

    def test_is_model_cached_ttl_holds_the_answer(self):
        calls = []

        def fake(model_id):
            calls.append(model_id)
            return True

        with patch("generator._is_model_cached_uncached", fake):
            self.assertTrue(generator.is_model_cached("flux2-klein-4b"))
            self.assertTrue(generator.is_model_cached("flux2-klein-4b"))
            self.assertEqual(len(calls), 1, "second call within the TTL must be served from cache")
            generator._model_cached_cache = (generator._model_cached_cache[0] - 60, "flux2-klein-4b", True)
            self.assertTrue(generator.is_model_cached("flux2-klein-4b"))
            self.assertEqual(len(calls), 2, "the TTL must expire")

    def test_is_model_cached_ttl_is_keyed_per_model(self):
        with patch("generator._is_model_cached_uncached", lambda m: True):
            generator.is_model_cached("flux2-klein-4b")
            generator.is_model_cached("krea2-turbo")
            self.assertEqual(generator._model_cached_cache[1], "krea2-turbo")


class PresetTimeClaimTests(unittest.TestCase):
    """Preset labels quote a real measured time, not a guess.

    Every number in a preset label is the median generation_time of real runs of
    that exact (model, width, height, steps) taken from the gallery sidecars, or
    - for realvis-xl-v5-lightning, whose guidance changed - that median divided
    by the measured guidance speedup. Configs with no samples lost their claim
    rather than keep a guess.

    The bug this guards against: qwen-image-2.1's labels read "25s" and "40s"
    for presets that run 25 and 40 steps, because the step count had been written
    into the label as seconds. Real medians are 790.8s and 816.0s, so those
    claims were 32x optimistic, and z-image-turbo's 1024x1024 preset claimed
    35s against a measured 444.6s.
    """

    # Measured median generation_time per preset, from the gallery sidecars of
    # the public tree (1469 sidecars, n=1..107 per row). realvis-xl-v5-lightning
    # is its pre-change median divided by the measured guidance speedup.
    MEASURED_MEDIANS = {
        ("flux2-klein-4b", "draft"): 56.1,
        ("flux2-klein-4b", "fast"): 93.8,
        ("flux2-klein-4b", "quality"): 121.3,
        ("flux2-klein-9b", "draft"): 109.9,
        ("juggernaut-xl-lightning", "draft"): 10.0,
        ("juggernaut-xl-lightning", "photo"): 72.3,
        ("juggernaut-xl-lightning", "portrait"): 62.7,
        ("juggernaut-xl-lightning", "cinematic"): 43.4,
        ("realvis-xl-v5-lightning", "draft"): 16.0,
        ("realvis-xl-v5-lightning", "photo"): 56.0,
        ("realvis-xl-v5-lightning", "portrait"): 108.0,
        ("realvis-xl-v5", "draft"): 48.7,
        ("juggernaut-xi", "draft"): 44.7,
        # z-image moved 8 -> 6 steps on 2026-09-29, so these are the 6-step
        # medians: 512x768 min 84.92 / median 91.76, 1024x1024 293.17,
        # 1280x720 295.28. The 8-step numbers were 149.1 (n=83 in the gallery),
        # 444.6 and 359.07.
        ("z-image-turbo", "draft"): 91.8,
        ("z-image-turbo", "turbo"): 293.2,
        ("z-image-turbo", "wide"): 295.3,
        ("krea2-turbo", "draft"): 198.3,
        ("krea2-turbo", "turbo"): 216.3,
        ("krea2-turbo", "portrait"): 329.4,
        ("krea2-turbo", "fast"): 147.7,
        ("qwen-image-2.1", "draft"): 715.8,
        ("qwen-image-2.1", "wide"): 709.1,
        ("qwen-image-2.1", "landscape"): 540.7,
    }
    # 816.0s was measured at 768x512 / 40 steps and 910.0s at 512x768 / 40 steps.
    # Both 40-step presets are gone, so those measurements no longer back any
    # label. They are kept here as the evidence for the 25-step ceiling.
    # Rounding to "~1.8min" / "~13min" loses a few percent; allow 15% either way.
    TOLERANCE = 0.15

    @staticmethod
    def _claimed_seconds(label):
        m = re.search(r"~(\d+(?:\.\d+)?)\s*(s|min)\b", label)
        if not m:
            return None
        return float(m.group(1)) * (60.0 if m.group(2) == "min" else 1.0)

    def test_claims_match_the_measured_medians(self):
        """Two-sided: a label may not understate or wildly overstate the median."""
        problems = []
        for (model_id, preset_id), median in self.MEASURED_MEDIANS.items():
            preset = next(
                (x for x in generator.MODELS[model_id].get("presets", []) if x["id"] == preset_id),
                None,
            )
            self.assertIsNotNone(preset, f"{model_id}/{preset_id} missing from the registry")
            claimed = self._claimed_seconds(preset["label"])
            if claimed is None:
                problems.append(f"{model_id}/{preset_id}: lost its time claim (median {median:.0f}s)")
                continue
            low = median * (1 - self.TOLERANCE)
            high = median * (1 + self.TOLERANCE)
            if not (low <= claimed <= high):
                problems.append(
                    f"{model_id}/{preset_id}: claims {claimed:.0f}s, measured median {median:.0f}s "
                    f"(allowed {low:.0f}-{high:.0f}s)"
                )
        self.assertEqual(problems, [], "\n".join(problems))

    def test_no_preset_outside_the_measured_set_claims_a_time(self):
        known = set(self.MEASURED_MEDIANS)
        problems = [
            f"{model_id}/{preset['id']}: {preset['label']}"
            for model_id, minfo in generator.MODELS.items()
            for preset in minfo.get("presets") or []
            if (model_id, preset["id"]) not in known
            and self._claimed_seconds(preset.get("label", "")) is not None
        ]
        self.assertEqual(
            problems, [],
            "a preset may only quote a time it has a measured median for:\n" + "\n".join(problems),
        )

    def test_step_counts_are_not_pasted_in_as_seconds(self):
        # The specific regression: a label whose seconds equal the step count.
        offenders = [
            f"{model_id}/{preset['id']}: {preset['label']}"
            for model_id, minfo in generator.MODELS.items()
            for preset in minfo.get("presets") or []
            if self._claimed_seconds(preset.get("label", "")) == float(preset["steps"])
        ]
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_labels_state_their_size(self):
        offenders = [
            f"{model_id}/{preset['id']}: {preset['label']}"
            for model_id, minfo in generator.MODELS.items()
            for preset in minfo.get("presets") or []
            if f"{preset['width']}×{preset['height']}" not in preset["label"]
        ]
        self.assertEqual(offenders, [], "\n".join(offenders))


class LoraDiscoveryLockTests(unittest.TestCase):
    """Startup LoRA discovery must not hold _loras_lock across slow work.

    Discovery reads every safetensors header, streams a full-file SHA-256 for
    any registry entry that lacks one, and makes a Civitai HTTP call (8s
    timeout). It used to do all of that inside _loras_lock, which is the lock
    /api/loras and /api/gallery/loras need - so one entry missing its sha256
    stalled the gallery filter for the length of a 4-12s hash plus the HTTP
    round trip, on a background thread, at every startup.
    """

    def setUp(self):
        self.saved_cache = state._loras_cache
        self.saved_file = state.LORAS_FILE
        self.saved_files_dir = state.LORA_FILES_DIR
        self.saved_sdxl_dir = state.SDXL_LORA_DIR
        self.saved_data_dir = state.DATA_DIR
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        state._loras_cache = {}
        state.DATA_DIR = root
        # A registry entry with no sha256 and no civitai_version_id is the case
        # that forces the slow path: a full-file hash plus an HTTP lookup.
        state.LORAS_FILE = root / "loras.json"
        state.LORA_FILES_DIR = root / "lora_files"
        state.SDXL_LORA_DIR = root / "SDXL"
        state.LORA_FILES_DIR.mkdir()
        state.SDXL_LORA_DIR.mkdir()
        state.LORAS_FILE.write_text(
            json.dumps([{"name": "needs-metadata", "path": str(root / "needs-metadata.safetensors"),
                         "scale": 1.0, "triggers": [], "base_model": "flux2"}]),
            encoding="utf-8",
        )

    def tearDown(self):
        state._loras_cache = self.saved_cache
        state.LORAS_FILE = self.saved_file
        state.LORA_FILES_DIR = self.saved_files_dir
        state.SDXL_LORA_DIR = self.saved_sdxl_dir
        state.DATA_DIR = self.saved_data_dir
        self.temp.cleanup()

    def test_registry_stays_readable_while_discovery_runs(self):
        slow_started = threading.Event()
        slow_release = threading.Event()

        def slow_sync(entry):
            slow_started.set()
            slow_release.wait(5.0)
            return False

        with patch.object(state, "sync_lora_entry_with_civitai", slow_sync), \
             patch.object(state, "_KNOWN_TRIGGERS", {}, create=True):
            worker = threading.Thread(target=state._discover_local_loras, daemon=True)
            worker.start()
            self.assertTrue(slow_started.wait(5.0), "discovery never reached the slow call")

            # A LoRA endpoint read must not be stuck behind it.
            t0 = time.perf_counter()
            state._read_loras()
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 0.5, f"read blocked for {elapsed * 1000:.0f}ms behind discovery")

            slow_release.set()
            worker.join(timeout=10)
            self.assertFalse(worker.is_alive())

    def test_source_does_not_wrap_discovery_in_the_lock(self):
        import inspect
        self.assertNotIn("with _loras_lock", inspect.getsource(state._discover_local_loras))


class BackendModelDefaultTests(unittest.TestCase):
    """Persisted per-model defaults must be enforced server-side, not echoed.

    app_settings.apply_model_defaults() is only called from GET /api/models, so
    the backend's own default resolution in generate() reads the raw registry
    dict. A saved override therefore only took effect because the frontend sent
    the value back on every request - and app_settings allows steps up to 50 and
    cache_interval up to 10, which on a 4-step distilled model is a 12x slowdown
    that no server-side bound would catch.
    """

    def test_effective_steps_are_bounded_for_distilled_models(self):
        import app_settings
        for model_id, minfo in generator.MODELS.items():
            if not minfo.get("is_distilled"):
                continue
            resolved = generator._effective_steps(minfo, None, {})
            self.assertLessEqual(
                resolved, 12,
                f"{model_id} is a {minfo['default_steps']}-step distilled model; "
                f"resolving to {resolved} steps would be a large slowdown",
            )

    def test_distilled_models_are_capped_even_if_a_request_asks_for_more(self):
        import app_settings
        minfo = generator.MODELS["flux2-klein-4b"]
        self.assertLessEqual(generator._effective_steps(minfo, 50, {}), 12)

    def test_non_distilled_models_are_not_capped(self):
        minfo = generator.MODELS["realvis-xl-v5"]
        self.assertEqual(generator._effective_steps(minfo, 25, {}), 25)


class RequestDefaultStepsTests(unittest.TestCase):
    """An omitted `steps` must resolve to the model's own default.

    The schema hardcoded `steps: int = Field(default=4)`, applied to every model,
    so a direct API caller that omitted the field got 4 steps on juggernaut-xi
    (default 25), realvis-xl-v5 (25), qwen-image-2.1 (25), z-image-turbo (8) and
    krea2-turbo (8) - badly under-sampled images. On realvis-xl-v5-lightning it
    also produced exactly the 4-step config the A/B contact sheets showed to be
    visibly softer than the 6-step default. guidance and sampler already
    deferred to the registry; steps did not.
    """

    def test_schema_leaves_steps_unset(self):
        from state import GenerateRequest
        self.assertIsNone(GenerateRequest(prompt="x").steps)

    def test_explicit_steps_are_preserved(self):
        from state import GenerateRequest
        self.assertEqual(GenerateRequest(prompt="x", steps=17).steps, 17)

    def test_every_model_resolves_to_its_registry_default(self):
        from state import GenerateRequest
        for model_id, minfo in generator.MODELS.items():
            request = GenerateRequest(prompt="x", model=model_id)
            resolved = minfo["default_steps"] if request.steps is None else request.steps
            self.assertEqual(
                resolved, minfo["default_steps"],
                f"{model_id} would not get its own default step count",
            )

    def test_generator_signature_accepts_none(self):
        import inspect
        signature = inspect.signature(generator.generate)
        self.assertIsNone(signature.parameters["steps"].default)

    def test_worker_resolves_steps_for_the_progress_readout(self):
        from state import GenerateRequest, _effective_request_steps
        for model_id, minfo in generator.MODELS.items():
            self.assertEqual(
                _effective_request_steps(GenerateRequest(prompt="x", model=model_id)),
                minfo["default_steps"],
                f"{model_id} progress readout would show the wrong total",
            )

    def test_worker_keeps_an_explicit_step_count(self):
        from state import GenerateRequest, _effective_request_steps
        request = GenerateRequest(prompt="x", model="realvis-xl-v5-lightning", steps=11)
        self.assertEqual(_effective_request_steps(request), 11)

    def test_progress_math_survives_an_omitted_step_count(self):
        # The bug this guards: on_step computed `req.steps - done` with
        # req.steps None, raising TypeError and failing the whole job.
        from state import GenerateRequest, _effective_request_steps
        request = GenerateRequest(prompt="x", model="realvis-xl-v5-lightning")
        total = _effective_request_steps(request)
        for done in range(1, total + 1):
            eta = 1.0 / done * (total - done)
            self.assertIsInstance(eta, float)


class QwenGuardrailTests(unittest.TestCase):
    """Qwen-Image 2.1 is fenced in on this machine, by owner decision.

    Two traps, both reported by the user:
      * 1024x1024 is a trap - the engine's pixel cap is 589824, so the schema's
        old hardcoded 1024x1024 default made a minimal API request impossible.
      * never run 40 steps again.

    Measured medians back the step ceiling: 790.8s at 512x768 / 25 steps
    against 910.0s at 512x768 / 40 steps, so the extra 15 steps cost ~15% more
    time for no measured gain.
    """

    QWEN_PIXEL_CAP = 589824

    def test_every_model_declares_a_default_size(self):
        for model_id, minfo in generator.MODELS.items():
            self.assertIn("default_width", minfo, f"{model_id} has no default_width")
            self.assertIn("default_height", minfo, f"{model_id} has no default_height")
            self.assertLessEqual(
                minfo["default_width"] * minfo["default_height"], 2048 * 2048, model_id
            )

    def test_no_model_default_exceeds_the_qwen_pixel_cap(self):
        # Otherwise switching to qwen with an omitted size would still fail.
        minfo = generator.MODELS["qwen-image-2.1"]
        self.assertLessEqual(minfo["default_width"] * minfo["default_height"], self.QWEN_PIXEL_CAP)

    def test_a_minimal_request_resolves_to_something_qwen_accepts(self):
        from state import GenerateRequest
        from routers.jobs import _prepare_request
        request = _prepare_request(GenerateRequest(prompt="x", model="qwen-image-2.1"))
        self.assertIsNotNone(request.width)
        self.assertIsNotNone(request.height)
        self.assertLessEqual(request.width * request.height, self.QWEN_PIXEL_CAP)

    def test_a_minimal_request_works_for_every_model(self):
        from state import GenerateRequest
        from routers.jobs import _prepare_request
        for model_id in generator.MODELS:
            request = _prepare_request(GenerateRequest(prompt="x", model=model_id))
            self.assertIsNotNone(request.width, model_id)
            self.assertGreaterEqual(request.width, 128, model_id)

    def test_explicit_size_is_preserved(self):
        from state import GenerateRequest
        from routers.jobs import _prepare_request
        request = _prepare_request(
            GenerateRequest(prompt="x", model="qwen-image-2.1", width=768, height=512)
        )
        self.assertEqual((request.width, request.height), (768, 512))

    def test_qwen_never_offers_more_than_25_steps(self):
        for preset in generator.MODELS["qwen-image-2.1"]["presets"]:
            self.assertLessEqual(
                preset["steps"], generator._QWEN_MAX_STEPS,
                f"preset {preset['id']} offers {preset['steps']} steps",
            )

    def test_qwen_default_step_count_is_within_the_ceiling(self):
        minfo = generator.MODELS["qwen-image-2.1"]
        self.assertLessEqual(minfo["default_steps"], generator._QWEN_MAX_STEPS)

    def test_the_qwen_presets_use_only_known_safe_sizes(self):
        # The user named these three explicitly.
        allowed = {(512, 768), (768, 512), (768, 768)}
        for preset in generator.MODELS["qwen-image-2.1"]["presets"]:
            size = (preset["width"], preset["height"])
            self.assertIn(size, allowed, f"preset {preset['id']} uses {size}")
            self.assertLessEqual(size[0] * size[1], self.QWEN_PIXEL_CAP)

    def test_forty_steps_is_refused_at_submit_time_not_in_the_worker(self):
        # A queued job that fails a second later is worse UX than a 400, and it
        # still burns a queue slot.
        from fastapi import HTTPException
        from state import GenerateRequest
        from routers.jobs import _prepare_request
        with self.assertRaises(HTTPException) as ctx:
            _prepare_request(
                GenerateRequest(prompt="x", model="qwen-image-2.1", width=512, height=768, steps=40)
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("25 steps", str(ctx.exception.detail))

    def test_twenty_five_steps_is_accepted_at_submit_time(self):
        from state import GenerateRequest
        from routers.jobs import _prepare_request
        request = _prepare_request(
            GenerateRequest(prompt="x", model="qwen-image-2.1", width=512, height=768, steps=25)
        )
        self.assertEqual(request.steps, 25)

    def test_the_engine_also_refuses_more_than_25_steps(self):
        # Defence in depth: the engine is a separate process with its own check,
        # so a request that somehow bypassed the parent still cannot run 40 steps.
        # (generate() keeps its general 1-50 bound, which is correct for the
        # non-distilled models that legitimately run 25 steps.)
        engine = Path(__file__).resolve().parent / "qwen_engine.py"
        self.assertIn("steps > 25", engine.read_text(encoding="utf-8"))


class WiredBudgetTests(unittest.TestCase):
    """The generic wired budget is 9GB, matching krea2 (owner decision).

    FLUX.2-klein, FLUX.2-klein-9b, Z-Image and SDXL all shared a 7GB budget while
    krea2 and qwen sat on 9GB, and AGENTS.md records the krea2 raise as ~18%
    faster for identical output. Both paths now share one helper and one memory
    fraction so the two cannot drift apart again.
    """

    def test_generic_budget_is_nine_gib(self):
        self.assertAlmostEqual(generator._wired_limit_bytes() / (1 << 30), 9.0, places=1)

    def test_krea_budget_is_unchanged(self):
        self.assertAlmostEqual(generator._krea_wired_limit_bytes() / (1 << 30), 9.0, places=1)

    def test_sdxl_still_gets_its_own_lower_cap(self):
        # _generate_sdxl clamps to 6.5GB independently, so the generic raise
        # cannot push the SDXL daemon past what its UNet can page.
        generic = generator._wired_limit_bytes()
        self.assertAlmostEqual(min(generic, int(6.5 * (1 << 30))) / (1 << 30), 6.5, places=1)

    @unittest.skipUnless(_MLX_AVAILABLE, "MLX runtime is unavailable")
    def test_budget_never_exceeds_apples_recommended_working_set(self):
        import mlx.core as mx
        info = mx.device_info()
        cap = info.get("max_recommended_working_set_size") or 0
        if cap:
            self.assertLessEqual(generator._wired_limit_bytes(), int(cap))

    @unittest.skipUnless(_MLX_AVAILABLE, "MLX runtime is unavailable")
    def test_budget_never_exceeds_the_shared_memory_fraction(self):
        import mlx.core as mx
        mem = mx.device_info().get("memory_size", 0)
        if mem:
            ceiling = int(mem * generator._WIRED_MEMORY_FRACTION)
            for value in (generator._wired_limit_bytes(), generator._krea_wired_limit_bytes()):
                self.assertLessEqual(value, ceiling)

    def test_zero_disables_the_budget(self):
        with patch.object(generator, "_wired_limit_gb", return_value=0):
            self.assertEqual(generator._wired_limit_bytes(), 0)
        with patch.object(generator, "_krea_wired_limit_gb", return_value=0):
            self.assertEqual(generator._krea_wired_limit_bytes(), 0)

    def test_both_paths_share_one_helper(self):
        import inspect
        self.assertIn("_wired_budget_bytes", inspect.getsource(generator._wired_limit_bytes))
        self.assertIn("_wired_budget_bytes", inspect.getsource(generator._krea_wired_limit_bytes))


if __name__ == "__main__":
    unittest.main()
