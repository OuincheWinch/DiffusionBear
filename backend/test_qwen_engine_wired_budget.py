"""The Qwen engine must receive the generic wired budget, not Krea's.

H3: `_generate_qwen_subprocess` built its engine request with
`"wired_limit_bytes": _krea_wired_limit_bytes()`, while the code's own
comments place qwen-image-2.1 under the GENERIC budget (generator.py:575 and
the note at :2824). Krea's budget is tuned for a different pipeline's
resident set; handing it to Qwen sizes the wrong engine's memory ceiling.

The break: req["wired_limit_bytes"] equals the Krea budget instead of the
generic one.

Settings are pinned (generic 4.0 GB, krea 9.0 GB) through a get_setting
wrapper, so the test is hermetic -- the real settings.json is never touched.
4.0 is chosen deliberately, not arbitrarily: it sits below every supported
machine's device cap (8 GB floor for this app), so the resolver returns it
exactly as 4 GiB, while 9.0 resolves to the device cap or 9 GiB -- different
from 4 GiB on every machine. The expected value is therefore a hand-derived
literal. A second assertion checks the request differs from the live
Krea-resolved value, so a machine where the two budgets could coincide fails
loudly instead of passing vacuously.

What is real: the request construction, the registry minfo, the budget
resolution chain, image post-processing and sidecar write. Faked (the engine
subprocess boundary only): the write captures the REAL request JSON the
test asserts on, and the read returns one canned reply with the complete
field structure. The seeded "engine" is a live `sleep`, so reuse takes the
same path a warm daemon would.
"""
import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import app_settings
import generator

try:
    import mlx.core  # noqa: F401

    _MLX_AVAILABLE = True
except ModuleNotFoundError:
    _MLX_AVAILABLE = False

GENERIC_PIN_GB = 4.0
KREA_PIN_GB = 9.0
EXPECTED_BYTES = 4 * (1 << 30)


@unittest.skipUnless(_MLX_AVAILABLE, "MLX runtime is unavailable")
class QwenEngineBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(
            generator, "GENERATED_DIR", self.tmp / "generated"))
        (self.tmp / "generated").mkdir(parents=True)
        # Saved BEFORE patching: _pinned_settings falls through to it, and
        # looking it up via the module inside the side effect would recurse
        # into the mock itself.
        self._real_get_setting = app_settings.get_setting
        self.enterContext(mock.patch.object(
            app_settings, "get_setting", side_effect=self._pinned_settings))

        self._old_proc = generator._qwen_process
        self.addCleanup(self._restore_qwen_process)
        self.addCleanup(generator._cancel_qwen_watchdog)

        self.proc = subprocess.Popen(["sleep", "120"], start_new_session=True)
        self.addCleanup(self._kill_proc)
        generator._qwen_process = self.proc

        self.captured = {}

    def enterContext(self, cm):
        result = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return result

    def _pinned_settings(self, key, default=None):
        if key == "memory_wired_limit_gb":
            return GENERIC_PIN_GB
        if key == "memory_krea_wired_limit_gb":
            return KREA_PIN_GB
        return self._real_get_setting(key, default)

    def _restore_qwen_process(self):
        generator._qwen_process = self._old_proc

    def _kill_proc(self):
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass

    def _fake_write_request(self, proc, payload: bytes, cancel_events, timeout_s):
        """The engine's half of the contract: capture the request, render a
        real PNG at dest. Qwen's engine writes an image file (unlike SDXL's
        raw RGB dump), so the fake must too, or post-processing reads
        nothing."""
        from PIL import Image

        req = json.loads(bytes(payload).decode("utf-8"))
        self.captured.update(req)
        Image.new("RGB", (req["width"], req["height"])).save(req["dest"])
        return None

    def _fake_read_message(self, proc, messages, done, cancel_events,
                           timeout_s, label, tail=None):
        return json.dumps({
            "width": 64, "height": 64, "steps": 2, "sampler": "linear",
            "guidance": 1.0, "quantization": 4,
            "generation_time": 0.05, "load_time": 0.0,
        })

    def test_qwen_request_carries_the_generic_budget(self):
        self.enterContext(mock.patch.object(
            generator, "_write_process_request",
            side_effect=self._fake_write_request))
        self.enterContext(mock.patch.object(
            generator, "_read_engine_message",
            side_effect=self._fake_read_message))
        meta = generator._generate_qwen_subprocess(
            "a watchmaker", 64, 64, 2, 1.0, 42,
            None, None, "qwen-image-2.1", threading.Event(), "", "linear",
        )
        self.assertEqual(
            self.captured["wired_limit_bytes"], EXPECTED_BYTES,
            f"Qwen was handed {self.captured['wired_limit_bytes']} bytes -- "
            f"the Krea budget, not the generic {EXPECTED_BYTES}",
        )
        self.assertNotEqual(
            self.captured["wired_limit_bytes"],
            generator._krea_wired_limit_bytes(),
            "generic and Krea budgets coincide on this machine -- the test "
            "cannot discriminate here",
        )
        self.assertTrue(
            (self.tmp / "generated" / f"{meta['id']}.png").is_file(),
            "the run must still complete end to end",
        )


if __name__ == "__main__":
    unittest.main()
