"""The SDXL daemon must survive across generations for the same model.

C1: `_generate_sdxl` killed the engine unconditionally at entry, so the
`--serve` persistence both sides were built for never engaged. The engine
(`sdxl_engine.get_pipeline`, keyed on model_dir + quantize, LoRAs applied per
request) and the backend (`_get_sdxl_daemon` returning a live proc,
`_current_sdxl_model` tracking identity, the idle watchdog) all assume reuse;
one line before any of them ran threw it away, and `is_daemon_alive` was dead
logic that could only ever be False. Every SDXL run paid a ~6.5 GB reload.

The break these catch: a second generation with an UNCHANGED model identity
respawns the engine process (test 1), or a CHANGED identity fails to replace
it (test 2 -- the guard against an over-eager fix that reuses everything).

What is real and what is faked, stated up front because the line matters:

  REAL: the lock, the kill path (`_kill_sdxl_daemon` really SIGTERMs a real
  `sleep` stand-in), the identity comparison, the registry `minfo`, the
  diffusers layout check, LoRA normalize/enrich, Lightning injection, image
  post-processing, sidecar write, watchdog arm/cancel, gc.

  FAKED (the engine-subprocess boundary only, the slow/external level):
  `_get_sdxl_daemon` (spawning a 6.5 GB engine is the thing under test's
  cost, not its subject -- but it mirrors the real reuse-then-spawn
  structure exactly), `_write_process_request` (parses the REAL request JSON
  and writes the w*h*3 raw bytes to the REAL dest path, which is the
  engine's half of the contract), `_read_engine_message` (one canned reply
  with the complete field structure the post-processing reads),
  `resolve_local_model_path` (returns a tmp dir, so the test cannot depend
  on this machine's settings.json overrides -- the identity STRING
  comparison it feeds stays real), ASSET_DIR/GENERATED_DIR (tmp, so no
  gallery pollution).

The assertion is on `generator._sdxl_daemon` -- real module state -- never on
a mock. A fake that accepts anything verifies nothing, so the fake getter
mirrors the real one's contract: live proc returned as-is, fresh spawn
otherwise.
"""
import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import generator

try:
    import mlx.core  # noqa: F401

    _MLX_AVAILABLE = True
except ModuleNotFoundError:
    _MLX_AVAILABLE = False


def _make_model_dir(root: Path, model_id: str) -> str:
    """A diffusers layout just complete enough to pass sdxl_layout_report.

    Only presence is checked (model_index.json, unet/*.safetensors), never
    contents -- an empty weight file is fine because no engine ever reads it.
    """
    model_dir = root / "models" / model_id
    (model_dir / "unet").mkdir(parents=True, exist_ok=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    (model_dir / "unet" / "placeholder.safetensors").write_bytes(b"")
    return str(model_dir)


def _spawn_stand_in():
    """A real, alive, harmless process in place of the engine.

    Real so the kill path (`poll`, SIGTERM via process group, `wait`) runs
    for real; harmless so killing it can never touch anything that matters.
    Mirrors `_get_sdxl_daemon`'s spawn flags.
    """
    proc = subprocess.Popen(
        ["sleep", "120"], start_new_session=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    proc._mlx_process_group = True
    return proc


RESULT = {
    "width": 64,
    "height": 64,
    "steps": 2,
    "sampler": "euler_trailing",
    "guidance": 1.0,
    "quantization": 4,
    "generation_time": 0.05,
    "load_time": 0.0,
}


@unittest.skipUnless(_MLX_AVAILABLE, "MLX runtime is unavailable")
class SdxlDaemonLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.asset_dir = self.tmp / "asset"
        self.generated_dir = self.tmp / "generated"
        self.generated_dir.mkdir(parents=True)

        # Module state this test borrows -- every one of these is restored,
        # because a leaked daemon global is a leaked daemon for the suite.
        self._old_daemon = generator._sdxl_daemon
        self._old_model = generator._current_sdxl_model
        self.addCleanup(self._restore_module_state)

        self.enterContext(mock.patch.object(generator, "ASSET_DIR", self.asset_dir))
        self.enterContext(mock.patch.object(generator, "GENERATED_DIR", self.generated_dir))
        self.dirs = {}
        self.enterContext(mock.patch.object(
            generator, "resolve_local_model_path",
            side_effect=lambda model_id: self.dirs[model_id],
        ))

        self.procs = []
        self.addCleanup(self._kill_stand_ins)
        self.addCleanup(generator._cancel_sdxl_watchdog)

    def enterContext(self, cm):
        result = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return result

    def _restore_module_state(self):
        generator._sdxl_daemon = self._old_daemon
        generator._current_sdxl_model = self._old_model

    def _kill_stand_ins(self):
        for proc in self.procs:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass

    def _fake_get_daemon(self):
        """Mirror of the real `_get_sdxl_daemon` contract: live proc back
        untouched, fresh spawn otherwise -- INCLUDING assigning the module
        global on the spawn path, which is how the replacement becomes
        observable. No pid file -- the stand-in is tracked through the
        module global, which is what the tests assert."""
        live = generator._sdxl_daemon
        if live is not None and live.poll() is None:
            return live
        proc = _spawn_stand_in()
        self.procs.append(proc)
        generator._sdxl_daemon = proc
        return proc

    def _fake_write_request(self, proc, payload: bytes, cancel_events, timeout_s):
        """The engine's half of the contract: raw RGB bytes land at dest."""
        req = json.loads(bytes(payload).decode("utf-8"))
        dest = Path(req["dest"])
        dest.write_bytes(bytes(req["width"] * req["height"] * 3))
        return None

    def _fake_read_message(self, proc, messages, done, cancel_events,
                           timeout_s, label, tail=None):
        return json.dumps(RESULT)

    def _run(self, model_id):
        self.enterContext(mock.patch.object(
            generator, "_get_sdxl_daemon", side_effect=lambda: self._fake_get_daemon()))
        self.enterContext(mock.patch.object(
            generator, "_write_process_request", side_effect=self._fake_write_request))
        self.enterContext(mock.patch.object(
            generator, "_read_engine_message", side_effect=self._fake_read_message))
        return generator._generate_sdxl(
            "a brass diving helmet", 64, 64, 2, 1.0, 42, [],
            None, model_id, threading.Event(), "", "euler_trailing",
        )

    def _seed_live_daemon(self, model_id):
        self.dirs["juggernaut-xl-lightning"] = _make_model_dir(
            self.asset_dir, "juggernaut-xl-lightning")
        self.dirs["realvis-xl-v5-lightning"] = _make_model_dir(
            self.asset_dir, "realvis-xl-v5-lightning")
        proc = _spawn_stand_in()
        self.procs.append(proc)
        generator._sdxl_daemon = proc
        generator._current_sdxl_model = self.dirs[model_id]
        return proc

    def test_same_model_reuses_the_daemon(self):
        """The regression: the second run killed a live same-model daemon and
        paid a full reload. The SAME process object must serve both runs."""
        proc = self._seed_live_daemon("juggernaut-xl-lightning")
        meta = self._run("juggernaut-xl-lightning")
        self.assertIs(
            generator._sdxl_daemon, proc,
            "the live daemon was killed and respawned for an unchanged model -- "
            "every SDXL generation pays a ~6.5 GB reload",
        )
        self.assertTrue(
            (self.generated_dir / f"{meta['id']}.png").is_file(),
            "the reused-daemon run must still complete end to end",
        )

    def test_dead_daemon_is_replaced_not_reused(self):
        """The third branch of the kill decision: a daemon object that is no
        longer alive must never be handed a request, even when the model
        matches. Without the `poll()` clause the dead process would be reused
        and the run would fail deep in the engine protocol instead of
        respawning cleanly here."""
        proc = self._seed_live_daemon("juggernaut-xl-lightning")
        proc.kill()
        proc.wait(timeout=10)
        self.assertIsNotNone(proc.poll())
        self._run("juggernaut-xl-lightning")
        self.assertIsNot(
            generator._sdxl_daemon, proc,
            "a dead daemon was reused for a matching model",
        )
        self.assertIsNone(
            generator._sdxl_daemon.poll(),
            "the replacement daemon is not alive",
        )

    def test_changed_model_replaces_the_daemon(self):
        """The guard the other way: reuse must be keyed on identity, so a new
        model still gets a fresh engine instead of rendering with stale
        weights."""
        proc = self._seed_live_daemon("juggernaut-xl-lightning")
        self._run("realvis-xl-v5-lightning")
        self.assertIsNot(
            generator._sdxl_daemon, proc,
            "a changed model kept the old daemon -- reuse is not keyed on identity",
        )
        self.assertIsNotNone(
            proc.poll(),
            "the replaced daemon was never actually stopped",
        )


if __name__ == "__main__":
    unittest.main()
