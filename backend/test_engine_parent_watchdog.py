"""Engines must die when their parent dies, not linger holding ~10 GB.

The gap: graceful shutdown is fully handled (SIGTERM handlers, lifespan
cleanup, atexit, next-launch stray-killer), but a backend killed with SIGKILL
(or a crash, or an OOM kill) leaves the SDXL/Qwen grandchildren alive --
reparented to launchd, each holding its resident weights -- until the next app
launch. That window is the orphan leak.

The fix under test: the backend passes its own pid as --parent-pid at spawn,
and each engine runs a watchdog that exits when getppid() no longer matches.
Three properties:

1. HELPER: parent_alive(expected) is True iff getppid() equals expected.
   getppid is mocked -- the only seam; everything else is real.
2. ORPHAN EXITS: a watchdog armed with a dead pid (a pid that cannot be ours
   -- the test's own grandparent chain guarantees it) stops promptly.
3. LIVE PARENT SURVIVES: a watchdog armed with our own pid is still alive
   after the same grace period; we then stop it cleanly (no leaked thread).

The watchdog entry point starts a real daemon thread and returns a stop
callable -- the subprocess boundary the engines share is os.getppid, so a
thread test here plus the flag wiring there covers both sides. No engine
module is imported: engine_watchdog.py has zero third-party imports, which
is also what lets both engine interpreters (main venv AND venv-sdxl) share
it without dragging MLX across.
"""
import os
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import engine_watchdog


class ParentAliveTests(unittest.TestCase):
    def test_matching_pid_is_alive(self):
        with mock.patch.object(os, "getppid", return_value=os.getpid()):
            self.assertTrue(engine_watchdog.parent_alive(os.getpid()))

    def test_reparented_to_launchd_is_dead(self):
        with mock.patch.object(os, "getppid", return_value=1):
            self.assertFalse(engine_watchdog.parent_alive(os.getpid()))

    def test_other_pid_is_dead(self):
        with mock.patch.object(os, "getppid", return_value=999999999):
            self.assertFalse(engine_watchdog.parent_alive(12345))


class WatchdogThreadTests(unittest.TestCase):
    def test_orphan_watchdog_stops_promptly(self):
        stop, done = engine_watchdog.start_parent_watchdog(
            expected_ppid=999999999, interval_s=0.05)
        try:
            self.assertTrue(
                done.wait(timeout=5),
                "watchdog armed with a dead parent pid never fired",
            )
        finally:
            stop()

    def test_live_parent_watchdog_stays_quiet(self):
        stop, done = engine_watchdog.start_parent_watchdog(
            expected_ppid=os.getppid(), interval_s=0.05)
        try:
            time.sleep(0.4)
            self.assertFalse(
                done.is_set(),
                "watchdog fired while the parent was alive",
            )
        finally:
            stop()
        self.assertFalse(done.is_set())


class ParentPidFromArgvTests(unittest.TestCase):
    def test_absent_flag_is_none(self):
        self.assertIsNone(engine_watchdog.parent_pid_from_argv(["prog", "--serve"]))

    def test_value_parsed(self):
        self.assertEqual(
            engine_watchdog.parent_pid_from_argv(
                ["prog", "--serve", "--parent-pid", "4242"]),
            4242)

    def test_missing_value_is_none(self):
        self.assertIsNone(
            engine_watchdog.parent_pid_from_argv(["prog", "--parent-pid"]))

    def test_non_integer_is_none(self):
        self.assertIsNone(
            engine_watchdog.parent_pid_from_argv(["prog", "--parent-pid", "soon"]))

    def test_non_positive_is_none(self):
        for bad in ("0", "-3"):
            with self.subTest(bad=bad):
                self.assertIsNone(
                    engine_watchdog.parent_pid_from_argv(["prog", "--parent-pid", bad]))


class OrphanExitTests(unittest.TestCase):
    """arm_orphan_exit in real subprocesses: the engine-facing contract.

    A dead parent pid must end the process promptly (this is the orphan
    case); a live parent pid must leave it running. Both run the real helper
    in a real child interpreter -- no engine modules, no MLX, just the
    zero-dependency watchdog file.
    """

    def _spawn(self, expected: str) -> "subprocess.Popen":
        import subprocess
        import sys

        backend = str(Path(__file__).resolve().parent)
        return subprocess.Popen(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r); "
             "from engine_watchdog import arm_orphan_exit; "
             "arm_orphan_exit(%s, interval_s=0.05); "
             "import time; time.sleep(30)" % (backend, expected)],
        )

    def test_dead_parent_ends_the_process(self):
        proc = self._spawn("999999999")
        try:
            code = proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
        self.assertEqual(code, 2, "orphaned process did not exit promptly")

    def test_live_parent_keeps_it_running(self):
        # The child's parent is this test process: pass OUR pid.
        proc = self._spawn(str(os.getpid()))
        try:
            time.sleep(1.0)
            self.assertIsNone(
                proc.poll(), "process exited while its parent was alive")
        finally:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
