"""Parent-death watchdog for the engine subprocesses.

The backend spawns the SDXL and Qwen engines with start_new_session=True, so a
backend killed with SIGKILL (crash, force-quit, OOM kill) orphans them: they
are reparented to launchd and keep their ~10 GB resident weights until the
next app launch cleans them up. Graceful shutdown is fully handled elsewhere
(SIGTERM handlers, lifespan cleanup, atexit); this covers the ungraceful half.

The backend passes its own pid as --parent-pid at spawn; each engine arms
this watchdog in --serve mode. Only an explicit expected pid arms anything --
engines launched by hand for debugging get no watchdog and behave exactly as
before.

Zero third-party imports, deliberately: both engine interpreters (the main
venv and venv-sdxl) import this without dragging MLX across.
"""

from __future__ import annotations

import os
import threading


def parent_alive(expected_ppid: int) -> bool:
    """True iff our parent is still the process that spawned us."""
    try:
        return os.getppid() == int(expected_ppid)
    except (OSError, TypeError, ValueError):
        return False


def start_parent_watchdog(
    expected_ppid: int, interval_s: float = 5.0
) -> tuple[callable, threading.Event]:
    """Poll the parent pid; set the event and return when it changes.

    Returns (stop, done): done is set if and only if orphanhood was
    observed; stop() ends the thread without setting it. The thread is a
    daemon, so a wedged main loop cannot keep the process alive through it;
    conversely the thread never kills anything itself, it only reports, so
    the engine chooses its own exit path (drain, release, exit).
    """
    done = threading.Event()
    stop_requested = threading.Event()

    def _watch() -> None:
        while not stop_requested.is_set():
            if not parent_alive(expected_ppid):
                done.set()
                return
            stop_requested.wait(interval_s)

    thread = threading.Thread(target=_watch, name="parent-watchdog", daemon=True)
    thread.start()

    def _stop() -> None:
        stop_requested.set()
        thread.join(timeout=interval_s + 5.0)

    return _stop, done


def parent_pid_from_argv(argv: list[str]) -> int | None:
    """The --parent-pid value from an argv list, or None if absent/invalid.

    Kept here so both engines parse the flag identically instead of each
    hand-rolling argv scanning with its own off-by-one.
    """
    try:
        i = list(argv).index("--parent-pid")
    except ValueError:
        return None
    try:
        value = int(argv[i + 1])
    except (IndexError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def arm_orphan_exit(expected_ppid: int, interval_s: float = 5.0) -> None:
    """Exit the process when the parent dies. For engine --serve loops.

    An orphaned engine holds ~10 GB nobody will ever read from, so on
    orphanhood this exits promptly (code 2) after one stderr line instead of
    draining: there is no consumer left for a graceful result, and every
    second lingering is memory the next launch cannot use. No cleanup beyond
    the interpreter's own teardown is attempted, for the same reason -- the
    parent that would observe it is gone.
    """
    import sys

    def _watch_and_exit() -> None:
        stop, done = start_parent_watchdog(expected_ppid, interval_s)
        done.wait()
        if not parent_alive(expected_ppid):
            print(
                f"[engine_watchdog] parent {expected_ppid} gone; exiting",
                file=sys.stderr,
                flush=True,
            )
            os._exit(2)

    thread = threading.Thread(
        target=_watch_and_exit, name="orphan-exit", daemon=True)
    thread.start()
