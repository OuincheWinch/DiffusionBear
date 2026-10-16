"""A cancelled Civitai download must not poison the retry.

C3: the download loop opened the destination in "ab" mode afresh for every
1 MB chunk. Three defects in one loop: fd churn on multi-GB files, appends
onto a partial left by a cancelled run (retry produced partial+full, i.e. a
corrupt safetensors), and no cleanup on cancel or error -- the partial sat
there waiting for the next attempt to append to.

Two tests, one per broken promise:

1. RETRY CORRECTNESS: cancel after two chunks, run again to completion. The
   final file must be byte-identical to the served bytes. Before the fix it
   was partial+full (or missing, when validation caught it -- either way not
   equal).

2. NO ORPHANS: a mid-stream connection drop must leave no partial, temp or
   otherwise, in the target dir. Before the fix the first chunk remained.

What is real: the worker function itself, task registry updates, progress
bookkeeping, and the safetensors validation -- the served bytes are a REAL
minimal valid header+8-data-bytes file, so `is_valid_safetensors` runs
unmocked. ASSET_DIR is patched to tmp, so nothing touches the real tree.

Faked (the network boundary only): `open_public_https_stream` returns a
scripted object mirroring the slice of requests' streaming API the worker
touches -- headers dict, iter_content generator, context-manager protocol.
"""
import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import civitai_service
import generator
from routers import downloads as dl


def _valid_payload() -> bytes:
    """A real minimal valid safetensors file: header + 8 data bytes."""
    header = json.dumps(
        {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
        separators=(",", ":"),
    ).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + bytes(8)


PAYLOAD = _valid_payload()


class _ScriptedStream:
    """The slice of a streaming response the worker touches, scripted.

    mode="full": yield every chunk.
    mode="cancel": yield two chunks, arm the cancel event, yield one more --
      the worker's per-iteration cancel check then raises DownloadCancelled,
      exactly as a user pressing cancel mid-download.
    mode="drop": yield one chunk, then raise ConnectionError, as a severed
      connection surfaces from requests' iter_content.
    """

    FILENAME = "probe.safetensors"

    def __init__(self, payload: bytes, mode: str, cancel_event=None):
        self._payload = payload
        self._mode = mode
        self._cancel_event = cancel_event
        self.headers = {
            "Content-Disposition": f'attachment; filename="{self.FILENAME}"',
            "Content-Length": str(len(payload)),
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size=None):
        size = max(1, len(self._payload) // 3)
        parts = [
            self._payload[i:i + size] for i in range(0, len(self._payload), size)
        ]
        if self._mode == "full":
            yield from parts
        elif self._mode == "cancel":
            yield parts[0]
            yield parts[1]
            self._cancel_event.set()
            yield parts[2]
        elif self._mode == "drop":
            yield parts[0]
            raise ConnectionError("boom")
        else:  # pragma: no cover - test scaffolding guard
            raise AssertionError(f"unknown stream mode {self._mode!r}")


class _CivitaiDownloadCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(generator, "ASSET_DIR", self.tmp))
        self.target_dir = self.tmp / "models" / "probe-model"
        self._tids = []
        self.addCleanup(self._drop_tasks)

    def enterContext(self, cm):
        result = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return result

    def _register_task(self):
        # A retry is a NEW task: the route mints a fresh task_id per attempt,
        # and _finish_model_task only completes entries still marked
        # downloading. Reusing one entry across both runs would test a flow
        # the app never performs.
        tid = f"civitai-atomic-probe-{len(self._tids)}"
        with dl._MODEL_DOWNLOAD_LOCK:
            dl.MODEL_DOWNLOAD_TASKS[tid] = {"id": tid, "status": "downloading"}
        self._tids.append(tid)
        return tid

    def _drop_tasks(self):
        with dl._MODEL_DOWNLOAD_LOCK:
            for tid in self._tids:
                dl.MODEL_DOWNLOAD_TASKS.pop(tid, None)

    def _run(self, stream, cancel_event):
        tid = self._register_task()
        # The URL is inert: open_public_https_stream is mocked, and this
        # worker never reads final_url (the filename comes from
        # Content-Disposition). Loopback keeps test_repo_security's
        # outbound-host guard green without touching the guard itself.
        self.enterContext(mock.patch.object(
            civitai_service, "open_public_https_stream",
            return_value=(stream, "http://127.0.0.1/file"),
        ))
        dl._run_civitai_model_download(
            tid, 12345, "probe-model", None, cancel_event,
        )
        with dl._MODEL_DOWNLOAD_LOCK:
            return dict(dl.MODEL_DOWNLOAD_TASKS[tid])

    def _files(self):
        if not self.target_dir.is_dir():
            return set()
        return {p.name for p in self.target_dir.iterdir()}


class TestCancelThenRetryIsExact(_CivitaiDownloadCase):
    def test_retry_produces_byte_identical_file(self):
        cancel = threading.Event()
        first = self._run(
            _ScriptedStream(PAYLOAD, "cancel", cancel_event=cancel), cancel)
        self.assertEqual(
            first["status"], "cancelled",
            f"the cancel path never engaged: {first}",
        )
        second = self._run(
            _ScriptedStream(PAYLOAD, "full"), threading.Event())
        self.assertEqual(second["status"], "done", f"{second}")
        self.assertEqual(
            (self.target_dir / "probe.safetensors").read_bytes(), PAYLOAD,
            "retry appended onto the cancelled partial instead of replacing it",
        )
        self.assertEqual(self._files(), {"probe.safetensors"})


class TestFailedDownloadLeavesNoPartials(_CivitaiDownloadCase):
    def test_dropped_connection_leaves_empty_dir(self):
        task = self._run(
            _ScriptedStream(PAYLOAD, "drop"), threading.Event())
        self.assertEqual(task["status"], "error", f"{task}")
        self.assertEqual(
            self._files(), set(),
            "a partial from the failed download was left behind for the "
            "next attempt to append to",
        )


class TestContentDispositionFilename(unittest.TestCase):
    def test_extracts_filename_from_quoted_and_unquoted_headers(self):
        self.assertEqual(
            dl._content_disposition_filename('attachment; filename="model.safetensors"'),
            "model.safetensors",
        )
        self.assertEqual(
            dl._content_disposition_filename('attachment; filename=model.safetensors'),
            "model.safetensors",
        )

    def test_handles_rfc5987_and_extra_parameters(self):
        header = 'attachment; filename="foo.safetensors"; size=1024'
        self.assertEqual(dl._content_disposition_filename(header), "foo.safetensors")

    def test_empty_or_malformed_returns_none(self):
        self.assertIsNone(dl._content_disposition_filename(""))
        self.assertIsNone(dl._content_disposition_filename(None))


if __name__ == "__main__":
    unittest.main()
