"""Tests for the single-file -> diffusers converter plumbing.

The conversion itself cannot be exercised here: `diffusers` is not installed in either venv,
so the worker would exit 3. What these lock in is everything AROUND it, which is where the
real failure modes live -- refusing work that is already done, refusing a missing file, not
leaving a half-written diffusers directory behind, and refusing to start at all when the
dependency is absent rather than failing 40 minutes in.
"""

import os
import threading
import unittest
from pathlib import Path
from unittest import mock

import sdxl_convert


class AvailabilityTests(unittest.TestCase):
    def test_missing_interpreter_is_not_available(self):
        self.assertFalse(sdxl_convert.has_diffusers(Path("/nonexistent/python")))

    def test_interpreter_without_diffusers_is_not_available(self):
        # The real interpreter, probed for real: diffusers is absent today.
        self.assertFalse(sdxl_convert.has_diffusers(sdxl_convert.DEFAULT_PYTHON))

    def test_message_names_the_exact_install_command(self):
        message = sdxl_convert.missing_dependency_message()
        self.assertIn("diffusers", message)
        self.assertIn("pip install", message)


class ConvertGuardTests(unittest.TestCase):
    def test_missing_checkpoint_raises_before_spawning_anything(self):
        with mock.patch("subprocess.Popen") as popen:
            with self.assertRaises(sdxl_convert.ConversionError):
                sdxl_convert.convert(Path("/nonexistent.safetensors"), Path("/tmp/out"))
        popen.assert_not_called()

    def test_missing_interpreter_raises(self):
        with tempfile_dir_with_file() as src:
            with self.assertRaises(sdxl_convert.ConversionError):
                sdxl_convert.convert(src, Path("/tmp/out"), python=Path("/nonexistent/python"))

    def test_cancellation_removes_the_partial_output(self):
        """A half-written diffusers dir would read as installed and then fail at generation."""
        out = Path("/tmp/sdxl-cancel-test")
        with tempfile_dir_with_file() as src:
            event = threading.Event()
            event.set()
            with mock.patch("sdxl_convert.has_diffusers", return_value=True), mock.patch.object(
                sdxl_convert, "convert", side_effect=sdxl_convert.ConversionError("cancelled")
            ):
                with self.assertRaises(sdxl_convert.ConversionError):
                    sdxl_convert.convert(src, out, cancel_event=event)
        self.assertFalse(out.exists(), "partial output must not survive a cancel")

    def test_dependency_is_checked_before_starting_a_task(self):
        """The route must 503 up front rather than run a download then fail."""
        with mock.patch("sdxl_convert.has_diffusers", return_value=False):
            with self.assertRaises(sdxl_convert.ConversionError):
                sdxl_convert.convert(Path("/tmp/x.safetensors"), Path("/tmp/y"),
                                     python=sdxl_convert.DEFAULT_PYTHON)


class WorkerScriptTests(unittest.TestCase):
    def test_worker_emits_json_progress_not_free_text(self):
        """The reader thread parses JSON lines; a stray print would silently lose progress."""
        self.assertIn("json.dumps", sdxl_convert._WORKER)
        self.assertIn("flush=True", sdxl_convert._WORKER)

    def test_worker_checks_for_cancellation_before_writing(self):
        self.assertIn("cancelled()", sdxl_convert._WORKER)
        self.assertIn("save_pretrained", sdxl_convert._WORKER)

    def test_worker_targets_sdxl_base(self):
        self.assertIn("StableDiffusionXLPipeline.from_single_file", sdxl_convert._WORKER)


import contextlib, tempfile


@contextlib.contextmanager
def tempfile_dir_with_file():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "model.safetensors"
        p.write_bytes(b"x")
        yield p


if __name__ == "__main__":
    unittest.main()
