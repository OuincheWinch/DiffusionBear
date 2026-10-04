"""The model-download worker must not raise a bare NameError.

Shipped broken: v0.3.1 could not download a single mflux model. The worker
called `_hf_repo_cache_dir(repo)`, which lives in generator.py, without
qualifying it and without importing it. Every registry model that is not
SDXL and not an HF-browser download reaches that line, so the task failed
immediately with:

    Download failed: name '_hf_repo_cache_dir' is not defined

and every fresh install had no way to obtain a model. SDXL downloads
survived because they set `target_dir` from `model_dir` and never evaluate
the fallback, which is why the bug looked model-specific rather than total.

No test exercised this path, so the fix is pinned here. The assertions are
deliberately about *reaching* the network layer rather than about any
particular file: what must not happen is a NameError before the first byte.
"""

import ast
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
DOWNLOADS = BACKEND / "routers" / "downloads.py"
GENERATOR = BACKEND / "generator.py"

def _bound_names(tree: ast.AST) -> set[str]:
    """Every name this module can legitimately call.

    Function and class definitions, module-level and local assignments, loop and
    with targets, comprehension variables, function parameters and imports. A
    name outside this set that is then called can only be a builtin or a bug.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
    return names


_TREE = ast.parse(DOWNLOADS.read_text(encoding="utf-8"))
_BOUND = _bound_names(_TREE)


class ModelDownloadPathTests(unittest.TestCase):
    def test_no_unqualified_call_to_a_generator_private_helper(self):
        """Every bare `name(...)` must resolve in this module or be imported.

        A NameError here is invisible until a user clicks Download, and then it
        is the only thing they can do, so it is worth a static check.
        """
        import builtins

        unknown = set()
        for node in ast.walk(_TREE):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not isinstance(fn, ast.Name):
                continue
            name = fn.id
            if name in _BOUND or hasattr(builtins, name):
                continue
            unknown.add(name)
        self.assertEqual(
            unknown,
            set(),
            "routers/downloads.py calls these without defining or importing "
            "them in this module; each raises NameError the moment that line is "
            "reached, which for the download worker is the only thing a user can do",
        )

    def test_the_cache_dir_helper_is_qualified_in_the_worker(self):
        """Pin the specific regression, with the reason it was missed."""
        source = DOWNLOADS.read_text(encoding="utf-8")
        self.assertNotRegex(
            source,
            r"(?<!generator\.)(?<![\w.])_hf_repo_cache_dir\(",
            "_hf_repo_cache_dir is defined in generator.py; calling it "
            "unqualified raises NameError and breaks every mflux model download",
        )
        self.assertIn(
            "def _hf_repo_cache_dir",
            GENERATOR.read_text(encoding="utf-8"),
            "generator.py must still define it; downloads.py depends on it",
        )

    def test_generator_exposes_the_helper(self):
        """The call site is inside a try, so import-time absence is the failure."""
        import sys

        sys.path.insert(0, str(BACKEND))
        try:
            import generator
        finally:
            sys.path.remove(str(BACKEND))
        self.assertTrue(
            callable(getattr(generator, "_hf_repo_cache_dir", None)),
            "downloads.py calls generator._hf_repo_cache_dir; it must exist",
        )

    def test_sdxl_downloads_do_not_depend_on_the_fallback(self):
        """Why the bug looked partial: SDXL sets target_dir and skips the line.

        If this ever changes, the failure mode becomes every model rather than
        most of them, so the distinction is worth recording.
        """
        source = DOWNLOADS.read_text(encoding="utf-8")
        self.assertIn(
            'elif minfo.get("engine") == "sdxl":',
            source,
            "SDXL branch must keep setting target_dir from model_dir",
        )


if __name__ == "__main__":
    unittest.main()


class DownloadRepoMatchesLoaderTests(unittest.TestCase):
    """The repo a model is downloaded from must be the repo it is loaded from.

    flux2-klein-4b shipped downloading black-forest-labs/FLUX.2-klein-4B, 22.1 GB
    of upstream BFL weights, while the pipeline reads the repo in its MODELS
    entry (mlx-community/flux2-klein-4b-4bit, 4.3 GB). A fresh install therefore
    downloaded 17.8 GB it never used, and the progress bar sat at 99% for hours
    because the byte total belonged to weights that were irrelevant.

    The special case was justified by a comment asserting mflux falls back to
    its own default repo when model_path is absent. It does not: the 4B branch
    passes model_path=local_arg, so the configured repo wins.
    """

    def test_every_download_repo_is_the_models_entry_repo(self):
        import sys

        sys.path.insert(0, str(BACKEND))
        try:
            import generator
        finally:
            sys.path.remove(str(BACKEND))
        for model_id, minfo in generator.MODELS.items():
            if minfo.get("engine") == "sdxl":
                continue  # diffusers repos install into model_dir, not the hub cache
            if model_id == "krea2-turbo":
                continue  # local bundle, no remote source
            self.assertEqual(
                generator.model_download_repo(model_id, minfo),
                minfo.get("repo"),
                f"{model_id} downloads from a different repo than the one its "
                f"pipeline loads; a fresh install would fetch weights it never uses",
            )

    def test_no_model_downloads_from_an_unconverted_upstream_repo(self):
        """The upstream bf16 repos carry duplicate and demo files.

        black-forest-labs/FLUX.2-klein-4B holds both a sharded transformer and a
        single 7.4 GB checkpoint plus demo JPEGs, so it is several times the size
        of the MLX conversion actually used.
        """
        import sys

        sys.path.insert(0, str(BACKEND))
        try:
            import generator
        finally:
            sys.path.remove(str(BACKEND))
        upstream = {"black-forest-labs/FLUX.2-klein-4B", "black-forest-labs/FLUX.2-klein-9B"}
        for model_id, minfo in generator.MODELS.items():
            repo = generator.model_download_repo(model_id, minfo)
            self.assertNotIn(
                repo,
                upstream,
                f"{model_id} downloads the unconverted upstream repo; use the "
                f"MLX conversion the pipeline actually loads",
            )


class DownloadStallWatchdogTests(unittest.TestCase):
    """A hung transfer must become an error, not a bar that never moves.

    huggingface_hub raises nothing when a transfer stops. A fresh flux2-klein-4b
    install sat at 99% indefinitely: one 2 GB shard stopped at 575 MB with zero
    bytes of progress, holding its blob lock, while the byte counter had already
    saturated at the total and so read as complete. There was no timeout, so the
    only symptom was a progress bar that stopped moving.
    """

    def setUp(self):
        import sys

        sys.path.insert(0, str(BACKEND))
        try:
            from routers import downloads
        finally:
            sys.path.remove(str(BACKEND))
        self.downloads = downloads
        self._saved = dict(downloads.MODEL_DOWNLOAD_TASKS)
        downloads.MODEL_DOWNLOAD_TASKS.clear()
        self._saved_limit = downloads._DOWNLOAD_STALL_LIMIT_S

    def tearDown(self):
        self.downloads.MODEL_DOWNLOAD_TASKS.clear()
        self.downloads.MODEL_DOWNLOAD_TASKS.update(self._saved)
        self.downloads._DOWNLOAD_STALL_LIMIT_S = self._saved_limit

    def _task(self, **over):
        base = {
            "id": "t1", "source": "model", "model_id": "flux2-klein-4b",
            "model_name": "FLUX.2-klein 4B", "status": "downloading",
            "downloaded_bytes": 500 << 20, "total_bytes": 2000 << 20,
            "worker_active": True, "progress": 0.25, "error": None,
        }
        base.update(over)
        return base

    def test_a_frozen_transfer_is_failed(self):
        import time

        self.downloads._DOWNLOAD_STALL_LIMIT_S = 1.0
        self.downloads.MODEL_DOWNLOAD_TASKS["t1"] = self._task(
            _last_progress_at=time.monotonic() - 60
        )
        self.downloads._enforce_download_stall_watchdog()
        task = self.downloads.MODEL_DOWNLOAD_TASKS["t1"]
        self.assertEqual(task["status"], "error")
        self.assertIn("stalled", task["error"])
        self.assertFalse(task["worker_active"])

    def test_the_error_says_what_happened_and_what_to_do(self):
        import time

        self.downloads._DOWNLOAD_STALL_LIMIT_S = 1.0
        self.downloads.MODEL_DOWNLOAD_TASKS["t1"] = self._task(
            _last_progress_at=time.monotonic() - 60
        )
        self.downloads._enforce_download_stall_watchdog()
        message = self.downloads.MODEL_DOWNLOAD_TASKS["t1"]["error"]
        self.assertIn("no data received", message)
        self.assertIn("resumes", message)

    def test_a_healthy_transfer_is_left_alone(self):
        import time

        self.downloads._DOWNLOAD_STALL_LIMIT_S = 300.0
        self.downloads.MODEL_DOWNLOAD_TASKS["t1"] = self._task(
            _last_progress_at=time.monotonic()
        )
        self.downloads._enforce_download_stall_watchdog()
        self.assertEqual(self.downloads.MODEL_DOWNLOAD_TASKS["t1"]["status"], "downloading")

    def test_no_bytes_yet_is_not_treated_as_a_stall(self):
        """A slow start on a large first file is not a hang."""
        import time

        self.downloads._DOWNLOAD_STALL_LIMIT_S = 1.0
        self.downloads.MODEL_DOWNLOAD_TASKS["t1"] = self._task(
            downloaded_bytes=0, _last_progress_at=time.monotonic() - 600
        )
        self.downloads._enforce_download_stall_watchdog()
        self.assertEqual(self.downloads.MODEL_DOWNLOAD_TASKS["t1"]["status"], "downloading")

    def test_the_heartbeat_key_is_not_exposed_by_the_api(self):
        import inspect

        src = inspect.getsource(self.downloads.get_model_downloads)
        self.assertIn('"_last_progress_at"', src)


class TokenPersistenceTests(unittest.TestCase):
    """A saved token must be readable after a relaunch.

    POST /api/tokens writes DATA_DIR/hf_token.txt. hf_service.get_hf_token read
    Path(__file__).parent / "data" / "hf_token.txt", which inside the standalone
    app resolves inside the signed bundle -- a path nothing writes. So the token
    was stored correctly and never read back: the UI reported it unset after
    every relaunch and gated downloads had nothing to authenticate with.

    civitai_service resolves its token through app_settings.DATA_DIR; this pins
    the same contract for Hugging Face so the two cannot drift again.
    """

    def _write_and_read(self, tmp: Path):
        import sys

        sys.path.insert(0, str(BACKEND))
        saved = sys.modules.get("app_settings")
        try:
            import app_settings
            import hf_service
            app_settings.DATA_DIR = tmp
            hf_service.app_settings = app_settings
            # neutralise the env and the HF cache fallback so only the file matters
            for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                import os

                os.environ.pop(var, None)
            hf_service.Path.home = staticmethod(lambda: tmp / "nohome")
            (tmp / "hf_token.txt").write_text("hf_" + "a" * 34, encoding="utf-8")
            return hf_service.get_hf_token()
        finally:
            if saved is not None:
                sys.modules["app_settings"] = saved

    def test_a_token_written_to_the_data_dir_is_found(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as d:
            self.assertEqual(self._write_and_read(Path(d)), "hf_" + "a" * 34)

    def test_the_reader_does_not_hardcode_a_bundle_relative_path(self):
        src = (BACKEND / "hf_service.py").read_text(encoding="utf-8")
        self.assertNotIn(
            'Path(__file__).resolve().parent / "data" / "hf_token.txt"',
            src.replace(
                'token_file = Path(__file__).resolve().parent / "data" / "hf_token.txt"',
                "", 1  # the fallback arm is allowed; the bare read is not
            ),
            "the token must be read through app_settings.DATA_DIR, the same place "
            "the token route writes it",
        )


class ModelPickerOrderingTests(unittest.TestCase):
    """Installed models first, then alphabetical.

    One list feeds the Generate dropdown, the Models tab and the defaults
    section. Sorting in each view is how they drift apart, so it is asserted
    here on the shared endpoint.
    """

    def _items(self):
        return [
            {"id": "z-model", "label": "Zeta Turbo", "installed": True},
            {"id": "a-off", "label": "Alpha Offline", "installed": False},
            {"id": "m-adopted", "label": "My Adopted SDXL", "installed": True},
            {"id": "b-model", "label": "beta Image", "installed": False},
            {"id": "c-model", "label": "Civet Krea", "installed": True},
        ]

    def test_installed_come_first(self):
        import sys

        sys.path.insert(0, str(BACKEND))
        try:
            from routers.loras import _sort_models_for_picker
        finally:
            sys.path.remove(str(BACKEND))
        order = [e["id"] for e in _sort_models_for_picker(self._items())]
        installed_flags = [
            e["installed"] for e in _sort_models_for_picker(self._items())
        ]
        # True sorts before False only by intent, not by Python's bool ordering,
        # so compare against reverse-sorted explicitly.
        self.assertEqual(installed_flags, sorted(installed_flags, reverse=True))
        self.assertEqual(order[:3], ["c-model", "m-adopted", "z-model"])

    def test_alphabetical_within_each_group_and_case_insensitive(self):
        import sys

        sys.path.insert(0, str(BACKEND))
        try:
            from routers.loras import _sort_models_for_picker
        finally:
            sys.path.remove(str(BACKEND))
        labels = [e["label"] for e in _sort_models_for_picker(self._items())]
        installed = sorted(
            [e["label"] for e in self._items() if e["installed"]], key=str.casefold
        )
        offline = sorted(
            [e["label"] for e in self._items() if not e["installed"]], key=str.casefold
        )
        self.assertEqual(labels, installed + offline)

    def test_an_adopted_model_is_not_pushed_to_the_end(self):
        """A hand-installed checkpoint must sort by name, not by being 'extra'."""
        import sys

        sys.path.insert(0, str(BACKEND))
        try:
            from routers.loras import _sort_models_for_picker
        finally:
            sys.path.remove(str(BACKEND))
        items = [
            {"id": "z", "label": "Zeta", "installed": True},
            {"id": "adopted", "label": "Adopted One", "installed": True},
        ]
        order = [e["id"] for e in _sort_models_for_picker(items)]
        self.assertEqual(order, ["adopted", "z"])


class StorageAdoptionTests(unittest.TestCase):
    """A directory nothing references should be adoptable, not merely reported.

    The storage panel has always flagged unrecognised model directories -- a model
    downloaded or unpacked by hand. Reporting them without offering anything left the
    user to work out that the fix was the same model_paths override the downloader
    already writes.
    """

    def _entry(self, name):
        import sys

        sys.path.insert(0, str(BACKEND))
        try:
            import storage
        finally:
            sys.path.remove(str(BACKEND))
        return storage.Entry(
            name=name, path="/tmp/x", bytes=1024, files=4,
            detail={"status": "unrecognised"},
        )

    def test_a_klein_variant_is_suggested_for_the_klein_model(self):
        import sys

        sys.path.insert(0, str(BACKEND))
        try:
            import storage
        finally:
            sys.path.remove(str(BACKEND))
        for name in ("FLUX.2-Klein-4B-6bit", "flux2-klein-4b-8bit"):
            suggestion = storage._suggest_adoption(self._entry(name))
            self.assertIsNotNone(suggestion, f"{name} should get a suggestion")
            self.assertEqual(suggestion["model_id"], "flux2-klein-4b")

    def test_nothing_is_suggested_for_an_unrelated_directory(self):
        """A confident wrong guess is worse than no guess at all."""
        import sys

        sys.path.insert(0, str(BACKEND))
        try:
            import storage
        finally:
            sys.path.remove(str(BACKEND))
        self.assertIsNone(storage._suggest_adoption(self._entry("my-photos-backup")))

    def test_adopt_rejects_a_path_outside_the_model_store(self):
        import sys

        sys.path.insert(0, str(BACKEND))
        try:
            from routers import storage as rs
        finally:
            sys.path.remove(str(BACKEND))
        self.assertTrue(
            any(getattr(r, "path", "") == "/api/storage/adopt" for r in rs.router.routes),
            "the adopt endpoint must exist",
        )
        source = (BACKEND / "routers" / "storage.py").read_text(encoding="utf-8")
        self.assertIn("relative_to(models_root)", source)
        self.assertIn("contains no weights", source)
