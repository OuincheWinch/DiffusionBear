"""Model installs must stay inside ASSET_DIR/models/.

C2: the HF-browser install path built `ASSET_DIR / "models" / name` where the
name came from `safe_install_name`, which permits dots -- `safe_install_name(
"..")` is `".."`, verified, not assumed. Containment was then checked against
ASSET_DIR rather than models/, and ASSET_DIR/models/.. IS ASSET_DIR, so the
check passed and multi-GB model files could land in the asset root, next to
settings.json, the gallery index and the LoRA registry.

Two doors, two tests, because there are two places that must refuse:

1. ROUTE: POST /api/models/download with install_name ".." (or ".") must
   400. Before the fix it returned 200 already_installed -- the asset root
   presented as an installed model -- because the escaped path exists and the
   route short-circuits on existence instead of validating.

2. WORKER: `_run_model_download` handed an install_dir outside models/ must
   fail BEFORE any mkdir or download. Before the fix it proceeded to the HF
   download. The HfApi mock below is scaffolding that stands in for the
   network; the assertions are on worker behavior (task failed, download
   never attempted), never on the mock -- and a control test proves the mock
   actually intercepts the real call site, so "not called" cannot pass
   vacuously through a mistargeted patch.
"""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import generator
from routers import downloads as dl


class TestInstallRouteRejectsTraversal(unittest.TestCase):
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.include_router(dl.router)
        return TestClient(app, raise_server_exceptions=False)

    def test_dot_names_are_rejected_not_installed(self):
        """Both ".." (the asset root) and "." (models/ itself) resolve outside
        a model directory. Answering 200 already_installed for either is wrong
        twice over: it is not an installed model, and it masks the traversal."""
        for name in ("..", "."):
            with self.subTest(install_name=name):
                response = self.client().post(
                    "/api/models/download",
                    json={
                        "model_id": "probe-traversal",
                        "repo_id": "someone/probe-model",
                        "install_name": name,
                    },
                )
                self.assertEqual(
                    response.status_code, 400,
                    f"install_name {name!r} was accepted "
                    f"({response.status_code}): {response.text[:200]}",
                )


class TestInstallWorkerRefusesEscape(unittest.TestCase):
    def setUp(self):
        self.tid = "traversal-probe-task"
        with dl._MODEL_DOWNLOAD_LOCK:
            dl.MODEL_DOWNLOAD_TASKS[self.tid] = {
                "id": self.tid, "status": "downloading",
            }
        self.addCleanup(self._drop_task)

    def _drop_task(self):
        with dl._MODEL_DOWNLOAD_LOCK:
            dl.MODEL_DOWNLOAD_TASKS.pop(self.tid, None)

    def _run(self, install_dir):
        dl._run_model_download(
            self.tid, "probe-model",
            {"engine": "mflux", "label": "probe"},
            threading.Event(),
            repo_override="someone/probe-model",
            install_dir=install_dir,
        )
        with dl._MODEL_DOWNLOAD_LOCK:
            return dict(dl.MODEL_DOWNLOAD_TASKS[self.tid])

    def test_install_dir_outside_models_is_refused_before_download(self):
        """The escaped target itself: the asset root. The worker must fail
        here, not proceed to mkdir and download into it."""
        with mock.patch.object(
            dl, "HfApi",
            side_effect=AssertionError("network reached from outside models/"),
        ) as hfapi:
            task = self._run(generator.ASSET_DIR)
        hfapi.assert_not_called()
        self.assertEqual(task["status"], "error")
        self.assertIn("models", task["error"].lower())

    def test_mock_intercepts_the_real_download_call_site(self):
        """Scaffolding validation, not a behavior test: a legit inside-models/
        install must REACH the download step, proving the HfApi patch above
        sits on the real call path and "not called" means refused, not
        mistargeted. Uses a tmp asset root so nothing touches the real tree."""
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with (
            mock.patch.object(generator, "ASSET_DIR", tmp),
            mock.patch.object(
                dl, "HfApi", side_effect=RuntimeError("probe-download-intercepted")),
        ):
            task = self._run(tmp / "models" / "probe-legit")
        self.assertEqual(task["status"], "error")
        self.assertIn("probe-download-intercepted", task["error"])

    def enterContext(self, cm):
        result = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return result


if __name__ == "__main__":
    unittest.main()
