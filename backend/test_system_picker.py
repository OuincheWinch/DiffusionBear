import unittest
from unittest.mock import patch, MagicMock
import subprocess
from fastapi.testclient import TestClient

from main import app


class TestSystemPicker(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("sys.platform", "darwin")
    @patch("subprocess.run")
    def test_pick_folder_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="/Volumes/MockDrive/models/my-model/\n",
            stderr="",
        )
        resp = self.client.post("/api/system/pick-path", json={"kind": "folder"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["cancelled"])
        self.assertEqual(data["path"], "/Volumes/MockDrive/models/my-model")
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "osascript")
        self.assertIn("choose folder", args[2])

    @patch("sys.platform", "darwin")
    @patch("subprocess.run")
    def test_pick_file_with_extensions(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="/Volumes/MockDrive/loras/my_lora.safetensors\n",
            stderr="",
        )
        resp = self.client.post(
            "/api/system/pick-path",
            json={"kind": "file", "title": "Select LoRA", "extensions": [".safetensors", "bin"]},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["cancelled"])
        self.assertEqual(data["path"], "/Volumes/MockDrive/loras/my_lora.safetensors")
        args = mock_run.call_args[0][0]
        self.assertIn("choose file", args[2])
        self.assertIn('"safetensors", "bin"', args[2])
        self.assertIn("Select LoRA", args[2])

    @patch("sys.platform", "darwin")
    @patch("subprocess.run")
    def test_pick_user_cancelled(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="execution error: User cancelled. (-128)\n",
        )
        resp = self.client.post("/api/system/pick-path", json={"kind": "folder"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["cancelled"])
        self.assertIsNone(data["path"])

    @patch("sys.platform", "darwin")
    @patch("subprocess.run")
    def test_pick_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="osascript", timeout=120)
        resp = self.client.post("/api/system/pick-path", json={"kind": "folder"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["cancelled"])
        self.assertEqual(data.get("reason"), "timeout")

    @patch("sys.platform", "linux")
    def test_pick_non_darwin(self):
        resp = self.client.post("/api/system/pick-path", json={"kind": "folder"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("macOS", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()
