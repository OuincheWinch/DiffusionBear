"""Task 5: HF + Civitai acquisition for unlisted models.

Hermetic: mocked download cores (HfApi / hf_hub_download /
open_public_https_stream), fake flux2-layout dirs, temp DATA_DIR registry +
temp ASSET_DIR, TestClient routes. No network, no weights.
"""

import json
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient

import civitai_service
import custom_models
import generator
import main
import state
from routers import downloads as dl


def _write_flux_index(model_dir: Path, doubles: int, singles: int,
                      shards: int = 2) -> None:
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for i in range(doubles):
        shard = f"model-{i % shards:05d}-of-{shards:05d}.safetensors"
        weight_map[f"transformer.transformer_blocks.{i}.attn.to_q.weight"] = shard
    for i in range(singles):
        shard = f"model-{(i + doubles) % shards:05d}-of-{shards:05d}.safetensors"
        weight_map[f"transformer.single_transformer_blocks.{i}.attn.to_q.weight"] = shard
    (transformer_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    for shard in set(weight_map.values()):
        (transformer_dir / shard).write_bytes(b"\x00")


def _valid_payload() -> bytes:
    header = json.dumps(
        {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
        separators=(",", ":"),
    ).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + bytes(8)


PAYLOAD = _valid_payload()


class _ScriptedStream:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.headers = {
            "Content-Disposition": 'attachment; filename="probe.safetensors"',
            "Content-Length": str(len(payload)),
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size=None):
        size = max(1, len(self._payload) // 3)
        for i in range(0, len(self._payload), size):
            yield self._payload[i:i + size]


class _RegistryTempDir(unittest.TestCase):
    def enterContext(self, cm):
        result = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return result

    def setUp(self):
        super().setUp()
        self.saved_file = custom_models.CUSTOM_MODELS_FILE
        self.saved_cache = custom_models._custom_models_cache
        self.temp = tempfile.TemporaryDirectory(dir=state.DATA_DIR)
        custom_models._custom_models_cache = {}
        custom_models.CUSTOM_MODELS_FILE = Path(self.temp.name) / "custom_models.json"
        self.asset_tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(generator, "ASSET_DIR", self.asset_tmp))
        self.enterContext(mock.patch(
            "app_settings.update_settings", return_value={}))
        self._tids = []
        self.addCleanup(self._drop_tasks)

    def tearDown(self):
        custom_models.CUSTOM_MODELS_FILE = self.saved_file
        custom_models._custom_models_cache = self.saved_cache
        self.temp.cleanup()
        super().tearDown()

    def _drop_tasks(self):
        with dl._MODEL_DOWNLOAD_LOCK:
            for tid in self._tids:
                dl.MODEL_DOWNLOAD_TASKS.pop(tid, None)

    def _new_task(self, extra=None):
        tid = f"task5-probe-{len(self._tids)}-{time.monotonic_ns()}"
        task = {"id": tid, "status": "downloading"}
        if extra:
            task.update(extra)
        with dl._MODEL_DOWNLOAD_LOCK:
            dl.MODEL_DOWNLOAD_TASKS[tid] = task
        self._tids.append(tid)
        return tid

    def _task_snapshot(self, tid):
        with dl._MODEL_DOWNLOAD_LOCK:
            return dict(dl.MODEL_DOWNLOAD_TASKS[tid])


def _fake_hf_api():
    api = mock.Mock()
    api.model_info.return_value = SimpleNamespace(
        sha="a" * 40,
        siblings=[
            SimpleNamespace(rfilename="transformer/model-00001-of-00002.safetensors", size=10),
            SimpleNamespace(rfilename="transformer/model-00002-of-00002.safetensors", size=10),
        ],
    )
    return api


def _hf_hub_side_effect(**kwargs):
    local_dir = Path(kwargs["local_dir"])
    _write_flux_index(local_dir, 5, 20)
    return str(local_dir / "transformer" / "model-00001-of-00002.safetensors")


class HFWorkerTailTests(_RegistryTempDir):
    def _run_hf(self, declaration=None, install_name="probe-hf"):
        install_dir = self.asset_tmp / "models" / install_name
        extra = {}
        if declaration is not None:
            extra["declaration"] = declaration
        tid = self._new_task(extra)
        with (
            mock.patch.object(dl, "HfApi", return_value=_fake_hf_api()),
            mock.patch("huggingface_hub.hf_hub_download",
                       side_effect=_hf_hub_side_effect),
        ):
            dl._run_model_download(
                tid, "probe-custom-hf",
                {"id": "probe-custom-hf", "label": "Probe", "engine": "mflux"},
                threading.Event(),
                repo_override="org/probe",
                install_dir=install_dir,
            )
        return self._task_snapshot(tid)

    def test_hf_worker_registers_flux2_unlisted(self):
        task = self._run_hf({
            "id": None, "label": "Probe Custom", "family": "flux2",
            "variant": "4b", "source": "hf", "hf_repo": "org/probe",
        })
        self.assertEqual(task["status"], "done", task)
        entries = custom_models.read_custom_models()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["source"], "hf")
        self.assertEqual(entry["family"], "flux2")
        self.assertEqual(entry["variant"], "4b")
        self.assertEqual(entry["hf_repo"], "org/probe")
        self.assertEqual(entry["id"], "probe-hf")

    def test_hf_worker_size_mismatch_fails_task_without_entry(self):
        task = self._run_hf({
            "id": None, "label": "Probe", "family": "flux2",
            "variant": "9b", "source": "hf", "hf_repo": "org/probe",
        })
        self.assertEqual(task["status"], "error", task)
        self.assertIn("size mismatch", task["error"])
        self.assertEqual(custom_models.read_custom_models(), [])

    def test_hf_worker_without_declaration_keeps_old_behavior(self):
        task = self._run_hf(None)
        self.assertEqual(task["status"], "done", task)
        self.assertEqual(custom_models.read_custom_models(), [])


class CivitaiWorkerTailTests(_RegistryTempDir):
    def _run_civitai(self, declaration=None, name="probe-civitai", version=12345):
        extra = {}
        if declaration is not None:
            extra["declaration"] = declaration
        tid = self._new_task(extra)

        def _fake_stream(url, **kwargs):
            _write_flux_index(self.asset_tmp / "models" / name, 5, 20)
            return (_ScriptedStream(PAYLOAD), "http://127.0.0.1/file")

        with mock.patch.object(civitai_service, "open_public_https_stream",
                               side_effect=_fake_stream):
            dl._run_civitai_model_download(
                tid, version, name, None, threading.Event())
        return self._task_snapshot(tid)

    def test_civitai_worker_registers_flux2_unlisted(self):
        task = self._run_civitai({
            "id": None, "label": "Probe Civ", "family": "flux2",
            "variant": "4b", "source": "civitai", "civitai_ids": [12345],
            "civitai_model_id": 7, "civitai_version_id": 12345,
        })
        self.assertEqual(task["status"], "done", task)
        entries = custom_models.read_custom_models()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["source"], "civitai")
        self.assertEqual(entry["family"], "flux2")
        self.assertEqual(entry["civitai_ids"], ["12345"])
        self.assertEqual(entry["civitai_model_id"], 7)
        self.assertEqual(entry["civitai_version_id"], 12345)

    def test_civitai_worker_refusal_fails_task_without_entry(self):
        task = self._run_civitai({
            "id": None, "label": "Probe", "family": "flux2",
            "variant": "9b", "source": "civitai", "civitai_ids": [12345],
            "civitai_model_id": 7, "civitai_version_id": 12345,
        })
        self.assertEqual(task["status"], "error", task)
        self.assertIn("size mismatch", task["error"])
        self.assertEqual(custom_models.read_custom_models(), [])


class DownloadRouteDeclarationTests(_RegistryTempDir):
    def setUp(self):
        super().setUp()
        self.client = TestClient(main.app)

    def _wait_settled(self, tid, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            task = self._task_snapshot(tid)
            if task.get("status") != "downloading":
                return task
            time.sleep(0.05)
        return self._task_snapshot(tid)

    def test_hf_unlisted_requires_family(self):
        r = self.client.post("/api/models/download", json={
            "model_id": "some-new-model",
            "repo_id": "org/probe",
            "install_name": "probe-no-family",
        })
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("family", r.json()["detail"])

    def test_hf_unlisted_accepts_declaration(self):
        with mock.patch.object(dl, "HfApi",
                               side_effect=RuntimeError("no network")):
            r = self.client.post("/api/models/download", json={
                "model_id": "Probe Custom",
                "repo_id": "org/probe",
                "install_name": "probe-declared",
                "family": "flux2",
                "variant": "4b",
            })
        self.assertEqual(r.status_code, 200, r.text)
        tid = r.json()["task_id"]
        self._tids.append(tid)
        task = self._wait_settled(tid)
        declaration = task.get("declaration") or {}
        self.assertEqual(declaration.get("family"), "flux2")
        self.assertEqual(declaration.get("source"), "hf")

    def test_civitai_requires_family(self):
        r = self.client.post("/api/civitai/models/download", json={
            "model_id": 7,
            "model_version_id": 12345,
        })
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("family", r.json()["detail"])

    def test_civitai_traversal_name_rejected(self):
        r = self.client.post("/api/civitai/models/download", json={
            "model_id": 7,
            "model_version_id": 12345,
            "name": "..",
            "family": "sdxl",
        })
        self.assertEqual(r.status_code, 400, r.text)

    def test_civitai_accepts_declaration(self):
        with mock.patch.object(
                civitai_service, "open_public_https_stream",
                side_effect=RuntimeError("no network")):
            r = self.client.post("/api/civitai/models/download", json={
                "model_id": 7,
                "model_version_id": 12345,
                "name": "probe-civ-route",
                "family": "flux2",
                "variant": "4b",
            })
        self.assertEqual(r.status_code, 200, r.text)
        tid = r.json()["task_id"]
        self._tids.append(tid)
        task = self._wait_settled(tid)
        declaration = task.get("declaration") or {}
        self.assertEqual(declaration.get("family"), "flux2")
        self.assertEqual(declaration.get("source"), "civitai")


def _version_info(**overrides):
    info = {
        "id": 12345,
        "modelId": 7,
        "name": "v1.0",
        "type": "LORA",
        "baseModel": "SDXL",
        "trainedWords": ["trigger1"],
        "model": {"name": "Probe Model", "type": "LORA"},
        "files": [{
            "name": "probe.safetensors",
            "downloadUrl": "https://civitai.com/api/download/models/12345",
            "sizeKB": 100.0,
            "hashes": {"SHA256": "A" * 64},
        }],
    }
    info.update(overrides)
    return info


class CheckpointMetadataTests(unittest.TestCase):
    def test_checkpoint_type_accepted_with_same_shape(self):
        meta = civitai_service.extract_civitai_checkpoint_metadata(
            _version_info(type="Checkpoint",
                          model={"name": "Probe Model", "type": "Checkpoint"}))
        self.assertEqual(
            set(meta),
            set(civitai_service.extract_civitai_metadata(_version_info())),
        )
        self.assertEqual(meta["triggers"], [])
        self.assertIn("Probe Model", meta["civitai_model_name"])

    def test_checkpoint_name_default_differs_from_lora(self):
        checkpoint = civitai_service.extract_civitai_checkpoint_metadata(
            _version_info(type="Checkpoint",
                          model={"type": "Checkpoint"},
                          name=None))
        lora = civitai_service.extract_civitai_metadata(
            _version_info(model={}, name=None))
        self.assertIn("Checkpoint", checkpoint["civitai_model_name"])
        self.assertIn("LoRA", lora["civitai_model_name"])

    def test_checkpoint_type_match_is_case_insensitive(self):
        meta = civitai_service.extract_civitai_checkpoint_metadata(
            _version_info(type="checkpoint",
                          model={"name": "M", "type": "checkpoint"}))
        self.assertEqual(meta["civitai_version_id"], 12345)

    def test_lora_path_untouched(self):
        meta = civitai_service.extract_civitai_metadata(_version_info())
        self.assertEqual(meta["triggers"], ["trigger1"])

    def test_other_types_rejected_on_both_paths(self):
        info = _version_info(type="TextualInversion",
                             model={"name": "M", "type": "TextualInversion"})
        with self.assertRaises(ValueError):
            civitai_service.extract_civitai_metadata(info)
        with self.assertRaises(ValueError):
            civitai_service.extract_civitai_checkpoint_metadata(info)


class QwenMappingTests(unittest.TestCase):
    def test_qwen_base_models_map_to_qwen(self):
        for raw in ("Qwen-Image", "qwen image 2.0", "QWEN"):
            self.assertEqual(civitai_service.normalize_base_model(raw), "qwen", raw)

    def test_unknown_string_keeps_sdxl_default(self):
        self.assertEqual(
            civitai_service.normalize_base_model("SomeFutureArch-XL"), "sdxl")

    def test_prefill_suggestion_mapping(self):
        suggest = civitai_service.suggest_family_for_base_model
        self.assertEqual(suggest("Qwen-Image"), "qwen")
        self.assertEqual(suggest("SomeFutureArch-XL"), "sdxl")
        self.assertEqual(suggest("Krea"), "krea")
        self.assertEqual(suggest("z-image-turbo"), "zit")
        self.assertIsNone(suggest(None))
        self.assertIsNone(suggest("SD 1.5"))


class PrefillRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_qwen_version_prefills_qwen(self):
        info = _version_info(baseModel="Qwen-Image")
        with mock.patch.object(civitai_service, "fetch_by_version_id",
                               return_value=info):
            r = self.client.get("/api/civitai/models/version/12345/prefill")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["suggested_family"], "qwen")
        self.assertEqual(body["model_name"], "Probe Model")
        self.assertEqual(body["version_name"], "v1.0")

    def test_unknown_base_model_prefills_sdxl_default(self):
        info = _version_info(baseModel="SomeFutureArch-XL")
        with mock.patch.object(civitai_service, "fetch_by_version_id",
                               return_value=info):
            r = self.client.get("/api/civitai/models/version/12345/prefill")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["suggested_family"], "sdxl")

    def test_unsupported_base_model_suggests_nothing(self):
        info = _version_info(baseModel="SD 1.5")
        with mock.patch.object(civitai_service, "fetch_by_version_id",
                               return_value=info):
            r = self.client.get("/api/civitai/models/version/12345/prefill")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIsNone(r.json()["suggested_family"])

    def test_missing_version_is_404(self):
        with mock.patch.object(civitai_service, "fetch_by_version_id",
                               return_value=None):
            r = self.client.get("/api/civitai/models/version/999999/prefill")
        self.assertEqual(r.status_code, 404)


class PrefillModelIdTests(unittest.TestCase):
    """Version-only inputs (bare ids, download links) need a model id to
    start the download route. The prefill endpoint already fetches the
    version payload carrying modelId -- it just never returned it, so the
    UI had no way to resolve one."""

    def setUp(self):
        from fastapi.testclient import TestClient
        import main

        self.client = TestClient(main.app)

    def test_prefill_returns_the_model_id(self):
        info = _version_info()
        with mock.patch.object(civitai_service, "fetch_by_version_id",
                               return_value=info):
            r = self.client.get("/api/civitai/models/version/12345/prefill")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["model_id"], 7)

    def test_prefill_without_model_id_is_none_not_missing(self):
        info = _version_info()
        del info["modelId"]
        with mock.patch.object(civitai_service, "fetch_by_version_id",
                               return_value=info):
            r = self.client.get("/api/civitai/models/version/12345/prefill")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("model_id", r.json())
        self.assertIsNone(r.json()["model_id"])


class SingleFileSDXLOnboardingTests(_RegistryTempDir):
    def setUp(self):
        super().setUp()
        self.client = TestClient(main.app)

    def test_single_file_checkpoint_refused_without_flag(self):
        pony_dir = self.asset_tmp / "models" / "Pony"
        pony_dir.mkdir(parents=True, exist_ok=True)
        (pony_dir / "pony.safetensors").write_bytes(b"\x00" * 1024)
        r = self.client.post("/api/models/onboard", json={
            "family": "sdxl",
            "local_path": str(pony_dir),
            "label": "Pony Test",
            "convert_if_single_file": False,
        })
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("single-file checkpoint", r.json()["detail"])

    def test_single_file_checkpoint_starts_conversion_with_flag(self):
        pony_dir = self.asset_tmp / "models" / "Pony"
        pony_dir.mkdir(parents=True, exist_ok=True)
        (pony_dir / "pony.safetensors").write_bytes(b"\x00" * 1024)
        with mock.patch("sdxl_convert.has_diffusers", return_value=True), \
             mock.patch("routers.downloads.start_sdxl_conversion", return_value={"task_id": "test1234", "status": "downloading", "model_name": "Pony"}) as mock_convert:
            r = self.client.post("/api/models/onboard", json={
                "family": "sdxl",
                "local_path": str(pony_dir),
                "label": "Pony Test",
                "convert_if_single_file": True,
            })
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body["status"], "converting")
            self.assertEqual(body["task_id"], "test1234")
            mock_convert.assert_called_once()

    def test_single_file_direct_path_refused_without_flag(self):
        pony_dir = self.asset_tmp / "models" / "Pony"
        pony_dir.mkdir(parents=True, exist_ok=True)
        file_path = pony_dir / "reapony_v110.safetensors"
        file_path.write_bytes(b"\x00" * 1024)
        r = self.client.post("/api/models/onboard", json={
            "family": "sdxl",
            "local_path": str(file_path),
            "label": "Pony Test",
            "convert_if_single_file": False,
        })
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("single-file checkpoint", r.json()["detail"])

    def test_single_file_direct_path_starts_conversion_with_flag(self):
        pony_dir = self.asset_tmp / "models" / "Pony"
        pony_dir.mkdir(parents=True, exist_ok=True)
        file_path = pony_dir / "reapony_v110.safetensors"
        file_path.write_bytes(b"\x00" * 1024)
        with mock.patch("sdxl_convert.has_diffusers", return_value=True), \
             mock.patch("routers.downloads.start_sdxl_conversion", return_value={"task_id": "test1234", "status": "downloading", "model_name": "Pony"}) as mock_convert:
            r = self.client.post("/api/models/onboard", json={
                "family": "sdxl",
                "local_path": str(file_path),
                "label": "Pony Test",
                "convert_if_single_file": True,
            })
            self.assertEqual(r.status_code, 200, r.text)
            body = r.json()
            self.assertEqual(body["status"], "converting")
            self.assertEqual(body["task_id"], "test1234")
            mock_convert.assert_called_once()

    def test_civitai_single_file_auto_converts_sdxl(self):
        name = "civitai-pony-test"
        dest_dir = self.asset_tmp / "models" / name
        dest_dir.mkdir(parents=True, exist_ok=True)

        def _fake_stream(url, **kwargs):
            (dest_dir / "reapony.safetensors").write_bytes(b"\x00" * 1024)
            return (_ScriptedStream(PAYLOAD), "http://127.0.0.1/reapony.safetensors")

        def _fake_convert(src, out_dir, **kwargs):
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "model_index.json").write_text(
                json.dumps({"unet": ["diffusers", "UNet2DConditionModel"]}), encoding="utf-8"
            )
            (out_dir / "unet").mkdir(parents=True, exist_ok=True)
            (out_dir / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"\x00")

        declaration = {
            "id": None, "label": "ReaPony", "family": "sdxl",
            "source": "civitai", "civitai_ids": [3003924],
            "civitai_model_id": 1353456, "civitai_version_id": 3003924,
        }
        tid = self._new_task({"declaration": declaration})
        version_id = 3003924
        with mock.patch.object(civitai_service, "open_public_https_stream", side_effect=_fake_stream), \
             mock.patch("sdxl_convert.has_diffusers", return_value=True), \
             mock.patch("sdxl_convert.convert", side_effect=_fake_convert):
            dl._run_civitai_model_download(tid, version_id, name, None, threading.Event())

        entries = custom_models.read_custom_models()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["family"], "sdxl")
        self.assertEqual(entries[0]["label"], "ReaPony")
        self.assertTrue((dest_dir / "model_index.json").is_file())


if __name__ == "__main__":
    unittest.main()

