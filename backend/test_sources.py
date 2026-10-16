"""Tests for the shared source shapes.

`Item`, `Provenance` and `LicenseInfo` are the contract every source adapter fills and
the frontend reads. Field names are frozen here: renaming one later silently breaks the
JSON the UI already parses, so this file is the record of that contract.
"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sources  # noqa: E402


class ItemShapeTests(unittest.TestCase):
    def test_item_serialises_with_the_keys_the_frontend_reads(self):
        item = sources.Item(
            id="civitai/123/456", name="A LoRA", source="civitai",
            kind="lora", architecture="SDXL 1.0", quantization=None,
            size_bytes=1234, preview_url=None, license=None,
            provenance=sources.Provenance("civitai", "https://civitai.com/x",
                                          "123", "456", "v1.0"))
        self.assertEqual(sorted(item.to_dict()), [
            "architecture", "id", "kind", "license", "name", "preview_url",
            "provenance", "quantization", "size_bytes", "source"])

    def test_item_nests_its_provenance_as_a_dict(self):
        item = sources.Item(
            id="hf:x/y", name="n", source="huggingface", kind="diffusion",
            architecture=None, quantization="4bit", size_bytes=None,
            preview_url=None, license=None,
            provenance=sources.Provenance("huggingface", None, "x/y", None, None))
        self.assertEqual(item.to_dict()["provenance"]["model_id"], "x/y")

    def test_a_missing_field_serialises_as_null_not_a_guess(self):
        item = sources.Item(id="i", name="n", source="s", kind="lora",
                            architecture=None, quantization=None, size_bytes=None,
                            preview_url=None, license=None,
                            provenance=sources.Provenance("s", None, None, None, None))
        d = item.to_dict()
        self.assertIsNone(d["architecture"])
        self.assertIsNone(d["quantization"])
        self.assertIsNone(d["license"])


class LicenseInfoTests(unittest.TestCase):
    def test_license_info_carries_identifier_and_permissions(self):
        info = sources.LicenseInfo(identifier="other", url=None,
                                   source_url="https://civitai.com/x",
                                   permissions={"derivatives": True}, fetched_at=1.0)
        d = info.to_dict()
        self.assertEqual(d["identifier"], "other")
        self.assertIsNone(d["url"])
        self.assertEqual(d["permissions"], {"derivatives": True})
        self.assertEqual(sorted(d), ["fetched_at", "identifier", "permissions",
                                     "source_url", "url"])

    def test_an_absent_licence_is_none_not_an_empty_object(self):
        # A source that has nothing must serialise `null`, so the frontend can
        # tell "no data" from "licence with no terms".
        self.assertIsNone(sources.LicenseInfo().to_dict()["identifier"])
        self.assertIsNone(sources.LicenseInfo().to_dict()["permissions"])

    def test_the_other_identifier_survives_unchanged(self):
        # Civitai has no name; HF reports the literal "other". Global Constraint 1
        # forbids replacing either with an inferred name.
        self.assertEqual(
            sources.LicenseInfo(identifier="other").to_dict()["identifier"], "other")


class SourceBaseTests(unittest.TestCase):
    def test_source_declares_the_three_methods_adapters_implement(self):
        for method in ("search", "detail", "install"):
            with self.subTest(method=method):
                self.assertTrue(callable(getattr(sources.Source, method, None)))

    def test_source_has_a_name_attribute(self):
        self.assertTrue(hasattr(sources.Source, "name"))


class SourceAdapterTests(unittest.TestCase):
    """Adapters map the existing browse helpers onto `Item`.

    They deliberately do not re-implement filtering: `hf_browse.guess_architecture`,
    `guess_quantization` and `guess_kind` encode decisions that have been got wrong
    before, and duplicating them is how they drift apart a second time.
    """

    def test_hf_item_carries_repo_id_as_model_id(self):
        from sources import huggingface
        item = huggingface.to_item({
            "repo_id": "black-forest-labs/FLUX.1-dev",
            "label": "FLUX.1-dev",
            "architecture": "flux1",
            "architecture_label": "FLUX.1",
            "quantization": "q4",
            "kind": "diffusion",
            "install_name": "flux-1-dev",
            "downloads": 5, "likes": 2,
        })
        self.assertEqual(item.source, "huggingface")
        self.assertEqual(item.id, "hf:black-forest-labs/FLUX.1-dev")
        self.assertEqual(item.provenance.origin, "huggingface")
        self.assertEqual(item.provenance.model_id, "black-forest-labs/FLUX.1-dev")
        self.assertEqual(item.architecture, "FLUX.1")

    def test_hf_item_with_no_size_is_none_not_zero(self):
        from sources import huggingface
        item = huggingface.to_item({"repo_id": "a/b", "label": "b", "kind": "diffusion"})
        self.assertIsNone(item.size_bytes)

    def test_civitai_item_carries_model_and_version_ids_apart(self):
        from sources import civitai
        item = civitai.to_item({
            "id": "civitai/2890476",
            "source": "civitai",
            "civitai_model_id": 2890476,
            "civitai_version_id": 3306914,
            "label": "model.safetensors",
            "architecture": "flux2",
            "architecture_label": "FLUX.2",
            "quantization": None,
            "kind": "lora",
            "size_bytes": 2097152,
        })
        self.assertEqual(item.id, "civitai:2890476:3306914")
        self.assertEqual(item.provenance.model_id, "2890476")
        self.assertEqual(item.provenance.version_id, "3306914")
        self.assertEqual(item.size_bytes, 2097152)

    def test_civitai_item_with_no_version_is_still_returned(self):
        from sources import civitai
        item = civitai.to_item({
            "id": "civitai/1", "source": "civitai", "civitai_model_id": 1,
            "civitai_version_id": None, "label": "x", "kind": "lora",
        })
        self.assertIsNotNone(item)
        self.assertIsNone(item.provenance.version_id)
        self.assertEqual(item.id, "civitai:1:")

    def test_the_two_id_namespaces_cannot_collide(self):
        from sources import civitai, huggingface
        hf = huggingface.to_item({"repo_id": "x/y", "label": "y", "kind": "diffusion"})
        cv = civitai.to_item({"id": "civitai/x", "source": "civitai",
                              "civitai_model_id": "x", "label": "y", "kind": "diffusion"})
        self.assertNotEqual(hf.id, cv.id)
        self.assertTrue(hf.id.startswith("hf:"))
        self.assertTrue(cv.id.startswith("civitai:"))

    def test_both_adapters_expose_search(self):
        from sources import civitai, huggingface
        self.assertTrue(callable(getattr(huggingface.HuggingFaceSource, "search")))
        self.assertTrue(callable(getattr(civitai.CivitaiSource, "search")))

    def test_both_adapters_are_registered_under_their_names(self):
        import sources
        self.assertIn("huggingface", sources.SOURCES)
        self.assertIn("civitai", sources.SOURCES)


class DrawThingsSourceTests(unittest.TestCase):
    """Listing and identifying Draw Things `.ckpt` files.

    Builds its own tiny SQLite `.ckpt` files in a temp directory and passes that
    directory explicitly. It deliberately does NOT read the real user folder:
    `Path.home()` is monkeypatched process-wide by `test_model_download_path.py`, so a
    test that reaches for the real directory sees an empty temp dir whenever that test
    runs first -- which is exactly how three of these failed only under a full
    `unittest discover`. The logic under test is identical either way, and the real
    folder is exercised by Task 13's gate against real files.
    """

    def setUp(self):
        import sqlite3
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        # (filename, [tensor names]) -- a LoRA has up/down, a model has plain weights.
        self._mk("a_sdxl_lora.ckpt", ["__unet__[t-1-0]__up__", "__unet__[t-1-0]__down__"])
        self._mk("b_flux_lora.ckpt", ["__dit__[t-x_q-0-0]__up__", "__dit__[t-x_q-0-0]__down__"])
        self._mk("c_sdxl_model.ckpt", ["__unet__[t-2-0]", "__unet__[t-2-1]"])
        self._mk("d_flux_model.ckpt", ["__dit__[t-x_v-9-1]", "__dit__[t-x_k-9-1]"])
        self._mk("not_a_ckpt.txt", ["ignored"])
        self.addCleanup(self.tmp.cleanup)

    def _mk(self, name, tensor_names):
        import sqlite3
        path = self.dir / name
        db = sqlite3.connect(str(path))
        db.execute("CREATE TABLE tensors (name TEXT)")
        db.executemany("INSERT INTO tensors VALUES (?)",
                       [(n,) for n in tensor_names])
        db.commit()
        db.close()
        return path

    def _files(self):
        from sources import drawthings
        return drawthings.list_ckpt_files(self.dir)

    def test_only_ckpt_files_are_listed(self):
        names = {f.name for f in self._files()}
        self.assertEqual(names, {"a_sdxl_lora.ckpt", "b_flux_lora.ckpt",
                                 "c_sdxl_model.ckpt", "d_flux_model.ckpt"})

    def test_listing_is_sorted_by_name(self):
        got = [f.name for f in self._files()]
        self.assertEqual(got, sorted(got))

    def test_every_entry_reports_a_positive_size(self):
        for f in self._files():
            self.assertGreater(f.size_bytes, 0, f.name)

    def test_lora_and_model_are_told_apart(self):
        kinds = {f.name: f.kind for f in self._files()}
        self.assertEqual(kinds["a_sdxl_lora.ckpt"], "lora")
        self.assertEqual(kinds["b_flux_lora.ckpt"], "lora")
        self.assertEqual(kinds["c_sdxl_model.ckpt"], "model")
        self.assertEqual(kinds["d_flux_model.ckpt"], "model")

    def test_only_loras_carry_a_base_model(self):
        for f in self._files():
            if f.kind == "lora":
                self.assertIn(f.base_model, ("sdxl", "flux"), f.name)
            else:
                self.assertIsNone(f.base_model, f.name)

    def test_listing_never_writes(self):
        import hashlib
        def fp():
            h = hashlib.sha256()
            for p in sorted(self.dir.glob("*")):
                st = p.stat()
                h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}".encode())
            return h.hexdigest()
        before = fp()
        self._files()
        self.assertEqual(before, fp())

    def test_a_missing_directory_lists_empty_rather_than_raising(self):
        from sources import drawthings
        self.assertEqual(drawthings.list_ckpt_files(self.dir / "nope"), [])

    def test_flux_is_detected_from_any_dit_prefix(self):
        from sources import drawthings
        self.assertEqual(
            drawthings.detect_base_model_names(
                ["__unet__[t-25-0]__up__", "__dit__[t-x_q-0-0]__up__"]), "flux")

    def test_sdxl_is_detected_when_no_dit_prefix_is_present(self):
        from sources import drawthings
        self.assertEqual(
            drawthings.detect_base_model_names(["__unet__[t-25-0]__up__"]), "sdxl")

    def test_an_empty_tensor_list_is_unknown_not_sdxl(self):
        # "sdxl" as a default would mislabel anything we cannot read.
        from sources import drawthings
        self.assertIsNone(drawthings.detect_base_model_names([]))

    def test_an_unrecognisable_prefix_is_unknown(self):
        from sources import drawthings
        self.assertIsNone(drawthings.detect_base_model_names(["garbage", "nope"]))

    def test_kind_detection_is_pure_on_names(self):
        from sources import drawthings
        self.assertEqual(drawthings.detect_kind_names(
            ["__unet__[t-25-0]__up__", "__unet__[t-25-0]__down__"]), "lora")
        self.assertEqual(drawthings.detect_kind_names(
            ["__unet__[t-25-0]", "__dit__[t-x_v-9-1]"]), "model")
        self.assertIsNone(drawthings.detect_kind_names([]))

    def test_detection_reads_only_names_not_payloads(self):
        # Listing must stay fast on a 777 MB file, so it reads the tensors table's
        # name column and nothing else.
        from sources import drawthings
        self.assertEqual(drawthings.TENSOR_NAME_QUERY.strip(), "SELECT name FROM tensors")

    def test_search_filters_by_kind(self):
        from sources.drawthings import DrawThingsSource
        items = DrawThingsSource().search("", kind="lora", limit=100, directory=self.dir)
        self.assertEqual({i.name for i in items},
                         {"a_sdxl_lora.ckpt", "b_flux_lora.ckpt"})


class DrawThingsModelDirTests(unittest.TestCase):
    """`model_dir()` resolves lazily so a patched HOME is honoured per call."""

    def test_model_dir_is_a_function_not_a_baked_constant(self):
        from sources import drawthings
        self.assertTrue(callable(drawthings.model_dir))

    def test_model_dir_points_into_the_containers_tree(self):
        from sources import drawthings
        self.assertIn("com.liuliu.draw-things", str(drawthings.model_dir()))


class DiffusionBeeSourceTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from test_sources_diffusionbee import create_synthetic_tdict
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.assets = self.dir / "downloaded_assets"
        self.assets.mkdir()
        create_synthetic_tdict(self.assets / "my_lora.tdict", {
            "lora_unet_down.weight_lora_up": {"shape": [4, 64], "dtype": "float32"},
            "lora_unet_down.weight_lora_down": {"shape": [64, 4], "dtype": "float32"},
        })
        self.addCleanup(self.tmp.cleanup)

    def test_diffusionbee_adapter_is_registered(self):
        import sources
        self.assertIn("diffusionbee", sources.SOURCES)

    def test_diffusionbee_search_returns_item_shape(self):
        from sources.diffusionbee import DiffusionBeeSource
        source = DiffusionBeeSource()
        items = source.search("", limit=10, directories=[self.assets])
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.source, "diffusionbee")
        self.assertEqual(item.id, "diffusionbee:my_lora.tdict")
        self.assertEqual(item.name, "my_lora.tdict")
        self.assertEqual(item.kind, "lora")
        self.assertIsNone(item.license)
        self.assertEqual(item.provenance.origin, "diffusionbee")


if __name__ == "__main__":
    unittest.main()
