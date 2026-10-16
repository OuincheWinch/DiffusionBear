"""Which architectures DiffusionBear can actually execute, and how it decides.

Two separate questions that used to be conflated:

  * "is this architecture executable?" -- we have a pipeline that runs it.
  * "is this architecture *present* on disk?" -- a folder of weights exists.

The UI once showed RealVisXL beside the claim "not runnable here", which is exactly
the confusion these tests exist to prevent. Present is not runnable, and the registry
is the only authority on runnable.
"""

import json
import tempfile
import unittest
from pathlib import Path

import generator


class ExecutableArchitectureTests(unittest.TestCase):
    """The registry is the single source of truth for what runs."""

    def test_every_registry_entry_declares_an_engine_we_dispatch(self):
        """A model with no dispatch silently 400s at generate time.

        Adding a registry entry without wiring the engine is the easiest mistake to
        make and the most expensive to discover, so it is pinned here.
        """
        known = {"mflux", "qwen", "sdxl"}
        for key, info in generator.MODELS.items():
            engine = info.get("engine")
            self.assertIn(
                engine, known,
                f"{key}: engine={engine!r} is not one of {sorted(known)}. "
                f"Add the dispatch in generator.generate() or the model will be "
                f"listed in the UI and then fail at generation time.",
            )

    def test_every_registry_entry_can_run_at_some_quantization(self):
        """Guards the inverse: an engine that can never be reached at any setting."""
        pinned = ("krea2-turbo", "qwen-image-2.1", "qwen-image-2.1-turbo")
        for key, info in generator.MODELS.items():
            with self.subTest(model=key):
                if key in pinned:
                    # These are pinned to 4-bit because other widths are slower or
                    # break the text-encoder quantisation pass. Not a defect.
                    self.assertEqual(generator._effective_quantization(key, 4), 4)
                else:
                    self.assertEqual(generator._effective_quantization(key, 8), 8)

    def test_krea_and_qwen_refuse_other_quantizations_with_a_reason(self):
        """A silent clamp would make the user think they changed something."""
        for key in ("krea2-turbo", "qwen-image-2.1", "qwen-image-2.1-turbo"):
            with self.subTest(model=key):
                with self.assertRaises(ValueError) as ctx:
                    generator._effective_quantization(key, 8)
                self.assertIn("4-bit", str(ctx.exception))


class ArchitectureDetectionTests(unittest.TestCase):
    """Detect the architecture from what the checkpoint declares.

    The old path substring-matched the folder name -- "flux", "krea", "z-image" --
    which is a guess. It mis-classified anything whose name did not happen to contain
    the magic word, and it had no way to say "I do not recognise this".
    """

    def _write_model_index(self, root: Path, payload: dict) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "model_index.json").write_text(json.dumps(payload), encoding="utf-8")
        return root

    def test_reads_the_class_name_a_diffusers_repo_declares(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._write_model_index(Path(tmp) / "mystery", {
                "_class_name": "StableDiffusionXLPipeline",
                "_diffusers_version": "0.35.0",
            })
            self.assertEqual(
                generator.detect_architecture(root), "sdxl",
                "a repo that declares StableDiffusionXLPipeline is SDXL, whatever "
                "the folder is called",
            )

    def test_returns_none_for_an_architecture_we_cannot_execute(self):
        """Say 'unknown' rather than guessing. Guessing is what lies to the user."""
        with tempfile.TemporaryDirectory() as tmp:
            root = self._write_model_index(Path(tmp) / "something-else", {
                "_class_name": "AnimaPipeline",
            })
            self.assertIsNone(
                generator.detect_architecture(root),
                "Anima is a Cosmos Predict2 DiT we have no pipeline for; it must "
                "report unknown, not borrow another engine's identity",
            )

    def test_folder_name_alone_never_decides_the_architecture(self):
        """A folder called 'flux-something' that is not FLUX must not be claimed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "flux-not-really"
            root.mkdir(parents=True)
            (root / "model_index.json").write_text(
                json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8"
            )
            self.assertEqual(
                generator.detect_architecture(root), "sdxl",
                "the declaration wins over the folder name",
            )

    def test_a_directory_without_an_index_is_unknown_not_a_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "krea-lookalike"
            root.mkdir(parents=True)
            self.assertIsNone(generator.detect_architecture(root))

    def test_a_corrupt_index_is_unknown_rather_than_an_exception(self):
        """A half-written download must not crash the registration path."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "broken"
            root.mkdir(parents=True)
            (root / "model_index.json").write_text("{not json", encoding="utf-8")
            self.assertIsNone(generator.detect_architecture(root))


class PONYCheckpointTests(unittest.TestCase):
    """A PONY finetune is an SDXL checkpoint. It has to be registerable as one."""

    def test_pony_is_detected_as_sdxl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ponyDiffusionV6XL_v6StartWithThisOne"
            root.mkdir(parents=True)
            (root / "model_index.json").write_text(
                json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8"
            )
            (root / "model.safetensors").write_bytes(b"x" * 64)
            self.assertEqual(generator.detect_architecture(root), "sdxl")


class ArchitectureVocabularyTests(unittest.TestCase):
    """detect_architecture's vocabulary and the UI's label map must not drift.

    These were two different sets. detect_architecture returned "mflux" while the
    label map only knew "flux2", so every FLUX checkpoint fell through and kept the
    name-derived guess -- exactly the bug the shared function was meant to remove.
    """

    def test_every_detectable_architecture_has_a_ui_label(self):
        import routers.downloads as downloads

        mappable = set(getattr(downloads, "ARCHITECTURE_LABELS", {}))
        returned = {"sdxl", "mflux"}
        self.assertTrue(
            returned <= mappable,
            f"detect_architecture can return {sorted(returned - mappable)} but "
            f"routers.downloads.ARCHITECTURE_LABELS only knows {sorted(mappable)}",
        )


if __name__ == "__main__":
    unittest.main()