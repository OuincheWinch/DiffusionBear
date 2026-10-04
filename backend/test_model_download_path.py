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
