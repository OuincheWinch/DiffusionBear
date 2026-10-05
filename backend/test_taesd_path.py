"""TAESD must be reachable from the installed app, not just from the repo.

THE FAILURE THIS GUARDS
`get_taesd_decoder()` defaulted to

    Path(__file__).parent / "data" / "models" / "taesdxl" / ...

which is right in the repo checkout and wrong in the installed app: inside the
signed bundle `__file__` is `Contents/Resources/backend`, so the lookup resolved
to `Contents/Resources/backend/data/models/taesdxl` -- inside the signature. The
weights sit in the model store, so the file was never found.

The consequence was silent. `sdxl_engine.py` wraps the call in `except Exception`
and falls back to the full VAE, printing one line to stderr. So the "~0.5s
ultra-fast TAESD decode" was dead in every shipped build, and the only evidence
was a stderr line nobody reads. No test failed, because no test asked.

WHY THIS FILE DOES NOT `import taesd_mlx`
That module imports `mlx.core` at module scope, and CI runs on Linux where mlx
does not exist -- importing it made this test ERROR there, which is how it was
found. A test that cannot run in CI cannot guard anything. So the function under
test is extracted from the source and executed with only `Path` and
`app_settings` in scope, the same technique test_size_selector.py uses. The guard
now runs on every platform, including the one that ships the bundle.

WHAT IS ASSERTED
That the default resolves through ASSET_DIR (so it honours `store_path`), that it
never points back inside the bundle, and that it follows a relocated store --
which cannot be observed in the checkout, because there ASSET_DIR *is*
backend/data and the old path was accidentally correct.
"""

import ast
import re
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

import app_settings  # noqa: E402

MODULE_SOURCE = BACKEND / "taesd_mlx.py"


def _load_default_path_function():
    """Exec just `_default_taesd_path`, without importing mlx."""
    source = MODULE_SOURCE.read_text(encoding="utf-8")
    match = re.search(
        r"^def _default_taesd_path\(\).*?^\n(?=\S)", source, flags=re.S | re.M
    )
    if not match:
        return None
    namespace = {"Path": Path, "app_settings": app_settings}
    exec(compile(match.group(0), str(MODULE_SOURCE), "exec"), namespace)
    return namespace.get("_default_taesd_path")


_default_path = _load_default_path_function()


@unittest.skipIf(_default_path is None, "taesd_mlx.py no longer defines _default_taesd_path")
class DefaultTaesdPathTests(unittest.TestCase):
    def setUp(self):
        self.path = _default_path()

    def test_it_resolves_under_the_model_store(self):
        self.assertEqual(
            self.path.parent.parent.parent,
            Path(app_settings.ASSET_DIR).resolve(),
            "TAESD must be looked up in ASSET_DIR so store_path relocation is honoured",
        )

    def test_it_never_points_inside_the_bundle(self):
        """The exact defect: resolving to Contents/Resources/backend/data."""
        text = str(self.path)
        self.assertNotIn("Contents/Resources", text)
        self.assertNotIn("__file__", text)

    def test_it_follows_a_relocated_store(self):
        """The invariant that matters, and cannot be seen in the repo checkout.

        In the checkout ASSET_DIR *is* backend/data, so the old __file__/data path
        and the new one coincide -- the bug was invisible here and only appeared
        once the app was installed. So point ASSET_DIR somewhere else entirely and
        assert the lookup moves with it.
        """
        original = app_settings.ASSET_DIR
        with tempfile.TemporaryDirectory() as tmp:
            app_settings.ASSET_DIR = Path(tmp)
            try:
                moved = _default_path()
            finally:
                app_settings.ASSET_DIR = original
        self.assertEqual(
            moved,
            Path(tmp) / "models" / "taesdxl" / "diffusion_pytorch_model.safetensors",
            "the TAESD lookup must follow ASSET_DIR, not the module location",
        )

    def test_the_filename_is_still_the_one_on_disk(self):
        self.assertEqual(self.path.name, "diffusion_pytorch_model.safetensors")

    def test_the_weights_exist_where_it_points(self):
        if not self.path.exists():
            self.skipTest(f"TAESD weights are not on this machine: {self.path}")
        self.assertGreater(self.path.stat().st_size, 1_000_000, "weights look truncated")


class NoHardcodedStorePathTests(unittest.TestCase):
    """A source-level guard, so the regression cannot reappear in any refactor.

    Checked against EXECUTABLE CODE ONLY. A plain substring search fails here: the
    function's docstring explains the bug and therefore names both `__file__` and
    Contents/Resources, so grepping the raw file flags the very documentation that
    makes the defect understandable. Both assertions below walk the AST and skip
    docstrings.
    """

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(MODULE_SOURCE.read_text(encoding="utf-8"))

    @staticmethod
    def _docstring_nodes(tree):
        """Every string literal that IS a docstring, so it can be excluded."""
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    if isinstance(body[0].value.value, str):
                        found.add(id(body[0].value))
        return found

    def test_no_executable_string_mentions___file___or_the_bundle(self):
        docstrings = self._docstring_nodes(self.tree)
        offenders = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings:
                continue
            for bad in ("__file__", "Contents/Resources", "/Applications/"):
                if bad in node.value:
                    offenders.append(f"line {node.lineno}: {bad}")
        self.assertEqual(
            offenders, [],
            "taesd_mlx.py must resolve weights through ASSET_DIR, not a path literal",
        )

    def test___file___is_never_used_as_a_path_root(self):
        docstrings = self._docstring_nodes(self.tree)
        uses = [
            node.lineno
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "__file__"
            and id(getattr(node, "value", None)) not in docstrings
        ]
        self.assertEqual(
            uses, [],
            "taesd_mlx.py must not build a path from __file__; use app_settings.ASSET_DIR",
        )


if __name__ == "__main__":
    unittest.main()