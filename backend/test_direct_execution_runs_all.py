"""Direct execution must run the whole file, not just what's above unittest.main().

C4: test_i18n.py, test_review_bugs.py and test_hf_browse.py each had
`if __name__ == "__main__": unittest.main()` placed MID-FILE. `unittest.main()`
runs the tests defined SO FAR in `__main__`, so `python test_i18n.py` executed
only the classes above line 292 and silently skipped the five classes below --
including the language-registry and translation-integrity regressions.

`unittest discover` imports the full module, so the suite stayed green at 1050
and nobody noticed. The most common developer workflow -- running one file
directly -- was the one that dropped the tests.

The break this catches: any placement of the unittest.main() guard that leaves
test classes below it. It does NOT assert on the guard's text position (a grep
on source proves only that the source is the source). It runs each file as
`python <file>` in a subprocess -- the real direct-execution path -- and
compares the executed count against what the stdlib loader finds on full
import. Those two numbers must agree; today they do not.

The expected counts are deliberately NOT pinned as literals: a literal would
break on every legitimately added test. The contract is direct == import, plus
a floor so that "Ran 0 tests" on both sides cannot pass as agreement.
"""
import re
import subprocess
import sys
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parent

FILES = ("test_i18n.py", "test_review_bugs.py", "test_hf_browse.py")


def _import_count(module_name: str) -> int:
    """TestCases the stdlib loader finds on full module import.

    Import executes the whole file, so mid-file unittest.main() placement is
    irrelevant here -- this side is the reference, computed by unittest itself,
    not by anything under test.
    """
    return unittest.TestLoader().loadTestsFromName(module_name).countTestCases()


def _direct_run(path: Path) -> tuple[int, int, str]:
    """(returncode, executed count, stderr tail) of running the file directly."""
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        timeout=600,
    )
    match = re.search(r"^Ran (\d+) tests?", proc.stderr, flags=re.M)
    count = int(match.group(1)) if match else -1
    return proc.returncode, count, proc.stderr[-1500:]


class TestDirectExecutionRunsEveryTestClass(unittest.TestCase):
    def test_direct_execution_matches_import_discovery(self):
        for filename in FILES:
            with self.subTest(file=filename):
                path = BACKEND / filename
                imported = _import_count(path.stem)
                self.assertGreater(
                    imported, 0, f"{filename} has no tests at all -- nothing to compare"
                )
                returncode, executed, tail = _direct_run(path)
                self.assertEqual(
                    returncode, 0,
                    f"direct run of {filename} failed:\n{tail}",
                )
                self.assertEqual(
                    executed, imported,
                    f"{filename}: direct execution ran {executed} tests but full "
                    f"import finds {imported} -- classes below a mid-file "
                    f"unittest.main() are being skipped",
                )


if __name__ == "__main__":
    unittest.main()
