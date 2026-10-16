"""SDXL engine errors must stay readable without exposing tracebacks."""

import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parent

import generator  # noqa: E402


class SdxlEngineFailurePresentationTests(unittest.TestCase):
    def test_traceback_stays_in_diagnostics_not_user_error(self):
        engine_error = "TypeError: expected str, bytes or os.PathLike object, not NoneType"
        stderr = "\n".join(
            [
                "Traceback (most recent call last):",
                '  File "<engine>", line 42, in generate',
                engine_error,
            ]
        )
        user_message, diagnostic = generator._format_sdxl_engine_failure(
            engine_error, stderr, "abc123"
        )

        self.assertIn("TypeError: expected str", user_message)
        self.assertIn("abc123", user_message)
        self.assertNotIn("Traceback (most recent call last):", user_message)
        self.assertIn("Traceback (most recent call last):", diagnostic)

    def test_blank_engine_error_has_a_safe_fallback(self):
        user_message, _ = generator._format_sdxl_engine_failure("", "traceback", "abc123")
        self.assertEqual(
            user_message,
            "SDXL engine failed (engine diagnostics logged for generation abc123)",
        )

    def test_generation_path_uses_the_formatter(self):
        source = (BACKEND / "generator.py").read_text(encoding="utf-8")
        self.assertIn("_format_sdxl_engine_failure(", source)
        self.assertIn('result.get("error")', source)
        self.assertNotIn("--- engine traceback ---", source)


if __name__ == "__main__":
    unittest.main()
