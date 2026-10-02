"""The wired budget must be derived from the machine, not carried as a constant.

The generic default was a hardcoded 9 GB and krea2's was a persisted 9 GB pin, both
hand-tuned on ONE 16GB M1 and then stored in settings.json, where they travelled to
every other Mac. On this machine that stale krea pin was the reason the Krea 2
text-encoder quantisation pass no longer fit: it transiently holds a 7.5GB bf16 copy
AND its ~1.9GB q4 result, and died with a Metal CommandBuffer OOM during loading_model.
"""
import os
import re
import unittest
from pathlib import Path

GEN = Path(__file__).resolve().parent / "generator.py"
SETTINGS = Path(__file__).resolve().parent / "app_settings.py"


class DerivedBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = GEN.read_text(encoding="utf-8")

    def test_the_device_is_consulted_for_the_default(self):
        self.assertIn("max_recommended_working_set_size", self.src)
        self.assertIn("def _derived_wired_budget_bytes", self.src)
        self.assertRegex(self.src, r"mem \* _WIRED_MEMORY_FRACTION")

    def test_derived_never_exceeds_apples_own_recommendation(self):
        """The device's recommended working set is the ceiling; a fraction of RAM above it
        would ask Metal for memory Apple already said is unwise."""
        block = re.search(r"def _derived_wired_budget_bytes.*?(?=\ndef )", self.src, re.S)
        self.assertIsNotNone(block)
        self.assertIn("value = min(value, int(cap))", block.group(0))

    def test_the_derived_value_is_cached(self):
        self.assertIn("_device_budget_cache", self.src)

    def test_auto_means_derive_and_zero_means_unbounded(self):
        """They are different intents and must not collapse into each other."""
        block = re.search(r"def _resolve_wired_budget.*?(?=\ndef )", self.src, re.S).group(0)
        self.assertRegex(block, r"if setting_gb is None:\s*\n\s*return _derived_wired_budget_bytes\(\)")
        self.assertRegex(block, r"if setting_gb <= 0:\s*\n\s*return 0")

    def test_an_explicit_pin_is_still_honoured(self):
        """Auto is a default, not a policy: someone who knows their machine can still pin."""
        block = re.search(r"def _resolve_wired_budget.*?(?=\ndef )", self.src, re.S).group(0)
        self.assertRegex(block, r"return _wired_budget_bytes\(int\(setting_gb \* \(1 << 30\)\)")

    def test_no_hardcoded_gigabyte_default_remains(self):
        """The whole point: a constant tuned on one machine must not be the default."""
        for name in ("_WIRED_LIMIT_GB", "_KREA_WIRED_LIMIT_GB"):
            self.assertNotRegex(
                self.src,
                rf"{name}\s*=\s*int\(os\.environ\.get\([^)]*\"(?!auto)\d+\"",
                f"{name} still defaults to a fixed GB constant",
            )

    def test_auto_is_an_accepted_env_value(self):
        self.assertRegex(self.src, r"raw\.strip\(\)\.lower\(\) == \"auto\"")

    def test_an_invalid_env_value_falls_back_to_deriving(self):
        """Previously an unparseable value silently became 9 -- a constant from nowhere."""
        self.assertIn("deriving from device instead", self.src)

    def test_status_reports_configured_and_derived_separately(self):
        """The UI must be able to say "Auto". If only the derived figure were reported it
        would look typed, and the next save would persist the machine-specific constant
        straight back into settings.json."""
        block = re.search(r'status\["wired"\] = \{(.*?)\n    \}', self.src, re.S).group(1)
        self.assertIn('"generic_auto"', block)
        self.assertIn('"generic_derived_gb"', block)
        self.assertIn('"krea_auto"', block)
        self.assertIn('"krea_derived_gb"', block)


class TileSizeSettingTests(unittest.TestCase):
    def test_the_tile_size_is_a_validated_setting(self):
        s = SETTINGS.read_text(encoding="utf-8")
        self.assertIn('"krea_vae_tile_size": None', s)
        self.assertRegex(s, r'"krea_vae_tile_size":\s*lambda v:.*0 <= v <= 4096')

    def test_tiling_is_applied_to_the_krea2_pipeline(self):
        """Anchored on the Krea2 CONSTRUCTION, not on the model_id string.

        `elif model_id == "krea2-turbo"` appears more than once in this file, so matching
        the branch by its id silently picked an unrelated one and the assertion passed or
        failed for the wrong reason."""
        src = GEN.read_text(encoding="utf-8")
        ctor = src.index("_pipeline = Krea2(")
        # The tiling call must come after the pipeline exists, and before the branch ends.
        apply_at = src.index("_apply_krea_vae_tiling(_pipeline)")
        self.assertGreater(apply_at, ctor, "tiling configured before the pipeline was built")
        nxt = src.index('elif model_id == "qwen-image-2.1"', ctor)
        self.assertLess(apply_at, nxt, "tiling call fell outside the krea2 branch")

    def test_the_config_requests_more_than_one_tile(self):
        """VAEUtil.decode only tiles when tiles_per_dim > 1; a config with 1 or 0 is a
        silent no-op that would look like the flag had been applied."""
        src = GEN.read_text(encoding="utf-8")
        self.assertRegex(src, r"vae_decode_tiles_per_dim=8")

    def test_zero_tile_size_restores_untiled_decoding(self):
        src = GEN.read_text(encoding="utf-8")
        block = re.search(r"def _apply_krea_vae_tiling.*?(?=\ndef )", src, re.S).group(0)
        self.assertRegex(block, r"if size <= 0:\s*\n\s*return")


class MigrationRecordTests(unittest.TestCase):
    """The stale constants are gone from the live settings the app actually reads."""

    def test_the_running_settings_file_is_on_auto(self):
        path = Path(
            "/Volumes/Externe/IA/MLX-DIFFUSION OpenCode/backend/data/settings.json"
        )
        if not path.exists():
            self.skipTest("live settings file not present on this machine")
        import json

        data = json.loads(path.read_text())
        self.assertIsNone(
            data.get("memory_wired_limit_gb"),
            "a machine-specific constant is persisted again; that is what caused the OOM",
        )
        self.assertIsNone(data.get("memory_krea_wired_limit_gb"))

    def test_a_backup_of_the_previous_values_exists(self):
        path = Path(
            "/Volumes/Externe/IA/MLX-DIFFUSION OpenCode/backend/data/settings.json"
            ".bak-before-auto-wired-limit"
        )
        if not path.exists():
            self.skipTest("no backup on this machine")
        import json

        data = json.loads(path.read_text())
        self.assertEqual(data.get("memory_krea_wired_limit_gb"), 9.0)
        self.assertEqual(data.get("memory_wired_limit_gb"), 7.0)
