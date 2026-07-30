import unittest
from pathlib import Path

from aitk_telemetry import configure_telemetry_environment, telemetry_enabled


REPO_ROOT = Path(__file__).resolve().parents[1]


class TelemetryConfigurationTest(unittest.TestCase):
    def test_telemetry_is_disabled_by_default(self):
        env = {}

        self.assertFalse(configure_telemetry_environment(env))
        self.assertEqual(env["DISABLE_TELEMETRY"], "YES")
        self.assertEqual(env["HF_HUB_DISABLE_TELEMETRY"], "1")

    def test_explicit_opt_in_enables_library_telemetry(self):
        env = {"AITK_TELEMETRY_ENABLED": "true"}

        self.assertTrue(configure_telemetry_environment(env))
        self.assertEqual(env["DISABLE_TELEMETRY"], "NO")
        self.assertEqual(env["HF_HUB_DISABLE_TELEMETRY"], "0")

    def test_false_values_override_inherited_library_settings(self):
        env = {
            "AITK_TELEMETRY_ENABLED": "0",
            "DISABLE_TELEMETRY": "NO",
            "HF_HUB_DISABLE_TELEMETRY": "0",
        }

        self.assertFalse(telemetry_enabled(env))
        self.assertFalse(configure_telemetry_environment(env))
        self.assertEqual(env["DISABLE_TELEMETRY"], "YES")
        self.assertEqual(env["HF_HUB_DISABLE_TELEMETRY"], "1")

    def test_entrypoints_configure_telemetry_before_toolkit_imports(self):
        for filename in ("run.py", "run_modal.py"):
            with self.subTest(filename=filename):
                source = (REPO_ROOT / filename).read_text(encoding="utf-8")
                configure_index = source.index("\nconfigure_telemetry_environment()\n")
                toolkit_import_index = source.find("\nfrom toolkit.")
                if toolkit_import_index >= 0:
                    self.assertLess(configure_index, toolkit_import_index)


if __name__ == "__main__":
    unittest.main()
