import copy
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.request import url2pathname

from manager import env, spec


class ManagerNattenTests(unittest.TestCase):
    def setUp(self):
        self.detection = {
            "os": "windows",
            "arch": "x86_64",
            "backend": "cuda",
            "nvidia": {"cuda_version": "13.0", "gpus": [{"compute_cap": "12.0"}]},
        }

    def natten_packages(self, detection):
        return [p for p in spec.build_spec(detection).optional_packages if "natten" in p]

    def test_matching_windows_selects_real_wheel_with_current_abi(self):
        packages = self.natten_packages(self.detection)
        self.assertEqual(len(packages), 1)
        self.assertEqual(env._optional_names(packages[0]), ("natten", "natten"))
        url = urlsplit(packages[0])
        self.assertEqual(url.scheme, "file")
        wheel = Path(url2pathname(url.path))
        self.assertTrue(wheel.is_file())
        with zipfile.ZipFile(wheel) as archive:
            metadata = archive.read(next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))).decode()
            self.assertIn("Version: " + spec._natten_pin("cu130").split("==")[1], metadata)
            self.assertIn("natten/libnatten.cp312-win_amd64.pyd", archive.namelist())

    def test_unknown_or_unsupported_gpu_configurations_skip(self):
        for caps in ([], [None], ["[N/A]"], ["nan"], ["8.9"], ["12.1"], ["12.0", "8.9"], ["12.0", None]):
            with self.subTest(caps=caps):
                detection = copy.deepcopy(self.detection)
                detection["nvidia"]["gpus"] = [{"compute_cap": cap} for cap in caps]
                self.assertEqual(self.natten_packages(detection), [])

    def test_cuda_and_python_must_match(self):
        self.assertIsNone(spec._windows_natten_wheel(self.detection, "cu126", "3.12"))
        self.assertIsNone(spec._windows_natten_wheel(self.detection, "cu130", "3.11"))
        self.assertIsNone(spec._windows_natten_wheel(self.detection, "cu130", "3.13"))

    def test_pin_bumps_do_not_reuse_old_wheel(self):
        with patch.object(spec, "TORCH", dict(spec.TORCH, torch="2.14.0")):
            self.assertEqual(self.natten_packages(self.detection), [])
        with patch.object(spec, "NATTEN_VERSION", "0.99.0"):
            self.assertEqual(self.natten_packages(self.detection), [])

    def test_missing_artifact_skips_without_breaking_install(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(spec, "NATTEN_WINDOWS_WHEELS_DIR", directory):
            result = spec.build_spec(self.detection)
            self.assertFalse(any("natten" in p for p in result.optional_packages))
            self.assertTrue(any("No bundled NATTEN wheel matches" in note for note in result.notes))

    def test_paths_with_spaces_are_installable_file_urls(self):
        with tempfile.TemporaryDirectory(prefix="natten test ") as directory:
            original = self.natten_packages(self.detection)[0]
            name = Path(url2pathname(urlsplit(original).path)).name
            (Path(directory) / name).touch()
            with patch.object(spec, "NATTEN_WINDOWS_WHEELS_DIR", directory):
                selected = self.natten_packages(self.detection)[0]
            self.assertIn("%20", selected)
            self.assertEqual(Path(url2pathname(urlsplit(selected).path)), Path(directory) / name)

    def test_linux_keeps_upstream_wheel_selection(self):
        detection = dict(self.detection, os="linux")
        result = spec.build_spec(detection)
        self.assertIn(spec._natten_pin("cu130"), result.optional_packages)
        self.assertIn(spec.NATTEN_FIND_LINKS, result.find_links)
        self.assertFalse(any(p.startswith("file:") for p in result.optional_packages))

    def test_windows_arm_fallback_selects_x64_wheel(self):
        detection = dict(self.detection, arch="aarch64")
        with patch.dict(os.environ, {"AITK_SPARK_NATIVE": "0"}):
            result = spec.build_spec(detection)
        self.assertEqual(result.uv_python, "cpython-3.12-windows-x86_64-none")
        self.assertEqual(
            [p for p in result.optional_packages if "natten" in p],
            self.natten_packages(self.detection),
        )

    def test_existing_python_filter_rejects_wrong_interpreter(self):
        package = self.natten_packages(self.detection)[0]
        with patch.object(env, "venv_python_version", return_value="3.11"), patch.object(env, "warn"):
            self.assertEqual(env._filter_extras([package]), [])

    def test_runtime_validation_rejects_silent_native_import_failure(self):
        # NATTEN can import successfully even when its extension fails to load.
        # Exercise the real subprocess check with both possible module states.
        with tempfile.TemporaryDirectory() as directory:
            module = Path(directory) / "natten.py"
            for has_library in (False, True):
                with self.subTest(has_library=has_library):
                    module.write_text("HAS_LIBNATTEN = %r\n" % has_library)
                    child_env = dict(os.environ, PYTHONPATH=directory, PYTHONDONTWRITEBYTECODE="1")
                    with patch.object(env, "venv_python", return_value=sys.executable), patch.object(env, "clean_env", return_value=child_env):
                        self.assertEqual(env._venv_import_ok("natten"), has_library)


if __name__ == "__main__":
    unittest.main()
