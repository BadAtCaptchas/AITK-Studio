import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock


PATHS_FILE = Path(__file__).resolve().parents[1] / "toolkit" / "paths.py"


def _load_paths_module():
    spec = importlib.util.spec_from_file_location("_aitk_test_paths", PATHS_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ModelsPathPrecedenceTests(unittest.TestCase):
    def test_nonblank_environment_value_wins(self):
        with mock.patch.dict(os.environ, {"MODELS_PATH": "D:/shared/models"}):
            module = _load_paths_module()
        self.assertEqual(module.MODELS_PATH, "D:/shared/models")

    def test_blank_environment_value_falls_back_to_repository_models(self):
        with mock.patch.dict(os.environ, {"MODELS_PATH": "   "}):
            module = _load_paths_module()
        expected = os.path.join(module.TOOLKIT_ROOT, "models")
        self.assertEqual(module.MODELS_PATH, expected)


if __name__ == "__main__":
    unittest.main()
