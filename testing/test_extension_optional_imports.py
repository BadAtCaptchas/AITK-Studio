import unittest

import toolkit.extension as extension_module
from toolkit.extension import Extension


class GoodProcess:
    pass


class GoodExtension(Extension):
    uid = "good"
    name = "Good"

    @classmethod
    def get_process(cls):
        return GoodProcess


class MissingDependencyExtension(Extension):
    uid = "missing"
    name = "Missing Dependency"

    @classmethod
    def get_process(cls):
        raise ImportError("missing optional dependency")


class ExtensionOptionalImportTest(unittest.TestCase):
    def test_process_map_keeps_unavailable_extensions_lazy(self):
        original_get_all_extensions = extension_module.get_all_extensions
        extension_module.get_all_extensions = lambda: [
            GoodExtension,
            MissingDependencyExtension,
        ]
        try:
            process_dict = extension_module.get_all_extensions_process_dict()
        finally:
            extension_module.get_all_extensions = original_get_all_extensions

        self.assertIs(process_dict["good"], GoodProcess)
        self.assertIn("missing", process_dict)
        with self.assertRaisesRegex(
            ImportError,
            "Missing Dependency .*missing optional dependency",
        ):
            process_dict["missing"](0, None, {})


if __name__ == "__main__":
    unittest.main()
