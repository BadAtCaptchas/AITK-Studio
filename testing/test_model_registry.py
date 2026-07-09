import unittest
from types import SimpleNamespace
from unittest import mock

from toolkit.util import get_model


class ModelRegistryTest(unittest.TestCase):
    def test_repeated_discovery_does_not_mutate_builtins(self):
        original = list(get_model.BUILT_IN_MODELS)
        with mock.patch.object(get_model.pkgutil, 'iter_modules', return_value=[]):
            first = get_model.get_all_models()
            second = get_model.get_all_models()

        self.assertEqual(get_model.BUILT_IN_MODELS, original)
        self.assertEqual(first, original)
        self.assertEqual(second, original)

    def test_unknown_architecture_fails_instead_of_using_legacy_model(self):
        with mock.patch.object(get_model, 'get_all_models', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'Unsupported model architecture'):
                get_model.get_model_class(SimpleNamespace(arch='typo_model'))


if __name__ == '__main__':
    unittest.main()
