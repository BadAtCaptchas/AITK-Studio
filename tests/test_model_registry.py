import subprocess
import sys
import unittest
from unittest.mock import patch
from toolkit.model_registry import ModelDescriptor, MissingModelDependency, BrokenModelIntegration, CONTRACT
from toolkit.config_contract import config_contract_errors


class ModelRegistryTests(unittest.TestCase):
    def test_discovery_is_weight_free(self):
        subprocess.run([sys.executable, '-c', "import sys; import extensions_built_in.diffusion_models as registry; assert len(registry.AI_TOOLKIT_MODELS)>30; assert 'torch' not in sys.modules"], check=True)

    def test_missing_extra_and_broken_integration_are_distinct(self):
        descriptor = ModelDescriptor('test', 'test.model', 'Model', ('optional_extra',))
        with patch('toolkit.model_registry.import_module', side_effect=ModuleNotFoundError('missing', name='optional_extra')):
            with self.assertRaises(MissingModelDependency): descriptor.load()
        for error in (ImportError('missing public symbol'), ModuleNotFoundError('internal bug', name='toolkit.typo')):
            with patch('toolkit.model_registry.import_module', side_effect=error):
                with self.assertRaises(BrokenModelIntegration): descriptor.load()

    def test_choices_resolve_to_registered_architectures(self):
        arches = {item['arch'] for item in CONTRACT['models']}
        self.assertFalse({item['arch'] for item in CONTRACT['choices']} - arches)

    def test_numeric_and_namespaced_extension_boundary(self):
        config = {'config': {'process': [{'type': 'diffusion_trainer', 'model': {'arch': 'flux'}, 'train': {'steps': 1}}]}}
        self.assertEqual(config_contract_errors(config), [])
        for invalid in (float('nan'), float('inf'), -1, True, '100'):
            config['config']['process'][0]['train']['steps'] = invalid
            self.assertTrue(config_contract_errors(config))
        config = {'extensions': {'example.custom': {'processKinds': ['my_process'], 'modelArches': ['my_arch']}}, 'config': {'process': [{'type': 'my_process', 'model': {'arch': 'my_arch'}, 'custom': {'keep': 1}}]}}
        self.assertEqual(config_contract_errors(config), [])


if __name__ == '__main__': unittest.main()
