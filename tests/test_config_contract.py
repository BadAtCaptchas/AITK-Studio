import json
from pathlib import Path
import unittest
import yaml
from toolkit.config_contract import config_contract_errors
from toolkit.model_registry import CONTRACT, BY_ARCH


class ConfigContractTests(unittest.TestCase):
    def test_shared_guided_and_raw_cases(self):
        cases = json.loads((Path(__file__).parent / 'fixtures/config_contract_cases.json').read_text())
        for case in cases:
            with self.subTest(case=case['name']):
                before = json.dumps(case['config'], sort_keys=True)
                self.assertEqual(not config_contract_errors(case['config']), case['valid'])
                self.assertEqual(not config_contract_errors(yaml.safe_load(yaml.safe_dump(case['config']))), case['valid'])
                self.assertEqual(json.dumps(case['config'], sort_keys=True), before)

    def test_every_ui_choice_has_a_python_descriptor(self):
        aliases = {'sd15': 'sd1', 'flex1': 'flux'}
        for choice in CONTRACT['choices']:
            base = choice['name'].split(':')[0]
            self.assertIn(aliases.get(base, base), BY_ARCH, choice['name'])
            self.assertIn(choice['outputType'], ('image', 'video', 'audio'))
            self.assertEqual(choice['offloading']['layers'], 'model.layer_offloading' in choice['additionalSections'])


if __name__ == '__main__':
    unittest.main()
