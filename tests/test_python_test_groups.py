"""CI cannot silently omit a newly added test module."""

import unittest

from scripts.run_python_tests import validate_manifest


class PythonTestOwnershipTests(unittest.TestCase):
    def test_missing_or_stale_module_fails(self):
        with self.assertRaisesRegex(ValueError, 'Unassigned test suite: tests.new'):
            validate_manifest({'version': 1, 'suites': {}}, {'tests.new'})
        with self.assertRaisesRegex(ValueError, 'Missing test module: tests.old'):
            validate_manifest({'version': 1, 'suites': {'tests.old': {'group': 'runtime'}}}, set())

    def test_manual_and_gpu_require_documented_requirements(self):
        for group in ('manual', 'gpu'):
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, 'Explain'):
                validate_manifest({'version': 1, 'suites': {'testing.integration': {'group': group}}}, {'testing.integration'})

    def test_valid_groups_are_accepted(self):
        entries = {'tests.cpu': {'group': 'lightweight'}, 'testing.hardware': {'group': 'gpu', 'reason': 'CUDA FP4 GPU'}}
        self.assertEqual(validate_manifest({'version': 1, 'suites': entries}, set(entries)), entries)


if __name__ == '__main__':
    unittest.main()
