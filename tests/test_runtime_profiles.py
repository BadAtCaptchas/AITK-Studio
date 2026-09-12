import unittest
from scripts.install_runtime import PROFILES, ROOT, check_interpreter


class RuntimeProfilesTests(unittest.TestCase):
    def test_supported_isolated_interpreters(self):
        for version in ((3, 11), (3, 12)):
            check_interpreter(version, True)
        for version, isolated in (((3, 13), True), ((3, 12), False)):
            with self.assertRaises(ValueError):
                check_interpreter(version, isolated)

    def test_one_torchcodec_pin_per_complete_profile(self):
        self.assertNotIn('torchcodec==', (ROOT / 'requirements_base.txt').read_text())
        expected = {'legacy-cu128': '0.7.0', 'blackwell-cu128': '0.10.0', 'blackwell-cu130': '0.10.0', 'dgx-cu130': '0.10.0', 'macos': '0.11.0'}
        for name, files in PROFILES.items():
            for filename in files:
                self.assertTrue((ROOT / filename).is_file())
            self.assertIn('torchcodec==' + expected[name], (ROOT / files[0]).read_text())


if __name__ == '__main__':
    unittest.main()
