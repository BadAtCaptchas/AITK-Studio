import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import call, patch

from scripts.install_runtime import PROFILES, ROOT, check_interpreter, main


class RuntimeProfilesTests(unittest.TestCase):
    def run_installer(self, args, platform='linux'):
        with patch('scripts.install_runtime.sys.argv', ['install_runtime.py', *args]), \
             patch('scripts.install_runtime.sys.platform', platform), \
             patch('scripts.install_runtime.sys.executable', 'test-python'), \
             patch('scripts.install_runtime.check_interpreter'), \
             patch('scripts.install_runtime.subprocess.run') as run:
            main()
        return run

    def test_default_installs_cuda_13_and_checks_dependencies(self):
        for platform in ('linux', 'win32'):
            with self.subTest(platform=platform):
                run = self.run_installer([], platform)
                self.assertEqual(run.call_args_list, [
                    call(['test-python', '-m', 'pip', 'install',
                          '-r', str(ROOT / 'requirements_torch_blackwell_cu130.txt'),
                          '-r', str(ROOT / 'requirements.txt')], cwd=ROOT, check=True),
                    call(['test-python', '-m', 'pip', 'check'], check=True),
                ])

    def test_explicit_profiles_override_default(self):
        for name, filenames in PROFILES.items():
            with self.subTest(profile=name):
                run = self.run_installer(['--profile', name],
                                         'darwin' if name == 'macos' else 'linux')
                command = run.call_args_list[0].args[0]
                self.assertEqual(command[4:], [
                    '-r', str(ROOT / filenames[0]), '-r', str(ROOT / filenames[1]),
                ])

    def test_dry_run_reports_resolution_without_pip_check(self):
        run = self.run_installer(['--dry-run', '--report', 'resolution.json'])
        run.assert_called_once_with(
            ['test-python', '-m', 'pip', 'install',
             '-r', str(ROOT / 'requirements_torch_blackwell_cu130.txt'),
             '-r', str(ROOT / 'requirements.txt'),
             '--dry-run', '--ignore-installed', '--report', 'resolution.json'],
            cwd=ROOT, check=True,
        )

    def test_invalid_platform_profile_pairs_do_not_install(self):
        for platform, args in (
            ('darwin', []),
            ('darwin', ['--profile', 'legacy-cu128']),
            ('linux', ['--profile', 'macos']),
            ('win32', ['--profile', 'macos']),
        ):
            with self.subTest(platform=platform, args=args), \
                 patch('scripts.install_runtime.sys.argv', ['install_runtime.py', *args]), \
                 patch('scripts.install_runtime.sys.platform', platform), \
                 patch('scripts.install_runtime.check_interpreter'), \
                 patch('scripts.install_runtime.subprocess.run') as run, \
                 redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()

    def test_invalid_interpreter_does_not_install(self):
        with patch('scripts.install_runtime.sys.argv', ['install_runtime.py']), \
             patch('scripts.install_runtime.check_interpreter', side_effect=ValueError('Activate a virtual environment first.')), \
             patch('scripts.install_runtime.subprocess.run') as run, \
             redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

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
