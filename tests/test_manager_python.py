import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from manager import env, ffmpeg, gitwin, migrations, nodejs, util, uvbin
from manager.__main__ import cmd_check
from manager.spec import EnvSpec


class ManagerPythonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='aitk-manager-python-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / '.venv'
        self.spec = EnvSpec('cpu', {}, python_version='3.12')
        self.created_version = '3.12'
        self.created_platform = 'win-amd64'
        for module in (env, util):
            self.mock(module, 'REPO_ROOT', str(self.root))
        self.mock(env, 'venv_python_version', side_effect=lambda target=None: self.read(target, 'version'))
        self.mock(env, '_venv_platform', side_effect=lambda target=None: self.read(target, 'platform'))
        self.uv = self.mock(env, 'find_uv', return_value='test-uv')
        self.run = self.mock(env, 'run', side_effect=self.create)
        for name in ('info', 'warn', 'ok'):
            self.mock(env, name)
        self.mock(util, 'error')

    def mock(self, obj, name, *args, **kwargs):
        patcher = patch.object(obj, name, *args, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def read(self, target, name):
        path = Path(target or util.venv_dir()) / name
        return path.read_text() if path.is_file() else None

    def write_env(self, target, version, platform='win-amd64'):
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        (target / 'version').write_text(version)
        (target / 'platform').write_text(platform)
        python = Path(util.venv_python(str(target)))
        python.parent.mkdir(parents=True, exist_ok=True)
        python.touch()

    def create(self, command, **kwargs):
        target = command[2] if command[0] == 'test-uv' else command[-1]
        self.write_env(target, self.created_version, self.created_platform)
        return 0, None

    def test_matching_interpreter_is_reused_without_installing(self):
        self.write_env(self.target, '3.12')
        self.assertEqual(env.ensure_venv(self.spec), util.venv_python(str(self.target)))
        self.run.assert_not_called()
        self.assertEqual(list(self.root.glob('*.backup-*')), [])

    def test_update_replaces_unsupported_interpreter_and_preserves_old_packages(self):
        for version in ('3.10', '3.13'):
            with self.subTest(version=version):
                self.write_env(self.target, version)
                (self.target / 'old-packages').write_text(version)
                env.ensure_venv(self.spec)
                self.assertEqual(self.read(self.target, 'version'), '3.12')
                backups = list(self.root.glob('.venv.backup-*'))
                self.assertTrue(any((backup / 'old-packages').read_text() == version for backup in backups))
                self.assertFalse((self.target / 'old-packages').exists())
                self.assertEqual(self.run.call_args.args[0], [
                    'test-uv', 'venv', str(self.target), '--python', '3.12', '--seed',
                ])

    def test_new_environment_is_validated_before_being_declared_ready(self):
        self.created_version = '3.13'
        with self.assertRaises(SystemExit):
            env.ensure_venv(self.spec)
        env.ok.assert_not_called()

    def test_explicit_dgx_python_311_spec_is_preserved(self):
        self.spec.python_version = '3.11'
        self.write_env(self.target, '3.11')
        env.ensure_venv(self.spec)
        self.run.assert_not_called()

    def test_architecture_change_also_preserves_old_environment(self):
        self.spec.uv_python = 'cpython-3.12-windows-aarch64-none'
        self.created_platform = 'win-arm64'
        self.write_env(self.target, '3.12', 'win-amd64')
        env.ensure_venv(self.spec)
        self.assertEqual(self.read(self.target, 'platform'), 'win-arm64')
        backup, = self.root.glob('.venv.backup-*')
        self.assertEqual(self.read(backup, 'platform'), 'win-amd64')
        self.assertEqual(self.run.call_args.args[0][-2], self.spec.uv_python)

    def test_both_venv_directories_keep_the_selected_target_during_replacement(self):
        self.write_env(self.target, '3.10')
        legacy = self.root / 'venv'
        self.write_env(legacy, '3.11')
        env.ensure_venv(self.spec)
        self.assertEqual(self.read(self.target, 'version'), '3.12')
        self.assertEqual(self.read(legacy, 'version'), '3.11')
        self.assertEqual(self.run.call_args.args[0][2], str(self.target))

    def test_legacy_venv_location_is_retained(self):
        target = self.root / 'venv'
        self.write_env(target, '3.10')
        result = env.ensure_venv(self.spec)
        self.assertEqual(result, util.venv_python(str(target)))
        self.assertEqual(self.read(target, 'version'), '3.12')
        self.assertFalse(self.target.exists())

    def test_dry_run_does_not_replace_or_install(self):
        self.write_env(self.target, '3.10')
        env.ensure_venv(self.spec, dry_run=True)
        self.assertEqual(self.read(self.target, 'version'), '3.10')
        self.assertEqual(list(self.root.glob('*.backup-*')), [])
        self.run.assert_not_called()

    def test_unavailable_uv_does_not_fall_back_to_incompatible_python(self):
        self.write_env(self.target, '3.10')
        self.uv.return_value = None
        with patch.object(env.sys, 'version_info', (3, 13, 0)):
            with self.assertRaises(SystemExit):
                env.ensure_venv(self.spec)
        self.assertEqual(self.read(self.target, 'version'), '3.10')
        self.assertEqual(list(self.root.glob('*.backup-*')), [])
        self.run.assert_not_called()

    def test_compatible_system_python_can_create_environment_without_uv(self):
        self.uv.return_value = None
        with patch.object(env.sys, 'version_info', (3, 12, 0)):
            env.ensure_venv(self.spec)
        self.assertEqual(self.run.call_args.args[0][1:], ['-m', 'venv', str(self.target)])
        self.assertEqual(self.read(self.target, 'version'), '3.12')

    def test_creation_failure_restores_old_environment_and_retains_failed_attempt(self):
        self.write_env(self.target, '3.10')

        def fail(command, **kwargs):
            self.create(command, **kwargs)
            raise SystemExit(1)

        self.run.side_effect = fail
        with self.assertRaises(SystemExit):
            env.ensure_venv(self.spec)
        self.assertEqual(self.read(self.target, 'version'), '3.10')
        failed, = self.root.glob('.venv.failed-*')
        self.assertEqual(self.read(failed, 'version'), '3.12')
        self.assertEqual(list(self.root.glob('*.backup-*')), [])

    def test_wrong_created_version_restores_old_environment(self):
        self.write_env(self.target, '3.10')
        self.created_version = '3.13'
        with self.assertRaises(SystemExit):
            env.ensure_venv(self.spec)
        self.assertEqual(self.read(self.target, 'version'), '3.10')
        env.ok.assert_not_called()

    def test_failed_backup_does_not_start_installing(self):
        self.write_env(self.target, '3.10')
        with patch.object(env.os, 'rename', side_effect=PermissionError('in use')):
            with self.assertRaises(SystemExit):
                env.ensure_venv(self.spec)
        self.assertEqual(self.read(self.target, 'version'), '3.10')
        self.run.assert_not_called()

    def test_replacement_refuses_environment_outside_checkout(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / '.venv'
            self.write_env(target, '3.10')
            with patch.object(env, 'venv_dir', return_value=str(target)):
                with self.assertRaises(SystemExit):
                    env.ensure_venv(self.spec)
            self.assertEqual(self.read(target, 'version'), '3.10')
        self.run.assert_not_called()

    def test_check_does_not_claim_old_python_is_up_to_date_when_packages_match(self):
        self.write_env(self.target, '3.10')
        with patch('manager.__main__._resolve_spec', return_value=({}, self.spec)), \
             patch('manager.__main__._toolkit_version', return_value='test'), \
             patch('manager.__main__.gitops') as gitops, \
             patch.object(env, 'torch_matches', return_value=True), \
             patch.object(env, 'requirements_in_sync', return_value=True), \
             patch('manager.__main__.print_json') as report:
            gitops.behind_count.return_value = 0
            cmd_check(SimpleNamespace(json=True))
        self.assertFalse(report.call_args.args[0]['deps_in_sync'])
        self.assertTrue(report.call_args.args[0]['update_available'])

    def test_forced_sync_updates_interpreter_before_installing_packages(self):
        self.write_env(self.target, '3.10')
        for module, name in (
            (uvbin, 'ensure_uv'), (gitwin, 'ensure_git'),
            (ffmpeg, 'ensure_ffmpeg'), (nodejs, 'ensure_node'),
            (nodejs, 'ensure_ui_deps'), (migrations, 'run_pending'),
            (env, 'write_sitecustomize'),
        ):
            self.mock(module, name)

        def install_torch(*args, **kwargs):
            self.assertEqual(self.read(self.target, 'version'), '3.12')
            return False

        self.mock(env, 'ensure_torch', side_effect=install_torch)
        requirements = self.mock(env, 'ensure_requirements')
        env.sync(self.spec, {}, force=True)
        requirements.assert_called_once_with(self.spec, dry_run=False, force=True)

    def test_failed_interpreter_update_stops_forced_sync_before_package_install(self):
        self.write_env(self.target, '3.10')
        self.created_version = '3.13'
        self.mock(uvbin, 'ensure_uv')
        self.mock(gitwin, 'ensure_git')
        torch = self.mock(env, 'ensure_torch')
        requirements = self.mock(env, 'ensure_requirements')
        with self.assertRaises(SystemExit):
            env.sync(self.spec, {}, force=True)
        torch.assert_not_called()
        requirements.assert_not_called()
        self.assertEqual(self.read(self.target, 'version'), '3.10')


if __name__ == '__main__':
    unittest.main()
