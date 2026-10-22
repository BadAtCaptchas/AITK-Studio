import unittest
from unittest.mock import patch
from manager import nodejs


class NodeRuntimeTests(unittest.TestCase):
    def test_supported_versions_match_studio_engines(self):
        for version, accepted in [('20.20.0', False), ('22.11.0', False), ('22.12.0', True),
                ('22.12.0-rc.1', False), ('23.0.0', False), ('24.22.1', True), ('25.0.0', False), ('26.0.0', True)]:
            with self.subTest(version=version), patch.object(nodejs, '_node_info', return_value=(version, int(version.split('.')[0]), 'x64')):
                self.assertEqual(nodejs._usable_major('node') is not None, accepted)
        with patch.object(nodejs, 'IS_WINDOWS', True), patch.object(nodejs, '_node_info', return_value=('24.22.1', 24, 'arm64')):
            self.assertIsNone(nodejs._usable_major('node'))

    def test_pinned_windows_portable_copy_still_requires_x64(self):
        with patch.object(nodejs.os.path, 'isfile', return_value=True), patch.object(nodejs, 'IS_WINDOWS', True):
            for arch, expected in [('arm64', True), ('x64', False)]:
                with patch.object(nodejs, '_node_info', return_value=(nodejs.NODE_VERSION, 24, arch)):
                    self.assertEqual(nodejs.local_node_stale(), expected)

    def test_portable_copy_refreshes_but_supported_system_copy_is_preserved(self):
        local = nodejs.local_node_exe()
        with patch.object(nodejs, 'have_usable_node', return_value=(local, 24)), patch.object(nodejs, 'local_node_stale', return_value=True), \
                patch.object(nodejs, '_node_info', return_value=('24.11.1', 24, 'x64')), patch.object(nodejs, 'which', return_value=None), \
                patch.object(nodejs, '_dist_url', return_value=('https://nodejs.org/test', 'test')), patch.object(nodejs, 'download') as download, patch.object(nodejs, 'info') as info:
            self.assertFalse(nodejs.ensure_node({'os': 'linux', 'arch': 'x64'}, dry_run=True))
            self.assertTrue(any('would install' in str(call) for call in info.call_args_list))
            download.assert_not_called()
        with patch.object(nodejs, 'have_usable_node', return_value=('/system/node', 26)), patch.object(nodejs, 'local_node_stale', return_value=False), patch.object(nodejs, 'download') as download:
            self.assertFalse(nodejs.ensure_node({}, dry_run=False))
            download.assert_not_called()
