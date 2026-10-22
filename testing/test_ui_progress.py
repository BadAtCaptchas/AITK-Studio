"""Progress writes remain bounded while the database is slow or unavailable."""

import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from toolkit.ui_database import UIJobStore
from toolkit.ui_progress import UIProgressReporter


class UIProgressTests(unittest.TestCase):
    def test_slow_store_does_not_block_and_coalesces_progress(self):
        entered, release = threading.Event(), threading.Event()
        writes = []

        def write(key, value):
            if not entered.is_set():
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('Test did not release database write')
            writes.append((key, value))

        reporter = UIProgressReporter(SimpleNamespace(
            update_key=write, update_status=lambda status, info: writes.append(('status', status)),
        ))
        try:
            reporter.update_key('step', 0)
            self.assertTrue(entered.wait(5))
            for step in range(1, 10001):
                reporter.update_key('step', step)
                reporter.update_key('speed_string', str(step))
                reporter.update_status('running', 'Training')
            self.assertLessEqual(reporter.pending_count, 4)
            reporter.update_status('completed', 'Training completed')
            self.assertEqual(writes, [])
            release.set()
            self.assertTrue(reporter.flush())
            self.assertEqual(writes, [('step', 0), ('step', 10000), ('speed_string', '10000'), ('status', 'completed')])
            self.assertEqual(reporter.pending_count, 0)
        finally:
            release.set()
            reporter.close()

    def test_close_drains_terminal_and_does_not_keep_completed_work(self):
        writes = []
        reporter = UIProgressReporter(SimpleNamespace(
            update_key=lambda key, value: writes.append((key, value)),
            update_status=lambda status, info: writes.append(('status', status)),
        ))
        reporter.update_key('step', 17)
        reporter.update_status('error', 'Invalid gradients')
        reporter.update_status('running', 'Late callback')
        reporter.close()
        self.assertEqual(writes[-1], ('status', 'error'))
        self.assertIn(('step', 17), writes)
        self.assertEqual(reporter.pending_count, 0)
        self.assertFalse(reporter._thread.is_alive())
        reporter.update_key('step', 99)
        reporter.close()
        self.assertNotIn(('step', 99), writes)

    def test_failure_is_reported_without_losing_later_terminal_write(self):
        writes = []
        reporter = UIProgressReporter(SimpleNamespace(
            update_key=mock.Mock(side_effect=sqlite3.OperationalError('database is locked')),
            update_status=lambda status, info: writes.append(status),
        ))
        try:
            reporter.update_key('step', 1)
            reporter.update_status('error', 'Stopped')
            with self.assertWarnsRegex(RuntimeWarning, 'could not be persisted'):
                self.assertFalse(reporter.flush())
            self.assertEqual(writes, ['error'])
            self.assertTrue(reporter.flush())
        finally:
            reporter.close()

    def test_arbitrary_keys_cannot_make_unbounded_queue(self):
        reporter = UIProgressReporter(SimpleNamespace(update_key=mock.Mock(), update_status=mock.Mock()))
        try:
            with self.assertRaises(ValueError):
                reporter.update_key('unbounded-key', 1)
            self.assertEqual(reporter.pending_count, 0)
        finally:
            reporter.close()

    def test_attempt_fencing_is_preserved_on_background_connection(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'test.db'
            with closing(sqlite3.connect(path)) as conn:
                conn.execute('CREATE TABLE Job (id TEXT, attempt_id TEXT, step INTEGER)')
                conn.execute("INSERT INTO Job VALUES ('job', 'current', 7)")
                conn.commit()
            with mock.patch.dict('os.environ', {'AITK_DB_PROVIDER': 'sqlite', 'AITK_ATTEMPT_ID': 'obsolete', 'AITK_SQLITE_PATH': str(path)}):
                store = UIJobStore('job', str(path))
                reporter = UIProgressReporter(store)
                reporter.update_key('step', 999)
                reporter.close()
                store.close()
            with closing(sqlite3.connect(path)) as conn:
                self.assertEqual(conn.execute('SELECT step FROM Job').fetchone()[0], 7)


if __name__ == '__main__':
    unittest.main()
