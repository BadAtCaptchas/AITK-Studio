import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

from toolkit.ui_database import UIJobStore
from toolkit.attempt_process import register_attempt_process, stop_owned_process


class AttemptOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.filename = os.path.join(self.directory.name, 'attempts.sqlite')
        self.environment = patch.dict(os.environ, {
            'AITK_JOB_ID': 'job', 'AITK_ATTEMPT_ID': 'attempt-a',
            'AITK_DB_PROVIDER': 'sqlite', 'AITK_SQLITE_PATH': self.filename,
        })
        self.environment.start()
        with closing(sqlite3.connect(self.filename)) as db, db:
            db.execute('CREATE TABLE Job (id TEXT PRIMARY KEY, attempt_id TEXT, status TEXT, pid INTEGER, process_started_at REAL, info TEXT, stop INTEGER DEFAULT 0, return_to_queue INTEGER DEFAULT 0, sample_now INTEGER DEFAULT 0, save_now INTEGER DEFAULT 0, step INTEGER DEFAULT 0, updated_at TEXT)')
            db.execute("INSERT INTO Job (id,attempt_id,status) VALUES ('job','attempt-a','starting')")

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

    def row(self):
        with closing(sqlite3.connect(self.filename)) as db, db:
            db.row_factory = sqlite3.Row
            return dict(db.execute('SELECT * FROM Job').fetchone())

    def test_registration_records_birth_before_runtime_and_rejects_stale_attempt(self):
        register_attempt_process()
        registered = self.row()
        self.assertEqual(registered['pid'], os.getpid())
        self.assertEqual(registered['status'], 'running')
        self.assertGreater(registered['process_started_at'], 0)
        with patch.dict(os.environ, {'AITK_ATTEMPT_ID': 'stale'}):
            with self.assertRaises(RuntimeError):
                register_attempt_process()
        self.assertEqual(self.row()['attempt_id'], 'attempt-a')

    def test_old_attempt_cannot_write_or_consume_new_attempt_controls(self):
        store = UIJobStore('job', self.filename)
        with closing(sqlite3.connect(self.filename)) as db, db:
            db.execute("UPDATE Job SET attempt_id='attempt-b', status='running', sample_now=1, save_now=1")
        store.update_status('error', 'stale callback')
        store.update_key('step', 900)
        self.assertTrue(store.should_stop())
        self.assertFalse(store.should_save())
        self.assertFalse(store.consume_sample_request())
        row = self.row()
        self.assertEqual((row['status'], row['step'], row['sample_now']), ('running', 0, 1))

    def test_terminal_report_keeps_pid_until_supervisor_observes_exit(self):
        register_attempt_process()
        UIJobStore('job', self.filename).update_status('completed')
        self.assertEqual(self.row()['pid'], os.getpid())

    def test_stop_refuses_mismatched_birth_without_signaling(self):
        register_attempt_process()
        with closing(sqlite3.connect(self.filename)) as db, db:
            db.execute("UPDATE Job SET status='stopping', process_started_at=1")
        self.assertFalse(stop_owned_process('job', 'attempt-a', os.getpid(), 1))


if __name__ == '__main__':
    unittest.main()
