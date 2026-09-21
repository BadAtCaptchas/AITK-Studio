import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

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

    def set_launch_pid(self, pid):
        with closing(sqlite3.connect(self.filename)) as db, db:
            db.execute('UPDATE Job SET pid=?', (pid,))

    def test_registration_accepts_the_pid_already_recorded_by_the_launcher(self):
        self.set_launch_pid(os.getpid())
        register_attempt_process()
        self.assertEqual(self.row()['status'], 'running')

    def test_windows_redirector_hands_ownership_to_its_actual_python_child(self):
        self.set_launch_pid(100)
        process = Mock(pid=200)
        process.create_time.return_value = 1234.5
        process.ppid.return_value = 100
        with patch('toolkit.attempt_process.os.name', 'nt'), \
             patch('toolkit.attempt_process.os.getpid', return_value=200), \
             patch('toolkit.attempt_process.psutil.Process', return_value=process):
            register_attempt_process()
        registered = self.row()
        self.assertEqual(registered['pid'], 200)
        self.assertEqual(registered['process_started_at'], 1234.5)
        self.assertEqual(registered['status'], 'running')
        # A late launcher write cannot put the redirector PID back after registration.
        with closing(sqlite3.connect(self.filename)) as db, db:
            updated = db.execute(
                "UPDATE Job SET pid=100 WHERE id='job' AND attempt_id='attempt-a' AND status='starting'",
            ).rowcount
        self.assertEqual(updated, 0)
        self.assertEqual(self.row()['pid'], 200)

    def test_parent_handoff_cannot_claim_a_stale_stopped_or_unrelated_launch(self):
        for platform, attempt, status, recorded_pid in (
            ('posix', 'attempt-a', 'starting', 100),
            ('nt', 'stale', 'starting', 100),
            ('nt', 'attempt-a', 'stopping', 100),
            ('nt', 'attempt-a', 'running', 100),
            ('nt', 'attempt-a', 'starting', 999),
        ):
            with self.subTest(platform=platform, attempt=attempt, status=status, pid=recorded_pid):
                with closing(sqlite3.connect(self.filename)) as db, db:
                    db.execute('UPDATE Job SET status=?, pid=?', (status, recorded_pid))
                before = self.row()
                process = Mock(pid=200)
                process.create_time.return_value = 1234.5
                process.ppid.return_value = 100
                with patch.dict(os.environ, {'AITK_ATTEMPT_ID': attempt}), \
                     patch('toolkit.attempt_process.os.name', platform), \
                     patch('toolkit.attempt_process.os.getpid', return_value=200), \
                     patch('toolkit.attempt_process.psutil.Process', return_value=process):
                    with self.assertRaisesRegex(RuntimeError, 'no longer owns'):
                        register_attempt_process()
                self.assertEqual(self.row(), before)

    def test_real_python_subprocess_registers_after_launcher_records_its_pid(self):
        # Windows venv python.exe can be a redirector whose child has a different PID.
        # The stdin barrier guarantees that the launcher PID reaches SQLite first.
        code = (
            'import os, sys; sys.stdin.readline(); '
            'from toolkit.attempt_process import register_attempt_process; '
            'register_attempt_process(); print(os.getpid())'
        )
        with subprocess.Popen(
            [sys.executable, '-c', code],
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        ) as child:
            try:
                self.set_launch_pid(child.pid)
                stdout, stderr = child.communicate('\n', timeout=15)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate(timeout=15)
        self.assertEqual(child.returncode, 0, stderr)
        registered = self.row()
        self.assertEqual(registered['pid'], int(stdout.strip()))
        self.assertEqual(registered['status'], 'running')
        self.assertGreater(registered['process_started_at'], 0)

    def test_mongodb_handoff_keeps_attempt_status_and_parent_pid_constraints(self):
        for platform, pids in (('nt', [None, 200, 100]), ('posix', [None, 200])):
            with self.subTest(platform=platform):
                store = Mock(
                    available=True, provider='mongodb',
                    identity={'id': 'job', 'attempt_id': 'attempt-a'},
                )
                store._jobs.update_one.return_value.matched_count = 1
                process = Mock(pid=200)
                process.create_time.return_value = 1234.5
                process.ppid.return_value = 100
                with patch('toolkit.attempt_process.UIJobStore', return_value=store), \
                     patch('toolkit.attempt_process.os.name', platform), \
                     patch('toolkit.attempt_process.os.getpid', return_value=200), \
                     patch('toolkit.attempt_process.psutil.Process', return_value=process):
                    register_attempt_process()
                    query, update = store._jobs.update_one.call_args.args
                    self.assertEqual(query, {
                        'id': 'job', 'attempt_id': 'attempt-a', 'status': 'starting',
                        '$or': [{'pid': pid} for pid in pids],
                    })
                    self.assertEqual(update['$set']['pid'], 200)
                    self.assertEqual(update['$set']['process_started_at'], 1234.5)
                    self.assertEqual(update['$set']['status'], 'running')
                    store._jobs.update_one.return_value.matched_count = 0
                    with self.assertRaisesRegex(RuntimeError, 'no longer owns'):
                        register_attempt_process()

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
