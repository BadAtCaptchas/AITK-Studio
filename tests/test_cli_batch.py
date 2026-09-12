"""Exercise the actual CLI functions with tiny jobs, without importing training backends."""
import argparse
import ast
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace


class BatchOutcomeTests(unittest.TestCase):
    def run_batch(self, outcomes, recover=False):
        source = Path(__file__).resolve().parents[1] / 'run.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        functions = ast.Module([n for n in tree.body if isinstance(n, ast.FunctionDef)], [])
        logs, executed = [], []

        class Stop(Exception):
            pass

        class UserError(Exception):
            pass

        def run(job):
            executed.append(job)
            if outcomes[job] == 'fail':
                raise UserError('expected fixture failure')
            if outcomes[job] == 'stop':
                raise Stop()
            if outcomes[job] == 'interrupt':
                raise KeyboardInterrupt()

        namespace = dict(argparse=argparse, accelerator=SimpleNamespace(is_main_process=True),
                         get_job=lambda config, name: config, run_job_instance=run,
                         print_acc=logs.append, setup_log_to_file=lambda _: None,
                         JobStopRequested=Stop, UserFacingError=UserError)
        exec(compile(functions, str(source), 'exec'), namespace)
        old_argv = sys.argv
        sys.argv = ['run.py', *outcomes, *(['--recover'] if recover else [])]
        try:
            try:
                status = namespace['main']()
            except SystemExit as error:
                status = error.code
        finally:
            sys.argv = old_argv
        return status, executed, logs

    def test_recovered_failure_is_not_success(self):
        status, executed, logs = self.run_batch({'bad': 'fail', 'good': 'ok'}, True)
        self.assertEqual(status, 1)
        self.assertEqual(executed, ['bad', 'good'])
        self.assertEqual(logs.count('Result:'), 1)

    def test_success_and_fail_fast(self):
        self.assertEqual(self.run_batch({'good': 'ok'})[0], 0)
        status, executed, _ = self.run_batch({'bad': 'fail', 'good': 'ok'})
        self.assertEqual((status, executed), (1, ['bad']))

    def test_stop_does_not_mask_failure(self):
        self.assertEqual(self.run_batch({'stop': 'stop'})[0], 0)
        self.assertEqual(self.run_batch({'bad': 'fail', 'stop': 'stop'}, True)[0], 1)
        self.assertEqual(self.run_batch({'interrupt': 'interrupt'})[0], 130)


if __name__ == '__main__':
    unittest.main()
