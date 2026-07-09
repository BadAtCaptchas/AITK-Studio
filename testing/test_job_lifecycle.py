import unittest
from collections import OrderedDict

from jobs.BaseJob import BaseJob
from toolkit.job import run_job_instance


class FakeProcess:
    def __init__(self, run_error=None, cleanup_error=None):
        self.run_error = run_error
        self.cleanup_error = cleanup_error
        self.run_calls = 0
        self.cleanup_calls = 0

    def run(self):
        self.run_calls += 1
        if self.run_error is not None:
            raise self.run_error

    def cleanup(self):
        self.cleanup_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


class JobLifecycleTest(unittest.TestCase):
    def make_base_job(self):
        return BaseJob(
            OrderedDict(
                [
                    ("job", "test"),
                    ("config", OrderedDict([("name", "lifecycle-test")])),
                ]
            )
        )

    def test_run_processes_retains_the_process_that_failed(self):
        job = self.make_base_job()
        first = FakeProcess()
        failure = RuntimeError("second process failed")
        second = FakeProcess(run_error=failure)
        third = FakeProcess()
        job.process = [first, second, third]

        with self.assertRaises(RuntimeError) as raised:
            job.run_processes()

        self.assertIs(raised.exception, failure)
        self.assertIs(job.active_process, second)
        self.assertEqual(first.run_calls, 1)
        self.assertEqual(second.run_calls, 1)
        self.assertEqual(third.run_calls, 0)

    def test_cleanup_attempts_every_process(self):
        job = self.make_base_job()
        failure = RuntimeError("first cleanup failed")
        first = FakeProcess(cleanup_error=failure)
        second = FakeProcess()
        job.process = [first, second]

        with self.assertRaises(RuntimeError) as raised:
            job.cleanup()

        self.assertIs(raised.exception, failure)
        self.assertEqual(first.cleanup_calls, 1)
        self.assertEqual(second.cleanup_calls, 1)

    def test_run_job_instance_cleans_up_and_preserves_primary_failure(self):
        primary_failure = RuntimeError("run failed")
        cleanup_failure = RuntimeError("cleanup failed")

        class FailingJob:
            def __init__(self):
                self.cleanup_calls = 0

            def run(self):
                raise primary_failure

            def cleanup(self):
                self.cleanup_calls += 1
                raise cleanup_failure

        job = FailingJob()
        with self.assertRaises(RuntimeError) as raised:
            run_job_instance(job)

        self.assertIs(raised.exception, primary_failure)
        self.assertEqual(job.cleanup_calls, 1)
        notes = getattr(raised.exception, "__notes__", [])
        if notes:
            self.assertIn("cleanup failed", notes[0])

    def test_cleanup_failure_is_reported_after_successful_run(self):
        cleanup_failure = RuntimeError("cleanup failed")

        class CleanupFailingJob:
            def run(self):
                return None

            def cleanup(self):
                raise cleanup_failure

        with self.assertRaises(RuntimeError) as raised:
            run_job_instance(CleanupFailingJob())

        self.assertIs(raised.exception, cleanup_failure)


if __name__ == "__main__":
    unittest.main()
