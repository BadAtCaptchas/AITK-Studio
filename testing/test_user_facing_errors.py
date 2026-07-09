import sys
import unittest
from io import StringIO
from unittest import mock

import run
from toolkit.exceptions import UserFacingError


class FakeJob:
    process = []

    def __init__(self):
        self.active_process = None
        self.cleaned_up = False

    def run(self):
        raise UserFacingError("Access required for a gated Hugging Face repo.")

    def cleanup(self):
        self.cleaned_up = True


class UserFacingErrorRunnerTest(unittest.TestCase):
    def test_run_exits_without_reraising_user_facing_error_traceback(self):
        argv = ["run.py", "fake_config.yaml"]
        output = StringIO()
        job = FakeJob()

        with mock.patch.object(sys, "argv", argv), \
             mock.patch("run.get_job", return_value=job), \
             mock.patch("sys.stdout", output):
            with self.assertRaises(SystemExit) as raised:
                run.main()

        self.assertEqual(raised.exception.code, 1)
        self.assertTrue(job.cleaned_up)
        text = output.getvalue()
        self.assertIn("Error running job: Access required for a gated Hugging Face repo.", text)
        self.assertIn(" - 1 failure", text)


if __name__ == "__main__":
    unittest.main()

