import sys
import unittest
from io import StringIO
from unittest import mock

import run
from toolkit.exceptions import UserFacingError


class FakeJob:
    process = []

    def run(self):
        raise UserFacingError("Access required for a gated Hugging Face repo.")

    def cleanup(self):
        raise AssertionError("cleanup should not run after a failed job")


class UserFacingErrorRunnerTest(unittest.TestCase):
    def test_run_exits_without_reraising_user_facing_error_traceback(self):
        argv = ["run.py", "fake_config.yaml"]
        output = StringIO()

        with mock.patch.object(sys, "argv", argv), \
             mock.patch("run.get_job", return_value=FakeJob()), \
             mock.patch("sys.stdout", output):
            with self.assertRaises(SystemExit) as raised:
                run.main()

        self.assertEqual(raised.exception.code, 1)
        text = output.getvalue()
        self.assertIn("Error running job: Access required for a gated Hugging Face repo.", text)
        self.assertIn(" - 1 failure", text)


if __name__ == "__main__":
    unittest.main()

