import tempfile
import io
import unittest
from pathlib import Path

from toolkit.print import Logger


class Cp1252Terminal:
    encoding = "cp1252"

    def __init__(self):
        self.messages = []

    def write(self, message):
        message.encode(self.encoding)
        self.messages.append(message)

    def flush(self):
        pass

    def isatty(self):
        return False


class LoggerEncodingTest(unittest.TestCase):
    def test_logger_replaces_unencodable_terminal_text_but_keeps_utf8_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "job.log"
            logger = Logger(log_path)
            try:
                terminal = Cp1252Terminal()
                logger.terminal = terminal

                logger.write("✅ No prompt tuning\n")

                self.assertEqual(terminal.messages, ["? No prompt tuning\n"])
                self.assertEqual(log_path.read_text(encoding="utf-8"), "✅ No prompt tuning\n")
            finally:
                logger.log.close()

    def test_stdout_and_stderr_loggers_share_one_handle_without_double_writes(self):
        log = io.StringIO()
        stdout_terminal = io.StringIO()
        stderr_terminal = io.StringIO()
        stdout = Logger(stdout_terminal, log)
        stderr = Logger(stderr_terminal, log)

        stdout.write("out\n")
        stderr.write("error\n")

        self.assertEqual(log.getvalue(), "out\nerror\n")
        self.assertEqual(stdout_terminal.getvalue(), "out\n")
        self.assertEqual(stderr_terminal.getvalue(), "error\n")


if __name__ == "__main__":
    unittest.main()
