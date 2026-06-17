import tempfile
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


if __name__ == "__main__":
    unittest.main()
