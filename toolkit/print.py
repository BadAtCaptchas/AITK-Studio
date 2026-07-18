import sys
import os
from toolkit.accelerator import get_accelerator


def print_acc(*args, **kwargs):
    if get_accelerator().is_local_main_process:
        print(*args, **kwargs)


class Logger:
    def __init__(self, terminal, log_file=None):
        # The optional one-argument form is retained for small integrations and
        # tests; setup_log_to_file uses the shared-handle form below.
        if log_file is None:
            self.terminal = sys.stdout
            self.log = open(terminal, 'a', encoding='utf-8', errors='replace')
        else:
            self.terminal = terminal
            self.log = log_file

    def _safe_terminal_write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            encoding = getattr(self.terminal, "encoding", None) or "utf-8"
            safe_message = message.encode(encoding, errors="replace").decode(
                encoding, errors="replace"
            )
            self.terminal.write(safe_message)

    def write(self, message):
        self._safe_terminal_write(message)
        self.log.write(message)
        self.log.flush()  # Make sure it's written immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def isatty(self):
        return self.terminal.isatty()


def setup_log_to_file(filename):
    if get_accelerator().is_local_main_process:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
    # Capture both real streams before replacing either one. Sharing the file
    # handle prevents stderr from being routed through the stdout logger and
    # written twice.
    log_file = open(filename, 'a', encoding='utf-8', errors='replace')
    stdout = sys.stdout
    stderr = sys.stderr
    sys.stdout = Logger(stdout, log_file)
    sys.stderr = Logger(stderr, log_file)
