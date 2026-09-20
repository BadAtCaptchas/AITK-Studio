"""Job status for resident processes through Studio's attempt-fenced database API."""
import os
import threading
from typing import Callable

from toolkit.ui_database import UIJobStore


class UIJobStatus:
    def __init__(self, sqlite_db_path: str = "./aitk_db.db"):
        self.job_id = os.environ.get("AITK_JOB_ID") or None
        self.store = UIJobStore(self.job_id, sqlite_db_path)
        self.is_ui_job = self.store.available
        self._closed = threading.Event()
        self._watcher = None
        self._last_info = None

    def update_status(self, status: str, info: str | None = None):
        self.store.update_status(status, info)

    def update_info(self, info: str):
        if info != self._last_info:
            self.store.update_key("info", info)
            self._last_info = info

    def update_key(self, key: str, value):
        self.store.update_key(key, value)

    def should_stop(self) -> bool:
        return self.store.should_stop()

    def should_return_to_queue(self) -> bool:
        return self.store.should_return_to_queue()

    def start_stop_watcher(self, on_stop: Callable[[str], None], interval_sec: float = 2.0):
        if not self.is_ui_job or self._watcher is not None:
            return

        def watch():
            while not self._closed.wait(interval_sec):
                try:
                    reason = "queued" if self.should_return_to_queue() else "stopped" if self.should_stop() else None
                except Exception:
                    # A transient database outage must not kill the stop watcher.
                    continue
                if reason:
                    on_stop(reason)
                    return

        self._watcher = threading.Thread(target=watch, daemon=True)
        self._watcher.start()

    def close(self):
        self._closed.set()
        if self._watcher is not None and self._watcher is not threading.current_thread():
            self._watcher.join(timeout=3)
        self.store.close()
