import json
import os
import tempfile
import threading
import time
import importlib
from contextlib import AbstractContextManager
from typing import Any, Dict, Optional


PROGRESS_PATH_ENV = "AITK_HF_DOWNLOAD_PROGRESS_PATH"
WRITE_INTERVAL_SECONDS = 0.25

_installed = False


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _coerce_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


class HFDownloadProgressReporter:
    def __init__(self, progress_path: str):
        self.progress_path = os.path.abspath(progress_path)
        self._lock = threading.RLock()
        self._active: Dict[int, Dict[str, Any]] = {}
        self._last_completed: Optional[Dict[str, Any]] = None
        self._last_error: Optional[str] = None
        self._last_write = 0.0
        self._next_id = 1

        os.makedirs(os.path.dirname(self.progress_path), exist_ok=True)
        self.write(force=True, status="idle", message="Waiting for Hugging Face downloads")

    def start(self, desc: Optional[str], total: Any, initial: Any, source: Optional[str]) -> int:
        with self._lock:
            transfer_id = self._next_id
            self._next_id += 1
            file_name = str(desc or "Hugging Face file")
            initial_value = _coerce_number(initial) or 0.0
            total_value = _coerce_number(total)
            now = _now_iso()
            self._active[transfer_id] = {
                "id": transfer_id,
                "fileName": file_name,
                "source": source or "huggingface_hub",
                "bytesDownloaded": initial_value,
                "bytesTotal": total_value,
                "startedAt": now,
                "updatedAt": now,
            }
            self._last_error = None
            self.write(force=True, status="downloading")
            return transfer_id

    def update(self, transfer_id: int, amount: Any):
        increment = _coerce_number(amount)
        if increment is None:
            return
        with self._lock:
            transfer = self._active.get(transfer_id)
            if transfer is None:
                return
            transfer["bytesDownloaded"] = float(transfer.get("bytesDownloaded") or 0.0) + increment
            transfer["updatedAt"] = _now_iso()
            self.write(status="downloading")

    def finish(self, transfer_id: int, failed: bool = False, error: Optional[str] = None):
        with self._lock:
            transfer = self._active.pop(transfer_id, None)
            if transfer is not None:
                transfer["updatedAt"] = _now_iso()
                self._last_completed = transfer
            if failed:
                self._last_error = error or "Download failed"
                self.write(force=True, status="failed", message=self._last_error)
            elif self._active:
                self.write(force=True, status="downloading")
            else:
                self.write(force=True, status="completed", message="Hugging Face downloads complete")

    def snapshot(self, status: str, message: Optional[str] = None) -> Dict[str, Any]:
        active = list(self._active.values())
        summary_items = active or ([self._last_completed] if self._last_completed is not None else [])
        bytes_downloaded = sum(float(item.get("bytesDownloaded") or 0.0) for item in summary_items)
        total_values = [item.get("bytesTotal") for item in summary_items]
        has_unknown_total = any(value is None for value in total_values)
        bytes_total = None if has_unknown_total else sum(float(value or 0.0) for value in total_values)
        percent = None

        if bytes_total and bytes_total > 0:
            percent = max(0.0, min(100.0, (bytes_downloaded / bytes_total) * 100.0))

        primary = active[-1] if active else self._last_completed
        file_name = primary.get("fileName") if primary else None

        if message is None:
            if status == "downloading":
                message = f"Downloading {file_name}" if file_name else "Downloading Hugging Face files"
            elif status == "completed":
                message = "Hugging Face downloads complete"
            elif status == "failed":
                message = self._last_error or "Download failed"
            else:
                message = "Waiting for Hugging Face downloads"

        return {
            "version": 1,
            "status": status,
            "message": message,
            "fileName": file_name,
            "activeCount": len(active),
            "bytesDownloaded": int(bytes_downloaded),
            "bytesTotal": int(bytes_total) if bytes_total is not None else None,
            "percent": round(percent, 2) if percent is not None else None,
            "downloads": [
                {
                    **item,
                    "bytesDownloaded": int(float(item.get("bytesDownloaded") or 0.0)),
                    "bytesTotal": int(float(item["bytesTotal"])) if item.get("bytesTotal") is not None else None,
                }
                for item in active
            ],
            "error": self._last_error,
            "updatedAt": _now_iso(),
        }

    def write(self, force: bool = False, status: str = "downloading", message: Optional[str] = None):
        now = time.monotonic()
        if not force and now - self._last_write < WRITE_INTERVAL_SECONDS:
            return
        self._last_write = now

        payload = self.snapshot(status=status, message=message)
        directory = os.path.dirname(self.progress_path)
        fd, tmp_path = tempfile.mkstemp(prefix=".hf_download_progress.", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
            os.replace(tmp_path, self.progress_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


class _TrackedProgress:
    def __init__(self, progress: Any, reporter: HFDownloadProgressReporter, transfer_id: int):
        self._progress = progress
        self._reporter = reporter
        self._transfer_id = transfer_id

    def update(self, amount: Any = 1):
        self._reporter.update(self._transfer_id, amount)
        return self._progress.update(amount)

    def __getattr__(self, name: str):
        return getattr(self._progress, name)


class _TrackedProgressContext(AbstractContextManager):
    def __init__(
        self,
        context: AbstractContextManager,
        reporter: HFDownloadProgressReporter,
        desc: Optional[str],
        total: Any,
        initial: Any,
        source: Optional[str],
    ):
        self._context = context
        self._reporter = reporter
        self._desc = desc
        self._total = total
        self._initial = initial
        self._source = source
        self._transfer_id: Optional[int] = None

    def __enter__(self):
        progress = self._context.__enter__()
        self._transfer_id = self._reporter.start(self._desc, self._total, self._initial, self._source)
        return _TrackedProgress(progress, self._reporter, self._transfer_id)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._transfer_id is not None:
            self._reporter.finish(
                self._transfer_id,
                failed=exc_type is not None,
                error=str(exc_value) if exc_value is not None else None,
            )
        return self._context.__exit__(exc_type, exc_value, traceback)


def install_hf_download_progress(progress_path: Optional[str] = None) -> bool:
    global _installed

    if _installed:
        return True

    path = progress_path or os.environ.get(PROGRESS_PATH_ENV)
    if not path:
        return False

    try:
        import huggingface_hub.file_download as file_download
        hf_tqdm = importlib.import_module("huggingface_hub.utils.tqdm")
    except Exception as exc:
        print(f"[AITK] Warning: could not install Hugging Face download progress hook: {exc}")
        return False

    reporter = HFDownloadProgressReporter(path)
    original_context = getattr(file_download, "_get_progress_bar_context", None) or hf_tqdm._get_progress_bar_context

    def tracked_progress_context(*args, **kwargs):
        desc = kwargs.get("desc")
        total = kwargs.get("total")
        initial = kwargs.get("initial", 0)
        source = kwargs.get("name")

        if len(args) > 0 and desc is None:
            desc = args[0]

        context = original_context(*args, **kwargs)
        return _TrackedProgressContext(context, reporter, desc, total, initial, source)

    file_download._get_progress_bar_context = tracked_progress_context
    hf_tqdm._get_progress_bar_context = tracked_progress_context
    _installed = True
    print(f"[AITK] Hugging Face download progress enabled: {os.path.abspath(path)}")
    return True
