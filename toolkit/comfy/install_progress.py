import json
import os
import tempfile
import time
from typing import Any, Dict, Optional


PROGRESS_PATH_ENV = "AITK_COMFY_INSTALL_PROGRESS_PATH"
REPLACE_RETRY_COUNT = 5
REPLACE_RETRY_DELAY_SECONDS = 0.05


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_unlink(path: str):
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


class ComfyInstallProgressReporter:
    def __init__(self, progress_path: Optional[str]):
        self.progress_path = os.path.abspath(progress_path) if progress_path else None
        self.started_at = _now_iso()

    @classmethod
    def from_env(cls):
        return cls(os.environ.get(PROGRESS_PATH_ENV))

    @property
    def enabled(self) -> bool:
        return bool(self.progress_path)

    def write(
        self,
        status: str,
        step: str,
        message: str,
        percent: Optional[float] = None,
        root: Optional[str] = None,
        error: Optional[str] = None,
    ):
        if not self.progress_path:
            return

        payload: Dict[str, Any] = {
            "version": 1,
            "status": status,
            "step": step,
            "message": message,
            "root": root,
            "percent": round(max(0.0, min(100.0, float(percent))), 2) if percent is not None else None,
            "error": error,
            "startedAt": self.started_at,
            "updatedAt": _now_iso(),
        }
        directory = os.path.dirname(self.progress_path)
        tmp_path = None
        try:
            os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=".comfy_install_progress.", suffix=".tmp", dir=directory)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))

            last_error = None
            for attempt in range(REPLACE_RETRY_COUNT):
                try:
                    os.replace(tmp_path, self.progress_path)
                    tmp_path = None
                    return
                except PermissionError as exc:
                    last_error = exc
                    time.sleep(REPLACE_RETRY_DELAY_SECONDS * (attempt + 1))

            if last_error is not None:
                print(f"[AITK] Warning: could not update ComfyUI install progress: {last_error}")
        except OSError as exc:
            print(f"[AITK] Warning: could not write ComfyUI install progress: {exc}")
        finally:
            if tmp_path is not None:
                _safe_unlink(tmp_path)

    def failed(self, step: str, message: str, root: Optional[str] = None, error: Optional[str] = None):
        self.write(
            status="failed",
            step=step,
            message=message,
            percent=None,
            root=root,
            error=error or message,
        )
