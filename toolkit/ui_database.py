import os
import sqlite3
import time
from typing import Any, Callable, Optional


DB_PROVIDER_ENV = "AITK_DB_PROVIDER"
SQLITE_PATH_ENV = "AITK_SQLITE_PATH"
MONGODB_URI_ENV = "AITK_MONGODB_URI"
MONGODB_DB_ENV = "AITK_MONGODB_DB"

DEFAULT_MONGODB_DB = "ai_toolkit"

JOB_UPDATE_FIELDS = {
    "status",
    "stop",
    "return_to_queue",
    "step",
    "info",
    "speed_string",
    "pid",
}


class DatabaseConfigError(RuntimeError):
    pass


def get_database_provider() -> str:
    provider = os.environ.get(DB_PROVIDER_ENV, "sqlite").strip().lower()
    if provider not in {"sqlite", "mongodb"}:
        raise DatabaseConfigError(
            f'Invalid {DB_PROVIDER_ENV} "{provider}". Expected "sqlite" or "mongodb".'
        )
    return provider


def get_mongodb_config() -> tuple[str, str]:
    uri = os.environ.get(MONGODB_URI_ENV, "").strip()
    if not uri:
        raise DatabaseConfigError(
            f"{MONGODB_URI_ENV} is required when {DB_PROVIDER_ENV}=mongodb."
        )
    db_name = os.environ.get(MONGODB_DB_ENV, DEFAULT_MONGODB_DB).strip()
    return uri, db_name or DEFAULT_MONGODB_DB


class UIJobStore:
    def __init__(self, job_id: Optional[str], sqlite_db_path: str):
        self.job_id = job_id.strip() if job_id else None
        self.sqlite_db_path = os.environ.get(SQLITE_PATH_ENV, sqlite_db_path)
        self.provider = get_database_provider()
        self._client = None
        self._jobs = None
        self.available = False
        self.description = ""

        if not self.job_id:
            return

        if self.provider == "sqlite":
            if not os.path.exists(self.sqlite_db_path):
                return
            self.available = True
            self.description = f"SQLite database at {self.sqlite_db_path}"
            return

        uri, db_name = get_mongodb_config()
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise ImportError(
                "MongoDB UI support requires pymongo. Install it with `pip install pymongo`."
            ) from exc

        self._client = MongoClient(uri)
        self._jobs = self._client[db_name]["jobs"]
        self.available = True
        self.description = f"MongoDB database {db_name}"

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None
            self._jobs = None

    def _db_connect(self):
        conn = sqlite3.connect(self.sqlite_db_path, timeout=30.0)
        conn.isolation_level = None
        return conn

    def _retry_sqlite_operation(
        self,
        operation_func: Callable[[], Any],
        max_retries: int = 3,
        base_delay: float = 2.0,
    ):
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return operation_func()
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc):
                    last_error = exc
                    if attempt < max_retries:
                        delay = base_delay * (2**attempt)
                        print(
                            f"[AITK] Database locked (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        print(
                            f"[AITK] Database locked after {max_retries + 1} attempts, giving up."
                        )
                else:
                    raise
        raise last_error

    def should_stop(self) -> bool:
        if not self.available:
            return False

        if self.provider == "mongodb":
            assert self._jobs is not None
            row = self._jobs.find_one({"id": self.job_id}, {"_id": 0, "stop": 1})
            return bool(row.get("stop")) if row else False

        def _check_stop():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT stop FROM Job WHERE id = ?", (self.job_id,))
                stop = cursor.fetchone()
                return False if stop is None else stop[0] == 1

        return bool(self._retry_sqlite_operation(_check_stop))

    def should_return_to_queue(self) -> bool:
        if not self.available:
            return False

        if self.provider == "mongodb":
            assert self._jobs is not None
            row = self._jobs.find_one(
                {"id": self.job_id}, {"_id": 0, "return_to_queue": 1}
            )
            return bool(row.get("return_to_queue")) if row else False

        def _check_return_to_queue():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT return_to_queue FROM Job WHERE id = ?", (self.job_id,)
                )
                return_to_queue = cursor.fetchone()
                return False if return_to_queue is None else return_to_queue[0] == 1

        return bool(self._retry_sqlite_operation(_check_return_to_queue))

    def update_key(self, key: str, value: Any):
        if not self.available:
            return
        if key not in JOB_UPDATE_FIELDS:
            raise ValueError(f"Unsupported job update field: {key}")

        if self.provider == "mongodb":
            assert self._jobs is not None
            self._jobs.update_one(
                {"id": self.job_id},
                {"$set": {key: value}, "$currentDate": {"updated_at": True}},
            )
            return

        def _do_update():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    if isinstance(value, str):
                        value_to_insert = value
                    elif isinstance(value, bool):
                        value_to_insert = 1 if value else 0
                    else:
                        value_to_insert = str(value)

                    update_query = f"UPDATE Job SET {key} = ? WHERE id = ?"
                    cursor.execute(update_query, (value_to_insert, self.job_id))
                finally:
                    cursor.execute("COMMIT")

        self._retry_sqlite_operation(_do_update)

    def update_status(self, status: str, info: Optional[str] = None):
        if not self.available:
            return

        if self.provider == "mongodb":
            assert self._jobs is not None
            patch = {"status": status}
            if info is not None:
                patch["info"] = info
            if status in {"stopped", "error", "completed"}:
                patch["pid"] = None
            self._jobs.update_one(
                {"id": self.job_id},
                {"$set": patch, "$currentDate": {"updated_at": True}},
            )
            return

        def _do_update():
            with self._db_connect() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE")
                try:
                    clear_pid = status in {"stopped", "error", "completed"}
                    if info is not None:
                        if clear_pid:
                            cursor.execute(
                                "UPDATE Job SET status = ?, info = ?, pid = NULL WHERE id = ?",
                                (status, info, self.job_id),
                            )
                        else:
                            cursor.execute(
                                "UPDATE Job SET status = ?, info = ? WHERE id = ?",
                                (status, info, self.job_id),
                            )
                    else:
                        if clear_pid:
                            cursor.execute(
                                "UPDATE Job SET status = ?, pid = NULL WHERE id = ?",
                                (status, self.job_id),
                            )
                        else:
                            cursor.execute(
                                "UPDATE Job SET status = ? WHERE id = ?",
                                (status, self.job_id),
                            )
                finally:
                    cursor.execute("COMMIT")

        self._retry_sqlite_operation(_do_update)
