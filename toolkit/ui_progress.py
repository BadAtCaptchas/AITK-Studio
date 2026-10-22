"""Bounded, coalesced UI reporting independent of the training event loop."""

import asyncio
from collections import OrderedDict
from threading import Condition, Thread
import warnings

from toolkit.ui_database import JOB_UPDATE_FIELDS


class UIProgressReporter:
    """Keep at most one queued write per supported field and one in flight.

    The store retains its attempt fencing and retry policy. Control reads stay
    synchronous in the trainer; only progress/status writes use this thread.
    """

    def __init__(self, store):
        self._store = store
        self._condition = Condition()
        self._pending = OrderedDict()
        self._inflight = False
        self._closing = False
        self._terminal = False
        self._failed = False
        self._thread = Thread(target=self._run, name='aitk-ui-progress', daemon=True)
        self._thread.start()

    @property
    def pending_count(self):
        with self._condition:
            return len(self._pending) + int(self._inflight)

    def _enqueue(self, key, value):
        with self._condition:
            if self._closing:
                return
            # Replacement belongs after all earlier writes, especially the final
            # step before a terminal status. Completed writes are never retained.
            self._pending.pop(key, None)
            self._pending[key] = value
            self._condition.notify_all()

    def update_key(self, key, value):
        if key not in JOB_UPDATE_FIELDS:
            raise ValueError(f'Unsupported job update field: {key}')
        self._enqueue(('key', key), value)

    def update_status(self, status, info=None):
        with self._condition:
            if self._terminal and status == 'running':
                return
            if status in {'stopped', 'queued', 'error', 'completed'}:
                self._terminal = True
            self._enqueue(('status', ''), (status, info))

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._closing)
                if not self._pending:
                    return
                (kind, key), value = self._pending.popitem(last=False)
                self._inflight = True
            try:
                if kind == 'status':
                    self._store.update_status(*value)
                else:
                    self._store.update_key(key, value)
            except Exception:
                # Never surface SQL values, credentials or dataset details here.
                with self._condition:
                    self._failed = True
            finally:
                with self._condition:
                    self._inflight = False
                    self._condition.notify_all()

    def flush(self) -> bool:
        with self._condition:
            self._condition.wait_for(lambda: not self._pending and not self._inflight)
            failed, self._failed = self._failed, False
        if failed:
            warnings.warn(
                'Some training progress could not be persisted. Check database availability and UI logs.',
                RuntimeWarning, stacklevel=2,
            )
        return not failed

    def close(self):
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join()
        self.flush()


class UIReportingMixin:
    """Shared reporting lifecycle for legacy UITrainer and DiffusionTrainer."""

    def _init_ui_reporting(self):
        self._progress_reporter = (
            UIProgressReporter(self.ui_job_store) if self.accelerator.is_main_process else None
        )

    def update_step(self):
        self.update_db_key('step', self.step_num)

    def update_db_key(self, key, value):
        reporter = getattr(self, '_progress_reporter', None)
        if reporter is not None:
            reporter.update_key(key, value)

    def update_status(self, status, info=None):
        reporter = getattr(self, '_progress_reporter', None)
        if reporter is not None:
            reporter.update_status(status, info)

    def flush_ui_reporting(self):
        reporter = getattr(self, '_progress_reporter', None)
        if reporter is not None:
            reporter.flush()

    async def wait_for_all_async(self):
        # Preserve the extension hook without owning or nesting an event loop.
        await asyncio.to_thread(self.flush_ui_reporting)

    def _close_ui_reporting(self):
        reporter = getattr(self, '_progress_reporter', None)
        if reporter is not None:
            reporter.close()
            self._progress_reporter = None
