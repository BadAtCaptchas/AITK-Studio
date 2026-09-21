"""Register process ownership before importing expensive training integrations."""
import os
import signal

import psutil

from toolkit.ui_database import UIJobStore


def register_attempt_process():
    """Register this process for the current job attempt, rejecting stale launches."""
    if not os.environ.get('AITK_ATTEMPT_ID'):
        return
    store = UIJobStore(os.environ.get('AITK_JOB_ID'), os.environ.get('AITK_SQLITE_PATH', ''))
    try:
        if not store.available:
            raise RuntimeError('Attempt database is unavailable')
        pid = os.getpid()
        process = psutil.Process(pid)
        started = process.create_time()
        launch_pids = [pid]
        if os.name == 'nt':
            # A Windows venv redirector stays alive while its Python child runs.
            # Node records that redirector's PID; only its actual child may take over.
            parent_pid = process.ppid()
            if parent_pid > 0:
                launch_pids.append(parent_pid)
        if store.provider == 'mongodb':
            result = store._jobs.update_one(
                {**store.identity, 'status': 'starting', '$or': [{'pid': None}, *({'pid': value} for value in launch_pids)]},
                {'$set': {'pid': pid, 'process_started_at': started, 'status': 'running', 'info': 'Loading training runtime'}},
            )
            accepted = result.matched_count == 1
        else:
            with store._db_connect() as connection:
                # Use explicit identity here; ordinary trainer queries are fenced by UIJobStore.
                accepted = connection.execute(
                    "UPDATE Job SET pid=?, process_started_at=?, status='running', info='Loading training runtime' "
                    "WHERE id=? AND attempt_id=? AND status='starting' AND "
                    "(pid IS NULL OR pid IN (" + ','.join('?' for _ in launch_pids) + "))",
                    (pid, started, store.job_id, store.attempt_id, *launch_pids),
                ).rowcount == 1
        if not accepted:
            raise RuntimeError('This launch no longer owns the job attempt; refusing to run')
    finally:
        store.close()


def stop_owned_process(job_id, attempt_id, pid, started_at):
    """Signal a live, fenced attempt without ever treating a PID as its identity."""
    store = UIJobStore(job_id, os.environ.get('AITK_SQLITE_PATH', ''))
    try:
        if not attempt_id or store.attempt_id != attempt_id or not store.available:
            return False
        if store.provider == 'mongodb':
            row = store._jobs.find_one({**store.identity, 'status': 'stopping', 'pid': pid})
            owned = row and row.get('process_started_at') == started_at
        else:
            with store._db_connect() as connection:
                owned = connection.execute(
                    "SELECT 1 FROM Job WHERE id=? AND attempt_id=? AND status='stopping' AND pid=? AND process_started_at=?",
                    (job_id, attempt_id, pid, started_at),
                ).fetchone()
        if not owned:
            return False
        process = psutil.Process(pid)
        if abs(process.create_time() - started_at) >= 0.01 or not process.is_running():
            return False
        # psutil checks PID reuse for send_signal/terminate on these instances.
        for target in reversed(process.children(recursive=True)):
            try:
                target.terminate() if os.name == 'nt' else target.send_signal(signal.SIGINT)
            except psutil.NoSuchProcess:
                pass
        process.terminate() if os.name == 'nt' else process.send_signal(signal.SIGINT)
        return True
    except psutil.NoSuchProcess:
        return False
    finally:
        store.close()
