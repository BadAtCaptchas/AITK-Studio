import { db } from './db';
import type { Job } from '../types';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { getToolkitPythonPath } from './pythonPath';

const ACTIVE_LOCAL_STATUSES = new Set(['starting', 'running', 'stopping']);
const LOCAL_JOB_PID_START_GRACE_MS = 2 * 60 * 1000;

function isLocalWorkerId(workerId: string | null | undefined) {
  return !workerId || workerId === 'local';
}

export function isProcessRunning(pid: number | null | undefined) {
  if (typeof pid !== 'number' || !Number.isInteger(pid) || pid <= 0) {
    return false;
  }

  try {
    process.kill(pid, 0);
    return true;
  } catch (error: any) {
    return error?.code === 'EPERM';
  }
}

function jobUpdatedAtMs(job: Job) {
  const value = job.updated_at instanceof Date ? job.updated_at.getTime() : Date.parse(String(job.updated_at));
  return Number.isFinite(value) ? value : Date.now();
}

export async function reconcileLocalJobProcess(job: Job | null): Promise<Job | null> {
  if (!job || !isLocalWorkerId(job.worker_id) || (!ACTIVE_LOCAL_STATUSES.has(job.status) && job.pid == null)) {
    return job;
  }

  if (job.pid == null) {
    if (Date.now() - jobUpdatedAtMs(job) > LOCAL_JOB_PID_START_GRACE_MS) {
      return db.jobs.updateIf(job.id, { attempt_id: job.attempt_id ?? null, status: job.status, updated_at: new Date(job.updated_at) }, {
        status: job.stop ? 'stopped' : 'error',
        pid: null,
        info: 'Job was active but had no recorded process after restart. Start it again if needed.',
      });
    }
    return job;
  }

  if (isProcessRunning(job.pid)) {
    // Legacy processes have no recorded birth time: preserve them rather than
    // claiming ownership from a PID alone. New attempts verify the OS identity.
    if (job.process_started_at == null) return job;
    try {
      const result = await promisify(execFile)(getToolkitPythonPath(), ['-c',
        'import sys,psutil; print(psutil.Process(int(sys.argv[1])).create_time())', String(job.pid)],
        { windowsHide: true, timeout: 5_000, maxBuffer: 4096 });
      if (Math.abs(Number(result.stdout.trim()) - job.process_started_at) < 0.01) return job;
    } catch { return job; }
  }

  return finishObservedJobProcess(job, 'error', `Job process ${job.pid} exited before reporting completion. Check the job log for launch errors.`);
}

/** Call only after observing this process exit or proving its recorded identity is absent. */
export async function finishObservedJobProcess(job: Job, outcome: 'completed' | 'error', info: string): Promise<Job | null> {
  const current = await db.jobs.findById(job.id);
  if (!current || current.attempt_id !== job.attempt_id) return null;
  const status = current.return_to_queue ? 'queued' : current.stop || current.status === 'stopping' ? 'stopped'
    : ACTIVE_LOCAL_STATUSES.has(current.status) ? outcome : current.status;
  const updated = await db.jobs.updateIf(job.id, { attempt_id: job.attempt_id ?? null, status: current.status, updated_at: new Date(current.updated_at) }, {
    status, pid: null, process_started_at: null,
    info: status === 'queued' ? 'Returned to queue' : status === 'stopped' ? 'Job stopped' : ACTIVE_LOCAL_STATUSES.has(current.status) ? info : current.info,
  });
  if (updated && job.attempt_id) {
    const key = `attempt:${job.attempt_id}`, row = await db.runtime.get(key);
    if (row && row.value && typeof row.value === 'object') await db.runtime.compareAndSwap(key, row.version, { ...row.value, status, endedAt: new Date().toISOString() });
  }
  return updated;
}
