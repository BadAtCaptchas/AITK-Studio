import { db } from './db';
import type { Job } from '../types';

const ACTIVE_LOCAL_STATUSES = new Set(['running', 'stopping']);
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
  if (!job || !isLocalWorkerId(job.worker_id) || !ACTIVE_LOCAL_STATUSES.has(job.status)) {
    return job;
  }

  if (job.pid == null) {
    if (Date.now() - jobUpdatedAtMs(job) > LOCAL_JOB_PID_START_GRACE_MS) {
      return db.jobs.update(job.id, {
        status: 'error',
        pid: null,
        info: 'Job was active but had no recorded process after restart. Start it again if needed.',
      });
    }
    return job;
  }

  if (isProcessRunning(job.pid)) {
    return job;
  }

  return db.jobs.update(job.id, {
    status: 'error',
    pid: null,
    info: `Job process ${job.pid} exited before reporting completion. Check the job log for launch errors.`,
  });
}
