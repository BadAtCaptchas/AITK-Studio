import { db } from './db';
import type { Job } from '../types';

const ACTIVE_LOCAL_STATUSES = new Set(['running', 'stopping']);

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

export async function reconcileLocalJobProcess(job: Job | null): Promise<Job | null> {
  if (!job || !isLocalWorkerId(job.worker_id) || !ACTIVE_LOCAL_STATUSES.has(job.status)) {
    return job;
  }

  if (job.pid == null) {
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
