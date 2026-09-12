import { db } from './db';
import { withProcessLease } from './processLease';
import { JobStartError } from './jobStart';
import { isRecord } from './commandInput';
import type { Job } from '../types';

/** A lost response to the same start command must never launch a second run. */
export async function idempotentStart(jobID: string, key: string | undefined, start: () => Promise<Job>): Promise<Job> {
  if (!key) return start();
  const recordKey = `start-command:${jobID}:${key}`;
  return withProcessLease(recordKey, async () => {
    const row = await db.runtime.get(recordKey);
    const job = await db.jobs.findById(jobID);
    if (!job) throw new JobStartError({ error: 'Job not found' }, 404);
    if (row && isRecord(row.value)) {
      if (
        row.value.accepted === true ||
        job.attempt_id !== row.value.beforeAttempt ||
        ['queued', 'starting', 'running', 'stopping'].includes(job.status)
      ) {
        await db.runtime.compareAndSwap(recordKey, row.version, { ...row.value, accepted: true });
        return job;
      }
    } else {
      if (
        !(await db.runtime.compareAndSwap(recordKey, null, {
          jobID,
          beforeAttempt: job.attempt_id ?? null,
          createdAt: new Date().toISOString(),
          accepted: false,
        }))
      )
        throw new JobStartError({ error: 'Start command is busy; retry' }, 409);
    }
    const result = await start();
    const current = await db.runtime.get(recordKey);
    if (current)
      await db.runtime.compareAndSwap(recordKey, current.version, {
        jobID,
        accepted: true,
        completedAt: new Date().toISOString(),
      });
    return result;
  });
}
