import { db } from './db';
import { decodeJobCursor, encodeJobCursor } from './jobsApiList';
import { reconcileLocalJobProcess } from './jobProcess';
import { syncRemoteJob, isLocalWorker } from './remoteClient';
import { withHFDownloadProgress } from './hfDownloadProgress';
import { withComfyInstallProgress } from './comfyInstallProgress';

export async function synchronizeJobPage(): Promise<void> {
  const cursor = await db.runtime.get('job-read-model:cursor');
  const before = decodeJobCursor(typeof cursor?.value === 'string' ? cursor.value : null);
  const jobs = await db.jobs.list({ limit: 25, before });
  // Active work is always visited; history advances independently to avoid starvation.
  const active = await db.jobs.list({ status: ['starting', 'running', 'remote-starting', 'stopping'], limit: 100 });
  const unique = new Map([...jobs, ...active].map(job => [job.id, job]));
  let failed = 0;
  for (const job of unique.values()) {
    try {
      const current = isLocalWorker(job.worker_id) ? await reconcileLocalJobProcess(job) : await syncRemoteJob(job);
      if (current) await withComfyInstallProgress(await withHFDownloadProgress(current));
    } catch (error) {
      failed++;
      console.error('Job synchronization failed', {
        jobID: job.id,
        attemptID: job.attempt_id,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }
  await db.runtime.compareAndSwap(
    'job-read-model:cursor',
    cursor?.version ?? null,
    jobs.length === 25 ? encodeJobCursor(jobs[jobs.length - 1]) : '',
  );
  const freshness = await db.runtime.get('job-read-model:freshness');
  await db.runtime.compareAndSwap('job-read-model:freshness', freshness?.version ?? null, {
    updatedAt: new Date().toISOString(),
    failed,
    checked: unique.size,
  });
}
