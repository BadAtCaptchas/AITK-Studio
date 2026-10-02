import { db } from '../../src/server/db';
import type { Job, Queue } from '../../src/types';
import { devicesOverlap } from '../../src/utils/jobIdentity';
import { reconcileLocalJobProcess } from '../../src/server/jobProcess';
import { isAnyRemoteOllamaCaptionJob } from '../../src/server/secureRemoteCaptionJobs';
import { withLocalQueueTransition } from '../../src/server/queueCoordination';
import { LeaseBusyError } from '../../src/server/processLease';
import startJob from './startJob';

function isSecureRemoteOllamaCaptionJobConfigJson(jobConfigJson: unknown) {
  if (typeof jobConfigJson !== 'string' || !jobConfigJson.trim()) return false;
  try {
    return isAnyRemoteOllamaCaptionJob(JSON.parse(jobConfigJson));
  } catch {
    return false;
  }
}

export default async function processQueue() {
  const queues: Queue[] = await db.queues.list('id', { worker_id: 'local' });

  for (const listedQueue of queues) {
    // Reconciliation probes the OS and must not hold the queue transition lease.
    // A second pass can select work after reconciling a process that has exited.
    for (let pass = 0; pass < 2; pass++) {
      let decision: QueueDecision;
      try {
        decision = await withLocalQueueTransition(listedQueue.gpu_ids, () => inspectQueue(listedQueue.gpu_ids), { wait: false });
      } catch (error) {
        if (error instanceof LeaseBusyError) break;
        throw error;
      }
      if (decision.kind === 'idle') break;
      if (decision.kind === 'start') {
        await startJob(decision.job.id);
        break;
      }
      const reconciledJob = await reconcileLocalJobProcess(decision.job);
      if (reconciledJob && ['starting', 'running', 'stopping'].includes(reconciledJob.status)) break;
    }
  }
}

type QueueDecision = { kind: 'idle' } | { kind: 'start' | 'reconcile'; job: Job };

/** Called only while holding this GPU queue's transition lease. */
async function inspectQueue(gpuIds: string): Promise<QueueDecision> {
  const queue = await db.queues.findByGpuIds(gpuIds, 'local');
  if (!queue) return { kind: 'idle' };
  if (!queue.is_running) {
    const runningJobs: Job[] = await db.jobs.list({
      status: 'running',
      gpu_ids: queue.gpu_ids,
      worker_id: 'local',
    });

    for (const job of runningJobs.filter(job => !isSecureRemoteOllamaCaptionJobConfigJson(job.job_config))) {
      console.log(`Stopping job ${job.id} on GPU(s) ${job.gpu_ids}`);
      await db.jobs.updateIf(job.id, { attempt_id: job.attempt_id ?? null, status: job.status }, {
        return_to_queue: true,
        info: 'Stopping job...',
      });
    }
    return { kind: 'idle' };
  }

  const runningJobs: Job[] = await db.jobs.list({
    status: ['starting', 'running', 'stopping'],
    worker_id: 'local',
  });
  const runningJob = runningJobs.find(job => devicesOverlap(job.gpu_ids, queue.gpu_ids) && !isSecureRemoteOllamaCaptionJobConfigJson(job.job_config));
  if (runningJob) return { kind: 'reconcile', job: runningJob };

  const nextJob: Job | null = await db.jobs.findFirst({
    status: 'queued',
    gpu_ids: queue.gpu_ids,
    worker_id: 'local',
    order: 'queue_asc',
  });
  if (nextJob) return { kind: 'start', job: nextJob };

  console.log(`No more jobs in queue for GPU(s) ${queue.gpu_ids}, stopping queue`);
  await db.queues.update(queue.id, { is_running: false });
  return { kind: 'idle' };
}
