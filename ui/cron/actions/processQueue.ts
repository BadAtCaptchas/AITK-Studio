import { db } from '../../src/server/db';
import type { Job, Queue } from '../../src/types';
import { reconcileLocalJobProcess } from '../../src/server/jobProcess';
import startJob from './startJob';

export default async function processQueue() {
  const queues: Queue[] = await db.queues.list('id', { worker_id: 'local' });

  for (const queue of queues) {
    if (!queue.is_running) {
      // stop any running jobs first
      const runningJobs: Job[] = await db.jobs.list({
        status: 'running',
        gpu_ids: queue.gpu_ids,
        worker_id: 'local',
      });

      for (const job of runningJobs) {
        console.log(`Stopping job ${job.id} on GPU(s) ${job.gpu_ids}`);
        await db.jobs.update(job.id, {
          return_to_queue: true,
          info: 'Stopping job...',
        });
      }
    }
    if (queue.is_running) {
      // first see if one is already running, status of running or stopping
      const runningJob: Job | null = await db.jobs.findFirst({
        status: ['running', 'stopping'],
        gpu_ids: queue.gpu_ids,
        worker_id: 'local',
      });

      if (runningJob) {
        const reconciledJob = await reconcileLocalJobProcess(runningJob);
        if (reconciledJob && ['running', 'stopping'].includes(reconciledJob.status)) {
          // already running, nothing to do
          continue; // skip to next queue
        }
      }

      // find the next job in the queue
      const nextJob: Job | null = await db.jobs.findFirst({
        status: 'queued',
        gpu_ids: queue.gpu_ids,
        worker_id: 'local',
        order: 'queue_asc',
      });
      if (nextJob) {
        console.log(`Starting job ${nextJob.id} on GPU(s) ${nextJob.gpu_ids}`);
        await startJob(nextJob.id);
      } else {
        // no more jobs, stop the queue
        console.log(`No more jobs in queue for GPU(s) ${queue.gpu_ids}, stopping queue`);
        await db.queues.update(queue.id, { is_running: false });
      }
    }
  }
}
