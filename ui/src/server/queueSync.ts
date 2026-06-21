import { db } from './db';
import { fetchWorkerQueues, getRemoteWorker, runRemoteBackgroundPoll } from './remoteClient';

export async function syncWorkerQueues() {
  const workers = await db.workerNodes.list({ enabled: true });
  await Promise.all(
    workers.map(async workerRecord => {
      try {
        const worker = await getRemoteWorker(workerRecord.id);
        const poll = await runRemoteBackgroundPoll(worker, 'queue sync', () => fetchWorkerQueues(worker));
        if ('reason' in poll) return;

        await Promise.all(
          (poll.value.queues || []).map(async remoteQueue => {
            const queue = await db.queues.findByGpuIds(remoteQueue.gpu_ids, worker.id);
            if (queue) {
              await db.queues.update(queue.id, { is_running: remoteQueue.is_running });
              return;
            }
            await db.queues.create({
              worker_id: worker.id,
              gpu_ids: remoteQueue.gpu_ids,
              is_running: remoteQueue.is_running,
            });
          }),
        );
      } catch (error) {
        console.error(
          `Failed to sync queues for worker ${workerRecord.name}: ${
            error instanceof Error ? error.message : 'Remote queue sync failed'
          }`,
        );
      }
    }),
  );
}

export async function listQueuesForQueueApi() {
  await syncWorkerQueues();
  return db.queues.list('gpu_ids');
}
