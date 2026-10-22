import { db } from './db';
import { fetchWorkerQueues, getRemoteWorker, runRemoteBackgroundPoll } from './remoteClient';
import { isRecord } from './commandInput';
import type { QueueFreshness } from '../utils/queueSnapshot';
const freshnessKey = (workerId: string) => `queue-sync:${workerId}`;

/** Only the supervised worker performs network I/O. Queue reads remain available offline. */
export async function syncWorkerQueues(): Promise<void> {
  const workers = await db.workerNodes.list({ enabled: true });
  await Promise.all(workers.map(async workerRecord => {
    const key = freshnessKey(workerRecord.id);
    const previous = await db.runtime.get(key);
    let updatedAt = isRecord(previous?.value) && typeof previous.value.updatedAt === 'string' ? previous.value.updatedAt : null;
    let error: string | null = null;
    try {
      const worker = await getRemoteWorker(workerRecord.id);
      const poll = await runRemoteBackgroundPoll(worker, 'queue sync', () => fetchWorkerQueues(worker));
      if ('reason' in poll) error = poll.reason;
      else {
        for (const remoteQueue of poll.value.queues) {
          const queue = await db.queues.findByGpuIds(remoteQueue.gpu_ids, worker.id);
          if (queue) await db.queues.update(queue.id, { is_running: remoteQueue.is_running });
          else await db.queues.create({ worker_id: worker.id, gpu_ids: remoteQueue.gpu_ids, is_running: remoteQueue.is_running });
        }
        const reported = new Set(poll.value.queues.map(queue => queue.gpu_ids));
        for (const saved of await db.queues.list('gpu_ids', { worker_id: worker.id })) {
          if (!reported.has(saved.gpu_ids) && saved.is_running) await db.queues.update(saved.id, { is_running: false });
        }
        updatedAt = new Date().toISOString();
      }
    } catch (failure) {
      error = failure instanceof Error ? failure.message : 'Remote queue sync failed';
    }
    await db.runtime.compareAndSwap(key, previous?.version ?? null, { updatedAt, checkedAt: new Date().toISOString(), error });
  }));
}

export async function listQueuesForQueueApi(localOnly = false) {
  return db.queues.list('gpu_ids', localOnly ? { worker_id: 'local' } : {});
}

export async function getQueueFreshness(): Promise<QueueFreshness[]> {
  const workers = await db.workerNodes.list();
  return Promise.all(workers.map(async worker => {
    const row = await db.runtime.get(freshnessKey(worker.id));
    const value = isRecord(row?.value) ? row.value : {};
    const updatedAt = typeof value.updatedAt === 'string' ? value.updatedAt : null;
    const error = !worker.enabled ? 'Worker is disabled; queue synchronization is paused.' : typeof value.error === 'string' ? value.error : null;
    return { workerId: worker.id, updatedAt, checkedAt: typeof value.checkedAt === 'string' ? value.checkedAt : '', error,
      stale: !!error || !updatedAt || !Number.isFinite(Date.parse(updatedAt)) || Date.now() - Date.parse(updatedAt) > 30_000 };
  }));
}
