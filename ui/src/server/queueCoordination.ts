import { setTimeout as delay } from 'timers/promises';
import { db } from './db';
import { acquireProcessLease, LeaseBusyError } from './processLease';
import type { Queue } from '../types';

/** Serialize short local queue transitions across the UI and cron processes. */
export async function withLocalQueueTransition<T>(
  gpuIds: string,
  work: () => Promise<T>,
  options: { wait?: boolean } = {},
): Promise<T> {
  const key = `queue-transition:${JSON.stringify(['local', gpuIds])}`;
  const deadline = Date.now() + 5_000;
  let lease: Awaited<ReturnType<typeof acquireProcessLease>>;
  while (true) {
    try {
      lease = await acquireProcessLease(key);
      break;
    } catch (error) {
      if (!(error instanceof LeaseBusyError) || options.wait === false || Date.now() >= deadline) throw error;
      await delay(25);
    }
  }
  try {
    return await work();
  } finally {
    await lease.release();
  }
}

export async function setLocalQueueRunning(gpuIds: string, isRunning: boolean): Promise<Queue | null> {
  return withLocalQueueTransition(gpuIds, async () => {
    const queue = await db.queues.findByGpuIds(gpuIds, 'local');
    if (queue) return db.queues.update(queue.id, { is_running: isRunning });
    if (!isRunning) return null;
    return db.queues.create({ worker_id: 'local', gpu_ids: gpuIds, is_running: true });
  });
}
