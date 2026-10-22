import type { Queue } from '../types';

export type QueueFreshness = { workerId: string; updatedAt: string | null; checkedAt: string; error: string | null; stale: boolean };
const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);

export function parseQueueSnapshot(value: unknown): { queues: Queue[]; freshness: QueueFreshness[] } {
  if (!record(value) || !Array.isArray(value.queues)) throw new Error('Invalid queue response');
  const queues = value.queues.map(queue => {
    if (!record(queue) || typeof queue.id !== 'number' || !Number.isSafeInteger(queue.id) ||
        typeof queue.worker_id !== 'string' || typeof queue.gpu_ids !== 'string' || typeof queue.is_running !== 'boolean')
      throw new Error('Invalid queue response');
    return { id: queue.id, worker_id: queue.worker_id, gpu_ids: queue.gpu_ids, is_running: queue.is_running };
  });
  if (value.freshness === undefined) return { queues, freshness: [] };
  if (!Array.isArray(value.freshness)) throw new Error('Invalid queue freshness');
  const freshness = value.freshness.map(item => {
    if (!record(item) || typeof item.workerId !== 'string' || typeof item.checkedAt !== 'string' || typeof item.stale !== 'boolean' ||
        (item.updatedAt !== null && typeof item.updatedAt !== 'string') || (item.error !== null && typeof item.error !== 'string'))
      throw new Error('Invalid queue freshness');
    return { workerId: item.workerId, checkedAt: item.checkedAt, stale: item.stale,
      updatedAt: typeof item.updatedAt === 'string' ? item.updatedAt : null, error: typeof item.error === 'string' ? item.error : null };
  });
  return { queues, freshness };
}
