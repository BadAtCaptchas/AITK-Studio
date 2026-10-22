import type { QueueFreshness } from '@/utils/queueSnapshot';
import type { WorkerNode } from '@/types';

export default function QueueSyncStatus({ freshness, workers }: { freshness: QueueFreshness[]; workers: Pick<WorkerNode, 'id' | 'name'>[] }) {
  const stale = freshness.filter(item => item.stale);
  if (!stale.length) return null;
  return <p role="status" className="mt-3 text-sm text-amber-800 dark:text-amber-300">
    Remote queue status may be out of date: {stale.map(item => workers.find(worker => worker.id === item.workerId)?.name || 'Remote worker').join(', ')}. Showing the last saved status.
  </p>;
}
