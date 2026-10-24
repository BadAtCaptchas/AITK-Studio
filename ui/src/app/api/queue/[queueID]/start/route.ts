import { db } from '@/server/db';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { setLocalQueueRunning } from '@/server/queueCoordination';
import { LeaseBusyError } from '@/server/processLease';

export async function POST(request: Request, { params }: { params: Promise<{ queueID: string }> }) {
  const { queueID } = await params;
  const workerID = new URL(request.url).searchParams.get('worker_id') || 'local';

  if (!isLocalWorker(workerID)) {
    const worker = await getRemoteWorker(workerID);
    const remoteQueue = await remoteJson(worker, `/api/queue/${encodeURIComponent(queueID)}/start`, { method: 'POST' });
    const queue = await db.queues.findByGpuIds(queueID, workerID);
    if (queue) {
      await db.queues.update(queue.id, { is_running: true });
    } else {
      await db.queues.create({ worker_id: workerID, gpu_ids: queueID, is_running: true });
    }
    return Response.json(remoteQueue);
  }

  try {
    return Response.json(await setLocalQueueRunning(queueID, true));
  } catch (error) {
    if (error instanceof LeaseBusyError) return Response.json({ error: 'Queue is busy. Retry shortly.' }, { status: 409 });
    throw error;
  }
}

export function GET() {
  return Response.json({ error: "Use POST for this command" }, { status: 405, headers: { Allow: "POST" } });
}
