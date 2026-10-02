import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { setLocalQueueRunning } from '@/server/queueCoordination';
import { LeaseBusyError } from '@/server/processLease';

export async function POST(request: NextRequest, { params }: { params: Promise<{ queueID: string }> }) {
  const { queueID } = await params;
  const workerID = request.nextUrl.searchParams.get('worker_id') || 'local';

  if (!isLocalWorker(workerID)) {
    const worker = await getRemoteWorker(workerID);
    const remoteQueue = await remoteJson(worker, `/api/queue/${encodeURIComponent(queueID)}/start`, { method: 'POST' });
    const queue = await db.queues.findByGpuIds(queueID, workerID);
    if (queue) {
      await db.queues.update(queue.id, { is_running: true });
    } else {
      await db.queues.create({ worker_id: workerID, gpu_ids: queueID, is_running: true });
    }
    return NextResponse.json(remoteQueue);
  }

  try {
    return NextResponse.json(await setLocalQueueRunning(queueID, true));
  } catch (error) {
    if (error instanceof LeaseBusyError) return NextResponse.json({ error: 'Queue is busy. Retry shortly.' }, { status: 409 });
    throw error;
  }
}

export function GET() {
  return NextResponse.json({ error: "Use POST for this command" }, { status: 405, headers: { Allow: "POST" } });
}
