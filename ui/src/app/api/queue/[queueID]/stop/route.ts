import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';

export async function POST(request: NextRequest, { params }: { params: Promise<{ queueID: string }> }) {
  const { queueID } = await params;
  const workerID = request.nextUrl.searchParams.get('worker_id') || 'local';

  if (!isLocalWorker(workerID)) {
    const worker = await getRemoteWorker(workerID);
    const remoteQueue = await remoteJson(worker, `/api/queue/${encodeURIComponent(queueID)}/stop`, { method: 'POST' });
    const queue = await db.queues.findByGpuIds(queueID, workerID);
    if (queue) {
      await db.queues.update(queue.id, { is_running: false });
    }
    return NextResponse.json(remoteQueue);
  }

  const queue = await db.queues.findByGpuIds(queueID, workerID);

  if (!queue) {
    return NextResponse.json({ error: 'Queue not found' }, { status: 404 });
  }

  await db.queues.update(queue.id, { is_running: false });

  return NextResponse.json(queue);
}

export function GET() {
  return NextResponse.json({ error: "Use POST for this command" }, { status: 405, headers: { Allow: "POST" } });
}
