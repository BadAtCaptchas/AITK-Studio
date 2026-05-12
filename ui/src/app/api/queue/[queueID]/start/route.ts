import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';

export async function GET(request: NextRequest, { params }: { params: { queueID: string } }) {
  const { queueID } = await params;
  const workerID = request.nextUrl.searchParams.get('worker_id') || 'local';

  if (!isLocalWorker(workerID)) {
    const worker = await getRemoteWorker(workerID);
    const remoteQueue = await remoteJson(worker, `/api/queue/${encodeURIComponent(queueID)}/start`);
    const queue = await db.queues.findByGpuIds(queueID, workerID);
    if (queue) {
      await db.queues.update(queue.id, { is_running: true });
    } else {
      await db.queues.create({ worker_id: workerID, gpu_ids: queueID, is_running: true });
    }
    return NextResponse.json(remoteQueue);
  }

  const queue = await db.queues.findByGpuIds(queueID, workerID);

  if (!queue) {
    // create it if it doesn't exist
    const newQueue = await db.queues.create({ worker_id: workerID, gpu_ids: queueID, is_running: true });
    return NextResponse.json(newQueue);
  }

  await db.queues.update(queue.id, { is_running: true });

  return NextResponse.json(queue);
}
