import { NextResponse } from 'next/server';
import { db } from '@/server/db';
import { fetchWorkerQueues, getRemoteWorker } from '@/server/remoteClient';

async function syncWorkerQueues() {
  const workers = await db.workerNodes.list({ enabled: true });
  await Promise.all(
    workers.map(async workerRecord => {
      try {
        const worker = await getRemoteWorker(workerRecord.id);
        const data = await fetchWorkerQueues(worker);
        await Promise.all(
          (data.queues || []).map(async remoteQueue => {
            const queue = await db.queues.findByGpuIds(remoteQueue.gpu_ids, worker.id);
            if (queue) {
              await db.queues.update(queue.id, { is_running: remoteQueue.is_running });
            } else {
              await db.queues.create({
                worker_id: worker.id,
                gpu_ids: remoteQueue.gpu_ids,
                is_running: remoteQueue.is_running,
              });
            }
          }),
        );
      } catch (error) {
        console.error(`Failed to sync queues for worker ${workerRecord.name}:`, error);
      }
    }),
  );
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);

  try {
    await syncWorkerQueues();
    const queues = await db.queues.list('gpu_ids');
    return NextResponse.json({ queues: queues });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch queue' }, { status: 500 });
  }
}
