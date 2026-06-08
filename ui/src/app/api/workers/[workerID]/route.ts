import { NextResponse } from 'next/server';
import { db, type WorkerNodeRecord } from '@/server/db';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function toPublicWorker(worker: WorkerNodeRecord) {
  const { api_token: _apiToken, ...publicWorker } = worker;
  return publicWorker;
}

export async function DELETE(_request: Request, { params }: { params: Promise<{ workerID: string }> }) {
  const { workerID } = await params;
  if (workerID === 'local') {
    return NextResponse.json({ error: 'The local worker cannot be deleted' }, { status: 400 });
  }

  const activeJobs = await db.jobs.list({ worker_id: workerID, status: ['queued', 'running', 'stopping'] });
  if (activeJobs.length > 0) {
    return NextResponse.json(
      { error: 'Cannot delete a worker with queued, running, or stopping jobs' },
      { status: 409 },
    );
  }

  const deletedQueues = await db.queues.deleteMany({ worker_id: workerID });
  const worker = await db.workerNodes.delete(workerID);
  return NextResponse.json({ worker: worker ? toPublicWorker(worker) : null, deletedQueues });
}
