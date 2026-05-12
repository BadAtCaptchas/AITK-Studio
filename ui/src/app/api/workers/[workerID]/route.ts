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
  const jobs = await db.jobs.list({ worker_id: workerID });
  if (jobs.length > 0) {
    return NextResponse.json({ error: 'Cannot delete a worker that still has jobs' }, { status: 409 });
  }

  const worker = await db.workerNodes.delete(workerID);
  return NextResponse.json(worker ? toPublicWorker(worker) : null);
}
