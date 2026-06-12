import { NextRequest, NextResponse } from 'next/server';
import { QueueReorderError, reorderQueueJobs } from '@/server/queueReorder';

export async function POST(request: NextRequest, { params }: { params: Promise<{ queueID: string }> }) {
  const { queueID } = await params;

  try {
    const body = await request.json();
    const workerID = typeof body.worker_id === 'string' && body.worker_id.trim() ? body.worker_id.trim() : 'local';
    const jobIDs = Array.isArray(body.job_ids) ? body.job_ids : [];
    const jobs = await reorderQueueJobs(queueID, jobIDs, workerID);
    return NextResponse.json({ jobs });
  } catch (error) {
    if (error instanceof QueueReorderError) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    console.error(error);
    return NextResponse.json({ error: 'Failed to reorder queue' }, { status: 500 });
  }
}
