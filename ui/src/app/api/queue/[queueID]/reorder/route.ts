import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { NextRequest, NextResponse } from 'next/server';
import { QueueReorderError, reorderQueueJobs } from '@/server/queueReorder';

async function postCommand(request: NextRequest, { params }: { params: Promise<{ queueID: string }> }) {
  const { queueID } = await params;

  try {
    const body = assertGlobalPayload(await readJsonCommand(request));
    const workerID = typeof body.worker_id === 'string' && body.worker_id.trim() ? body.worker_id.trim() : 'local';
    const jobs = await reorderQueueJobs(queueID, body.job_ids, workerID, undefined, {
      expectedJobIDs: body.expected_job_ids,
    });
    return NextResponse.json({ jobs });
  } catch (error) {
    if (error instanceof QueueReorderError) {
      return NextResponse.json({ error: error.message, code: error.code }, { status: error.status });
    }
    console.error(error);
    return NextResponse.json({ error: 'Failed to reorder queue' }, { status: 500 });
  }
}

export const POST = withCommandBoundary(postCommand);
