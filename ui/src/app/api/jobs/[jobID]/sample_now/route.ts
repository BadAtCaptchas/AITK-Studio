import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';
import {
  getRemoteWorker,
  isLocalWorker,
  remoteJson,
  syncRemoteJob,
} from '@/server/remoteClient';
import { assertProjectJobEnabled } from '@/server/projects';
import { isRequestAuthenticated } from '@/utils/authSession';

function errorDetails(error: unknown) {
  const value = error as { message?: unknown; status?: unknown };
  return {
    message: typeof value?.message === 'string' ? value.message : 'Project spaces are disabled',
    status: typeof value?.status === 'number' ? value.status : 403,
  };
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ jobID: string }> },
) {
  if (!(await isRequestAuthenticated(request, process.env.AI_TOOLKIT_AUTH))) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { jobID } = await params;
  const job = await db.jobs.findById(jobID);
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }
  try {
    await assertProjectJobEnabled(job, 'write');
  } catch (error) {
    const details = errorDetails(error);
    return NextResponse.json({ error: details.message }, { status: details.status });
  }
  if (job.job_type !== 'train') {
    return NextResponse.json(
      { error: 'Only training jobs can sample on demand' },
      { status: 400 },
    );
  }
  if (job.status !== 'running') {
    return NextResponse.json({ error: 'Job is not running' }, { status: 409 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) {
      return NextResponse.json(
        { error: 'Remote job has not been uploaded yet' },
        { status: 409 },
      );
    }
    try {
      const worker = await getRemoteWorker(job.worker_id);
      await remoteJson(
        worker,
        `/api/jobs/${encodeURIComponent(job.remote_job_id)}/sample_now`,
        { method: 'POST' },
      );
      return NextResponse.json(await syncRemoteJob(job));
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Failed to request remote sample';
      await db.jobs
        .update(jobID, { remote_error: message, remote_sync_at: new Date() })
        .catch(() => undefined);
      return NextResponse.json({ error: message }, { status: 502 });
    }
  }

  const updated = await db.jobs.update(jobID, { sample_now: true });
  console.log(`Job ${jobID} marked to sample on the next step`);
  return NextResponse.json(updated);
}
