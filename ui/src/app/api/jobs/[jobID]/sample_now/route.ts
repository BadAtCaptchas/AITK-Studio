import { db } from '@/server/db';
import { getRemoteWorker, isLocalWorker, remoteJson, syncRemoteJob } from '@/server/remoteClient';

import { isRequestAuthenticated } from '@/utils/authSession';

function errorDetails(error: unknown) {
  const value = error as { message?: unknown; status?: unknown };
  return {
    message: typeof value?.message === 'string' ? value.message : 'Unable to request a sample',
    status: typeof value?.status === 'number' ? value.status : 403,
  };
}

export async function POST(request: Request, { params }: { params: Promise<{ jobID: string }> }) {
  if (!(await isRequestAuthenticated(request, process.env.AI_TOOLKIT_AUTH))) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { jobID } = await params;
  const job = await db.jobs.findById(jobID);
  if (!job) {
    return Response.json({ error: 'Job not found' }, { status: 404 });
  }
  try {
  } catch (error) {
    const details = errorDetails(error);
    return Response.json({ error: details.message }, { status: details.status });
  }
  if (job.job_type !== 'train') {
    return Response.json({ error: 'Only training jobs can sample on demand' }, { status: 400 });
  }
  if (job.status !== 'running') {
    return Response.json({ error: 'Job is not running' }, { status: 409 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) {
      return Response.json({ error: 'Remote job has not been uploaded yet' }, { status: 409 });
    }
    try {
      const worker = await getRemoteWorker(job.worker_id);
      await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/sample_now`, { method: 'POST' });
      return Response.json(await syncRemoteJob(job));
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to request remote sample';
      await db.jobs.updateIf(jobID, { attempt_id: job.attempt_id ?? null, status: 'running' }, { remote_error: message, remote_sync_at: new Date() }).catch(() => undefined);
      return Response.json({ error: message }, { status: 502 });
    }
  }

  const updated = await db.jobs.updateIf(jobID, { attempt_id: job.attempt_id ?? null, status: 'running' }, { sample_now: true });
  if (!updated) return Response.json({ error: 'Job attempt changed' }, { status: 409 });
  console.log(`Job ${jobID} marked to sample on the next step`);
  return Response.json(updated);
}
