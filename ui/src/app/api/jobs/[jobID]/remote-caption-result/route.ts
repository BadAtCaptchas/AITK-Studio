import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { db } from '@/server/db';
import { isLocalWorker, syncRemoteJob } from '@/server/remoteClient';
import { getJobRemoteCaptionState } from '@/server/remoteCaptionJobs';
import { syncRemoteCaptionResultForJob } from '@/server/remoteCaptionResults';


async function postCommand(request: Request, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const job = await db.jobs.findById(jobID);
  if (!job) {
    return Response.json({ error: 'Job not found' }, { status: 404 });
  }
  if (isLocalWorker(job.worker_id) || !job.remote_job_id) {
    return Response.json({ error: 'Job is not a remote caption job' }, { status: 400 });
  }

  try {
    const body = await readJsonCommand(request);
    const synced = await syncRemoteJob(job, { background: false });
    const result = await syncRemoteCaptionResultForJob(synced, {
      force: body?.force === true,
      retryFailed: true,
      background: false,
    });
    const state = getJobRemoteCaptionState(result);
    if (state?.downloadStatus === 'failed') {
      return Response.json(
        { error: state.lastError || result.remote_error || 'Failed to sync remote caption result', job: result },
        { status: 500 },
      );
    }
    return Response.json(result);
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : 'Failed to sync remote caption result' },
      { status: 500 },
    );
  }
}

export const POST = withCommandBoundary(postCommand);
