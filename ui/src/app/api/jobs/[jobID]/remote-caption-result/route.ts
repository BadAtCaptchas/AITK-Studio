import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';
import { isLocalWorker, syncRemoteJob } from '@/server/remoteClient';
import { getJobRemoteCaptionState } from '@/server/remoteCaptionJobs';
import { syncRemoteCaptionResultForJob } from '@/server/remoteCaptionResults';
import { assertProjectJobEnabled } from '@/server/projects';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const job = await db.jobs.findById(jobID);
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }
  try {
    await assertProjectJobEnabled(job);
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Project spaces are disabled' }, { status: error?.status || 403 });
  }
  if (isLocalWorker(job.worker_id) || !job.remote_job_id) {
    return NextResponse.json({ error: 'Job is not a remote caption job' }, { status: 400 });
  }

  try {
    const body = await request.json().catch(() => ({}));
    const synced = await syncRemoteJob(job);
    const result = await syncRemoteCaptionResultForJob(synced, {
      force: body?.force === true,
      retryFailed: true,
    });
    const state = getJobRemoteCaptionState(result);
    if (state?.downloadStatus === 'failed') {
      return NextResponse.json(
        { error: state.lastError || result.remote_error || 'Failed to sync remote caption result', job: result },
        { status: 500 },
      );
    }
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to sync remote caption result' },
      { status: 500 },
    );
  }
}
