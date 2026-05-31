import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';
import { isLocalWorker, syncRemoteJob } from '@/server/remoteClient';
import { syncRemoteCaptionResultForJob } from '@/server/remoteCaptionResults';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(_request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const job = await db.jobs.findById(jobID);
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }
  if (isLocalWorker(job.worker_id) || !job.remote_job_id) {
    return NextResponse.json({ error: 'Job is not a remote caption job' }, { status: 400 });
  }

  try {
    const synced = await syncRemoteJob(job);
    const result = await syncRemoteCaptionResultForJob(synced, { force: true });
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to sync remote caption result' },
      { status: 500 },
    );
  }
}
