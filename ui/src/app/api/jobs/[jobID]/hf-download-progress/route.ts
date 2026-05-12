import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';
import { getHFDownloadProgress } from '@/server/hfDownloadProgress';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';

export async function GET(_request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) return NextResponse.json({ progress: null });
    try {
      const worker = await getRemoteWorker(job.worker_id);
      return NextResponse.json(
        await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/hf-download-progress`),
      );
    } catch {
      return NextResponse.json({ progress: null });
    }
  }

  return NextResponse.json({ progress: await getHFDownloadProgress(job) });
}
