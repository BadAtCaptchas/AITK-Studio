import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';
import { getRemoteWorker, isLocalWorker, remoteJson, syncRemoteJob } from '@/server/remoteClient';

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (job.remote_job_id) {
      try {
        const worker = await getRemoteWorker(job.worker_id);
        await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/mark_stopped`);
        const synced = await syncRemoteJob(job);
        return NextResponse.json(synced);
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to mark remote job as stopped';
        await db.jobs.update(jobID, { remote_error: message, remote_sync_at: new Date() }).catch(() => undefined);
        return NextResponse.json({ error: message }, { status: 502 });
      }
    }

    const updated = await db.jobs.update(jobID, {
      stop: true,
      status: 'stopped',
      info: 'Remote job stopped',
      pid: null,
    });
    return NextResponse.json(updated);
  }

  // update job status to 'running'
  await db.jobs.update(jobID, {
    stop: true,
    status: 'stopped',
    info: 'Job stopped',
    pid: null,
  });

  console.log(`Job ${jobID} marked as stopped`);

  return NextResponse.json(job);
}
