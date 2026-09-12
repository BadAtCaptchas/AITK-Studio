import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';
import { getRemoteWorker, isLocalWorker, remoteJson, syncRemoteJob } from '@/server/remoteClient';

export async function POST(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  if (job.job_type !== 'train') {
    return NextResponse.json({ error: 'Only training jobs can be saved on demand' }, { status: 400 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) {
      return NextResponse.json({ error: 'Remote job has not been uploaded yet' }, { status: 409 });
    }
    try {
      const worker = await getRemoteWorker(job.worker_id);
      await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/save_now`, { method: 'POST' });
      const synced = await syncRemoteJob(job);
      return NextResponse.json(synced);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to request remote save';
      await db.jobs.update(jobID, { remote_error: message, remote_sync_at: new Date() }).catch(() => undefined);
      return NextResponse.json({ error: message }, { status: 502 });
    }
  }

  const updated = await db.jobs.updateIf(jobID, { attempt_id: job.attempt_id ?? null, status: 'running' }, {
    save_now: true,
  });
  if (!updated) return NextResponse.json({ error: 'Job is no longer running' }, { status: 409 });

  console.log(`Job ${jobID} marked to save on next step`);

  return NextResponse.json(updated);
}

export function GET() {
  return NextResponse.json({ error: "Use POST for this command" }, { status: 405, headers: { Allow: "POST" } });
}
