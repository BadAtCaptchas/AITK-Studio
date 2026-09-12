import { NextResponse } from 'next/server';
import { db } from '@/server/db';
import { claimJobMaintenance } from '@/server/jobAttempts';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
export async function POST(_request: Request, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const job = await db.jobs.findById(jobID);
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  const owned = await claimJobMaintenance(job, 'editing');
  if (!owned) return NextResponse.json({ error: 'Stop the active process before marking this job stopped.' }, { status: 409 });
  try {
    if (!isLocalWorker(job.worker_id) && job.remote_job_id) {
      await remoteJson(await getRemoteWorker(job.worker_id), '/api/jobs/' + encodeURIComponent(job.remote_job_id) + '/mark_stopped', { method: 'POST' });
    }
    const stopped = await db.jobs.updateIf(jobID, { attempt_id: owned.attempt_id ?? null, status: 'editing' },
      { status: 'stopped', stop: false, return_to_queue: false, pid: null, process_started_at: null, info: 'Removed from queue' });
    return NextResponse.json(stopped);
  } catch (error) {
    await db.jobs.updateIf(jobID, { attempt_id: owned.attempt_id ?? null, status: 'editing' }, { status: job.status, info: 'Could not remove job from queue' });
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Unable to update job' }, { status: 502 });
  }
}
export function GET() { return NextResponse.json({ error: 'Use POST for this command' }, { status: 405, headers: { Allow: 'POST' } }); }
