import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';
import { assertProjectJobEnabled } from '@/server/projects';
import {
  getRemoteWorker,
  isLocalWorker,
  isRemoteJobMissingError,
  markRemoteJobMissing,
  remoteJson,
  syncRemoteJob,
} from '@/server/remoteClient';

const isWindows = process.platform === 'win32';

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
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

  if (job.status !== 'running') {
    return NextResponse.json({ error: 'Job is not running' }, { status: 409 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) {
      return NextResponse.json({ error: 'Remote job has not been uploaded yet' }, { status: 409 });
    }
    try {
      const worker = await getRemoteWorker(job.worker_id);
      await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/stop`);
      const synced = await syncRemoteJob(job);
      return NextResponse.json(synced);
    } catch (error) {
      if (isRemoteJobMissingError(error)) {
        return NextResponse.json(await markRemoteJobMissing(job));
      }
      const message = error instanceof Error ? error.message : 'Failed to stop remote job';
      await db.jobs.update(jobID, { remote_error: message, remote_sync_at: new Date() }).catch(() => undefined);
      return NextResponse.json({ error: message }, { status: 502 });
    }
  }

  await db.jobs.update(jobID, {
    stop: true,
    info: 'Stopping job...',
  });

  // Send SIGINT to the process if we have a PID
  if (job.pid != null) {
    console.log(`Attempting to stop job ${jobID} with PID ${job.pid}`);
    try {
      if (isWindows) {
        // Windows doesn't support SIGINT for arbitrary processes.
        // Use taskkill with /T (tree) to send a CTRL+C-like termination.
        const { execFileSync } = require('child_process');
        execFileSync('taskkill', ['/PID', String(job.pid), '/T', '/F'], { stdio: 'ignore' });
      } else {
        process.kill(job.pid, 'SIGINT');
      }
      // if it killed it, mark it stopped in the database
      await db.jobs.update(jobID, {
        status: 'stopped',
        info: 'Job stopped',
        pid: null,
      });
    } catch (e) {
      // Process may have already exited — that's fine
      console.error('Error sending signal to process:', e);
    }
  } else {
    console.warn(`No PID found for job ${jobID}, cannot send stop signal`);
  }

  return NextResponse.json(job);
}
