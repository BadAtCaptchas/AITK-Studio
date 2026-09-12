import { NextRequest, NextResponse } from 'next/server';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { db, getDatabaseConfig } from '@/server/db';
import { getToolkitPythonPath } from '@/server/pythonPath';
import { TOOLKIT_ROOT } from '@/paths';

import {
  getRemoteWorker,
  isLocalWorker,
  isRemoteJobMissingError,
  markRemoteJobMissing,
  remoteJson,
  syncRemoteJob,
} from '@/server/remoteClient';

const execFileAsync = promisify(execFile);

export async function POST(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  if (!['starting', 'running', 'stopping'].includes(job.status)) {
    return NextResponse.json({ error: 'Job is not running' }, { status: 409 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) {
      return NextResponse.json({ error: 'Remote job has not been uploaded yet' }, { status: 409 });
    }
    try {
      const worker = await getRemoteWorker(job.worker_id);
      await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/stop`, { method: 'POST' });
      const synced = await syncRemoteJob(job);
      return NextResponse.json(synced);
    } catch (error) {
      if (isRemoteJobMissingError(error)) {
        return NextResponse.json(await markRemoteJobMissing(job));
      }
      const message = error instanceof Error ? error.message : 'Failed to stop remote job';
      await db.jobs.updateIf(jobID, { attempt_id: job.attempt_id ?? null }, { remote_error: message, remote_sync_at: new Date() }).catch(() => undefined);
      return NextResponse.json({ error: message }, { status: 502 });
    }
  }

  const stopping = await db.jobs.updateIf(jobID, { attempt_id: job.attempt_id ?? null, status: job.status }, {
    stop: true,
    status: 'stopping',
    info: 'Stopping job...',
  });
  if (!stopping) return NextResponse.json({ error: 'Job attempt changed; refresh and retry' }, { status: 409 });

  // Send SIGINT to the process if we have a PID
  if (stopping.pid != null && stopping.attempt_id && stopping.process_started_at != null) {
    console.log(`Attempting to stop job ${jobID} with PID ${job.pid}`);
    try {
      const config = getDatabaseConfig();
      await execFileAsync(getToolkitPythonPath(), ['-c',
        'import sys; from toolkit.attempt_process import stop_owned_process; stop_owned_process(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))',
        jobID, stopping.attempt_id, String(stopping.pid), String(stopping.process_started_at)], {
        windowsHide: true, cwd: TOOLKIT_ROOT, timeout: 10_000, maxBuffer: 4096,
        env: { ...process.env, AITK_ATTEMPT_ID: stopping.attempt_id, AITK_DB_PROVIDER: config.provider,
          AITK_SQLITE_PATH: config.sqlitePath, AITK_MONGODB_URI: config.mongoUri || '', AITK_MONGODB_DB: config.mongoDb },
      });
    } catch (e) {
      // Process may have already exited — that's fine
      console.error('Error sending signal to process:', e);
    }
  } else {
    console.warn(`No PID found for job ${jobID}, cannot send stop signal`);
  }

  return NextResponse.json(stopping);
}

export function GET() {
  return NextResponse.json({ error: "Use POST for this command" }, { status: 405, headers: { Allow: "POST" } });
}
