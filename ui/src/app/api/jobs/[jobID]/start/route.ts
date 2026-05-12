import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';
import fsp from 'fs/promises';
import { createRemoteTrainingJobBundle } from '@/server/trainingJobBundle';
import {
  getRemoteWorker,
  isLocalWorker,
  remoteJson,
  syncRemoteJob,
  uploadBundleToWorker,
} from '@/server/remoteClient';

function ensureApiAccess(request: NextRequest): NextResponse | null {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) {
    return null;
  }

  const token = request.headers.get('authorization')?.split(' ')[1];
  if (token !== tokenToUse) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  return null;
}

function isValidJobId(jobID: string) {
  return /^[a-zA-Z0-9_-]+$/.test(jobID);
}

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) {
    return accessResponse;
  }

  const { jobID } = await params;

  if (!isValidJobId(jobID)) {
    return NextResponse.json({ error: 'Invalid job ID' }, { status: 400 });
  }

  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  if (!isLocalWorker(job.worker_id)) {
    try {
      const worker = await getRemoteWorker(job.worker_id);
      let remoteJobId = job.remote_job_id;

      if (!remoteJobId) {
        const bundle = await createRemoteTrainingJobBundle(jobID, { includeDatasets: true, checkpointMode: 'all' });
        try {
          const imported = await uploadBundleToWorker(worker, bundle.zipPath, job.gpu_ids);
          remoteJobId = imported.job.id;
          await db.jobs.update(jobID, {
            name: imported.job.name,
            gpu_ids: imported.job.gpu_ids,
            job_config: imported.job.job_config,
            remote_job_id: imported.job.id,
            remote_error: [...bundle.warnings, ...(imported.warnings || [])].join('\n') || null,
            remote_sync_at: new Date(),
          });
        } finally {
          await fsp.rm(bundle.zipPath, { force: true }).catch(() => undefined);
        }
      }

      await remoteJson(worker, `/api/jobs/${encodeURIComponent(remoteJobId)}/start`);
      await remoteJson(worker, `/api/queue/${encodeURIComponent(job.gpu_ids)}/start`);
      await db.queues
        .findByGpuIds(job.gpu_ids, job.worker_id)
        .then(queue =>
          queue
            ? db.queues.update(queue.id, { is_running: true })
            : db.queues.create({ worker_id: job.worker_id, gpu_ids: job.gpu_ids, is_running: true }),
        );
      const synced = await syncRemoteJob({
        ...(await db.jobs.findById(jobID))!,
        remote_job_id: remoteJobId,
      });
      return NextResponse.json(synced);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to start remote job';
      await db.jobs.update(jobID, { remote_error: message, remote_sync_at: new Date() }).catch(() => undefined);
      return NextResponse.json({ error: message }, { status: 502 });
    }
  }

  // get highest queue position
  const newQueuePosition = (await db.jobs.maxQueuePosition()) + 1000;

  await db.jobs.update(jobID, { queue_position: newQueuePosition });

  // make sure the queue is running
  const queue = await db.queues.findByGpuIds(job.gpu_ids);

  // if queue doesn't exist, create it
  if (!queue) {
    await db.queues.create({
      gpu_ids: job.gpu_ids,
      is_running: false,
    });
  }

  await db.jobs.update(jobID, {
    status: 'queued',
    stop: false,
    return_to_queue: false,
    info: 'Job queued',
  });

  // Return the response immediately
  return NextResponse.json(job);
}
