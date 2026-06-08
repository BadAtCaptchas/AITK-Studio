import { NextResponse } from 'next/server';
import { db, type WorkerNodeRecord } from '@/server/db';
import { fetchWorkerHealth } from '@/server/remoteClient';
import type { Job } from '@/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const ACTIVE_JOB_STATUSES = ['queued', 'running', 'stopping'];

function toPublicWorker(worker: WorkerNodeRecord) {
  const { api_token: _apiToken, ...publicWorker } = worker;
  return publicWorker;
}

type RemoteJobCleanupResult = {
  jobID: string;
  remoteJobID: string | null;
  status: 'not_uploaded' | 'worker_offline';
  error: string | null;
};

async function checkWorkerOffline(worker: WorkerNodeRecord | null) {
  if (!worker) {
    return { offline: true, error: 'Remote worker record was not found.' };
  }

  try {
    await fetchWorkerHealth(worker);
    return { offline: false, error: null };
  } catch (error) {
    return {
      offline: true,
      error: error instanceof Error ? error.message : 'Remote worker could not be reached.',
    };
  }
}

function activeJobCleanupForOfflineWorker(job: Job, offlineReason: string | null): RemoteJobCleanupResult {
  if (!job.remote_job_id) {
    return {
      jobID: job.id,
      remoteJobID: null,
      status: 'not_uploaded',
      error: null,
    };
  }

  return {
    jobID: job.id,
    remoteJobID: job.remote_job_id,
    status: 'worker_offline',
    error: offlineReason || 'Remote worker could not be reached.',
  };
}

async function markActiveJobForDeletedWorker(job: Job, cleanup: RemoteJobCleanupResult) {
  const workerOffline = cleanup.status === 'worker_offline';
  await db.jobs.update(job.id, {
    stop: true,
    return_to_queue: false,
    status: workerOffline ? 'error' : 'stopped',
    info: workerOffline
      ? 'Worker deleted while offline; remote job could not be canceled.'
      : cleanup.status === 'not_uploaded'
        ? 'Worker deleted before this job was uploaded.'
        : 'Worker deleted.',
    pid: null,
    remote_error: workerOffline ? cleanup.error || 'Remote worker could not be reached.' : null,
    remote_sync_at: new Date(),
  });
}

export async function DELETE(_request: Request, { params }: { params: Promise<{ workerID: string }> }) {
  const { workerID } = await params;
  if (workerID === 'local') {
    return NextResponse.json({ error: 'The local worker cannot be deleted' }, { status: 400 });
  }

  const activeJobs = await db.jobs.list({ worker_id: workerID, status: ACTIVE_JOB_STATUSES });
  const workerRecord = await db.workerNodes.findById(workerID);
  const offlineCheck = activeJobs.length > 0 ? await checkWorkerOffline(workerRecord) : null;
  if (offlineCheck && !offlineCheck.offline) {
    return NextResponse.json(
      {
        error: 'Cannot delete an online worker with queued, running, or stopping jobs. Stop those jobs first.',
        activeJobCount: activeJobs.length,
      },
      { status: 409 },
    );
  }

  const activeJobCleanup = await Promise.all(
    activeJobs.map(async job => {
      const cleanup = activeJobCleanupForOfflineWorker(job, offlineCheck?.error || null);
      await markActiveJobForDeletedWorker(job, cleanup);
      return cleanup;
    }),
  );

  const deletedQueues = await db.queues.deleteMany({ worker_id: workerID });
  const worker = await db.workerNodes.delete(workerID);
  return NextResponse.json({
    worker: worker ? toPublicWorker(worker) : null,
    deletedQueues,
    activeJobCleanup,
  });
}
