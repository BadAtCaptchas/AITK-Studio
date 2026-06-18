import { db } from './db';
import { getRemoteWorker, isLocalWorker, remoteJson } from './remoteClient';
import { areProjectsEnabled, PROJECT_SPACES_DISABLED_MESSAGE } from './settings';
import type { Job } from '../types';

const QUEUE_POSITION_STEP = 1000;

export class QueueReorderError extends Error {
  status: number;

  constructor(message: string, status = 400) {
    super(message);
    this.name = 'QueueReorderError';
    this.status = status;
  }
}

type QueueReorderDeps = {
  findJobById: (jobID: string) => Promise<Job | null>;
  updateJob: (jobID: string, data: { queue_position: number }) => Promise<Job>;
  getRemoteWorker: typeof getRemoteWorker;
  remoteJson: typeof remoteJson;
  isLocalWorker: typeof isLocalWorker;
};

const defaultDeps: QueueReorderDeps = {
  findJobById: jobID => db.jobs.findById(jobID),
  updateJob: (jobID, data) => db.jobs.update(jobID, data),
  getRemoteWorker,
  remoteJson,
  isLocalWorker,
};

function assertJobIds(jobIDs: string[]) {
  if (!Array.isArray(jobIDs) || jobIDs.length === 0) {
    throw new QueueReorderError('job_ids must include at least one queued job.');
  }

  const seen = new Set<string>();
  for (const jobID of jobIDs) {
    if (typeof jobID !== 'string' || !jobID.trim()) {
      throw new QueueReorderError('job_ids must contain valid job IDs.');
    }
    if (seen.has(jobID)) {
      throw new QueueReorderError('job_ids must not contain duplicates.');
    }
    seen.add(jobID);
  }
}

function assertSameQueue(job: Job, queueID: string, workerID: string) {
  if (job.worker_id !== workerID || job.gpu_ids !== queueID) {
    throw new QueueReorderError('All jobs must belong to the requested worker/GPU queue.');
  }
  if (job.status !== 'queued') {
    throw new QueueReorderError('Only queued jobs can be reordered.');
  }
}

async function loadQueuedJobs(queueID: string, workerID: string, jobIDs: string[], deps: QueueReorderDeps) {
  const jobs = await Promise.all(jobIDs.map(jobID => deps.findJobById(jobID)));
  const missingIndex = jobs.findIndex(job => !job);
  if (missingIndex >= 0) {
    throw new QueueReorderError(`Job not found: ${jobIDs[missingIndex]}`, 400);
  }

  const typedJobs = jobs as Job[];
  typedJobs.forEach(job => assertSameQueue(job, queueID, workerID));
  return typedJobs;
}

export async function reorderQueueJobs(
  queueID: string,
  jobIDs: string[],
  workerID = 'local',
  deps: QueueReorderDeps = defaultDeps,
) {
  if (typeof queueID !== 'string' || !queueID.trim()) {
    throw new QueueReorderError('Invalid queue ID.');
  }
  if (typeof workerID !== 'string' || !workerID.trim()) {
    throw new QueueReorderError('Invalid worker ID.');
  }
  assertJobIds(jobIDs);

  const jobs = await loadQueuedJobs(queueID, workerID, jobIDs, deps);
  if (jobs.some(job => job.project_id) && !(await areProjectsEnabled())) {
    throw new QueueReorderError(PROJECT_SPACES_DISABLED_MESSAGE, 403);
  }

  if (!deps.isLocalWorker(workerID)) {
    const remoteJobIDs = jobs.map(job => {
      if (!job.remote_job_id) {
        throw new QueueReorderError('Remote queue jobs must have a remote job ID.');
      }
      return job.remote_job_id;
    });
    const worker = await deps.getRemoteWorker(workerID);
    await deps.remoteJson(worker, `/api/queue/${encodeURIComponent(queueID)}/reorder`, {
      method: 'POST',
      body: JSON.stringify({ worker_id: 'local', job_ids: remoteJobIDs }),
    });
  }

  const updatedJobs = await Promise.all(
    jobIDs.map((jobID, index) =>
      deps.updateJob(jobID, {
        queue_position: (index + 1) * QUEUE_POSITION_STEP,
      }),
    ),
  );

  return updatedJobs;
}
