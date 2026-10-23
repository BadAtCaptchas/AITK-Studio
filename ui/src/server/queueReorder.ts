import { db } from './db';
import { getRemoteWorker, isLocalWorker, remoteJson, RemoteClientError } from './remoteClient';
import { withLocalQueueTransition } from './queueCoordination';
import { LeaseBusyError, withProcessLease } from './processLease';
import type { Job } from '../types';

const QUEUE_POSITION_STEP = 1000;

export class QueueReorderError extends Error {
  status: number;
  code: string;

  constructor(message: string, status = 400, code = 'INVALID_QUEUE_REORDER') {
    super(message);
    this.name = 'QueueReorderError';
    this.status = status;
    this.code = code;
  }
}

type QueueReorderDeps = {
  findJobById: (jobID: string) => Promise<Job | null>;
  updateJob: (jobID: string, data: { queue_position: number }) => Promise<Job>;
  listQueuedJobs: (queueID: string, workerID: string) => Promise<Job[]>;
  withQueueTransition: <T>(queueID: string, workerID: string, work: () => Promise<T>) => Promise<T>;
  getRemoteWorker: typeof getRemoteWorker;
  remoteJson: typeof remoteJson;
  isLocalWorker: typeof isLocalWorker;
};

const defaultDeps: QueueReorderDeps = {
  findJobById: jobID => db.jobs.findById(jobID),
  updateJob: (jobID, data) => db.jobs.update(jobID, data),
  listQueuedJobs: (queueID, workerID) =>
    db.jobs.list({ gpu_ids: queueID, worker_id: workerID, status: 'queued', order: 'queue_asc' }),
  withQueueTransition: (queueID, workerID, work) =>
    isLocalWorker(workerID)
      ? withLocalQueueTransition(queueID, work)
      : withProcessLease(`queue-reorder:${JSON.stringify([workerID, queueID])}`, work, { waitMs: 5_000 }),
  getRemoteWorker,
  remoteJson,
  isLocalWorker,
};

function assertJobIds(jobIDs: unknown, field = 'job_ids'): asserts jobIDs is string[] {
  if (!Array.isArray(jobIDs) || jobIDs.length === 0) {
    throw new QueueReorderError(`${field} must include at least one queued job.`);
  }

  const seen = new Set<string>();
  for (const jobID of jobIDs) {
    if (typeof jobID !== 'string' || !jobID.trim()) {
      throw new QueueReorderError(`${field} must contain valid job IDs.`);
    }
    if (seen.has(jobID)) {
      throw new QueueReorderError(`${field} must not contain duplicates.`);
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

  const typedJobs = jobs.filter((job): job is Job => job !== null);
  typedJobs.forEach(job => assertSameQueue(job, queueID, workerID));
  return typedJobs;
}

export async function reorderQueueJobs(
  queueID: string,
  jobIDs: unknown,
  workerID = 'local',
  deps: QueueReorderDeps = defaultDeps,
  options: { expectedJobIDs?: unknown } = {},
) {
  if (typeof queueID !== 'string' || !queueID.trim()) {
    throw new QueueReorderError('Invalid queue ID.');
  }
  if (typeof workerID !== 'string' || !workerID.trim()) {
    throw new QueueReorderError('Invalid worker ID.');
  }
  assertJobIds(jobIDs);
  let expectedJobIDs: string[] | undefined;
  if (options.expectedJobIDs !== undefined) {
    const expected = options.expectedJobIDs;
    assertJobIds(expected, 'expected_job_ids');
    expectedJobIDs = expected;
    const requested = new Set(jobIDs);
    if (expectedJobIDs.length !== jobIDs.length || expectedJobIDs.some(id => !requested.has(id))) {
      throw new QueueReorderError('expected_job_ids must contain the same jobs as job_ids.');
    }
  }

  try {
    return await deps.withQueueTransition(queueID, workerID, async () => {
      // Compare inside the same lease used by local queue insertion and cron selection.
      // Include every queued row: a paged/filtered snapshot must never replace the full lane.
      if (expectedJobIDs !== undefined) {
        const current = await deps.listQueuedJobs(queueID, workerID);
        if (!deps.isLocalWorker(workerID)) {
          // A worker breaks tied positions with its own IDs, which differ from mirror IDs.
          current.sort((a, b) => {
            const aID = a.remote_job_id || a.id,
              bID = b.remote_job_id || b.id;
            return a.queue_position - b.queue_position || (aID < bID ? -1 : aID > bID ? 1 : 0);
          });
        }
        if (
          current.length !== expectedJobIDs.length ||
          current.some((job, index) => job.id !== expectedJobIDs[index])
        ) {
          throw new QueueReorderError(
            'Queue changed while you were arranging jobs. Review the current order and try again.',
            409,
            'QUEUE_CHANGED',
          );
        }
      }
      const jobs = await loadQueuedJobs(queueID, workerID, jobIDs, deps);

      if (!deps.isLocalWorker(workerID)) {
        const remoteIDs = new Map(
          jobs.map(job => {
            if (!job.remote_job_id) {
              throw new QueueReorderError('Remote queue jobs must have a remote job ID.');
            }
            return [job.id, job.remote_job_id];
          }),
        );
        const remoteJobIDs = jobIDs.map(id => remoteIDs.get(id)!);
        assertJobIds(remoteJobIDs);
        const worker = await deps.getRemoteWorker(workerID);
        try {
          await deps.remoteJson(worker, `/api/queue/${encodeURIComponent(queueID)}/reorder`, {
            method: 'POST',
            body: JSON.stringify({
              worker_id: 'local',
              job_ids: remoteJobIDs,
              ...(expectedJobIDs === undefined
                ? {}
                : { expected_job_ids: expectedJobIDs.map(id => remoteIDs.get(id)!) }),
            }),
          });
        } catch (error) {
          if (error instanceof RemoteClientError && error.status === 409) {
            throw new QueueReorderError(
              'Remote queue changed or is busy. Refresh the queue and try again.',
              409,
              'QUEUE_CHANGED',
            );
          }
          throw error;
        }
      }

      // Both DB adapters support individual writes. Await every outstanding write even
      // after failure so another transition cannot start while this operation still writes.
      const results = await Promise.allSettled(
        jobIDs.map((jobID, index) => deps.updateJob(jobID, { queue_position: (index + 1) * QUEUE_POSITION_STEP })),
      );
      const failure = results.find(result => result.status === 'rejected');
      if (failure?.status === 'rejected') throw failure.reason;
      return results.flatMap(result => (result.status === 'fulfilled' ? [result.value] : []));
    });
  } catch (error) {
    if (error instanceof LeaseBusyError)
      throw new QueueReorderError('Queue is busy. Retry shortly.', 409, 'QUEUE_BUSY');
    throw error;
  }
}
