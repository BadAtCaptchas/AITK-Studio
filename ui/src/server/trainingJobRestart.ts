import { jobStorageKey } from '../utils/jobIdentity';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { db, type JobUpdateInput } from './db';
import { claimJobMaintenance } from './jobAttempts';
import { isPathWithinRoot } from './pathContainment';
import { getTrainingFolder } from './settings';
import { getJobTrainingRoot } from './trainingPaths';
import { getRemoteWorker, isLocalWorker, remoteJson, syncRemoteJob } from './remoteClient';
import {
  assertPreparedJobCanStart,
  JobStartError,
  prepareJobStart,
  isValidJobId,
  startPreparedJob,
  type PreparedJobStart,
} from './jobStart';
import type { EncryptedDatasetStartKey, Job } from '../types';

const RESTARTABLE_TRAINING_STATUSES = new Set(['queued', 'stopped', 'error', 'completed']);
const RESTART_QUEUE_INFO = 'Restarted from scratch and queued';

export type TrainingJobRestartOptions = {
  encryptedDatasetKeys?: EncryptedDatasetStartKey[];
  durableEncryptedDatasetKeys?: boolean;
};

export type TrainingJobRestartDeps = {
  findJobById: (jobID: string) => Promise<Job | null>;
  updateJob: (jobID: string, data: JobUpdateInput) => Promise<Job>;
  claimMaintenance: typeof claimJobMaintenance;
  updateOwnedJob: typeof db.jobs.updateIf;
  getTrainingFolder: () => Promise<string>;
  pathExists: (targetPath: string) => boolean;
  rmPath: typeof fsp.rm;
  deleteMetricsForJob: (jobID: string) => Promise<void>;
  prepareJobStart: typeof prepareJobStart;
  assertPreparedJobCanStart: typeof assertPreparedJobCanStart;
  startPreparedJob: typeof startPreparedJob;
  getRemoteWorker: typeof getRemoteWorker;
  remoteJson: typeof remoteJson;
  syncRemoteJob: typeof syncRemoteJob;
  ensureQueueRunning: (gpuIds: string, workerId: string) => Promise<void>;
};

export class TrainingJobRestartError extends Error {
  status: number;
  payload: { error: string };

  constructor(message: string, status: number) {
    super(message);
    this.name = 'TrainingJobRestartError';
    this.status = status;
    this.payload = { error: message };
  }
}

function failRestart(message: string, status: number): never {
  throw new TrainingJobRestartError(message, status);
}

export function resolveWithinRoot(root: string, target: unknown) {
  if (typeof target !== 'string' || target.trim().length === 0) {
    return null;
  }

  const resolvedRoot = path.resolve(root);
  const resolvedPath = path.resolve(resolvedRoot, target);
  const relativePath = path.relative(resolvedRoot, resolvedPath);

  if (relativePath === '' || relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
    return null;
  }

  return resolvedPath;
}

async function ensureQueueRunning(gpuIds: string, workerId: string) {
  const queue = await db.queues.findByGpuIds(gpuIds, workerId);
  if (queue) {
    await db.queues.update(queue.id, { is_running: true });
    return;
  }

  await db.queues.create({ worker_id: workerId, gpu_ids: gpuIds, is_running: true });
}

const defaultDeps: TrainingJobRestartDeps = {
  findJobById: jobID => db.jobs.findById(jobID),
  updateJob: (jobID, data) => db.jobs.update(jobID, data),
  claimMaintenance: claimJobMaintenance,
  updateOwnedJob: (jobID, expected, data) => db.jobs.updateIf(jobID, expected, data),
  getTrainingFolder,
  pathExists: targetPath => fs.existsSync(targetPath),
  rmPath: (targetPath, options) => fsp.rm(targetPath, options),
  deleteMetricsForJob: jobID => db.metrics.deleteForJob(jobID),
  prepareJobStart,
  assertPreparedJobCanStart,
  startPreparedJob,
  getRemoteWorker,
  remoteJson,
  syncRemoteJob,
  ensureQueueRunning,
};

function assertRestartableTrainingJob(job: Job) {
  if (job.job_type !== 'train') {
    failRestart('Only training jobs can be restarted from scratch', 400);
  }

  if (!RESTARTABLE_TRAINING_STATUSES.has(job.status)) {
    failRestart('Stop the job before restarting it from scratch', 409);
  }
}

async function clearLocalTrainingState(job: Job, deps: TrainingJobRestartDeps) {
  const trainingRoot = await deps.getTrainingFolder();
  const trainingFolder = resolveWithinRoot(trainingRoot, jobStorageKey(job));
  if (!trainingFolder) {
    failRestart('Invalid job path', 400);
  }

  if (deps.pathExists(trainingFolder)) {
    const [canonicalRoot, canonicalFolder] = await Promise.all([fsp.realpath(trainingRoot), fsp.realpath(trainingFolder)]);
    if (canonicalFolder === canonicalRoot || !isPathWithinRoot(canonicalRoot, canonicalFolder)) failRestart('Job folder escapes the training root', 400);
    await deps.rmPath(trainingFolder, { recursive: true, force: true });
  }

  await deps.deleteMetricsForJob(job.id);
}

function resetJobPatch(): JobUpdateInput {
  return {
    step: 0,
    speed_string: '',
    pid: null,
    stop: false,
    return_to_queue: false,
    save_now: false,
    sample_now: false,
    status: 'queued',
    info: 'Restarting job from scratch...',
  };
}

async function restartRemoteExistingJob(prepared: PreparedJobStart, deps: TrainingJobRestartDeps) {
  const { job, requiredEncryptedDatasets, encryptedKeysForLaunch, useDurableEncryptedKeys } = prepared;
  if (!job.remote_job_id) return null;

  await deps.assertPreparedJobCanStart(prepared);

  const owned = await deps.claimMaintenance(job, 'restarting');
  if (!owned) failRestart('Job changed before remote restart', 409);
  try {
    const worker = await deps.getRemoteWorker(job.worker_id);
    await deps.remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/restart-from-scratch`, {
      method: 'POST',
      body: JSON.stringify({
        encryptedDatasetKeys: requiredEncryptedDatasets.length > 0 ? encryptedKeysForLaunch : undefined,
        durableEncryptedDatasetKeys: useDurableEncryptedKeys,
      }),
    });
    await deps.ensureQueueRunning(job.gpu_ids, job.worker_id);
    const released = await deps.updateOwnedJob(job.id, { attempt_id: owned.attempt_id ?? null, status: 'restarting' }, { status: job.status });
    return deps.syncRemoteJob(released || owned);
  } catch (error) {
    if (error instanceof JobStartError) throw error;
    const message = error instanceof Error ? error.message : 'Failed to restart remote job from scratch';
    await deps.updateOwnedJob(job.id, { attempt_id: owned.attempt_id ?? null, status: 'restarting' }, { status: job.status, remote_error: message, remote_sync_at: new Date() }).catch(() => undefined);
    throw new JobStartError({ error: message }, 502);
  }
}

export async function restartTrainingJobFromScratch(
  jobID: string,
  options: TrainingJobRestartOptions = {},
  deps: TrainingJobRestartDeps = defaultDeps,
) {
  if (!isValidJobId(jobID)) {
    failRestart('Invalid job ID', 400);
  }

  const job = await deps.findJobById(jobID);
  if (!job) {
    failRestart('Job not found', 404);
  }
  assertRestartableTrainingJob(job);

  const prepared = await deps.prepareJobStart(
    jobID,
    options.encryptedDatasetKeys,
    options.durableEncryptedDatasetKeys === true,
    { allowQueued: true },
  );
  assertRestartableTrainingJob(prepared.job);

  if (!isLocalWorker(prepared.job.worker_id) && prepared.job.remote_job_id) {
    const remoteRestarted = await restartRemoteExistingJob(prepared, deps);
    if (remoteRestarted) return remoteRestarted;
  }

  await deps.assertPreparedJobCanStart(prepared);
  const owned = await deps.claimMaintenance(prepared.job, 'restarting');
  if (!owned) failRestart('Job changed or its process has not exited. Refresh and retry.', 409);
  let resetJob: Job;
  try {
    await clearLocalTrainingState(owned, deps);
    const reset = await deps.updateOwnedJob(jobID, { attempt_id: owned.attempt_id ?? null, status: 'restarting' }, resetJobPatch());
    if (!reset) failRestart('Restart ownership changed', 409);
    resetJob = reset;
  } catch (error) {
    await deps.updateOwnedJob(jobID, { attempt_id: owned.attempt_id ?? null, status: 'restarting' }, { status: 'error', info: 'Restart preparation failed. Review the error before retrying.' });
    throw error;
  }

  return deps.startPreparedJob(
    {
      ...prepared,
      job: resetJob,
    },
    {
      startQueue: true,
      queueInfo: RESTART_QUEUE_INFO,
    },
  );
}
