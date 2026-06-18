import fsp from 'fs/promises';
import { db } from './db';
import { createRemoteTrainingJobBundle } from './trainingJobBundle';
import {
  getRemoteWorker,
  isLocalWorker,
  remoteJson,
  syncRemoteJob,
  uploadBundleToWorker,
  withoutRemoteRedirects,
} from './remoteClient';
import {
  dispatchRemoteCaptionJob,
  isRemoteCaptionDispatchError,
} from './remoteCaptionDispatch';
import {
  getEncryptedDatasetsForJobConfig,
  getKeyForRequiredDataset,
  normalizeEncryptedKeyMap,
  validateEncryptedDatasetStartKey,
} from './encryptedDatasets';
import {
  getEncryptedKeyCoverage,
  isDurableEncryptedDatasetKeySecretError,
  storeDurableEncryptedDatasetKeys,
} from './encryptedDatasetSecrets';
import { isAnyRemoteOllamaCaptionJob } from './secureRemoteCaptionJobs';
import {
  syncRemoteDatasetsForJobConfig,
  type RemoteDatasetSyncMapping,
} from './remoteDatasetSync';
import { areProjectsEnabled, PROJECT_SPACES_DISABLED_MESSAGE } from './settings';
import { startJobNow } from '../../cron/actions/startJob';
import type { EncryptedDatasetStartKey, Job, RemoteStartProgress } from '../types';

type RequiredEncryptedDataset = { path: string; name: string };
type RemoteStartProgressCallback = (
  progress: Partial<
    Pick<
      RemoteStartProgress,
      | 'status'
      | 'message'
      | 'percent'
      | 'datasetName'
      | 'bytesProcessed'
      | 'bytesTotal'
      | 'warnings'
      | 'error'
      | 'remoteJobID'
    >
  >,
) => void;

export type PreparedJobStart = {
  jobID: string;
  job: Job;
  jobConfig: any;
  requiredEncryptedDatasets: RequiredEncryptedDataset[];
  encryptedKeysForLaunch: EncryptedDatasetStartKey[];
  useDurableEncryptedKeys: boolean;
};

export type JobStartErrorPayload = {
  error: string;
  code?: string;
  encryptedDatasets?: RequiredEncryptedDataset[];
  invalidEncryptedDatasets?: RequiredEncryptedDataset[];
};

export class JobStartError extends Error {
  status: number;
  payload: JobStartErrorPayload;

  constructor(payload: JobStartErrorPayload, status: number) {
    super(payload.error);
    this.name = 'JobStartError';
    this.status = status;
    this.payload = payload;
  }
}

function failStart(payload: JobStartErrorPayload, status: number): never {
  throw new JobStartError(payload, status);
}

export function isValidJobId(jobID: string) {
  return /^[a-zA-Z0-9_-]+$/.test(jobID);
}

function isSecureRemoteOllamaCaptionJobConfigJson(jobConfigJson: unknown) {
  if (typeof jobConfigJson !== 'string' || !jobConfigJson.trim()) return false;
  try {
    return isAnyRemoteOllamaCaptionJob(JSON.parse(jobConfigJson));
  } catch {
    return false;
  }
}

async function findInvalidEncryptedDatasetKeys(
  requiredDatasets: RequiredEncryptedDataset[],
  encryptedKeys: EncryptedDatasetStartKey[],
) {
  const keyMap = normalizeEncryptedKeyMap(encryptedKeys);
  const invalidDatasets: RequiredEncryptedDataset[] = [];

  for (const dataset of requiredDatasets) {
    const keyB64 = getKeyForRequiredDataset(keyMap, dataset);
    if (!keyB64) continue;
    try {
      await validateEncryptedDatasetStartKey(dataset, keyB64);
    } catch {
      invalidDatasets.push(dataset);
    }
  }

  return invalidDatasets;
}

export async function prepareJobStart(
  jobID: string,
  encryptedDatasetKeys?: EncryptedDatasetStartKey[],
  durableEncryptedDatasetKeys = false,
): Promise<PreparedJobStart> {
  if (!isValidJobId(jobID)) {
    failStart({ error: 'Invalid job ID' }, 400);
  }

  const job = await db.jobs.findById(jobID);
  if (!job) {
    failStart({ error: 'Job not found' }, 404);
  }
  if (job.project_id && !(await areProjectsEnabled())) {
    failStart({ error: PROJECT_SPACES_DISABLED_MESSAGE }, 403);
  }

  let jobConfig: any;
  try {
    jobConfig = JSON.parse(job.job_config);
  } catch {
    failStart({ error: 'Invalid job config' }, 400);
  }

  const requiredEncryptedDatasets = await getEncryptedDatasetsForJobConfig(jobConfig);
  let encryptedKeyCoverage = await getEncryptedKeyCoverage(jobID, requiredEncryptedDatasets, encryptedDatasetKeys);
  if (encryptedKeyCoverage.missingDatasets.length > 0) {
    failStart(
      {
        error: 'decryption key required',
        encryptedDatasets: encryptedKeyCoverage.missingDatasets,
      },
      409,
    );
  }

  let encryptedKeysForLaunch = encryptedKeyCoverage.combinedKeys;
  let useDurableEncryptedKeys = requiredEncryptedDatasets.length > 0 && encryptedKeyCoverage.durableKeys.length > 0;
  let invalidEncryptedDatasets = await findInvalidEncryptedDatasetKeys(requiredEncryptedDatasets, encryptedKeysForLaunch);
  if (invalidEncryptedDatasets.length > 0) {
    failStart(
      {
        error: 'invalid decryption key',
        encryptedDatasets: invalidEncryptedDatasets,
        invalidEncryptedDatasets,
      },
      409,
    );
  }

  if (durableEncryptedDatasetKeys && requiredEncryptedDatasets.length > 0) {
    try {
      await storeDurableEncryptedDatasetKeys(jobID, encryptedKeysForLaunch);
    } catch (error) {
      if (isDurableEncryptedDatasetKeySecretError(error)) {
        failStart({ error: error.message, code: 'durable_encrypted_key_secret_unavailable' }, 400);
      }
      throw error;
    }

    encryptedKeyCoverage = await getEncryptedKeyCoverage(jobID, requiredEncryptedDatasets, encryptedKeysForLaunch);
    encryptedKeysForLaunch = encryptedKeyCoverage.combinedKeys;
    useDurableEncryptedKeys = true;
    invalidEncryptedDatasets = await findInvalidEncryptedDatasetKeys(requiredEncryptedDatasets, encryptedKeysForLaunch);
    if (invalidEncryptedDatasets.length > 0) {
      failStart(
        {
          error: 'invalid decryption key',
          encryptedDatasets: invalidEncryptedDatasets,
          invalidEncryptedDatasets,
        },
        409,
      );
    }
  }

  return {
    jobID,
    job,
    jobConfig,
    requiredEncryptedDatasets,
    encryptedKeysForLaunch,
    useDurableEncryptedKeys,
  };
}

async function queueLocalJob(
  prepared: PreparedJobStart,
  options: { startQueue?: boolean; info?: string } = {},
) {
  const { job, jobID } = prepared;
  const newQueuePosition = (await db.jobs.maxQueuePosition()) + 1000;

  await db.jobs.update(jobID, { queue_position: newQueuePosition });

  const queue = await db.queues.findByGpuIds(job.gpu_ids);
  if (!queue) {
    await db.queues.create({
      gpu_ids: job.gpu_ids,
      is_running: options.startQueue === true,
    });
  } else if (options.startQueue === true && !queue.is_running) {
    await db.queues.update(queue.id, { is_running: true });
  }

  await db.jobs.update(jobID, {
    status: 'queued',
    stop: false,
    return_to_queue: false,
    info: options.info || 'Job queued',
  });

  return (await db.jobs.findById(jobID)) || job;
}

function scaleUploadPercent(loaded: number, total: number, start: number, end: number) {
  if (total <= 0) return start;
  return start + (end - start) * Math.min(1, loaded / total);
}

function addRemoteEncryptedDatasetKeyAliases(
  requiredEncryptedDatasets: RequiredEncryptedDataset[],
  encryptedKeysForLaunch: EncryptedDatasetStartKey[],
  datasetMappings: RemoteDatasetSyncMapping[],
) {
  if (requiredEncryptedDatasets.length === 0 || datasetMappings.length === 0) return encryptedKeysForLaunch;

  const nextKeys = [...encryptedKeysForLaunch];
  for (const mapping of datasetMappings) {
    const requiredDataset = requiredEncryptedDatasets.find(
      dataset => pathMatches(dataset.path, mapping.localDatasetPath) || dataset.name === mapping.datasetName,
    );
    if (!requiredDataset) continue;
    const keyB64 = getKeyForRequiredDataset(normalizeEncryptedKeyMap(encryptedKeysForLaunch), requiredDataset);
    if (!keyB64) continue;
    nextKeys.push({ datasetPath: mapping.remoteDatasetPath, keyB64 });
    if (mapping.remoteDatasetName !== mapping.datasetName) {
      nextKeys.push({ datasetPath: mapping.remoteDatasetName, keyB64 });
    }
  }
  return nextKeys;
}

function pathMatches(left: string, right: string) {
  return left.replace(/[\\/]+$/, '').toLowerCase() === right.replace(/[\\/]+$/, '').toLowerCase();
}

export async function assertPreparedJobCanStart(prepared: PreparedJobStart) {
  const { job, requiredEncryptedDatasets, useDurableEncryptedKeys } = prepared;

  if (!isLocalWorker(job.worker_id)) {
    if (requiredEncryptedDatasets.length === 0) return;
    const worker = await getRemoteWorker(job.worker_id);
    if (
      !worker.base_url.toLowerCase().startsWith('https://') &&
      process.env.AITK_ALLOW_INSECURE_REMOTE_ENCRYPTED_DATASETS !== '1'
    ) {
      failStart({ error: 'Remote encrypted training requires an HTTPS worker URL.' }, 400);
    }
    return;
  }

  if (requiredEncryptedDatasets.length === 0 || useDurableEncryptedKeys) return;

  const runningJobs = await db.jobs.list({
    status: ['running', 'stopping'],
    gpu_ids: job.gpu_ids,
    worker_id: 'local',
  });
  const runningJob = runningJobs.find(
    candidate => candidate.id !== job.id && !isSecureRemoteOllamaCaptionJobConfigJson(candidate.job_config),
  );
  if (runningJob) {
    failStart({ error: 'Encrypted jobs must start immediately; the selected local GPU is busy.' }, 409);
  }
}

export async function startPreparedJob(
  prepared: PreparedJobStart,
  options: { startQueue?: boolean; queueInfo?: string; onRemoteStartProgress?: RemoteStartProgressCallback } = {},
): Promise<Job> {
  const {
    job,
    jobID,
    jobConfig,
    requiredEncryptedDatasets,
    encryptedKeysForLaunch,
    useDurableEncryptedKeys,
  } = prepared;

  if (!isLocalWorker(job.worker_id)) {
    try {
      const worker = await getRemoteWorker(job.worker_id);
      const onProgress = options.onRemoteStartProgress;
      onProgress?.({
        status: 'preparing',
        message: `Preparing ${worker.name}`,
        percent: 2,
        datasetName: null,
        bytesProcessed: 0,
        bytesTotal: 0,
      });
      if (
        requiredEncryptedDatasets.length > 0 &&
        !worker.base_url.toLowerCase().startsWith('https://') &&
        process.env.AITK_ALLOW_INSECURE_REMOTE_ENCRYPTED_DATASETS !== '1'
      ) {
        failStart({ error: 'Remote encrypted training requires an HTTPS worker URL.' }, 400);
      }

      if (job.job_type === 'caption') {
        return dispatchRemoteCaptionJob({
          job,
          jobConfig,
          worker,
          encrypted: requiredEncryptedDatasets.length > 0,
          durableEncryptedDatasetKeys: useDurableEncryptedKeys,
          encryptedKeysForLaunch,
        });
      }

      let remoteJobId = job.remote_job_id;
      let remoteJobConfig = jobConfig;
      let remoteEncryptedKeysForLaunch = encryptedKeysForLaunch;
      let remoteWarnings: string[] = [];

      if (!remoteJobId) {
        const datasetSync = await syncRemoteDatasetsForJobConfig(jobConfig, worker, {
          onProgress: progress =>
            onProgress?.({
              status: progress.status,
              message: progress.message,
              percent: progress.percent,
              datasetName: progress.datasetName,
              bytesProcessed: progress.bytesProcessed ?? 0,
              bytesTotal: progress.bytesTotal ?? 0,
            }),
        });
        remoteJobConfig = datasetSync.jobConfig;
        remoteWarnings = datasetSync.warnings;
        remoteEncryptedKeysForLaunch = addRemoteEncryptedDatasetKeyAliases(
          requiredEncryptedDatasets,
          encryptedKeysForLaunch,
          datasetSync.mappings,
        );

        onProgress?.({
          status: 'zipping-job',
          message: 'Preparing remote job bundle',
          percent: 70,
          datasetName: null,
          bytesProcessed: 0,
          bytesTotal: 0,
          warnings: remoteWarnings,
        });
        const bundle = await createRemoteTrainingJobBundle(jobID, {
          includeDatasets: false,
          checkpointMode: 'all',
          targetWorker: worker,
          targetJobConfig: remoteJobConfig,
        });
        try {
          const bundleStat = await fsp.stat(bundle.zipPath);
          onProgress?.({
            status: 'uploading-job',
            message: 'Uploading remote job bundle',
            percent: 75,
            datasetName: null,
            bytesProcessed: 0,
            bytesTotal: bundleStat.size,
            warnings: remoteWarnings,
          });
          const imported = await uploadBundleToWorker(worker, bundle.zipPath, job.gpu_ids, progress => {
            const uploadComplete = progress.total > 0 && progress.loaded >= progress.total;
            onProgress?.({
              status: uploadComplete ? 'importing-job' : 'uploading-job',
              message: uploadComplete ? 'Importing remote job on worker' : 'Uploading remote job bundle',
              percent: scaleUploadPercent(progress.loaded, progress.total, 75, 88),
              datasetName: null,
              bytesProcessed: uploadComplete ? 0 : progress.loaded,
              bytesTotal: uploadComplete ? 0 : progress.total,
              warnings: remoteWarnings,
            });
          });
          remoteJobId = imported.job.id;
          const localJobConfig = {
            ...jobConfig,
            config: {
              ...(jobConfig?.config || {}),
              name: imported.job.name,
            },
          };
          await db.jobs.update(jobID, {
            name: imported.job.name,
            gpu_ids: imported.job.gpu_ids,
            job_config: JSON.stringify(localJobConfig),
            remote_job_id: imported.job.id,
            remote_error: [...remoteWarnings, ...bundle.warnings, ...(imported.warnings || [])].join('\n') || null,
            remote_sync_at: new Date(),
          });
          remoteWarnings = [...remoteWarnings, ...bundle.warnings, ...(imported.warnings || [])];
          onProgress?.({
            status: 'uploading-job',
            message: 'Remote job uploaded',
            percent: 89,
            bytesProcessed: bundleStat.size,
            bytesTotal: bundleStat.size,
            warnings: remoteWarnings,
            remoteJobID: imported.job.id,
          });
        } finally {
          await fsp.rm(bundle.zipPath, { force: true }).catch(() => undefined);
        }
      }

      onProgress?.({
        status: 'starting',
        message: 'Starting remote job',
        percent: 92,
        datasetName: null,
        bytesProcessed: 0,
        bytesTotal: 0,
        warnings: remoteWarnings,
        remoteJobID: remoteJobId,
      });
      const remoteStartHasKeys = requiredEncryptedDatasets.length > 0;
      const remoteStartInit: RequestInit = {
        method: 'POST',
        body: JSON.stringify({
          encryptedDatasetKeys: remoteStartHasKeys ? remoteEncryptedKeysForLaunch : undefined,
          durableEncryptedDatasetKeys: useDurableEncryptedKeys,
        }),
      };
      await remoteJson(
        worker,
        `/api/jobs/${encodeURIComponent(remoteJobId)}/start`,
        remoteStartHasKeys ? withoutRemoteRedirects(remoteStartInit) : remoteStartInit,
      );
      onProgress?.({
        status: 'starting',
        message: 'Starting remote queue',
        percent: 96,
        warnings: remoteWarnings,
        remoteJobID: remoteJobId,
      });
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
      onProgress?.({
        status: 'completed',
        message: 'Remote job started',
        percent: 100,
        warnings: remoteWarnings,
        remoteJobID: remoteJobId,
      });
      return synced;
    } catch (error) {
      if (error instanceof JobStartError) throw error;
      const message = error instanceof Error ? error.message : 'Failed to start remote job';
      options.onRemoteStartProgress?.({
        status: 'failed',
        message: 'Remote start failed',
        percent: 100,
        error: message,
      });
      await db.jobs.update(jobID, { remote_error: message, remote_sync_at: new Date() }).catch(() => undefined);
      failStart({ error: message }, isRemoteCaptionDispatchError(error) ? error.status : 502);
    }
  }

  if (isAnyRemoteOllamaCaptionJob(jobConfig)) {
    await startJobNow(jobID, {
      encryptedDatasetKeys: requiredEncryptedDatasets.length > 0 ? encryptedKeysForLaunch : undefined,
    });
    return (await db.jobs.findById(jobID)) || job;
  }

  if (requiredEncryptedDatasets.length > 0 && useDurableEncryptedKeys) {
    return queueLocalJob(prepared, { startQueue: options.startQueue, info: options.queueInfo });
  }

  if (requiredEncryptedDatasets.length > 0) {
    await assertPreparedJobCanStart(prepared);
    await startJobNow(jobID, { encryptedDatasetKeys: encryptedKeysForLaunch });
    return (await db.jobs.findById(jobID)) || job;
  }

  return queueLocalJob(prepared, { startQueue: options.startQueue, info: options.queueInfo });
}

export async function startJobFromRequest(
  jobID: string,
  encryptedDatasetKeys?: EncryptedDatasetStartKey[],
  durableEncryptedDatasetKeys = false,
) {
  return startPreparedJob(await prepareJobStart(jobID, encryptedDatasetKeys, durableEncryptedDatasetKeys));
}
