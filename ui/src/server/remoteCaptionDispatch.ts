import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { db, type WorkerNodeRecord } from './db';
import {
  getKeyForRequiredDataset,
  normalizeEncryptedKeyMap,
  resolveConfigPath,
} from './encryptedDatasets';
import { getDatasetsRoot } from './settings';
import { createDatasetExportArchive, datasetExportFileName } from './datasetTransfer';
import {
  remoteJson,
  syncRemoteJob,
  uploadDatasetArchiveToWorker,
} from './remoteClient';
import {
  buildInitialRemoteCaptionState,
  buildRemoteOllamaCaptionJobConfig,
  findCaptionProcess,
  patchRemoteCaptionState,
  remoteCaptionDatasetName,
  setRemoteCaptionState,
} from './remoteCaptionJobs';
import type { EncryptedDatasetStartKey, Job } from '../types';

export class RemoteCaptionDispatchError extends Error {
  status: number;

  constructor(message: string, status = 400) {
    super(message);
    this.name = 'RemoteCaptionDispatchError';
    this.status = status;
  }
}

export function isRemoteCaptionDispatchError(error: unknown): error is RemoteCaptionDispatchError {
  return error instanceof RemoteCaptionDispatchError;
}

function remoteCaptionRemoteJobName(job: Job) {
  return `${job.name}_remote`;
}

function encryptedKeyForRemoteDataset(
  originalDatasetPath: string,
  encryptedKeysForLaunch: EncryptedDatasetStartKey[],
  remoteDatasetPath: string,
) {
  const keyMap = normalizeEncryptedKeyMap(encryptedKeysForLaunch);
  const keyB64 = getKeyForRequiredDataset(keyMap, {
    path: originalDatasetPath,
    name: path.basename(originalDatasetPath),
  });
  return keyB64 ? [{ datasetPath: remoteDatasetPath, keyB64 }] : undefined;
}

async function startRemoteWorkerCaptionJob(
  worker: WorkerNodeRecord,
  remoteJobID: string,
  gpuIds: string,
  encryptedDatasetKeys?: EncryptedDatasetStartKey[],
) {
  await remoteJson(worker, `/api/jobs/${encodeURIComponent(remoteJobID)}/start`, {
    method: 'POST',
    body: JSON.stringify({
      encryptedDatasetKeys,
      durableEncryptedDatasetKeys: Array.isArray(encryptedDatasetKeys) && encryptedDatasetKeys.length > 0,
    }),
  });
  await remoteJson(worker, `/api/queue/${encodeURIComponent(gpuIds)}/start`);
  const queue = await db.queues.findByGpuIds(gpuIds, worker.id);
  if (queue) {
    await db.queues.update(queue.id, { is_running: true });
  } else {
    await db.queues.create({ worker_id: worker.id, gpu_ids: gpuIds, is_running: true });
  }
}

export async function dispatchRemoteCaptionJob(options: {
  job: Job;
  jobConfig: any;
  worker: WorkerNodeRecord;
  encrypted: boolean;
  durableEncryptedDatasetKeys: boolean;
  encryptedKeysForLaunch: EncryptedDatasetStartKey[];
}) {
  const { job, worker } = options;
  const captionInfo = findCaptionProcess(options.jobConfig);
  if (!captionInfo) {
    throw new RemoteCaptionDispatchError('Caption process not found in job config', 400);
  }

  const originalDatasetPath = resolveConfigPath(captionInfo.pathToCaption);
  if (!fs.existsSync(originalDatasetPath) || !fs.statSync(originalDatasetPath).isDirectory()) {
    throw new RemoteCaptionDispatchError('Caption dataset not found', 404);
  }
  if (options.encrypted && !options.durableEncryptedDatasetKeys) {
    throw new RemoteCaptionDispatchError(
      'Remote encrypted captioning requires durable encrypted dataset key opt-in.',
      409,
    );
  }

  let currentJobConfig = options.jobConfig;
  const now = new Date().toISOString();

  if (job.remote_job_id) {
    const state = currentJobConfig.config?.remote_caption;
    const encryptedKeys =
      options.encrypted && typeof state?.remoteDatasetPath === 'string'
        ? encryptedKeyForRemoteDataset(originalDatasetPath, options.encryptedKeysForLaunch, state.remoteDatasetPath)
        : undefined;
    await startRemoteWorkerCaptionJob(worker, job.remote_job_id, job.gpu_ids, encryptedKeys);
    const updated = await db.jobs.update(job.id, {
      job_config: JSON.stringify(
        patchRemoteCaptionState(currentJobConfig, {
          downloadStatus: 'running',
          lastError: null,
        }),
      ),
      remote_error: null,
      remote_sync_at: new Date(),
    });
    return syncRemoteJob(updated);
  }

  const initialState = buildInitialRemoteCaptionState({
    job,
    worker,
    originalDatasetPath,
    encrypted: options.encrypted,
    durableEncryptedKeys: options.durableEncryptedDatasetKeys,
    captionExtension: captionInfo.captionExtension,
    recaption: captionInfo.recaption,
  });
  currentJobConfig = setRemoteCaptionState(currentJobConfig, initialState);
  await db.jobs.update(job.id, {
    job_config: JSON.stringify(currentJobConfig),
    remote_error: null,
    remote_sync_at: new Date(),
  });

  const datasetsRoot = await getDatasetsRoot();
  const exportRoot = path.join(datasetsRoot, '.aitk-remote-caption-bundles');
  const originalDatasetName = path.basename(originalDatasetPath);
  const zipPath = path.join(exportRoot, datasetExportFileName(originalDatasetName));

  try {
    await createDatasetExportArchive(originalDatasetName, originalDatasetPath, zipPath);
    const importedDataset = await uploadDatasetArchiveToWorker(
      worker,
      zipPath,
      remoteCaptionDatasetName(job, originalDatasetName),
    );
    const remoteJobName = remoteCaptionRemoteJobName(job);
    const remoteJobConfig = buildRemoteOllamaCaptionJobConfig(options.jobConfig, {
      remoteDatasetPath: importedDataset.path,
      remoteJobName,
    });
    const remoteJob = await remoteJson<Job>(worker, '/api/jobs', {
      method: 'POST',
      body: JSON.stringify({
        name: remoteJobName,
        worker_id: 'local',
        gpu_ids: job.gpu_ids,
        job_config: remoteJobConfig,
        job_type: 'caption',
        job_ref: importedDataset.path,
      }),
    });

    const runningConfig = setRemoteCaptionState(options.jobConfig, {
      ...initialState,
      downloadStatus: 'running',
      remoteDatasetName: importedDataset.dataset.name,
      remoteDatasetPath: importedDataset.path,
      dispatchedAt: now,
      lastError: null,
    });
    currentJobConfig = runningConfig;
    const localJob = await db.jobs.update(job.id, {
      remote_job_id: remoteJob.id,
      job_config: JSON.stringify(runningConfig),
      remote_error: null,
      remote_sync_at: new Date(),
    });

    const encryptedKeys =
      options.encrypted && importedDataset.path
        ? encryptedKeyForRemoteDataset(originalDatasetPath, options.encryptedKeysForLaunch, importedDataset.path)
        : undefined;
    await startRemoteWorkerCaptionJob(worker, remoteJob.id, job.gpu_ids, encryptedKeys);
    return syncRemoteJob(localJob);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Remote caption dispatch failed';
    const failedConfig = patchRemoteCaptionState(currentJobConfig, {
      downloadStatus: 'failed',
      lastError: message,
    });
    await db.jobs
      .update(job.id, {
        job_config: JSON.stringify(failedConfig),
        remote_error: message,
        remote_sync_at: new Date(),
      })
      .catch(() => undefined);
    throw error;
  } finally {
    await fsp.rm(zipPath, { force: true }).catch(() => undefined);
  }
}
