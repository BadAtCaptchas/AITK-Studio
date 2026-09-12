import { beginOperationCommit, checkpointOperation, getOperation } from './operations';
import fsp from 'fs/promises';
import path from 'path';
import { db, type WorkerNodeRecord } from './db';
import {
  getKeyForRequiredDataset,
  normalizeEncryptedKeyMap,
  resolveConfigPath,
} from './encryptedDatasets';
import { getDatasetsRoot } from './settings';
import { resolveDatasetDirectoryInsideRoot } from './remoteCaptionSecurity';
import { createDatasetExportArchive, datasetExportFileName } from './datasetTransfer';
import {
  remoteJson,
  syncRemoteJob,
  uploadDatasetArchiveToWorker,
  withoutRemoteRedirects,
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
  operationID?: string,
) {
  const hasEncryptedDatasetKeys = Array.isArray(encryptedDatasetKeys) && encryptedDatasetKeys.length > 0;
  const startInit: RequestInit = {
    method: 'POST',
    body: JSON.stringify({
      encryptedDatasetKeys,
      idempotencyKey: operationID,
      durableEncryptedDatasetKeys: hasEncryptedDatasetKeys,
    }),
  };
  await remoteJson(
    worker,
    `/api/jobs/${encodeURIComponent(remoteJobID)}/start`,
    hasEncryptedDatasetKeys ? withoutRemoteRedirects(startInit) : startInit,
  );
  await remoteJson(worker, `/api/queue/${encodeURIComponent(gpuIds)}/start`, { method: 'POST' });
  const queue = await db.queues.findByGpuIds(gpuIds, worker.id);
  if (queue) {
    await db.queues.update(queue.id, { is_running: true });
  } else {
    await db.queues.create({ worker_id: worker.id, gpu_ids: gpuIds, is_running: true });
  }
}

export async function dispatchRemoteCaptionJob(options: {
  job: Job;
  operationID?: string;
  jobConfig: any;
  worker: WorkerNodeRecord;
  encrypted: boolean;
  durableEncryptedDatasetKeys: boolean;
  encryptedKeysForLaunch: EncryptedDatasetStartKey[];
}) {
  const { job, worker, operationID } = options;
  const ownedUpdate = async (patch: Parameters<typeof db.jobs.update>[1]) => {
    const updated = await db.jobs.updateIf(job.id, { attempt_id: job.attempt_id ?? null }, patch);
    if (!updated) throw new Error('Remote caption attempt ownership changed');
    return updated;
  };
  const checkpoint = async (phase: string, data: Record<string, unknown> = {}) => { if (operationID) await checkpointOperation(operationID, phase, data); };
  const prior = operationID ? await getOperation(operationID) : null;
  const captionInfo = findCaptionProcess(options.jobConfig);
  if (!captionInfo) {
    throw new RemoteCaptionDispatchError('Caption process not found in job config', 400);
  }

  const datasetsRoot = await getDatasetsRoot();
  const originalDatasetPath = resolveConfigPath(captionInfo.pathToCaption);
  let realOriginalDatasetPath: string;
  try {
    realOriginalDatasetPath = await resolveDatasetDirectoryInsideRoot(originalDatasetPath, datasetsRoot);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : '';
    const isOutsideDatasetsRoot = errorMessage.includes('inside the configured datasets folder');
    throw new RemoteCaptionDispatchError(
      isOutsideDatasetsRoot ? errorMessage : 'Caption dataset not found',
      isOutsideDatasetsRoot ? 400 : 404,
    );
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
    if (prior?.checkpoint.captionStartRequested !== true && typeof state?.remoteDatasetPath === 'string' && state.remoteDatasetPath.trim()) {
      const remoteJobName = remoteCaptionRemoteJobName(job);
      const remoteJobConfig = buildRemoteOllamaCaptionJobConfig(currentJobConfig, {
        remoteDatasetPath: state.remoteDatasetPath,
        remoteJobName,
      });
      await remoteJson<Job>(worker, '/api/jobs', {
        method: 'POST',
        body: JSON.stringify({
          id: job.remote_job_id,
          name: remoteJobName,
          worker_id: 'local',
          gpu_ids: job.gpu_ids,
          job_config: remoteJobConfig,
          job_type: 'caption',
          job_ref: state.remoteDatasetPath,
        }),
      });
    }
    const encryptedKeys =
      options.encrypted && typeof state?.remoteDatasetPath === 'string'
        ? encryptedKeyForRemoteDataset(realOriginalDatasetPath, options.encryptedKeysForLaunch, state.remoteDatasetPath)
        : undefined;
    if (operationID) await beginOperationCommit(operationID, 'starting-remote-job');
    await checkpoint('start-requested', { captionStartRequested: true, remoteJobID: job.remote_job_id });
    await startRemoteWorkerCaptionJob(worker, job.remote_job_id, job.gpu_ids, encryptedKeys, operationID);
    const updated = await ownedUpdate({
      job_config: JSON.stringify(
        patchRemoteCaptionState(currentJobConfig, {
          downloadStatus: 'running',
          lastError: null,
        }),
      ),
      remote_error: null,
      remote_sync_at: new Date(),
    });
    return syncRemoteJob(updated, { background: false });
  }

  const initialState = buildInitialRemoteCaptionState({
    job,
    worker,
    originalDatasetPath: realOriginalDatasetPath,
    encrypted: options.encrypted,
    durableEncryptedKeys: options.durableEncryptedDatasetKeys,
    captionExtension: captionInfo.captionExtension,
    recaption: captionInfo.recaption,
  });
  currentJobConfig = setRemoteCaptionState(currentJobConfig, initialState);
  await ownedUpdate({
    job_config: JSON.stringify(currentJobConfig),
    remote_error: null,
    remote_sync_at: new Date(),
  });

  const exportRoot = path.join(datasetsRoot, '.aitk-remote-caption-bundles');
  const originalDatasetName = path.basename(realOriginalDatasetPath);
  const cachedZip = prior?.checkpoint.captionZip;
  const zipPath = typeof cachedZip === 'string' ? cachedZip : path.join(exportRoot, datasetExportFileName(originalDatasetName));

  try {
    if (!await fsp.stat(zipPath).then(stat => stat.isFile()).catch(() => false)) await createDatasetExportArchive(originalDatasetName, realOriginalDatasetPath, zipPath);
    await checkpoint('caption-bundle-ready', { captionZip: zipPath });
    const importedDataset = await uploadDatasetArchiveToWorker(
      worker,
      zipPath,
      remoteCaptionDatasetName(job, originalDatasetName),
      undefined, operationID ? `caption-${operationID}` : undefined,
    );
    const remoteJobName = remoteCaptionRemoteJobName(job);
    const remoteJobConfig = buildRemoteOllamaCaptionJobConfig(options.jobConfig, {
      remoteDatasetPath: importedDataset.path,
      remoteJobName,
    });
    // The imported dataset path is unique to this operation; recover a create whose response was lost.
    const recovered = await remoteJson<Job | null>(worker, '/api/jobs?job_type=caption&job_ref=' + encodeURIComponent(importedDataset.path));
    const remoteJob = recovered || await remoteJson<Job>(worker, '/api/jobs', {
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

    await checkpoint('remote-job-identified', { remoteJobID: remoteJob.id });
    const runningConfig = setRemoteCaptionState(options.jobConfig, {
      ...initialState,
      downloadStatus: 'running',
      remoteDatasetName: importedDataset.dataset.name,
      remoteDatasetPath: importedDataset.path,
      dispatchedAt: now,
      lastError: null,
    });
    currentJobConfig = runningConfig;
    const localJob = await ownedUpdate({
      remote_job_id: remoteJob.id,
      job_config: JSON.stringify(runningConfig),
      remote_error: null,
      remote_sync_at: new Date(),
    });

    const encryptedKeys =
      options.encrypted && importedDataset.path
        ? encryptedKeyForRemoteDataset(realOriginalDatasetPath, options.encryptedKeysForLaunch, importedDataset.path)
        : undefined;
    if (operationID) await beginOperationCommit(operationID, 'starting-remote-job');
    await checkpoint('start-requested', { captionStartRequested: true, remoteJobID: remoteJob.id });
    await startRemoteWorkerCaptionJob(worker, remoteJob.id, job.gpu_ids, encryptedKeys, operationID);
    return syncRemoteJob(localJob, { background: false });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Remote caption dispatch failed';
    const failedConfig = patchRemoteCaptionState(currentJobConfig, {
      downloadStatus: 'failed',
      lastError: message,
    });
    await ownedUpdate({
        job_config: JSON.stringify(failedConfig),
        remote_error: message,
        remote_sync_at: new Date(),
      })
      .catch(() => undefined);
    throw error;
  } finally {
    if (!operationID) await fsp.rm(zipPath, { force: true }).catch(() => undefined);
  }
}
