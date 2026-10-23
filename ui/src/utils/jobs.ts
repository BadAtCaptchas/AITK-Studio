import { JobConfig } from '@/types';
import type { Job } from '@/types';
import { apiClient } from '@/utils/api';
import { getDisplayPath, getDownloadUrl } from '@/utils/media';
import type { EncryptedDatasetStartKey } from '@/types';
import type { RemoteStartProgress } from '@/types';
import type { AxiosProgressEvent } from 'axios';
export type TrainingJobCheckpointExportMode = 'latest' | 'all';
export type TrainingJobExportProgress = {
  exportID: string;
  jobID: string;
  includeDatasets: boolean;
  checkpointMode: TrainingJobCheckpointExportMode;
  status: 'queued' | 'preparing' | 'zipping' | 'finalizing' | 'completed' | 'failed' | 'canceling' | 'canceled';
  message: string;
  percent: number;
  entriesProcessed: number;
  entriesTotal: number;
  bytesProcessed: number;
  bytesTotal: number;
  zipPath: string | null;
  fileName: string | null;
  warnings: string[];
  error: string | null;
  cancelRequested: boolean;
  createdAt: string;
  updatedAt: string;
};
export type TrainingJobExportResult = {
  zipPath: string;
  fileName: string;
  warnings: string[];
};
export type TrainingJobImportProgress = {
  loaded: number;
  total: number | null;
  percent: number | null;
};
export type TrainingJobImportResult = {
  job: Job;
  warnings: string[];
};
const MAX_TRAINING_JOB_IMPORT_BYTES = 64 * 1024 * 1024 * 1024;
const TRAINING_JOB_IMPORT_CHUNK_BYTES = 64 * 1024 * 1024;
export type JobModelPrefetchResult = {
  handledValues: string[];
  downloads: Array<{
    value: string;
    path: string;
    kind: string;
    cached?: boolean;
  }>;
  warnings: string[];
  updatedConfig: boolean;
  job: Job;
};
export type StartJobOptions = {
  durableEncryptedDatasetKeys?: boolean;
  background?: boolean;
  idempotencyKey?: string;
  onRemoteStartProgress?: (progress: RemoteStartProgress) => void;
};
export class StartJobError extends Error {
  readonly status?: number;
  readonly code?: string;

  constructor(
    message: string,
    readonly outcome: 'unknown' | 'rejected',
    detail: { status?: number; code?: string } = {},
  ) {
    super(message);
    this.name = 'StartJobError';
    this.status = detail.status;
    this.code = detail.code;
  }
}
export function isStartJobError(error: unknown): error is StartJobError {
  return error instanceof StartJobError;
}
function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
function startActionError(error: unknown, fallback: string, accepted: boolean): StartJobError {
  if (isStartJobError(error)) return error;
  const response = isRecord(error) && isRecord(error.response) ? error.response : undefined;
  const data = response && isRecord(response.data) ? response.data : undefined;
  const status = typeof response?.status === 'number' && Number.isInteger(response.status) ? response.status : undefined;
  const code = typeof data?.code === 'string' ? data.code : undefined;
  const message = typeof data?.error === 'string' ? data.error : error instanceof Error ? error.message : fallback;
  // A failed progress read cannot reverse an already accepted start. Busy commands
  // likewise may still finish, so callers must retry their original command key.
  const rejected = !accepted && status !== undefined && status >= 400 && status < 500 &&
    ![408, 425].includes(status) && code !== 'START_BUSY';
  return new StartJobError(message, rejected ? 'rejected' : 'unknown', { status, code });
}
const remoteStartStatuses: readonly RemoteStartProgress['status'][] = [
  'queued', 'preparing', 'checking-datasets', 'zipping-dataset', 'uploading-dataset', 'importing-dataset',
  'zipping-job', 'uploading-job', 'importing-job', 'starting', 'completed', 'failed',
];
function isRemoteStartStatus(value: unknown): value is RemoteStartProgress['status'] {
  return typeof value === 'string' && remoteStartStatuses.some(status => status === value);
}
function parseRemoteStartProgress(value: unknown, jobID: string, startID: string): RemoteStartProgress {
  if (!isRecord(value) || value.jobID !== jobID || value.startID !== startID || !isRemoteStartStatus(value.status)) {
    throw new StartJobError('Invalid remote start progress response. Retry to recover the start result.', 'unknown');
  }
  return {
    jobID, startID, status: value.status,
    message: typeof value.message === 'string' ? value.message : '',
    percent: typeof value.percent === 'number' && Number.isFinite(value.percent) ? value.percent : 0,
    datasetName: typeof value.datasetName === 'string' ? value.datasetName : null,
    bytesProcessed: typeof value.bytesProcessed === 'number' ? value.bytesProcessed : 0,
    bytesTotal: typeof value.bytesTotal === 'number' ? value.bytesTotal : 0,
    warnings: Array.isArray(value.warnings) ? value.warnings.filter((warning): warning is string => typeof warning === 'string') : [],
    error: typeof value.error === 'string' ? value.error : null,
    remoteJobID: typeof value.remoteJobID === 'string' ? value.remoteJobID : null,
    createdAt: typeof value.createdAt === 'string' ? value.createdAt : '',
    updatedAt: typeof value.updatedAt === 'string' ? value.updatedAt : '',
    ...(typeof value.operationState === 'string' ? { operationState: value.operationState } : {}),
    ...(typeof value.phase === 'string' ? { phase: value.phase } : {}),
  };
}
const runStartLikeJobAction = (
  jobID: string,
  actionPath: 'start' | 'restart-from-scratch',
  successLabel: string,
  failureLabel: string,
  encryptedDatasetKeys?: EncryptedDatasetStartKey[],
  options: StartJobOptions = {},
) => {
  return new Promise<void>((resolve, reject) => {
    const idempotencyKey = options.idempotencyKey ?? globalThis.crypto.randomUUID();
    let accepted = false;
    const waitForRemoteStart = async (initial: RemoteStartProgress) => {
      let progress = initial;
      while (true) {
        options.onRemoteStartProgress?.(progress);
        if (progress.status === 'completed') return;
        if (progress.status === 'failed') {
          throw new StartJobError(progress.error || 'Remote start failed', 'rejected', { code: 'REMOTE_START_FAILED' });
        }
        await new Promise(waitResolve => window.setTimeout(waitResolve, 500));
        progress = await getRemoteStartProgress(jobID, initial.startID);
      }
    };
    const postStart = async () => {
      const res = await apiClient.post<unknown>(`/api/jobs/${jobID}/${actionPath}`, {
        ...(actionPath === 'start' ? { idempotencyKey } : {}),
        background: options.background === true && actionPath === 'start',
      });
      const data = res.data;
      if (isRecord(data) && typeof data.startID === 'string' && data.startID) {
        accepted = true;
        await waitForRemoteStart(parseRemoteStartProgress(data.progress, jobID, data.startID));
      } else if (!isRecord(data) || data.id !== jobID || typeof data.status !== 'string' || !data.status) {
        throw new StartJobError('Invalid job start response. Retry to recover the start result.', 'unknown');
      }
    };
    postStart()
      .then(() => {
        console.log(successLabel);
        resolve();
      })
      .catch((error: unknown) => {
        // Axios failures retain request bodies, headers and response configuration.
        console.error(`${failureLabel}:`);
        reject(startActionError(error, failureLabel, accepted));
      });
  });
};
export const startJob = (
  jobID: string,
  encryptedDatasetKeys?: EncryptedDatasetStartKey[],
  options: StartJobOptions = {},
) => {
  return runStartLikeJobAction(jobID, 'start', 'Job started', 'Failed to start job.', encryptedDatasetKeys, options);
};
export const restartJobFromScratch = (
  jobID: string,
  encryptedDatasetKeys?: EncryptedDatasetStartKey[],
  options: StartJobOptions = {},
) => {
  return runStartLikeJobAction(
    jobID,
    'restart-from-scratch',
    'Job restarted from scratch',
    'Failed to restart job from scratch.',
    encryptedDatasetKeys,
    options,
  );
};
export const stopJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .post(`/api/jobs/${jobID}/stop`)
      .then(() => {
        console.log('Job stopped:');
        resolve();
      })
      .catch(error => {
        console.error('Error stopping job:');
        reject(error);
      });
  });
};
export const deleteJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .post(`/api/jobs/${jobID}/delete`)
      .then(() => {
        console.log('Job deleted:');
        resolve();
      })
      .catch(error => {
        console.error('Error deleting job:');
        reject(error);
      });
  });
};
export const markJobAsStopped = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .post(`/api/jobs/${jobID}/mark_stopped`)
      .then(() => {
        console.log('Job marked as stopped:');
        resolve();
      })
      .catch(error => {
        console.error('Error marking job as stopped:');
        reject(error);
      });
  });
};
export const saveJobNow = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .post(`/api/jobs/${jobID}/save_now`)
      .then(() => {
        console.log('Job set to save on next step:');
        resolve();
      })
      .catch(error => {
        console.error('Error setting job to save on next step:');
        reject(error);
      });
  });
};
export const retryRemoteCaptionResult = (jobID: string) => {
  return apiClient.post(`/api/jobs/${jobID}/remote-caption-result`, { retryFailed: true }).then(res => res.data as Job);
};
export const exportTrainingJob = (
  jobID: string,
  includeDatasets: boolean,
  checkpointMode: TrainingJobCheckpointExportMode = 'latest',
) => {
  return apiClient
    .post(`/api/jobs/${jobID}/export`, { includeDatasets, checkpointMode })
    .then(res => res.data as TrainingJobExportResult);
};
export const startTrainingJobExport = (
  jobID: string,
  includeDatasets: boolean,
  checkpointMode: TrainingJobCheckpointExportMode = 'latest',
) => {
  return apiClient.post(`/api/jobs/${jobID}/export`, { includeDatasets, checkpointMode, background: true }).then(
    res =>
      res.data as {
        exportID: string;
        statusUrl: string;
        progress: TrainingJobExportProgress;
      },
  );
};
export const getTrainingJobExportProgress = (jobID: string, exportID: string) => {
  return apiClient.get(`/api/jobs/${jobID}/export/${exportID}`).then(res => res.data as TrainingJobExportProgress);
};
export const cancelTrainingJobExport = (jobID: string, exportID: string) => {
  return apiClient.delete(`/api/jobs/${jobID}/export/${exportID}`).then(res => res.data as TrainingJobExportProgress);
};
export const importTrainingJob = (
  file: File,
  gpuIDs: string | null,
  onUploadProgress?: (progress: TrainingJobImportProgress) => void,
) => {
  if (file.size > MAX_TRAINING_JOB_IMPORT_BYTES) {
    return Promise.reject(new Error('Training job archives must be 64 GB or smaller.'));
  }
  const uploadID = globalThis.crypto.randomUUID();
  const chunksTotal = Math.max(1, Math.ceil(file.size / TRAINING_JOB_IMPORT_CHUNK_BYTES));
  const commonParams = {
    uploadID,
    chunksTotal: String(chunksTotal),
    fileBytes: String(file.size),
  };
  const reportProgress = (loaded: number) => {
    const total = file.size || null;
    onUploadProgress?.({
      loaded: Math.min(loaded, file.size),
      total,
      percent: total ? Math.min(100, Math.round((loaded / total) * 100)) : 100,
    });
  };
  return (async () => {
    reportProgress(0);
    let uploadedBytes = 0;
    for (let chunkIndex = 0; chunkIndex < chunksTotal; chunkIndex += 1) {
      const start = chunkIndex * TRAINING_JOB_IMPORT_CHUNK_BYTES;
      const chunk = file.slice(start, Math.min(file.size, start + TRAINING_JOB_IMPORT_CHUNK_BYTES));
      const params = new URLSearchParams({
        aitk_upload: 'chunk',
        ...commonParams,
        chunkIndex: String(chunkIndex),
      });
      await apiClient.post(`/api/jobs/import?${params.toString()}`, chunk, {
        headers: {
          'Content-Type': 'application/octet-stream',
          'X-AITK-File-Name': encodeURIComponent(file.name),
        },
        onUploadProgress: (event: AxiosProgressEvent) => {
          reportProgress(uploadedBytes + Math.min(event.loaded, chunk.size));
        },
      });
      uploadedBytes += chunk.size;
      reportProgress(uploadedBytes);
    }
    const completeParams = new URLSearchParams({
      aitk_upload: 'complete',
      ...commonParams,
    });
    if (gpuIDs) completeParams.set('gpu_ids', gpuIDs);
    const response = await apiClient.post(`/api/jobs/import?${completeParams.toString()}`, undefined, {
      headers: { 'X-AITK-File-Name': encodeURIComponent(file.name) },
    });
    let result: unknown = response.data;
    const deadline = Date.now() + 60 * 60 * 1000;
    while (result && typeof result === 'object' && 'uploadID' in result && 'status' in result) {
      if (result.status === 'completed' && 'result' in result) { result = result.result; break; }
      if (result.status === 'failed') throw new Error('error' in result && typeof result.error === 'string' ? result.error : 'Archive import failed');
      if (Date.now() >= deadline) throw new Error('Import is still running. Its progress remains available on the server.');
      await new Promise(resolve => setTimeout(resolve, 1000));
      result = (await apiClient.get(`/api/jobs/import?aitk_upload=status&uploadID=${encodeURIComponent(uploadID)}`)).data;
    }
    if (!result || typeof result !== 'object' || !('job' in result) || !('warnings' in result)) throw new Error('Invalid import response');
    return result as TrainingJobImportResult;
  })();
};
export const sampleJobNow = async (jobID: string): Promise<void> => {
  await apiClient.post(`/api/jobs/${jobID}/sample_now`);
  console.log('Job set to sample on next step:');
};
export const getRemoteStartProgress = (jobID: string, startID: string) => {
  return apiClient.get<unknown>(`/api/jobs/${jobID}/start-progress/${startID}`).then(res => parseRemoteStartProgress(res.data, jobID, startID));
};
export const downloadJobModelReferences = (jobID: string) => {
  return apiClient.post(`/api/jobs/${jobID}/prefetch-models`).then(res => res.data as JobModelPrefetchResult);
};
export const downloadServerFile = (filePath: string, fileName?: string) => {
  const a = document.createElement('a');
  a.href = getDownloadUrl(filePath);
  a.download = fileName || getDisplayPath(filePath).split(/[\\/]/).pop() || 'download';
  document.body.appendChild(a);
  a.click();
  a.remove();
};
export const getJobConfig = (job: Job) => {
  return JSON.parse(job.job_config) as JobConfig;
};
export const getAvaliableJobActions = (job: Job) => {
  const jobConfig = getJobConfig(job);
  const isStopping = job.stop && job.status === 'running';
  const canDelete = ['queued', 'completed', 'stopped', 'error'].includes(job.status) && !isStopping;
  const canEdit = ['queued', 'completed', 'stopped', 'error'].includes(job.status) && !isStopping;
  const canRemoveFromQueue = job.status === 'queued';
  const canStop = ['starting', 'running'].includes(job.status) && !isStopping;
  let canStart = ['stopped', 'error'].includes(job.status) && !isStopping;
  const canRestartFromScratch =
    job.job_type === 'train' && ['queued', 'completed', 'stopped', 'error'].includes(job.status) && !isStopping;
  // can resume if more steps were added
  const totalSteps = getTotalSteps(job);
  if (job.status === 'completed' && totalSteps !== null && totalSteps > job.step && !isStopping) {
    canStart = true;
  }
  return { canDelete, canEdit, canStop, canStart, canRemoveFromQueue, canRestartFromScratch };
};
export const getNumberOfSamples = (job: Job) => {
  const jobConfig = getJobConfig(job);
  return jobConfig.config.process[0].sample?.prompts?.length || 0;
};
export const getTotalSteps = (job: Job): number | null => {
  const jobConfig = getJobConfig(job);
  return jobConfig.config.process[0].train?.steps || 0;
};
