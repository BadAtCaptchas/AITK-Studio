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
  onRemoteStartProgress?: (progress: RemoteStartProgress) => void;
};
function apiErrorMessage(error: unknown, fallback = 'Request failed') {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = error.response;
    if (response && typeof response === 'object' && 'data' in response) {
      const data = response.data;
      if (data && typeof data === 'object' && 'error' in data && typeof data.error === 'string') return data.error;
    }
  }
  return error instanceof Error ? error.message : fallback;
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
    const idempotencyKey = globalThis.crypto.randomUUID();
    const waitForRemoteStart = async (startID: string) => {
      while (true) {
        await new Promise(waitResolve => window.setTimeout(waitResolve, 500));
        const progress = await getRemoteStartProgress(jobID, startID);
        options.onRemoteStartProgress?.(progress);
        if (progress.status === 'completed') return;
        if (progress.status === 'failed') {
          throw new Error(progress.error || 'Remote start failed');
        }
      }
    };
    const postStart = async () => {
      const res = await apiClient.post(`/api/jobs/${jobID}/${actionPath}`, {
        ...(actionPath === 'start' ? { idempotencyKey } : {}),
        background: options.background === true && actionPath === 'start',
      });
      const data = res.data;
      if (data?.startID && data?.progress) {
        options.onRemoteStartProgress?.(data.progress);
        await waitForRemoteStart(data.startID);
      }
    };
    postStart()
      .then(() => {
        console.log(successLabel);
        resolve();
      })
      .catch(async error => {
        console.error(`${failureLabel}:`, error);
        reject(new Error(apiErrorMessage(error, failureLabel)));
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
      .then(res => res.data)
      .then(data => {
        console.log('Job stopped:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error stopping job:', error);
        reject(error);
      });
  });
};
export const deleteJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .post(`/api/jobs/${jobID}/delete`)
      .then(res => res.data)
      .then(data => {
        console.log('Job deleted:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error deleting job:', error);
        reject(error);
      });
  });
};
export const markJobAsStopped = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .post(`/api/jobs/${jobID}/mark_stopped`)
      .then(res => res.data)
      .then(data => {
        console.log('Job marked as stopped:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error marking job as stopped:', error);
        reject(error);
      });
  });
};
export const saveJobNow = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .post(`/api/jobs/${jobID}/save_now`)
      .then(res => res.data)
      .then(data => {
        console.log('Job set to save on next step:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error setting job to save on next step:', error);
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
  const response = await apiClient.post(`/api/jobs/${jobID}/sample_now`);
  console.log('Job set to sample on next step:', response.data);
};
export const getRemoteStartProgress = (jobID: string, startID: string) => {
  return apiClient.get(`/api/jobs/${jobID}/start-progress/${startID}`).then(res => res.data as RemoteStartProgress);
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
  if (jobConfig.config.process[0].train?.auto_train) return null;
  return jobConfig.config.process[0].train?.steps || 0;
};
