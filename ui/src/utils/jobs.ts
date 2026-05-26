import { JobConfig } from '@/types';
import type { Job } from '@/types';
import { apiClient } from '@/utils/api';
import { getDisplayPath, getDownloadUrl } from '@/utils/media';
import type { EncryptedDatasetStartKey } from '@/types';
import type { AxiosProgressEvent } from 'axios';
import {
  derivePasswordKey,
  exportRawAesKey,
  getRememberedEncryptedDatasetKey,
  rememberEncryptedDatasetKey,
} from '@/utils/encryptedDatasets';

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

export type JobModelPrefetchResult = {
  handledValues: string[];
  downloads: Array<{ value: string; path: string; kind: string; cached?: boolean }>;
  warnings: string[];
  updatedConfig: boolean;
  job: Job;
};

export type StartJobOptions = {
  durableEncryptedDatasetKeys?: boolean;
};

function basenameFromPath(value: string) {
  return value.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || value;
}

function promptForDurableEncryptedResume() {
  if (typeof window === 'undefined') return false;
  return window.confirm(
    'Store a wrapped copy of this encrypted dataset key on the server so the job can resume from the queue after an app restart? This requires AITK_DURABLE_DATASET_KEY_SECRET on the server; changing that secret invalidates durable resume for queued jobs.',
  );
}

async function resolveEncryptedDatasetStartKey(dataset: { path: string; name: string }) {
  const remembered =
    getRememberedEncryptedDatasetKey(dataset.path) || getRememberedEncryptedDatasetKey(dataset.name);
  if (remembered) {
    return { datasetPath: dataset.path, keyB64: remembered };
  }

  const datasetName = dataset.name || basenameFromPath(dataset.path);
  const res = await apiClient.post('/api/datasets/listImages', { datasetName });
  const manifest = res.data?.manifest;
  if (!manifest) throw new Error(`Encrypted dataset key required for ${datasetName}`);
  if (manifest.crypto?.kdf?.type !== 'PBKDF2-SHA256') {
    throw new Error(`Encrypted dataset ${datasetName} requires its key file. Unlock the dataset page first.`);
  }

  const password = window.prompt(`Password for encrypted dataset "${datasetName}"`);
  if (!password) throw new Error(`Encrypted dataset key required for ${datasetName}`);
  const key = await derivePasswordKey(password, manifest);
  const keyB64 = await exportRawAesKey(key);
  rememberEncryptedDatasetKey(dataset.path, keyB64);
  rememberEncryptedDatasetKey(datasetName, keyB64);
  return { datasetPath: dataset.path, keyB64 };
}

export const startJob = (
  jobID: string,
  encryptedDatasetKeys?: EncryptedDatasetStartKey[],
  options: StartJobOptions = {},
) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .post(`/api/jobs/${jobID}/start`, {
        encryptedDatasetKeys,
        durableEncryptedDatasetKeys: options.durableEncryptedDatasetKeys === true,
      })
      .then(res => res.data)
      .then(data => {
        console.log('Job started:', data);
        resolve();
      })
      .catch(async error => {
        const requiredDatasets = error.response?.status === 409 ? error.response?.data?.encryptedDatasets : null;
        if (Array.isArray(requiredDatasets)) {
          try {
            const supplied = encryptedDatasetKeys || [];
            const additional = await Promise.all(requiredDatasets.map(resolveEncryptedDatasetStartKey));
            const durableEncryptedDatasetKeys =
              options.durableEncryptedDatasetKeys ?? promptForDurableEncryptedResume();
            await apiClient.post(`/api/jobs/${jobID}/start`, {
              encryptedDatasetKeys: [...supplied, ...additional],
              durableEncryptedDatasetKeys,
            });
            resolve();
            return;
          } catch (keyError) {
            reject(keyError);
            return;
          }
        }
        console.error('Error starting job:', error);
        reject(error);
      });
  });
};

export const stopJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/jobs/${jobID}/stop`)
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
      .get(`/api/jobs/${jobID}/delete`)
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
      .get(`/api/jobs/${jobID}/mark_stopped`)
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
  return apiClient
    .post(`/api/jobs/${jobID}/export`, { includeDatasets, checkpointMode, background: true })
    .then(res => res.data as { exportID: string; statusUrl: string; progress: TrainingJobExportProgress });
};

export const getTrainingJobExportProgress = (jobID: string, exportID: string) => {
  return apiClient
    .get(`/api/jobs/${jobID}/export/${exportID}`)
    .then(res => res.data as TrainingJobExportProgress);
};

export const cancelTrainingJobExport = (jobID: string, exportID: string) => {
  return apiClient
    .delete(`/api/jobs/${jobID}/export/${exportID}`)
    .then(res => res.data as TrainingJobExportProgress);
};

export const importTrainingJob = (
  file: File,
  gpuIDs: string | null,
  onUploadProgress?: (progress: TrainingJobImportProgress) => void,
) => {
  const formData = new FormData();
  formData.append('file', file);
  if (gpuIDs) {
    formData.append('gpu_ids', gpuIDs);
  }

  return apiClient
    .post('/api/jobs/import', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (event: AxiosProgressEvent) => {
        const total = event.total && event.total > 0 ? event.total : file.size || null;
        onUploadProgress?.({
          loaded: event.loaded,
          total,
          percent: total ? Math.min(100, Math.round((event.loaded / total) * 100)) : null,
        });
      },
    })
    .then(res => res.data as TrainingJobImportResult);
};

export const downloadJobModelReferences = (jobID: string) => {
  return apiClient
    .post(`/api/jobs/${jobID}/prefetch-models`)
    .then(res => res.data as JobModelPrefetchResult);
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
  let canEdit = ['queued', 'completed', 'stopped', 'error'].includes(job.status) && !isStopping;
  const canRemoveFromQueue = job.status === 'queued';
  const canStop = job.status === 'running' && !isStopping;
  let canStart = ['stopped', 'error'].includes(job.status) && !isStopping;
  // can resume if more steps were added
  const totalSteps = getTotalSteps(job);
  if (job.status === 'completed' && totalSteps !== null && totalSteps > job.step && !isStopping) {
    canStart = true;
  }
  return { canDelete, canEdit, canStop, canStart, canRemoveFromQueue };
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
