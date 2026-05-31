import path from 'path';
import type { Job } from '../types';

export const REMOTE_CAPTION_STATE_VERSION = 1;

export type RemoteCaptionDownloadStatus =
  | 'not_started'
  | 'dispatching'
  | 'running'
  | 'downloading'
  | 'merged'
  | 'failed';

export type RemoteCaptionState = {
  version: typeof REMOTE_CAPTION_STATE_VERSION;
  downloadStatus: RemoteCaptionDownloadStatus;
  originalDatasetPath: string;
  originalDatasetName: string;
  remoteWorkerId: string;
  remoteWorkerName?: string;
  remoteDatasetName?: string;
  remoteDatasetPath?: string;
  encrypted: boolean;
  durableEncryptedKeys: boolean;
  captionExtension: string;
  recaption: boolean;
  dispatchedAt?: string;
  completedAt?: string;
  downloadStartedAt?: string;
  downloadedAt?: string;
  mergedAt?: string;
  importedFallbackPath?: string;
  cleanupError?: string | null;
  lastError?: string | null;
};

export type CaptionProcessInfo = {
  process: any;
  processIndex: number;
  caption: any;
  pathToCaption: string;
  captionExtension: string;
  recaption: boolean;
};

function cloneJson<T>(value: T): T {
  return JSON.parse(JSON.stringify(value));
}

function ensureJobConfigContainer(jobConfig: any) {
  if (!jobConfig.config || typeof jobConfig.config !== 'object') {
    jobConfig.config = {};
  }
  return jobConfig.config;
}

export function getRemoteCaptionState(jobConfig: any): RemoteCaptionState | null {
  const value = jobConfig?.config?.remote_caption;
  if (!value || typeof value !== 'object' || value.version !== REMOTE_CAPTION_STATE_VERSION) {
    return null;
  }
  return value as RemoteCaptionState;
}

export function hasRemoteCaptionState(jobConfig: any) {
  return getRemoteCaptionState(jobConfig) !== null;
}

export function setRemoteCaptionState(jobConfig: any, state: RemoteCaptionState) {
  const next = cloneJson(jobConfig);
  ensureJobConfigContainer(next).remote_caption = state;
  return next;
}

export function patchRemoteCaptionState(jobConfig: any, patch: Partial<RemoteCaptionState>) {
  const existing = getRemoteCaptionState(jobConfig);
  if (!existing) return jobConfig;
  return setRemoteCaptionState(jobConfig, { ...existing, ...patch });
}

export function getJobRemoteCaptionState(job: Job) {
  try {
    return getRemoteCaptionState(JSON.parse(job.job_config));
  } catch {
    return null;
  }
}

export function findCaptionProcess(jobConfig: any): CaptionProcessInfo | null {
  const processes = Array.isArray(jobConfig?.config?.process) ? jobConfig.config.process : [];
  for (let processIndex = 0; processIndex < processes.length; processIndex += 1) {
    const processConfig = processes[processIndex];
    const caption = processConfig?.caption;
    const pathToCaption = caption?.path_to_caption;
    if (!caption || typeof pathToCaption !== 'string' || !pathToCaption.trim()) continue;
    return {
      process: processConfig,
      processIndex,
      caption,
      pathToCaption: pathToCaption.trim(),
      captionExtension:
        typeof caption.caption_extension === 'string' && caption.caption_extension.trim()
          ? caption.caption_extension.trim().replace(/^\.+/, '')
          : 'txt',
      recaption: caption.recaption === true,
    };
  }
  return null;
}

export function isRemoteCaptionDispatchConfig(jobConfig: any) {
  const info = findCaptionProcess(jobConfig);
  if (!info) return false;
  return ['SecureRemoteOllamaCaptioner', 'OllamaCaptioner'].includes(String(info.process?.type || ''));
}

export function isRemoteCaptionMirrorJob(job: Job) {
  if (job.job_type !== 'caption') return false;
  return getJobRemoteCaptionState(job) !== null;
}

export function remoteCaptionDatasetName(job: Job, originalDatasetName: string) {
  const safeDataset = originalDatasetName
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 80);
  const safeJob = job.id.replace(/[^a-zA-Z0-9_-]+/g, '').slice(0, 12);
  return `${safeDataset || 'dataset'}_remote_caption_${safeJob}`;
}

export function buildInitialRemoteCaptionState(options: {
  job: Job;
  worker: { id: string; name?: string };
  originalDatasetPath: string;
  encrypted: boolean;
  durableEncryptedKeys: boolean;
  captionExtension: string;
  recaption: boolean;
}): RemoteCaptionState {
  return {
    version: REMOTE_CAPTION_STATE_VERSION,
    downloadStatus: 'dispatching',
    originalDatasetPath: options.originalDatasetPath,
    originalDatasetName: path.basename(options.originalDatasetPath),
    remoteWorkerId: options.worker.id,
    remoteWorkerName: options.worker.name,
    encrypted: options.encrypted,
    durableEncryptedKeys: options.durableEncryptedKeys,
    captionExtension: options.captionExtension,
    recaption: options.recaption,
    dispatchedAt: new Date().toISOString(),
    lastError: null,
  };
}

export function buildRemoteOllamaCaptionJobConfig(
  centralJobConfig: any,
  options: {
    remoteDatasetPath: string;
    remoteJobName: string;
  },
) {
  const next = cloneJson(centralJobConfig);
  ensureJobConfigContainer(next).name = options.remoteJobName;
  delete next.config.remote_caption;
  const info = findCaptionProcess(next);
  if (!info) {
    throw new Error('Caption process not found in job config');
  }
  info.process.type = 'OllamaCaptioner';
  info.process.device = 'cpu';
  info.caption.path_to_caption = options.remoteDatasetPath;
  delete info.caption.remote_worker_id;
  return next;
}
