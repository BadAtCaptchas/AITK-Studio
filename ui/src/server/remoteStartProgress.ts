import { randomUUID } from 'crypto';
import type { RemoteStartProgress, RemoteStartProgressStatus } from '../types';

type RemoteStartProgressPatch = Partial<
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
>;

type RemoteStartProgressStore = Map<string, RemoteStartProgress>;

const REMOTE_START_PROGRESS_MAX_AGE_MS = 60 * 60 * 1000;
const ACTIVE_REMOTE_START_STATUSES = new Set<RemoteStartProgressStatus>([
  'queued',
  'preparing',
  'checking-datasets',
  'zipping-dataset',
  'uploading-dataset',
  'importing-dataset',
  'zipping-job',
  'uploading-job',
  'importing-job',
  'starting',
]);

declare global {
  // eslint-disable-next-line no-var
  var __remoteStartProgressStore: RemoteStartProgressStore | undefined;
}

const remoteStartProgressStore: RemoteStartProgressStore =
  globalThis.__remoteStartProgressStore ?? new Map<string, RemoteStartProgress>();

if (!globalThis.__remoteStartProgressStore) {
  globalThis.__remoteStartProgressStore = remoteStartProgressStore;
}

function cloneProgress(progress: RemoteStartProgress) {
  return { ...progress, warnings: [...progress.warnings] };
}

function cleanupOldRemoteStartProgress() {
  const now = Date.now();
  for (const [startID, progress] of remoteStartProgressStore.entries()) {
    if (now - new Date(progress.updatedAt).getTime() > REMOTE_START_PROGRESS_MAX_AGE_MS) {
      remoteStartProgressStore.delete(startID);
    }
  }
}

export function createRemoteStartProgress(jobID: string) {
  cleanupOldRemoteStartProgress();

  const now = new Date().toISOString();
  const progress: RemoteStartProgress = {
    startID: randomUUID(),
    jobID,
    status: 'queued',
    message: 'Queued remote start',
    percent: 0,
    datasetName: null,
    bytesProcessed: 0,
    bytesTotal: 0,
    warnings: [],
    error: null,
    remoteJobID: null,
    createdAt: now,
    updatedAt: now,
  };

  remoteStartProgressStore.set(progress.startID, progress);
  return cloneProgress(progress);
}

export function getRemoteStartProgress(startID: string) {
  cleanupOldRemoteStartProgress();
  const progress = remoteStartProgressStore.get(startID);
  return progress ? cloneProgress(progress) : null;
}

export function updateRemoteStartProgress(startID: string, patch: RemoteStartProgressPatch) {
  const progress = remoteStartProgressStore.get(startID);
  if (!progress) return null;

  const updated: RemoteStartProgress = {
    ...progress,
    ...patch,
    percent: Math.max(0, Math.min(100, patch.percent ?? progress.percent)),
    datasetName: patch.datasetName !== undefined ? patch.datasetName : progress.datasetName,
    warnings: patch.warnings ? [...patch.warnings] : progress.warnings,
    updatedAt: new Date().toISOString(),
  };

  remoteStartProgressStore.set(startID, updated);
  return cloneProgress(updated);
}

export function hasActiveRemoteStartForJob(jobID: string) {
  cleanupOldRemoteStartProgress();
  for (const progress of remoteStartProgressStore.values()) {
    if (progress.jobID === jobID && ACTIVE_REMOTE_START_STATUSES.has(progress.status)) {
      return true;
    }
  }
  return false;
}
