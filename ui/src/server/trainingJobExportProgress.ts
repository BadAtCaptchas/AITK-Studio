import { randomUUID } from 'crypto';

export type TrainingJobExportStatus = 'queued' | 'preparing' | 'zipping' | 'finalizing' | 'completed' | 'failed';

export type TrainingJobExportProgressSnapshot = {
  exportID: string;
  jobID: string;
  includeDatasets: boolean;
  status: TrainingJobExportStatus;
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
  createdAt: string;
  updatedAt: string;
};

type TrainingJobExportProgressPatch = Partial<
  Pick<
    TrainingJobExportProgressSnapshot,
    | 'status'
    | 'message'
    | 'percent'
    | 'entriesProcessed'
    | 'entriesTotal'
    | 'bytesProcessed'
    | 'bytesTotal'
    | 'zipPath'
    | 'fileName'
    | 'warnings'
    | 'error'
  >
>;

type ExportProgressStore = Map<string, TrainingJobExportProgressSnapshot>;

const EXPORT_PROGRESS_MAX_AGE_MS = 60 * 60 * 1000;

declare global {
  // eslint-disable-next-line no-var
  var __trainingJobExportProgressStore: ExportProgressStore | undefined;
}

const exportProgressStore: ExportProgressStore =
  globalThis.__trainingJobExportProgressStore ?? new Map<string, TrainingJobExportProgressSnapshot>();

if (!globalThis.__trainingJobExportProgressStore) {
  globalThis.__trainingJobExportProgressStore = exportProgressStore;
}

function cleanupOldTrainingJobExportProgress() {
  const now = Date.now();
  for (const [exportID, progress] of exportProgressStore.entries()) {
    if (now - new Date(progress.updatedAt).getTime() > EXPORT_PROGRESS_MAX_AGE_MS) {
      exportProgressStore.delete(exportID);
    }
  }
}

export function createTrainingJobExportProgress(jobID: string, includeDatasets: boolean) {
  cleanupOldTrainingJobExportProgress();

  const now = new Date().toISOString();
  const progress: TrainingJobExportProgressSnapshot = {
    exportID: randomUUID(),
    jobID,
    includeDatasets,
    status: 'queued',
    message: 'Queued export',
    percent: 0,
    entriesProcessed: 0,
    entriesTotal: 0,
    bytesProcessed: 0,
    bytesTotal: 0,
    zipPath: null,
    fileName: null,
    warnings: [],
    error: null,
    createdAt: now,
    updatedAt: now,
  };

  exportProgressStore.set(progress.exportID, progress);
  return { ...progress };
}

export function getTrainingJobExportProgress(exportID: string) {
  cleanupOldTrainingJobExportProgress();
  const progress = exportProgressStore.get(exportID);
  return progress ? { ...progress, warnings: [...progress.warnings] } : null;
}

export function updateTrainingJobExportProgress(exportID: string, patch: TrainingJobExportProgressPatch) {
  const progress = exportProgressStore.get(exportID);
  if (!progress) return null;

  const updated: TrainingJobExportProgressSnapshot = {
    ...progress,
    ...patch,
    percent: Math.max(0, Math.min(100, patch.percent ?? progress.percent)),
    warnings: patch.warnings ? [...patch.warnings] : progress.warnings,
    updatedAt: new Date().toISOString(),
  };

  exportProgressStore.set(exportID, updated);
  return { ...updated, warnings: [...updated.warnings] };
}

