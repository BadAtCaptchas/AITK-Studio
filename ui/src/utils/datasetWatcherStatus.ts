export type DatasetWatcherListItem = {
  id: string;
  datasetName?: string;
  projectID?: string | null;
  enabled?: boolean;
};

export type DatasetWatcherStatus = {
  state?: string;
  lastScanAt?: string | null;
  lastImportedAt?: string | null;
  lastImportedCount?: number;
  lastCaptionedCount?: number;
  autoCaptionTotalCount?: number;
  autoCaptionPendingCount?: number;
  autoCaptionCompletedCount?: number;
  autoCaptionActivePath?: string | null;
  lastError?: string | null;
  warnings?: string[];
};

export type DatasetAutoCaptionProgress = {
  total: number;
  pending: number;
  completed: number;
  activePaths: string[];
};

function count(value: unknown) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : 0;
}

export function emptyAutoCaptionProgress(): DatasetAutoCaptionProgress {
  return { total: 0, pending: 0, completed: 0, activePaths: [] };
}

export function aggregateAutoCaptionProgress(
  watchers: DatasetWatcherListItem[],
  statuses: Record<string, DatasetWatcherStatus>,
  datasetName?: string | null,
) {
  const progress = emptyAutoCaptionProgress();
  for (const watcher of watchers) {
    if (watcher.enabled === false) continue;
    if (datasetName && watcher.datasetName !== datasetName) continue;
    const status = statuses[watcher.id];
    if (!status) continue;
    progress.total += count(status.autoCaptionTotalCount);
    progress.pending += count(status.autoCaptionPendingCount);
    progress.completed += count(status.autoCaptionCompletedCount);
    if (status.autoCaptionActivePath) progress.activePaths.push(status.autoCaptionActivePath);
  }
  return progress;
}

export function aggregateAutoCaptionProgressByDataset(
  watchers: DatasetWatcherListItem[],
  statuses: Record<string, DatasetWatcherStatus>,
) {
  const byDataset: Record<string, DatasetAutoCaptionProgress> = {};
  for (const watcher of watchers) {
    if (!watcher.datasetName || watcher.enabled === false) continue;
    const status = statuses[watcher.id];
    if (!status) continue;
    const progress = byDataset[watcher.datasetName] || emptyAutoCaptionProgress();
    progress.total += count(status.autoCaptionTotalCount);
    progress.pending += count(status.autoCaptionPendingCount);
    progress.completed += count(status.autoCaptionCompletedCount);
    if (status.autoCaptionActivePath) progress.activePaths.push(status.autoCaptionActivePath);
    byDataset[watcher.datasetName] = progress;
  }
  return byDataset;
}

export function hasAutoCaptionProgress(
  progress: DatasetAutoCaptionProgress | null | undefined,
): progress is DatasetAutoCaptionProgress {
  return Boolean(progress && progress.pending > 0);
}

export function autoCaptionProgressTitle(progress: DatasetAutoCaptionProgress) {
  const base =
    progress.total > 0
      ? `${progress.completed.toLocaleString()} of ${progress.total.toLocaleString()} auto-caption attempts finished`
      : 'Auto-captioning is running';
  const active = progress.activePaths.length > 0 ? ` Current: ${progress.activePaths[0]}` : '';
  return `${base}; ${progress.pending.toLocaleString()} left.${active}`;
}
