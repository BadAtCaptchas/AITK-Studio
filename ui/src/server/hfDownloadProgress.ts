import fs from 'fs/promises';
import path from 'path';
import type { Job } from '@/types';
import { getTrainingFolder } from '@/server/settings';

export const HF_DOWNLOAD_PROGRESS_FILE = '.hf_download_progress.json';

const ACTIVE_STALE_MS = 60_000;
const COMPLETED_GRACE_MS = 20_000;
const FAILED_GRACE_MS = 10 * 60_000;

export type HFDownloadStatus = 'idle' | 'downloading' | 'completed' | 'failed';

export type HFDownloadItem = {
  id: number;
  fileName: string;
  source?: string;
  bytesDownloaded: number;
  bytesTotal: number | null;
  startedAt: string;
  updatedAt: string;
};

export type HFDownloadProgress = {
  version: number;
  status: HFDownloadStatus;
  message: string;
  fileName: string | null;
  activeCount: number;
  bytesDownloaded: number;
  bytesTotal: number | null;
  percent: number | null;
  downloads: HFDownloadItem[];
  error: string | null;
  updatedAt: string;
};

function isPathWithin(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}

function numberOrNull(value: unknown): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return value;
}

function stringOrNull(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function normalizeProgress(raw: unknown): HFDownloadProgress | null {
  if (!raw || typeof raw !== 'object') return null;

  const record = raw as Record<string, unknown>;
  const status = record.status;
  if (status !== 'idle' && status !== 'downloading' && status !== 'completed' && status !== 'failed') {
    return null;
  }

  const downloads = Array.isArray(record.downloads)
    ? record.downloads
        .map((item): HFDownloadItem | null => {
          if (!item || typeof item !== 'object') return null;
          const download = item as Record<string, unknown>;
          const id = numberOrNull(download.id);
          const fileName = stringOrNull(download.fileName);
          const bytesDownloaded = numberOrNull(download.bytesDownloaded);
          const startedAt = stringOrNull(download.startedAt);
          const updatedAt = stringOrNull(download.updatedAt);
          if (id == null || fileName == null || bytesDownloaded == null || startedAt == null || updatedAt == null) {
            return null;
          }
          return {
            id,
            fileName,
            source: stringOrNull(download.source) || undefined,
            bytesDownloaded,
            bytesTotal: numberOrNull(download.bytesTotal),
            startedAt,
            updatedAt,
          };
        })
        .filter((item): item is HFDownloadItem => item !== null)
    : [];

  const updatedAt = stringOrNull(record.updatedAt);
  if (!updatedAt) return null;

  return {
    version: numberOrNull(record.version) || 1,
    status,
    message: stringOrNull(record.message) || 'Hugging Face download',
    fileName: stringOrNull(record.fileName),
    activeCount: numberOrNull(record.activeCount) || downloads.length,
    bytesDownloaded: numberOrNull(record.bytesDownloaded) || 0,
    bytesTotal: numberOrNull(record.bytesTotal),
    percent: numberOrNull(record.percent),
    downloads,
    error: stringOrNull(record.error),
    updatedAt,
  };
}

function isVisible(progress: HFDownloadProgress) {
  const updatedMs = new Date(progress.updatedAt).getTime();
  if (!Number.isFinite(updatedMs)) return false;
  const age = Date.now() - updatedMs;

  if (progress.status === 'downloading') return age <= ACTIVE_STALE_MS;
  if (progress.status === 'completed') return age <= COMPLETED_GRACE_MS;
  if (progress.status === 'failed') return age <= FAILED_GRACE_MS;
  return false;
}

export async function getHFDownloadProgress(job: Job): Promise<HFDownloadProgress | null> {
  const trainingRoot = await getTrainingFolder();
  const jobFolder = path.resolve(trainingRoot, job.name);
  if (!isPathWithin(trainingRoot, jobFolder)) return null;

  const progressPath = path.join(jobFolder, HF_DOWNLOAD_PROGRESS_FILE);
  if (!isPathWithin(jobFolder, progressPath)) return null;

  try {
    const raw = await fs.readFile(progressPath, 'utf-8');
    const progress = normalizeProgress(JSON.parse(raw));
    return progress && isVisible(progress) ? progress : null;
  } catch (error: any) {
    if (error?.code !== 'ENOENT') {
      console.error('Error reading Hugging Face download progress:', error);
    }
    return null;
  }
}

export async function withHFDownloadProgress<T extends Job>(job: T): Promise<T & { hf_download_progress: HFDownloadProgress | null }> {
  return {
    ...job,
    hf_download_progress: await getHFDownloadProgress(job),
  };
}
