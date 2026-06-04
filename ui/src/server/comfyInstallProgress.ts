import fs from 'fs/promises';
import path from 'path';
import type { Job } from '@/types';
import { getTrainingFolder } from '@/server/settings';

export const COMFY_INSTALL_PROGRESS_FILE = '.comfy_install_progress.json';

const ACTIVE_STALE_MS = 60 * 60_000;
const COMPLETED_GRACE_MS = 30_000;
const FAILED_GRACE_MS = 10 * 60_000;

export type ComfyInstallStatus = 'idle' | 'checking' | 'installing' | 'launching' | 'ready' | 'completed' | 'failed';

export type ComfyInstallProgress = {
  version: number;
  status: ComfyInstallStatus;
  step: string;
  message: string;
  root: string | null;
  percent: number | null;
  error: string | null;
  startedAt: string;
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

function normalizeProgress(raw: unknown): ComfyInstallProgress | null {
  if (!raw || typeof raw !== 'object') return null;

  const record = raw as Record<string, unknown>;
  const status = record.status;
  if (
    status !== 'idle' &&
    status !== 'checking' &&
    status !== 'installing' &&
    status !== 'launching' &&
    status !== 'ready' &&
    status !== 'completed' &&
    status !== 'failed'
  ) {
    return null;
  }

  const startedAt = stringOrNull(record.startedAt);
  const updatedAt = stringOrNull(record.updatedAt);
  if (!startedAt || !updatedAt) return null;

  const percent = numberOrNull(record.percent);

  return {
    version: numberOrNull(record.version) || 1,
    status,
    step: stringOrNull(record.step) || 'install',
    message: stringOrNull(record.message) || 'Preparing managed ComfyUI',
    root: stringOrNull(record.root),
    percent: percent == null ? null : Math.max(0, Math.min(100, percent)),
    error: stringOrNull(record.error),
    startedAt,
    updatedAt,
  };
}

function isVisible(progress: ComfyInstallProgress) {
  const updatedMs = new Date(progress.updatedAt).getTime();
  if (!Number.isFinite(updatedMs)) return false;
  const age = Date.now() - updatedMs;

  if (progress.status === 'checking' || progress.status === 'installing' || progress.status === 'launching' || progress.status === 'ready') {
    return age <= ACTIVE_STALE_MS;
  }
  if (progress.status === 'completed') return age <= COMPLETED_GRACE_MS;
  if (progress.status === 'failed') return age <= FAILED_GRACE_MS;
  return false;
}

export async function getComfyInstallProgress(job: Job): Promise<ComfyInstallProgress | null> {
  const trainingRoot = await getTrainingFolder();
  const jobFolder = path.resolve(trainingRoot, job.name);
  if (!isPathWithin(trainingRoot, jobFolder)) return null;

  const progressPath = path.join(jobFolder, COMFY_INSTALL_PROGRESS_FILE);
  if (!isPathWithin(jobFolder, progressPath)) return null;

  return getComfyInstallProgressAtPath(progressPath);
}

export async function getComfyInstallProgressAtPath(progressPath: string): Promise<ComfyInstallProgress | null> {
  try {
    const raw = await fs.readFile(progressPath, 'utf-8');
    const progress = normalizeProgress(JSON.parse(raw));
    return progress && isVisible(progress) ? progress : null;
  } catch (error: any) {
    if (error?.code !== 'ENOENT') {
      console.error('Error reading ComfyUI install progress:', error);
    }
    return null;
  }
}

export async function withComfyInstallProgress<T extends Job>(job: T): Promise<T & { comfy_install_progress: ComfyInstallProgress | null }> {
  return {
    ...job,
    comfy_install_progress: await getComfyInstallProgress(job),
  };
}
