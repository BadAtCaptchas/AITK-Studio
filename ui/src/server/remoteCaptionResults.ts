import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import { db, type JobUpdateInput } from './db';
import {
  getKeyForRequiredDataset,
  normalizeEncryptedKeyMap,
} from './encryptedDatasets';
import {
  clearDurableEncryptedDatasetKeys,
  getDurableEncryptedDatasetKeys,
} from './encryptedDatasetSecrets';
import { areProjectsEnabled, getDatasetsRoot } from './settings';
import { resolveDatasetDirectoryInsideRoot } from './remoteCaptionSecurity';
import {
  extractZipSafely,
  getExtractedDatasetPath,
  readDatasetExportManifest,
} from './datasetTransfer';
import {
  getRemoteWorker,
  isLocalWorker,
  remoteFetch,
  remoteJson,
  syncRemoteJob,
} from './remoteClient';
import {
  getJobRemoteCaptionState,
  getRemoteCaptionState,
  patchRemoteCaptionState,
  type RemoteCaptionState,
} from './remoteCaptionJobs';
import {
  mergeEncryptedCaptionDataset,
  mergePlainCaptionDataset,
} from './remoteCaptionMerge';
import {
  nextAvailablePath,
} from './trainingJobTransfer';
import type { Job } from '../types';

declare global {
  // eslint-disable-next-line no-var
  var __aitkRemoteCaptionResultSyncs: Set<string> | undefined;
}

function activeSyncs() {
  if (!globalThis.__aitkRemoteCaptionResultSyncs) {
    globalThis.__aitkRemoteCaptionResultSyncs = new Set();
  }
  return globalThis.__aitkRemoteCaptionResultSyncs;
}

function nowIso() {
  return new Date().toISOString();
}

const DOWNLOAD_STALE_MS = 10 * 60 * 1000;

function isFreshDownloadInProgress(state: RemoteCaptionState) {
  if (state.downloadStatus !== 'downloading') return false;
  if (!state.downloadStartedAt) return false;
  const startedAt = Date.parse(state.downloadStartedAt);
  return Number.isFinite(startedAt) && Date.now() - startedAt < DOWNLOAD_STALE_MS;
}

async function writeResponseBodyToFile(response: Response, targetPath: string) {
  if (!response.body) throw new Error('Remote worker returned an empty dataset archive');
  await fsp.mkdir(path.dirname(targetPath), { recursive: true });
  await pipeline(Readable.fromWeb(response.body as any), fs.createWriteStream(targetPath));
}

async function updateRemoteCaptionState(job: Job, patch: Partial<RemoteCaptionState>, extraJobPatch: JobUpdateInput = {}) {
  const jobConfig = JSON.parse(job.job_config);
  const nextConfig = patchRemoteCaptionState(jobConfig, patch);
  return db.jobs.update(job.id, {
    ...extraJobPatch,
    job_config: JSON.stringify(nextConfig),
    remote_sync_at: new Date(),
  });
}


async function durableKeyForRemoteCaption(job: Job, state: RemoteCaptionState) {
  const durableKeys = await getDurableEncryptedDatasetKeys(job.id);
  const keyMap = normalizeEncryptedKeyMap(durableKeys);
  return getKeyForRequiredDataset(keyMap, {
    path: state.originalDatasetPath,
    name: state.originalDatasetName,
  });
}

async function importDatasetFallback(datasetsRoot: string, sourceDatasetPath: string, preferredName: string) {
  const targetPath = await nextAvailablePath(datasetsRoot, preferredName);
  await fsp.cp(sourceDatasetPath, targetPath, { recursive: true, force: false, errorOnExist: true });
  return targetPath;
}

async function mergeRemoteCaptionDataset(
  job: Job,
  state: RemoteCaptionState,
  sourceDatasetPath: string,
  manifestName: string,
) {
  const originalDatasetPath = path.resolve(state.originalDatasetPath);
  const datasetsRoot = await getDatasetsRoot();
  const originalExists = fs.existsSync(originalDatasetPath) && fs.statSync(originalDatasetPath).isDirectory();
  let realOriginalDatasetPath = originalDatasetPath;

  if (originalExists) {
    realOriginalDatasetPath = await resolveDatasetDirectoryInsideRoot(originalDatasetPath, datasetsRoot);
  }

  if (!originalExists) {
    const importedFallbackPath = await importDatasetFallback(
      datasetsRoot,
      sourceDatasetPath,
      manifestName || state.originalDatasetName,
    );
    return {
      jobPatch: { job_ref: importedFallbackPath },
      statePatch: { importedFallbackPath },
      mergeStats: { copied: 0, skipped: 0, fallback: true },
    };
  }

  if (state.encrypted) {
    const keyB64 = await durableKeyForRemoteCaption(job, state);
    if (!keyB64) {
      throw new Error('Durable encrypted dataset key is required to merge remote captions');
    }
    const mergeStats = await mergeEncryptedCaptionDataset(sourceDatasetPath, realOriginalDatasetPath, {
      keyB64,
      recaption: state.recaption,
    });
    return { jobPatch: {}, statePatch: {}, mergeStats };
  }

  const mergeStats = await mergePlainCaptionDataset(sourceDatasetPath, realOriginalDatasetPath, {
    captionExtension: state.captionExtension,
    recaption: state.recaption,
  });
  return { jobPatch: {}, statePatch: {}, mergeStats };
}

async function cleanupRemoteDataset(job: Job, state: RemoteCaptionState) {
  if (!state.remoteDatasetName || isLocalWorker(job.worker_id)) return null;
  try {
    const worker = await getRemoteWorker(job.worker_id);
    await remoteJson(worker, '/api/datasets/delete', {
      method: 'POST',
      body: JSON.stringify({ name: state.remoteDatasetName }),
    });
    return null;
  } catch (error) {
    return error instanceof Error ? error.message : 'Failed to delete remote staging dataset';
  }
}

export async function syncRemoteCaptionResultForJob(
  job: Job,
  options: { force?: boolean; retryFailed?: boolean } = {},
) {
  if (isLocalWorker(job.worker_id) || !job.remote_job_id) return job;
  const state = getJobRemoteCaptionState(job);
  if (!state) return job;
  if (state.downloadStatus === 'merged' && !options.force) return job;
  if (state.downloadStatus === 'failed' && !options.force && !options.retryFailed) return job;
  if (isFreshDownloadInProgress(state) && !options.force) return job;
  if (job.status !== 'completed') return job;
  if (!state.remoteDatasetName) return job;

  const syncs = activeSyncs();
  if (syncs.has(job.id)) return job;
  syncs.add(job.id);

  const datasetsRoot = await getDatasetsRoot();
  let workRoot: string | null = null;
  let workingJob = job;

  try {
    const downloadStartedAt = nowIso();
    workingJob = await updateRemoteCaptionState(workingJob, {
      downloadStatus: 'downloading',
      completedAt: state.completedAt || nowIso(),
      downloadStartedAt,
      lastError: null,
    });

    await fsp.mkdir(datasetsRoot, { recursive: true });
    workRoot = await fsp.mkdtemp(path.join(datasetsRoot, `.aitk-remote-caption-result-${job.id}-`));
    const zipPath = path.join(workRoot, 'dataset.zip');
    const extractRoot = path.join(workRoot, 'extract');

    const worker = await getRemoteWorker(workingJob.worker_id);
    const remoteResponse = await remoteFetch(worker, '/api/datasets/export', {
      method: 'POST',
      body: JSON.stringify({ datasetName: state.remoteDatasetName }),
      headers: { 'Content-Type': 'application/json' },
    });
    await writeResponseBodyToFile(remoteResponse, zipPath);
    await extractZipSafely(zipPath, extractRoot);

    const manifest = await readDatasetExportManifest(extractRoot);
    const datasetSource = getExtractedDatasetPath(extractRoot, manifest.dataset.archivePath);
    if (!fs.existsSync(datasetSource) || !fs.statSync(datasetSource).isDirectory()) {
      throw new Error('Remote caption dataset payload missing from archive');
    }

    const currentState = getRemoteCaptionState(JSON.parse(workingJob.job_config)) || state;
    const merged = await mergeRemoteCaptionDataset(
      workingJob,
      currentState,
      datasetSource,
      manifest.dataset.name,
    );
    const cleanupError = await cleanupRemoteDataset(workingJob, currentState);
    const mergedAt = nowIso();
    const importedFallbackPath = 'importedFallbackPath' in merged.statePatch ? merged.statePatch.importedFallbackPath : '';
    const usedFallback = 'fallback' in merged.mergeStats && merged.mergeStats.fallback === true;
    const updated = await updateRemoteCaptionState(
      workingJob,
      {
        ...merged.statePatch,
        downloadStatus: 'merged',
        downloadedAt: mergedAt,
        mergedAt,
        downloadStartedAt: undefined,
        cleanupError,
        lastError: null,
      },
      {
        ...merged.jobPatch,
        remote_error: cleanupError,
        info: usedFallback
          ? `Remote captions imported as ${path.basename(importedFallbackPath || '')}`
          : `Remote captions merged (${merged.mergeStats.copied} copied, ${merged.mergeStats.skipped} skipped)`,
      },
    );
    await clearDurableEncryptedDatasetKeys(workingJob.id).catch(error =>
      console.error('Error clearing durable encrypted dataset keys:', error),
    );
    return updated;
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Remote caption result download failed';
    const latest = await db.jobs.findById(workingJob.id).catch(() => null);
    const latestState = latest ? getJobRemoteCaptionState(latest) : null;
    if (latest && latestState?.downloadStatus === 'merged') {
      return latest;
    }
    return updateRemoteCaptionState(workingJob, {
      downloadStatus: 'failed',
      downloadStartedAt: undefined,
      lastError: message,
    }, {
      info: `Remote caption sync failed: ${message}`,
      remote_error: message,
    });
  } finally {
    syncs.delete(job.id);
    if (workRoot) {
      await fsp.rm(workRoot, { recursive: true, force: true }).catch(() => undefined);
    }
  }
}

export async function syncRemoteCaptionResults() {
  const jobs = await db.jobs.list({ job_type: 'caption' });
  const includeProjectJobs = await areProjectsEnabled();
  const results: Job[] = [];
  for (const job of jobs) {
    if (job.project_id && !includeProjectJobs) continue;
    if (isLocalWorker(job.worker_id) || !job.remote_job_id || !getJobRemoteCaptionState(job)) continue;
    const synced = await syncRemoteJob(job);
    results.push(await syncRemoteCaptionResultForJob(synced));
  }
  return results;
}
