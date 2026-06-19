import path from 'path';
import fs from 'fs/promises';
import { createReadStream, existsSync } from 'fs';
import { randomUUID } from 'crypto';
import { Readable } from 'stream';
import { db, type WorkerNodeRecord } from './db';
import { guardedFetch } from './networkPolicy';
import { clearDurableEncryptedDatasetKeys } from './encryptedDatasetSecrets';
import { getJobRemoteCaptionState } from './remoteCaptionJobs';
import {
  collectDatasetReferences,
  collectSameWorkerRemoteDatasetReferences,
  isRemoteReference,
  resolveConfigPath,
} from './trainingJobTransfer';
import type { Job, Queue, GPUApiResponse, CpuInfo } from '../types';

const REMOTE_DISCOVERY_TIMEOUT_MS = 15_000;

type RemoteRequestInit = RequestInit & { timeoutMs?: number };

export class RemoteClientError extends Error {
  status: number;
  body: string;

  constructor(message: string, status: number, body = '') {
    super(message);
    this.name = 'RemoteClientError';
    this.status = status;
    this.body = body;
  }
}

export const REMOTE_JOB_MISSING_MESSAGE =
  'Remote job was not found on the worker. It may have been deleted there while the central UI was offline.';

type RemoteDiscoveryErrorLogState = {
  signature: string;
  lastLoggedAt: number;
  suppressedCount: number;
};

const REMOTE_DISCOVERY_ERROR_LOG_INTERVAL_MS = 5 * 60 * 1000;

declare global {
  // eslint-disable-next-line no-var
  var __remoteDiscoveryErrorLogState: Map<string, RemoteDiscoveryErrorLogState> | undefined;
}

const remoteDiscoveryErrorLogState =
  globalThis.__remoteDiscoveryErrorLogState ?? new Map<string, RemoteDiscoveryErrorLogState>();

if (!globalThis.__remoteDiscoveryErrorLogState) {
  globalThis.__remoteDiscoveryErrorLogState = remoteDiscoveryErrorLogState;
}

export function isLocalWorker(workerId: string | null | undefined) {
  return !workerId || workerId === 'local';
}

export function normalizeWorkerBaseUrl(baseUrl: string) {
  const trimmed = baseUrl.trim().replace(/\/+$/, '');
  if (!/^https?:\/\//i.test(trimmed)) {
    throw new Error('Worker base URL must start with http:// or https://');
  }
  return trimmed;
}

export async function getRemoteWorker(workerId: string): Promise<WorkerNodeRecord> {
  const worker = await db.workerNodes.findById(workerId);
  if (!worker) throw new Error(`Remote worker not found: ${workerId}`);
  if (!worker.enabled) throw new Error(`Remote worker is disabled: ${worker.name}`);
  if (!worker.api_token) throw new Error(`Remote worker has no API token: ${worker.name}`);
  return {
    ...worker,
    base_url: normalizeWorkerBaseUrl(worker.base_url),
  };
}

function remoteUrl(worker: WorkerNodeRecord, routePath: string) {
  const suffix = routePath.startsWith('/') ? routePath : `/${routePath}`;
  return `${normalizeWorkerBaseUrl(worker.base_url)}${suffix}`;
}

async function remoteRequest(worker: WorkerNodeRecord, routePath: string, init: RemoteRequestInit = {}) {
  const { timeoutMs, ...fetchInit } = init;
  const headers = new Headers(fetchInit.headers);
  headers.set('Authorization', `Bearer ${worker.api_token}`);

  let signal = fetchInit.signal;
  let timeout: ReturnType<typeof setTimeout> | null = null;
  let onAbort: (() => void) | null = null;
  if (timeoutMs != null) {
    const controller = new AbortController();
    signal = controller.signal;
    timeout = setTimeout(() => controller.abort(), timeoutMs);
    onAbort = () => controller.abort(fetchInit.signal?.reason);
    fetchInit.signal?.addEventListener('abort', onAbort, { once: true });
  }

  const url = remoteUrl(worker, routePath);
  const response = await guardedFetch(
    url,
    {
      ...fetchInit,
      headers,
      cache: 'no-store',
      redirect: fetchInit.redirect ?? 'manual',
      signal,
    },
    `remote worker ${worker.name}`,
  ).finally(() => {
    if (timeout) clearTimeout(timeout);
    if (onAbort) fetchInit.signal?.removeEventListener('abort', onAbort);
  });

  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new RemoteClientError(
      `Remote worker ${worker.name} returned ${response.status} for ${routePath}`,
      response.status,
      body,
    );
  }

  return response;
}

export async function remoteFetch(worker: WorkerNodeRecord, routePath: string, init: RemoteRequestInit = {}) {
  return remoteRequest(worker, routePath, init);
}

export async function remoteJson<T>(
  worker: WorkerNodeRecord,
  routePath: string,
  init: RemoteRequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body != null && !headers.has('Content-Type') && !(init.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }
  const response = await remoteRequest(worker, routePath, { ...init, headers });
  return response.json() as Promise<T>;
}

export function withoutRemoteRedirects(init: RequestInit): RequestInit {
  // Prevent 307/308 responses from replaying secret-bearing POST bodies to another URL.
  return { ...init, redirect: 'manual' };
}

export async function remoteProxyFetch(worker: WorkerNodeRecord, routePath: string, headersToForward: Headers) {
  const headers = new Headers();
  const range = headersToForward.get('range');
  if (range) headers.set('range', range);
  return remoteRequest(worker, routePath, { headers });
}

export async function fetchRemoteJob(workerId: string, remoteJobId: string) {
  const worker = await getRemoteWorker(workerId);
  return fetchWorkerJob(worker, remoteJobId);
}

async function fetchWorkerJob(worker: WorkerNodeRecord, remoteJobId: string) {
  return remoteJson<Job | null>(worker, `/api/jobs?id=${encodeURIComponent(remoteJobId)}`);
}

export function isRemoteJobMissingError(error: unknown) {
  return (
    error instanceof RemoteClientError &&
    error.status === 404 &&
    (/job not found/i.test(error.body) || /job not found/i.test(error.message))
  );
}

export function remoteJobMissingUpdate() {
  return {
    remote_job_id: null,
    status: 'error',
    stop: false,
    return_to_queue: false,
    pid: null,
    speed_string: '',
    info: 'Remote job was deleted on the worker.',
    remote_error: REMOTE_JOB_MISSING_MESSAGE,
    remote_sync_at: new Date(),
  };
}

export async function markRemoteJobMissing(localJob: Job) {
  return db.jobs.update(localJob.id, remoteJobMissingUpdate());
}

export async function fetchWorkerJobs(worker: WorkerNodeRecord, jobType?: string | null) {
  const query = new URLSearchParams({ local_only: '1' });
  if (jobType) query.set('job_type', jobType);
  return remoteJson<{ jobs: Job[] }>(worker, `/api/jobs?${query.toString()}`, {
    timeoutMs: REMOTE_DISCOVERY_TIMEOUT_MS,
  });
}

function remoteJobPatch(
  remoteJob: Job,
  workerId: string,
  remoteJobId: string,
  name: string,
  existingLocalJob?: Job | null,
) {
  const patch = {
    name,
    worker_id: workerId,
    remote_job_id: remoteJobId,
    gpu_ids: remoteJob.gpu_ids,
    job_config: remoteJob.job_config,
    status: remoteJob.status,
    stop: remoteJob.stop,
    return_to_queue: remoteJob.return_to_queue,
    step: remoteJob.step,
    info: remoteJob.info,
    speed_string: remoteJob.speed_string,
    queue_position: remoteJob.queue_position,
    pid: null,
    job_type: remoteJob.job_type,
    job_ref: remoteJob.job_ref,
    save_now: remoteJob.save_now ?? false,
    remote_sync_at: new Date(),
    remote_error: null,
  };

  if (existingLocalJob && shouldPreserveLocalJobConfig(existingLocalJob, workerId)) {
    patch.job_config = existingLocalJob.job_config;
  }

  if (existingLocalJob && getJobRemoteCaptionState(existingLocalJob)) {
    patch.name = existingLocalJob.name;
    patch.job_config = existingLocalJob.job_config;
    patch.job_ref = existingLocalJob.job_ref;
  }

  return patch;
}

function shouldPreserveLocalJobConfig(existingLocalJob: Job, workerId: string) {
  try {
    const jobConfig = JSON.parse(existingLocalJob.job_config);
    if (collectSameWorkerRemoteDatasetReferences(jobConfig, workerId).length > 0) return true;
    return collectDatasetReferences(jobConfig).some(ref => {
      if (isRemoteReference(ref.value)) return false;
      return existsSync(resolveConfigPath(ref.value));
    });
  } catch {
    return false;
  }
}

async function resolveRemoteMirrorName(worker: WorkerNodeRecord, remoteJob: Job, localJobId?: string) {
  const baseName = remoteJob.name || remoteJob.id;
  const existing = await db.jobs.findByNameInScope(baseName, null);
  if (!existing || existing.id === localJobId) return baseName;
  if (existing.worker_id === worker.id && existing.remote_job_id === remoteJob.id) return baseName;

  const workerScopedName = `${baseName} (${worker.name})`;
  const scopedExisting = await db.jobs.findByNameInScope(workerScopedName, null);
  if (!scopedExisting || scopedExisting.id === localJobId) return workerScopedName;
  if (scopedExisting.worker_id === worker.id && scopedExisting.remote_job_id === remoteJob.id) return workerScopedName;

  return `${baseName} (${worker.name}, ${remoteJob.id.slice(0, 8)})`;
}

async function upsertRemoteJobMirror(worker: WorkerNodeRecord, remoteJob: Job) {
  const existing = await db.jobs.findByRemoteId(worker.id, remoteJob.id);
  const name = await resolveRemoteMirrorName(worker, remoteJob, existing?.id);
  const patch = remoteJobPatch(remoteJob, worker.id, remoteJob.id, name, existing);

  const synced = existing
    ? await db.jobs.update(existing.id, patch)
    : await db.jobs.create({
        name: patch.name,
        worker_id: patch.worker_id,
        remote_job_id: patch.remote_job_id,
        gpu_ids: patch.gpu_ids,
        job_config: patch.job_config,
        status: patch.status,
        stop: patch.stop,
        return_to_queue: patch.return_to_queue,
        step: patch.step,
        info: patch.info,
        speed_string: patch.speed_string,
        queue_position: patch.queue_position,
        pid: patch.pid,
        job_type: patch.job_type,
        job_ref: patch.job_ref,
        save_now: patch.save_now,
        remote_sync_at: patch.remote_sync_at,
        remote_error: patch.remote_error,
      });

  if (remoteJob.status === 'completed' && !getJobRemoteCaptionState(synced)) {
    await clearDurableEncryptedDatasetKeys(synced.id).catch(error =>
      console.error('Error clearing durable encrypted dataset keys:', error),
    );
  }

  return synced;
}

export async function syncRemoteJob(localJob: Job) {
  if (isLocalWorker(localJob.worker_id) || !localJob.remote_job_id) return localJob;

  try {
    const worker = await getRemoteWorker(localJob.worker_id);
    const remoteJob = await fetchWorkerJob(worker, localJob.remote_job_id);
    if (!remoteJob) {
      return markRemoteJobMissing(localJob);
    }

    const latestLocalJob = await db.jobs.findById(localJob.id);
    const localJobForPatch = latestLocalJob || localJob;
    const name = await resolveRemoteMirrorName(worker, remoteJob, localJob.id);
    const synced = await db.jobs.update(
      localJob.id,
      remoteJobPatch(remoteJob, worker.id, remoteJob.id, name, localJobForPatch),
    );
    if (remoteJob.status === 'completed' && !getJobRemoteCaptionState(synced)) {
      await clearDurableEncryptedDatasetKeys(localJob.id).catch(error =>
        console.error('Error clearing durable encrypted dataset keys:', error),
      );
    }
    return synced;
  } catch (error) {
    if (isRemoteJobMissingError(error)) {
      return markRemoteJobMissing(localJob);
    }
    return db.jobs.update(localJob.id, {
      remote_sync_at: new Date(),
      remote_error: error instanceof Error ? error.message : 'Remote sync failed',
    });
  }
}

export async function syncRemoteJobs(jobs: Job[], alreadySyncedJobIds = new Set<string>()) {
  return Promise.all(jobs.map(job => (alreadySyncedJobIds.has(job.id) ? job : syncRemoteJob(job))));
}

function remoteDiscoveryErrorMessage(workerName: string, error: unknown) {
  if (error instanceof RemoteClientError) {
    const cloudflareTunnelUnavailable =
      error.status === 530 &&
      (/Cloudflare Tunnel error/i.test(error.body) || /Error<\/span>\s*<span>1033<\/span>/i.test(error.body));
    const detail = cloudflareTunnelUnavailable ? 'Cloudflare tunnel is unavailable' : error.message;
    return `Failed to discover jobs for worker ${workerName}: ${detail}`;
  }
  return `Failed to discover jobs for worker ${workerName}: ${
    error instanceof Error ? error.message : 'Remote worker discovery failed'
  }`;
}

function logRemoteDiscoveryError(workerId: string, workerName: string, error: unknown) {
  const message = remoteDiscoveryErrorMessage(workerName, error);
  const signature = error instanceof RemoteClientError ? `${error.status}:${message}` : message;
  const now = Date.now();
  const state = remoteDiscoveryErrorLogState.get(workerId);

  if (!state || state.signature !== signature || now - state.lastLoggedAt >= REMOTE_DISCOVERY_ERROR_LOG_INTERVAL_MS) {
    const suffix = state?.suppressedCount
      ? ` (${state.suppressedCount} repeated discovery error${state.suppressedCount === 1 ? '' : 's'} suppressed)`
      : '';
    console.warn(`${message}${suffix}`);
    remoteDiscoveryErrorLogState.set(workerId, {
      signature,
      lastLoggedAt: now,
      suppressedCount: 0,
    });
    return;
  }

  state.suppressedCount += 1;
}

function clearRemoteDiscoveryErrorLog(workerId: string) {
  const state = remoteDiscoveryErrorLogState.get(workerId);
  if (state?.suppressedCount) {
    console.info(
      `Remote worker discovery recovered for ${workerId} (${state.suppressedCount} repeated discovery error${
        state.suppressedCount === 1 ? '' : 's'
      } suppressed)`,
    );
  }
  remoteDiscoveryErrorLogState.delete(workerId);
}

export async function discoverRemoteJobs(jobType?: string | null) {
  const workers = await db.workerNodes.list({ enabled: true });
  const syncedJobIds = new Set<string>();
  await Promise.all(
    workers.map(async workerRecord => {
      try {
        const worker = await getRemoteWorker(workerRecord.id);
        const data = await fetchWorkerJobs(worker, jobType);
        const syncedJobs = await Promise.all(
          (data.jobs || []).map(remoteJob => upsertRemoteJobMirror(worker, remoteJob)),
        );
        syncedJobs.forEach(job => syncedJobIds.add(job.id));
        clearRemoteDiscoveryErrorLog(workerRecord.id);
      } catch (error) {
        logRemoteDiscoveryError(workerRecord.id, workerRecord.name, error);
      }
    }),
  );
  return syncedJobIds;
}

type FileUploadProgress = {
  loaded: number;
  total: number;
};

type RemoteArchiveImportStatus<T> = {
  uploadID: string;
  status: 'importing' | 'completed' | 'failed';
  result: T | null;
  error: string | null;
};

const DEFAULT_REMOTE_ARCHIVE_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024;

function appendQueryParam(routePath: string, name: string, value?: string | null) {
  if (value == null || value === '') return routePath;
  const separator = routePath.includes('?') ? '&' : '?';
  return `${routePath}${separator}${encodeURIComponent(name)}=${encodeURIComponent(value)}`;
}

function appendQueryParams(routePath: string, params: Record<string, string | number | undefined | null>) {
  return Object.entries(params).reduce(
    (nextPath, [name, value]) => appendQueryParam(nextPath, name, value == null ? null : String(value)),
    routePath,
  );
}

function remoteArchiveUploadChunkBytes() {
  const configured = Number(process.env.AITK_REMOTE_UPLOAD_CHUNK_MB || '');
  if (Number.isFinite(configured) && configured > 0) {
    return Math.max(256 * 1024, Math.floor(configured * 1024 * 1024));
  }
  return DEFAULT_REMOTE_ARCHIVE_UPLOAD_CHUNK_BYTES;
}

function isRemoteArchiveImportStatus<T>(value: unknown): value is RemoteArchiveImportStatus<T> {
  return (
    !!value &&
    typeof value === 'object' &&
    'uploadID' in value &&
    'status' in value &&
    ((value as { status?: unknown }).status === 'importing' ||
      (value as { status?: unknown }).status === 'completed' ||
      (value as { status?: unknown }).status === 'failed')
  );
}

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForRemoteArchiveImport<T>(worker: WorkerNodeRecord, routePath: string, uploadID: string) {
  let firstPoll = true;
  while (true) {
    if (firstPoll) {
      firstPoll = false;
    } else {
      await sleep(1000);
    }
    const status = await remoteJson<RemoteArchiveImportStatus<T>>(
      worker,
      appendQueryParams(routePath, {
        aitk_upload: 'status',
        uploadID,
      }),
    );
    if (status.status === 'completed') {
      if (status.result == null) {
        throw new Error('Remote archive import completed without a result.');
      }
      return status.result;
    }
    if (status.status === 'failed') {
      throw new Error(status.error || 'Remote archive import failed.');
    }
  }
}

async function remoteZipFileJson<T>(
  worker: WorkerNodeRecord,
  routePath: string,
  options: {
    filePath: string;
    fileName?: string;
    onProgress?: (progress: FileUploadProgress) => void;
    backgroundComplete?: boolean;
  },
) {
  const fileStat = await fs.stat(options.filePath);
  const fileName = options.fileName || path.basename(options.filePath);
  let uploadedFileBytes = 0;
  const reportProgress = () => {
    options.onProgress?.({
      loaded: Math.min(uploadedFileBytes, fileStat.size),
      total: fileStat.size,
    });
  };

  const uploadID = randomUUID();
  const chunkBytes = remoteArchiveUploadChunkBytes();
  const chunksTotal = Math.max(1, Math.ceil(fileStat.size / chunkBytes));
  reportProgress();

  for (let chunkIndex = 0; chunkIndex < chunksTotal; chunkIndex += 1) {
    const start = chunkIndex * chunkBytes;
    const end = Math.min(fileStat.size, start + chunkBytes) - 1;
    const chunkSize = Math.max(0, end - start + 1);
    const body =
      chunkSize > 0
        ? (Readable.toWeb(createReadStream(options.filePath, { start, end })) as unknown as BodyInit)
        : (Readable.toWeb(Readable.from([Buffer.alloc(0)])) as unknown as BodyInit);

    await remoteJson(
      worker,
      appendQueryParams(routePath, {
        aitk_upload: 'chunk',
        uploadID,
        chunkIndex,
        chunksTotal,
      }),
      {
        method: 'POST',
        body,
        headers: {
          'Content-Type': 'application/octet-stream',
          'Content-Length': String(chunkSize),
          'X-AITK-File-Name': fileName,
        },
        duplex: 'half',
      } as RequestInit & { duplex: 'half' },
    );

    uploadedFileBytes = Math.min(fileStat.size, uploadedFileBytes + chunkSize);
    reportProgress();
  }

  const completeResult = await remoteJson<T | RemoteArchiveImportStatus<T>>(
    worker,
    appendQueryParams(routePath, {
      aitk_upload: 'complete',
      uploadID,
      chunksTotal,
      background: options.backgroundComplete ? '1' : null,
    }),
    {
      method: 'POST',
      headers: {
        'X-AITK-File-Name': fileName,
      },
    },
  );

  if (options.backgroundComplete && isRemoteArchiveImportStatus<T>(completeResult)) {
    if (completeResult.status === 'completed') {
      if (completeResult.result == null) {
        throw new Error('Remote archive import completed without a result.');
      }
      return completeResult.result;
    }
    if (completeResult.status === 'failed') {
      throw new Error(completeResult.error || 'Remote archive import failed.');
    }
    return waitForRemoteArchiveImport<T>(worker, routePath, uploadID);
  }

  return completeResult as T;
}

export async function uploadBundleToWorker(
  worker: WorkerNodeRecord,
  zipPath: string,
  gpuIds: string,
  onProgress?: (progress: FileUploadProgress) => void,
) {
  return remoteZipFileJson<{ job: Job; warnings: string[] }>(
    worker,
    appendQueryParam('/api/jobs/import', 'gpu_ids', gpuIds),
    {
      filePath: zipPath,
      fileName: path.basename(zipPath),
      onProgress,
    },
  );
}

export async function fetchWorkerHealth(worker: WorkerNodeRecord) {
  return remoteJson<{
    ok: boolean;
    app: string;
    cloudflared: unknown;
    ollama?: unknown;
    timestamp: string;
  }>(worker, '/api/remote/health');
}

export async function uploadDatasetArchiveToWorker(
  worker: WorkerNodeRecord,
  zipPath: string,
  preferredName?: string,
  onProgress?: (progress: FileUploadProgress) => void,
) {
  return remoteZipFileJson<{
    dataset: { name: string; encrypted: boolean; path?: string };
    path: string;
    renamed: boolean;
  }>(worker, appendQueryParam('/api/datasets/import-archive', 'preferredName', preferredName), {
    filePath: zipPath,
    fileName: path.basename(zipPath),
    onProgress,
    backgroundComplete: true,
  });
}

export async function fetchWorkerGpu(worker: WorkerNodeRecord) {
  return remoteJson<GPUApiResponse>(worker, '/api/gpu');
}

export async function fetchWorkerCpu(worker: WorkerNodeRecord) {
  return remoteJson<CpuInfo>(worker, '/api/cpu');
}

export async function fetchWorkerQueues(worker: WorkerNodeRecord) {
  return remoteJson<{ queues: Queue[] }>(worker, '/api/queue');
}
