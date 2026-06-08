import path from 'path';
import fs from 'fs/promises';
import { createReadStream, existsSync } from 'fs';
import { randomUUID } from 'crypto';
import { Readable } from 'stream';
import { db, type WorkerNodeRecord } from './db';
import { clearDurableEncryptedDatasetKeys } from './encryptedDatasetSecrets';
import { getJobRemoteCaptionState } from './remoteCaptionJobs';
import {
  collectDatasetReferences,
  collectSameWorkerRemoteDatasetReferences,
  isRemoteReference,
  resolveConfigPath,
} from './trainingJobTransfer';
import type { Job, Queue, GPUApiResponse, CpuInfo } from '../types';

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

async function remoteRequest(worker: WorkerNodeRecord, routePath: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers);
  headers.set('Authorization', `Bearer ${worker.api_token}`);

  const response = await fetch(remoteUrl(worker, routePath), {
    ...init,
    headers,
    cache: 'no-store',
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

export async function remoteFetch(worker: WorkerNodeRecord, routePath: string, init: RequestInit = {}) {
  return remoteRequest(worker, routePath, init);
}

export async function remoteJson<T>(worker: WorkerNodeRecord, routePath: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body != null && !headers.has('Content-Type') && !(init.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }
  const response = await remoteRequest(worker, routePath, { ...init, headers });
  return response.json() as Promise<T>;
}

export async function remoteProxyFetch(
  worker: WorkerNodeRecord,
  routePath: string,
  headersToForward: Headers,
) {
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

export async function fetchWorkerJobs(worker: WorkerNodeRecord, jobType?: string | null) {
  const query = jobType ? `?job_type=${encodeURIComponent(jobType)}` : '';
  return remoteJson<{ jobs: Job[] }>(worker, `/api/jobs${query}`);
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
  const existing = await db.jobs.findByName(baseName);
  if (!existing || existing.id === localJobId) return baseName;
  if (existing.worker_id === worker.id && existing.remote_job_id === remoteJob.id) return baseName;

  const workerScopedName = `${baseName} (${worker.name})`;
  const scopedExisting = await db.jobs.findByName(workerScopedName);
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
      return db.jobs.update(localJob.id, {
        remote_error: 'Remote job was not found on the worker.',
        remote_sync_at: new Date(),
      });
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
    return db.jobs.update(localJob.id, {
      remote_sync_at: new Date(),
      remote_error: error instanceof Error ? error.message : 'Remote sync failed',
    });
  }
}

export async function syncRemoteJobs(jobs: Job[], alreadySyncedJobIds = new Set<string>()) {
  return Promise.all(jobs.map(job => (alreadySyncedJobIds.has(job.id) ? job : syncRemoteJob(job))));
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
      } catch (error) {
        console.error(`Failed to discover jobs for worker ${workerRecord.name}:`, error);
      }
    }),
  );
  return syncedJobIds;
}

type FileUploadProgress = {
  loaded: number;
  total: number;
};

function escapeMultipartValue(value: string) {
  return value.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\r|\n/g, '_');
}

async function remoteMultipartFileJson<T>(
  worker: WorkerNodeRecord,
  routePath: string,
  options: {
    filePath: string;
    fileFieldName: string;
    fileName?: string;
    contentType?: string;
    fields?: Record<string, string | undefined | null>;
    onProgress?: (progress: FileUploadProgress) => void;
  },
) {
  const fileStat = await fs.stat(options.filePath);
  const fileName = options.fileName || path.basename(options.filePath);
  const contentType = options.contentType || 'application/octet-stream';
  const boundary = `----aitk-${randomUUID()}`;
  const parts: Buffer[] = [];

  Object.entries(options.fields || {}).forEach(([name, value]) => {
    if (value == null) return;
    parts.push(
      Buffer.from(
        `--${boundary}\r\nContent-Disposition: form-data; name="${escapeMultipartValue(name)}"\r\n\r\n${value}\r\n`,
      ),
    );
  });

  const fileHeader = Buffer.from(
    `--${boundary}\r\nContent-Disposition: form-data; name="${escapeMultipartValue(
      options.fileFieldName,
    )}"; filename="${escapeMultipartValue(fileName)}"\r\nContent-Type: ${contentType}\r\n\r\n`,
  );
  const fileFooter = Buffer.from(`\r\n--${boundary}--\r\n`);
  const contentLength =
    parts.reduce((total, part) => total + part.length, 0) + fileHeader.length + fileStat.size + fileFooter.length;

  let uploadedFileBytes = 0;
  const reportProgress = () => {
    options.onProgress?.({
      loaded: Math.min(uploadedFileBytes, fileStat.size),
      total: fileStat.size,
    });
  };

  async function* multipartBody() {
    for (const part of parts) {
      yield part;
    }

    yield fileHeader;
    reportProgress();

    for await (const chunk of createReadStream(options.filePath)) {
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      uploadedFileBytes += buffer.length;
      reportProgress();
      yield buffer;
    }

    yield fileFooter;
    reportProgress();
  }

  return remoteJson<T>(worker, routePath, {
    method: 'POST',
    body: Readable.toWeb(Readable.from(multipartBody())) as unknown as BodyInit,
    headers: {
      'Content-Type': `multipart/form-data; boundary=${boundary}`,
      'Content-Length': String(contentLength),
    },
    duplex: 'half',
  } as RequestInit & { duplex: 'half' });
}

export async function uploadBundleToWorker(
  worker: WorkerNodeRecord,
  zipPath: string,
  gpuIds: string,
  onProgress?: (progress: FileUploadProgress) => void,
) {
  return remoteMultipartFileJson<{ job: Job; warnings: string[] }>(worker, '/api/jobs/import', {
    filePath: zipPath,
    fileFieldName: 'file',
    fileName: path.basename(zipPath),
    contentType: 'application/zip',
    fields: { gpu_ids: gpuIds },
    onProgress,
  });
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
  return remoteMultipartFileJson<{
    dataset: { name: string; encrypted: boolean; path?: string };
    path: string;
    renamed: boolean;
  }>(worker, '/api/datasets/import-archive', {
    filePath: zipPath,
    fileFieldName: 'file',
    fileName: path.basename(zipPath),
    contentType: 'application/zip',
    fields: { preferredName },
    onProgress,
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
