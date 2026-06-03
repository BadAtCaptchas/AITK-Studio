import path from 'path';
import fs from 'fs/promises';
import net from 'net';
import { db, type WorkerNodeRecord } from './db';
import type { Job, Queue, GPUApiResponse, CpuInfo } from '@/types';

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
  const trimmed = baseUrl.trim();
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    throw new Error('Worker base URL must be a valid http:// or https:// URL');
  }

  if (!/^https?:$/i.test(parsed.protocol)) {
    throw new Error('Worker base URL must start with http:// or https://');
  }

  if (parsed.username || parsed.password) {
    throw new Error('Worker base URL cannot include credentials');
  }

  if ((parsed.pathname && parsed.pathname !== '/') || parsed.search || parsed.hash) {
    throw new Error('Worker base URL must be an origin without path, query, or fragment');
  }

  const host = parsed.hostname.toLowerCase();
  const ipType = net.isIP(host);
  if (host === 'localhost' || host.endsWith('.localhost')) {
    throw new Error('Worker base URL cannot target localhost');
  }
  if ((ipType === 4 && isPrivateIPv4(host)) || (ipType === 6 && isPrivateIPv6(host))) {
    throw new Error('Worker base URL cannot target private or loopback IPs');
  }

  parsed.pathname = '';
  return parsed.toString().replace(/\/$/, '');
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
  return new URL(suffix, `${normalizeWorkerBaseUrl(worker.base_url)}/`).toString();
}

function isPrivateIPv4(ip: string) {
  const octets = ip.split('.').map(part => Number(part));
  if (octets.length !== 4 || octets.some(octet => Number.isNaN(octet))) return false;
  const [a, b] = octets;
  return (
    a === 10 ||
    a === 127 ||
    (a === 169 && b === 254) ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168)
  );
}

function isPrivateIPv6(ip: string) {
  const normalized = ip.toLowerCase();
  return normalized === '::1' || normalized.startsWith('fc') || normalized.startsWith('fd') || normalized.startsWith('fe80:');
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
  return remoteJson<Job | null>(worker, `/api/jobs?id=${encodeURIComponent(remoteJobId)}`);
}

export async function syncRemoteJob(localJob: Job) {
  if (isLocalWorker(localJob.worker_id) || !localJob.remote_job_id) return localJob;

  try {
    const remoteJob = await fetchRemoteJob(localJob.worker_id, localJob.remote_job_id);
    if (!remoteJob) {
      return db.jobs.update(localJob.id, {
        remote_error: 'Remote job was not found on the worker.',
        remote_sync_at: new Date(),
      });
    }

    return db.jobs.update(localJob.id, {
      name: remoteJob.name,
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
      remote_sync_at: new Date(),
      remote_error: null,
    });
  } catch (error) {
    return db.jobs.update(localJob.id, {
      remote_sync_at: new Date(),
      remote_error: error instanceof Error ? error.message : 'Remote sync failed',
    });
  }
}

export async function syncRemoteJobs(jobs: Job[]) {
  return Promise.all(jobs.map(job => syncRemoteJob(job)));
}

export async function uploadBundleToWorker(worker: WorkerNodeRecord, zipPath: string, gpuIds: string) {
  const form = new FormData();
  const buffer = await fs.readFile(zipPath);
  const blob = new Blob([buffer], { type: 'application/zip' });
  form.append('file', blob, path.basename(zipPath));
  form.append('gpu_ids', gpuIds);
  return remoteJson<{ job: Job; warnings: string[] }>(worker, '/api/jobs/import', {
    method: 'POST',
    body: form,
  });
}

export async function fetchWorkerHealth(worker: WorkerNodeRecord) {
  return remoteJson<{
    ok: boolean;
    app: string;
    cloudflared: unknown;
    timestamp: string;
  }>(worker, '/api/remote/health');
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
