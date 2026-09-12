import { isLegacyScopedRecord } from '../utils/obsoleteWorkspaceGuard';
import { ownResponseBody, readBoundedResponseText } from './responseBody';
import path from 'path';
import fs from 'fs/promises';
import { createReadStream, existsSync } from 'fs';
import { randomUUID } from 'crypto';
import { Readable } from 'stream';
import { db, type WorkerNodeRecord } from './db';
import {
  getOfflineBypassHostnames,
  guardedFetch,
  isLocalPrivateIp,
  isOfflineModeEnabled,
  normalizeHostname,
  OfflineModeError,
} from './networkPolicy';
import { activeOperationForResource } from './operations';
import { clearDurableEncryptedDatasetKeys, getDurableKeySnapshot } from './encryptedDatasetSecrets';
import { getJobRemoteCaptionState } from './remoteCaptionJobs';
import {
  collectDatasetReferences,
  collectSameWorkerRemoteDatasetReferences,
  isRemoteReference,
  resolveConfigPath,
} from './trainingJobTransfer';
import type { Job, Queue, GPUApiResponse, CpuInfo } from '../types';

const REMOTE_BACKGROUND_POLL_TIMEOUT_MS = 5_000;
const REMOTE_BACKGROUND_COOLDOWN_MS = 30_000;
const REMOTE_BACKGROUND_LOG_INTERVAL_MS = 5 * 60 * 1000;

type RemoteRequestInit = RequestInit & { timeoutMs?: number; headerTimeoutMs?: number; bodyIdleMs?: number; acceptHttpStatus?: boolean };

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

type RemoteBackgroundPollLogState = {
  signature: string;
  lastLoggedAt: number;
  suppressedCount: number;
};

type RemoteBackgroundPollCooldownState = {
  reason: string;
  until: number;
};

type RemoteBackgroundPollEligibility =
  | {
      allowed: true;
    }
  | {
      allowed: false;
      reason: string;
    };

type RemoteBackgroundPollResult<T> =
  | {
      skipped: false;
      value: T;
    }
  | {
      skipped: true;
      reason: string;
    };

declare global {
  var __remoteDiscoveryErrorLogState: Map<string, RemoteDiscoveryErrorLogState> | undefined;
  var __remoteBackgroundPollLogState: Map<string, RemoteBackgroundPollLogState> | undefined;
  var __remoteBackgroundPollCooldownState: Map<string, RemoteBackgroundPollCooldownState> | undefined;
}

const remoteDiscoveryErrorLogState =
  globalThis.__remoteDiscoveryErrorLogState ?? new Map<string, RemoteDiscoveryErrorLogState>();

if (!globalThis.__remoteDiscoveryErrorLogState) {
  globalThis.__remoteDiscoveryErrorLogState = remoteDiscoveryErrorLogState;
}

const remoteBackgroundPollLogState =
  globalThis.__remoteBackgroundPollLogState ?? new Map<string, RemoteBackgroundPollLogState>();

if (!globalThis.__remoteBackgroundPollLogState) {
  globalThis.__remoteBackgroundPollLogState = remoteBackgroundPollLogState;
}

const remoteBackgroundPollCooldownState =
  globalThis.__remoteBackgroundPollCooldownState ?? new Map<string, RemoteBackgroundPollCooldownState>();

if (!globalThis.__remoteBackgroundPollCooldownState) {
  globalThis.__remoteBackgroundPollCooldownState = remoteBackgroundPollCooldownState;
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

function workerPollKey(worker: Pick<WorkerNodeRecord, 'id' | 'base_url'>) {
  return worker.id || normalizeWorkerBaseUrl(worker.base_url);
}

function workerHostname(worker: Pick<WorkerNodeRecord, 'base_url'>) {
  try {
    return normalizeHostname(new URL(normalizeWorkerBaseUrl(worker.base_url)).hostname);
  } catch {
    return '';
  }
}

function isLocalHostname(hostname: string) {
  return hostname === 'localhost' || hostname.endsWith('.localhost');
}

async function isWorkerAllowedInOfflineMode(worker: WorkerNodeRecord) {
  if (worker.offline_bypass_enabled) return true;
  const hostname = workerHostname(worker);
  if (!hostname) return false;
  if (isLocalHostname(hostname) || isLocalPrivateIp(hostname)) return true;
  const allowedHosts = await getOfflineBypassHostnames();
  return allowedHosts.has(hostname);
}

function logRemoteBackgroundPoll(worker: WorkerNodeRecord, feature: string, message: string, level: 'info' | 'warn') {
  const key = `${workerPollKey(worker)}:${feature}`;
  const signature = message;
  const now = Date.now();
  const state = remoteBackgroundPollLogState.get(key);

  if (!state || state.signature !== signature || now - state.lastLoggedAt >= REMOTE_BACKGROUND_LOG_INTERVAL_MS) {
    const suffix = state?.suppressedCount
      ? ` (${state.suppressedCount} repeated background poll message${state.suppressedCount === 1 ? '' : 's'} suppressed)`
      : '';
    const log = level === 'warn' ? console.warn : console.info;
    log(`${message}${suffix}`);
    remoteBackgroundPollLogState.set(key, {
      signature,
      lastLoggedAt: now,
      suppressedCount: 0,
    });
    return;
  }

  state.suppressedCount += 1;
}

function clearRemoteBackgroundPollLog(worker: WorkerNodeRecord, feature: string) {
  const key = `${workerPollKey(worker)}:${feature}`;
  const state = remoteBackgroundPollLogState.get(key);
  if (state?.suppressedCount) {
    console.info(
      `Remote background poll recovered for ${worker.name} ${feature} (${state.suppressedCount} repeated message${
        state.suppressedCount === 1 ? '' : 's'
      } suppressed)`,
    );
  }
  remoteBackgroundPollLogState.delete(key);
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error || 'Remote worker polling failed');
}

function getErrorName(error: unknown) {
  if (error instanceof Error && error.name) return error.name;
  if (typeof error === 'object' && error !== null) {
    const name = (error as Record<string, unknown>).name;
    return typeof name === 'string' ? name : '';
  }
  return '';
}

function getNestedErrorCode(error: unknown): string | null {
  if (!error || typeof error !== 'object') return null;
  const record = error as Record<string, unknown>;
  const code = record.code;
  if (typeof code === 'string' && code) return code;
  return getNestedErrorCode(record.cause);
}

function isCloudflareTunnelUnavailable(error: unknown) {
  return (
    error instanceof RemoteClientError &&
    error.status === 530 &&
    (/Cloudflare Tunnel error/i.test(error.body) || /Error<\/span>\s*<span>1033<\/span>/i.test(error.body))
  );
}

const TRANSIENT_REMOTE_ERROR_CODES = new Set([
  'EAI_AGAIN',
  'ENOTFOUND',
  'ECONNREFUSED',
  'ECONNRESET',
  'ETIMEDOUT',
  'EHOSTUNREACH',
  'ENETUNREACH',
  'UND_ERR_CONNECT_TIMEOUT',
  'UND_ERR_HEADERS_TIMEOUT',
  'UND_ERR_SOCKET',
]);

const TRANSIENT_REMOTE_HTTP_STATUSES = new Set([408, 429, 502, 503, 504, 522, 523, 524, 530]);

export function isTransientRemoteBackgroundPollError(error: unknown) {
  if (isCloudflareTunnelUnavailable(error)) return true;
  if (error instanceof RemoteClientError) return TRANSIENT_REMOTE_HTTP_STATUSES.has(error.status);
  if (error instanceof OfflineModeError) return /DNS lookup failed/i.test(error.message);
  if (getErrorName(error) === 'AbortError' || /aborted|timed out|timeout/i.test(getErrorMessage(error))) return true;
  const code = getNestedErrorCode(error);
  return !!code && TRANSIENT_REMOTE_ERROR_CODES.has(code);
}

function transientRemoteErrorReason(error: unknown) {
  if (isCloudflareTunnelUnavailable(error)) return 'Cloudflare tunnel is unavailable';
  const code = getNestedErrorCode(error);
  const message = getErrorMessage(error);
  return code ? `${code}: ${message}` : message;
}

export async function getRemoteBackgroundPollEligibility(
  worker: WorkerNodeRecord,
  feature = 'background polling',
): Promise<RemoteBackgroundPollEligibility> {
  const key = workerPollKey(worker);
  const now = Date.now();
  const cooldown = remoteBackgroundPollCooldownState.get(key);
  if (cooldown && now < cooldown.until) {
    const seconds = Math.max(1, Math.ceil((cooldown.until - now) / 1000));
    const reason = `cooling down for ${seconds}s after ${cooldown.reason}`;
    logRemoteBackgroundPoll(worker, feature, `Skipping ${feature} for worker ${worker.name}: ${reason}.`, 'info');
    return { allowed: false, reason };
  }
  if (cooldown) {
    remoteBackgroundPollCooldownState.delete(key);
  }

  if (!(await isOfflineModeEnabled())) return { allowed: true };
  if (await isWorkerAllowedInOfflineMode(worker)) return { allowed: true };

  const reason = 'offline mode is enabled and this worker is not allowed for offline polling';
  logRemoteBackgroundPoll(worker, feature, `Skipping ${feature} for worker ${worker.name}: ${reason}.`, 'info');
  return { allowed: false, reason };
}

export function noteRemoteBackgroundPollSuccess(worker: WorkerNodeRecord, feature = 'background polling') {
  remoteBackgroundPollCooldownState.delete(workerPollKey(worker));
  clearRemoteBackgroundPollLog(worker, feature);
}

export function noteRemoteBackgroundPollFailure(
  worker: WorkerNodeRecord,
  error: unknown,
  feature = 'background polling',
) {
  if (!isTransientRemoteBackgroundPollError(error)) return false;
  const reason = transientRemoteErrorReason(error);
  remoteBackgroundPollCooldownState.set(workerPollKey(worker), {
    reason,
    until: Date.now() + REMOTE_BACKGROUND_COOLDOWN_MS,
  });
  logRemoteBackgroundPoll(
    worker,
    feature,
    `Remote ${feature} for worker ${worker.name} failed; background polling will retry after cooldown: ${reason}.`,
    'warn',
  );
  return true;
}

export async function runRemoteBackgroundPoll<T>(
  worker: WorkerNodeRecord,
  feature: string,
  task: () => Promise<T>,
): Promise<RemoteBackgroundPollResult<T>> {
  const eligibility = await getRemoteBackgroundPollEligibility(worker, feature);
  if ('reason' in eligibility) return { skipped: true, reason: eligibility.reason };

  try {
    const value = await task();
    noteRemoteBackgroundPollSuccess(worker, feature);
    return { skipped: false, value };
  } catch (error) {
    if (noteRemoteBackgroundPollFailure(worker, error, feature)) {
      return { skipped: true, reason: transientRemoteErrorReason(error) };
    }
    throw error;
  }
}

export function resetRemoteBackgroundPollingStateForTests() {
  remoteBackgroundPollLogState.clear();
  remoteBackgroundPollCooldownState.clear();
}

async function remoteRequest(worker: WorkerNodeRecord, routePath: string, init: RemoteRequestInit = {}) {
  const { timeoutMs, headerTimeoutMs = 30_000, bodyIdleMs = 60_000, acceptHttpStatus = false, ...fetchInit } = init;
  const headers = new Headers(fetchInit.headers);
  headers.set('Authorization', `Bearer ${worker.api_token}`);

  fetchInit.signal?.throwIfAborted();
  const controller = new AbortController();
  const signal = controller.signal;
  const duration = timeoutMs ?? 60 * 60 * 1000;
  if (!Number.isFinite(duration) || duration <= 0) throw new Error('Invalid remote request timeout');
  const timeout = setTimeout(() => controller.abort(new Error('Remote request deadline exceeded')), duration);
  const headerTimeout = setTimeout(() => controller.abort(new Error('Remote response header deadline exceeded')), Math.min(duration, headerTimeoutMs));
  headerTimeout.unref?.();
  timeout.unref?.();
  const onAbort = () => controller.abort(fetchInit.signal?.reason);
  fetchInit.signal?.addEventListener('abort', onAbort, { once: true });
  const dispose = () => {
    clearTimeout(timeout);
    clearTimeout(headerTimeout);
    fetchInit.signal?.removeEventListener('abort', onAbort);
  };

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
  ).then(result => { clearTimeout(headerTimeout); return ownResponseBody(result, signal, dispose, bodyIdleMs); }).catch(error => {
    dispose();
    throw error;
  });

  if (!response.ok && !acceptHttpStatus) {
    const body = await readBoundedResponseText(response, 64 * 1024, true).catch(() => 'Remote error body unavailable');
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
  const response = await remoteRequest(worker, routePath, { timeoutMs: 30_000, ...init, headers });
  const payload: unknown = JSON.parse(await readBoundedResponseText(response, 8 * 1024 * 1024));
  return payload as T;
}

export function withoutRemoteRedirects(init: RequestInit): RequestInit {
  // Prevent 307/308 responses from replaying secret-bearing POST bodies to another URL.
  return { ...init, redirect: 'manual' };
}

export async function remoteProxyFetch(
  worker: WorkerNodeRecord,
  routePath: string,
  headersToForward: Headers,
  method: 'GET' | 'HEAD' = 'GET',
  signal?: AbortSignal,
) {
  const headers = new Headers();
  for (const name of ['range', 'if-range', 'if-none-match', 'if-match', 'if-modified-since', 'if-unmodified-since']) {
    const value = headersToForward.get(name);
    if (value) headers.set(name, value);
  }
  return remoteRequest(worker, routePath, { headers, method, signal, acceptHttpStatus: true });
}

export async function fetchRemoteJob(workerId: string, remoteJobId: string) {
  const worker = await getRemoteWorker(workerId);
  return fetchWorkerJob(worker, remoteJobId);
}

async function fetchWorkerJob(worker: WorkerNodeRecord, remoteJobId: string, init: RemoteRequestInit = {}) {
  const job = await remoteJson<Job | null>(worker, `/api/jobs?id=${encodeURIComponent(remoteJobId)}`, init);
  return isLegacyScopedRecord(job) ? null : job;
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
  return await db.jobs.updateIf(localJob.id, { attempt_id: localJob.attempt_id ?? null, status: localJob.status, updated_at: new Date(localJob.updated_at) }, remoteJobMissingUpdate()) || (await db.jobs.findById(localJob.id)) || localJob;
}

export async function fetchWorkerJobs(worker: WorkerNodeRecord, jobType?: string | null) {
  const cursorKey = `remote-discovery-cursor:${worker.id}:${jobType || 'all'}`;
  const prior = await db.runtime.get(cursorKey);
  const query = new URLSearchParams({ local_only: '1', summary: '0', limit: '20' });
  if (typeof prior?.value === 'string' && prior.value) query.set('cursor', prior.value);
  if (jobType) query.set('job_type', jobType);
  const result = await remoteJson<{ jobs: Job[]; nextCursor?: string | null }>(worker, `/api/jobs?${query.toString()}`, {
    timeoutMs: REMOTE_BACKGROUND_POLL_TIMEOUT_MS,
  });
  if (!Array.isArray(result.jobs)) throw new Error('Worker returned an invalid job page');
  await db.runtime.compareAndSwap(cursorKey, prior?.version ?? null, result.nextCursor || '');
  return { jobs: result.jobs.filter(job => !isLegacyScopedRecord(job)) };
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
    sample_now: remoteJob.sample_now ?? false,
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
  if (existing && (['editing', 'deleting', 'restarting', 'remote-starting'].includes(existing.status) || await activeOperationForResource(`job:${existing.id}`))) return existing;
  const keySnapshot = existing ? await getDurableKeySnapshot(existing.id) : null;
  const name = await resolveRemoteMirrorName(worker, remoteJob, existing?.id);
  const patch = remoteJobPatch(remoteJob, worker.id, remoteJob.id, name, existing);

  const synced = existing
    ? await db.jobs.updateIf(existing.id, { attempt_id: existing.attempt_id ?? null, status: existing.status, updated_at: new Date(existing.updated_at) }, patch) || existing
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
        sample_now: patch.sample_now,
        remote_sync_at: patch.remote_sync_at,
        remote_error: patch.remote_error,
      });

  if (remoteJob.status === 'completed' && !getJobRemoteCaptionState(synced)) {
    await clearDurableEncryptedDatasetKeys(synced.id, keySnapshot).catch(error =>
      console.error('Error clearing durable encrypted dataset keys:', error),
    );
  }

  return synced;
}

export async function syncRemoteJob(localJob: Job, options: { background?: boolean } = {}) {
  if (isLocalWorker(localJob.worker_id) || !localJob.remote_job_id || ['editing', 'deleting', 'restarting'].includes(localJob.status)) return localJob;
  if (options.background !== false && await activeOperationForResource(`job:${localJob.id}`)) return localJob;

  const keySnapshot = await getDurableKeySnapshot(localJob.id);
  try {
    const worker = await getRemoteWorker(localJob.worker_id);
    let remoteJob: Job | null;
    if (options.background === false) {
      remoteJob = await fetchWorkerJob(worker, localJob.remote_job_id);
    } else {
      const poll = await runRemoteBackgroundPoll(worker, `job sync ${localJob.id}`, () =>
        fetchWorkerJob(worker, localJob.remote_job_id as string, {
          timeoutMs: REMOTE_BACKGROUND_POLL_TIMEOUT_MS,
        }),
      );
      if ('reason' in poll) return localJob;
      remoteJob = poll.value;
    }
    if (!remoteJob || isLegacyScopedRecord(remoteJob)) {
      return markRemoteJobMissing(localJob);
    }

    const latestLocalJob = await db.jobs.findById(localJob.id);
    const localJobForPatch = latestLocalJob || localJob;
    const name = await resolveRemoteMirrorName(worker, remoteJob, localJob.id);
    const synced = await db.jobs.updateIf(
      localJob.id, { attempt_id: localJob.attempt_id ?? null, status: localJob.status, updated_at: new Date(localJob.updated_at) },
      remoteJobPatch(remoteJob, worker.id, remoteJob.id, name, localJobForPatch),
    );
    if (!synced) return latestLocalJob || localJob;
    if (remoteJob.status === 'completed' && !getJobRemoteCaptionState(synced)) {
      await clearDurableEncryptedDatasetKeys(localJob.id, keySnapshot).catch(error =>
        console.error('Error clearing durable encrypted dataset keys:', error),
      );
    }
    return synced;
  } catch (error) {
    if (isRemoteJobMissingError(error)) {
      return markRemoteJobMissing(localJob);
    }
    return await db.jobs.updateIf(localJob.id, { attempt_id: localJob.attempt_id ?? null, status: localJob.status, updated_at: new Date(localJob.updated_at) }, {
      remote_sync_at: new Date(),
      remote_error: error instanceof Error ? error.message : 'Remote sync failed',
    }) || (await db.jobs.findById(localJob.id)) || localJob;
  }
}

export async function syncRemoteJobs(jobs: Job[], alreadySyncedJobIds = new Set<string>()) {
  return Promise.all(jobs.map(job => (alreadySyncedJobIds.has(job.id) ? job : syncRemoteJob(job))));
}

function remoteDiscoveryErrorMessage(workerName: string, error: unknown) {
  if (error instanceof RemoteClientError) {
    const detail = isCloudflareTunnelUnavailable(error) ? 'Cloudflare tunnel is unavailable' : error.message;
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
        const poll = await runRemoteBackgroundPoll(worker, 'job discovery', () => fetchWorkerJobs(worker, jobType));
        if ('reason' in poll) return;
        const data = poll.value;
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
  const deadline = Date.now() + 60 * 60 * 1000;
  while (true) {
    if (Date.now() > deadline) throw new Error('Remote import is still pending; its operation can be resumed.');
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
    uploadID?: string;
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

  const uploadID = options.uploadID || randomUUID();
  if (options.uploadID) {
    try {
      const prior: unknown = await remoteJson(worker, appendQueryParams(routePath, { aitk_upload: 'status', uploadID }));
      if (isRemoteArchiveImportStatus<T>(prior)) {
        if (prior.status === 'completed' && prior.result !== null) return prior.result;
        if (prior.status === 'failed') throw new Error(prior.error || 'Remote import failed');
        return waitForRemoteArchiveImport<T>(worker, routePath, uploadID);
      }
    } catch (error) { if (!(error instanceof RemoteClientError && error.status === 404)) throw error; }
  }
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
        fileBytes: fileStat.size,
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
        timeoutMs: 600_000, headerTimeoutMs: 600_000,
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
      fileBytes: fileStat.size,
      background: options.backgroundComplete ? '1' : null,
    }),
    {
      method: 'POST',
      headers: {
        'X-AITK-File-Name': fileName,
      },
    },
  );

  if (isRemoteArchiveImportStatus<T>(completeResult)) {
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
  uploadID?: string,
) {
  return remoteZipFileJson<{ job: Job; warnings: string[] }>(
    worker,
    appendQueryParams('/api/jobs/import', {
      gpu_ids: gpuIds,
    }),
    {
      filePath: zipPath,
      fileName: path.basename(zipPath),
      onProgress,
      uploadID,
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
  }>(worker, '/api/remote/health', { timeoutMs: REMOTE_BACKGROUND_POLL_TIMEOUT_MS });
}

export async function uploadDatasetArchiveToWorker(
  worker: WorkerNodeRecord,
  zipPath: string,
  preferredName?: string,
  onProgress?: (progress: FileUploadProgress) => void,
  uploadID?: string,
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
    uploadID,
  });
}

export async function fetchWorkerGpu(worker: WorkerNodeRecord) {
  return remoteJson<GPUApiResponse>(worker, '/api/gpu', { timeoutMs: REMOTE_BACKGROUND_POLL_TIMEOUT_MS });
}

export async function fetchWorkerCpu(worker: WorkerNodeRecord) {
  return remoteJson<CpuInfo>(worker, '/api/cpu', { timeoutMs: REMOTE_BACKGROUND_POLL_TIMEOUT_MS });
}

export async function fetchWorkerQueues(worker: WorkerNodeRecord) {
  return remoteJson<{ queues: Queue[] }>(worker, '/api/queue', { timeoutMs: REMOTE_BACKGROUND_POLL_TIMEOUT_MS });
}
