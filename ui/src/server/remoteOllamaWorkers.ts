import { randomUUID } from 'crypto';
import { db } from './db';
import {
  getOllamaStatus,
  listOllamaModels,
  type OllamaEndpointConfig,
  type OllamaModel,
  type OllamaStatus,
} from './ollama';
import type { RemoteOllamaWorker } from '../types';

export const REMOTE_OLLAMA_WORKERS_SETTING_KEY = 'REMOTE_OLLAMA_WORKERS';

export type RemoteOllamaWorkerRecord = RemoteOllamaWorker & {
  auth_token: string;
};

type RemoteOllamaWorkerInput = {
  id?: string;
  name?: string;
  base_url?: string;
  auth_token?: string;
  clear_auth_token?: boolean;
  enabled?: boolean;
};

function nowIso() {
  return new Date().toISOString();
}

function asString(value: unknown) {
  return typeof value === 'string' ? value.trim() : '';
}

function asNullableString(value: unknown) {
  const stringValue = asString(value);
  return stringValue || null;
}

function asNullableNumber(value: unknown) {
  if (value === null || value === undefined || value === '') return null;
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : null;
}

export function isRemoteOllamaWorkersSettingKey(key: string) {
  return key === REMOTE_OLLAMA_WORKERS_SETTING_KEY;
}

export function normalizeRemoteOllamaBaseUrl(baseUrl: string) {
  const trimmed = baseUrl.trim().replace(/\/+$/, '');
  if (!/^https?:\/\//i.test(trimmed)) {
    throw new Error('Remote Ollama URL must start with http:// or https://');
  }
  return trimmed;
}

function normalizeRemoteOllamaWorker(raw: any): RemoteOllamaWorkerRecord | null {
  if (!raw || typeof raw !== 'object') return null;
  const id = asString(raw.id);
  const name = asString(raw.name);
  const baseUrl = asString(raw.base_url);
  if (!id || !name || !baseUrl) return null;
  const createdAt = asString(raw.created_at) || nowIso();
  return {
    id,
    name,
    base_url: baseUrl.replace(/\/+$/, ''),
    auth_token: asString(raw.auth_token),
    enabled: raw.enabled !== false,
    last_status: asString(raw.last_status) || 'unknown',
    last_error: asNullableString(raw.last_error),
    last_checked_at: asNullableString(raw.last_checked_at),
    model_count: asNullableNumber(raw.model_count),
    created_at: createdAt,
    updated_at: asString(raw.updated_at) || createdAt,
  };
}

function sortRemoteOllamaWorkers(workers: RemoteOllamaWorkerRecord[]) {
  return workers.sort((left, right) => left.name.localeCompare(right.name));
}

export function toPublicRemoteOllamaWorker(worker: RemoteOllamaWorkerRecord): RemoteOllamaWorker {
  const { auth_token: _authToken, ...publicWorker } = worker;
  return publicWorker;
}

async function readRemoteOllamaWorkers() {
  const row = await db.settings.get(REMOTE_OLLAMA_WORKERS_SETTING_KEY);
  if (!row?.value) return [];
  try {
    const parsed = JSON.parse(row.value);
    if (!Array.isArray(parsed)) return [];
    return sortRemoteOllamaWorkers(parsed.map(normalizeRemoteOllamaWorker).filter(Boolean) as RemoteOllamaWorkerRecord[]);
  } catch {
    return [];
  }
}

async function writeRemoteOllamaWorkers(workers: RemoteOllamaWorkerRecord[]) {
  const normalized = sortRemoteOllamaWorkers(workers).map(worker => ({
    ...worker,
    base_url: normalizeRemoteOllamaBaseUrl(worker.base_url),
  }));
  await db.settings.upsert(REMOTE_OLLAMA_WORKERS_SETTING_KEY, JSON.stringify(normalized));
  return normalized;
}

function assertUniqueRemoteOllamaName(workers: RemoteOllamaWorkerRecord[], name: string, id: string) {
  const nameKey = name.toLowerCase();
  const duplicate = workers.find(worker => worker.id !== id && worker.name.toLowerCase() === nameKey);
  if (duplicate) {
    throw new Error('Remote Ollama name already exists');
  }
}

export async function listRemoteOllamaWorkers(options: { enabled?: boolean } = {}) {
  const workers = await readRemoteOllamaWorkers();
  return typeof options.enabled === 'boolean' ? workers.filter(worker => worker.enabled === options.enabled) : workers;
}

export async function findRemoteOllamaWorker(id: string) {
  const workers = await readRemoteOllamaWorkers();
  return workers.find(worker => worker.id === id) || null;
}

export async function getRemoteOllamaWorker(id: string, options: { requireEnabled?: boolean } = {}) {
  const worker = await findRemoteOllamaWorker(id);
  if (!worker) throw new Error(`Remote Ollama worker not found: ${id}`);
  if (options.requireEnabled !== false && !worker.enabled) {
    throw new Error(`Remote Ollama worker is disabled: ${worker.name}`);
  }
  return {
    ...worker,
    base_url: normalizeRemoteOllamaBaseUrl(worker.base_url),
  };
}

export function endpointForRemoteOllamaWorker(worker: RemoteOllamaWorkerRecord): OllamaEndpointConfig {
  return { baseUrl: worker.base_url, authToken: worker.auth_token };
}

export async function saveRemoteOllamaWorker(input: RemoteOllamaWorkerInput) {
  const id = asString(input.id) || randomUUID();
  const name = asString(input.name);
  const baseUrl = normalizeRemoteOllamaBaseUrl(asString(input.base_url));
  const now = nowIso();
  const workers = await readRemoteOllamaWorkers();
  const existing = workers.find(worker => worker.id === id) || null;

  if (!name) throw new Error('Remote Ollama name is required');
  assertUniqueRemoteOllamaName(workers, name, id);

  const authToken =
    input.clear_auth_token === true
      ? ''
      : typeof input.auth_token === 'string' && input.auth_token.trim()
        ? input.auth_token.trim()
        : existing?.auth_token || '';

  const nextWorker: RemoteOllamaWorkerRecord = {
    id,
    name,
    base_url: baseUrl,
    auth_token: authToken,
    enabled: input.enabled !== false,
    last_status: existing?.last_status || 'unknown',
    last_error: existing?.last_error || null,
    last_checked_at: existing?.last_checked_at || null,
    model_count: existing?.model_count ?? null,
    created_at: existing?.created_at || now,
    updated_at: now,
  };

  const nextWorkers = existing
    ? workers.map(worker => (worker.id === id ? nextWorker : worker))
    : [...workers, nextWorker];
  await writeRemoteOllamaWorkers(nextWorkers);
  return nextWorker;
}

export async function deleteRemoteOllamaWorker(id: string) {
  const workers = await readRemoteOllamaWorkers();
  const existing = workers.find(worker => worker.id === id) || null;
  if (!existing) return null;
  await writeRemoteOllamaWorkers(workers.filter(worker => worker.id !== id));
  return existing;
}

export async function updateRemoteOllamaWorkerStatus(
  id: string,
  status: OllamaStatus,
): Promise<RemoteOllamaWorkerRecord> {
  const workers = await readRemoteOllamaWorkers();
  const existing = workers.find(worker => worker.id === id);
  if (!existing) throw new Error(`Remote Ollama worker not found: ${id}`);
  const updated: RemoteOllamaWorkerRecord = {
    ...existing,
    last_status: status.ok ? 'online' : 'error',
    last_error: status.ok ? null : status.error || 'Remote Ollama is unavailable',
    last_checked_at: nowIso(),
    model_count: status.modelCount,
    updated_at: nowIso(),
  };
  await writeRemoteOllamaWorkers(workers.map(worker => (worker.id === id ? updated : worker)));
  return updated;
}

export async function checkRemoteOllamaWorker(id: string) {
  const worker = await getRemoteOllamaWorker(id, { requireEnabled: false });
  const status = await getOllamaStatus(endpointForRemoteOllamaWorker(worker));
  return {
    worker: await updateRemoteOllamaWorkerStatus(id, status),
    status,
  };
}

export async function listRemoteOllamaWorkerModels(id: string): Promise<{
  worker: RemoteOllamaWorkerRecord;
  status: OllamaStatus;
  models: OllamaModel[];
}> {
  const worker = await getRemoteOllamaWorker(id);
  const endpoint = endpointForRemoteOllamaWorker(worker);
  try {
    const models = await listOllamaModels(endpoint);
    const status: OllamaStatus = {
      ok: true,
      baseUrl: worker.base_url,
      modelCount: models.length,
      error: null,
    };
    return {
      worker: await updateRemoteOllamaWorkerStatus(id, status),
      status,
      models,
    };
  } catch (error) {
    const status: OllamaStatus = {
      ok: false,
      baseUrl: worker.base_url,
      modelCount: 0,
      error: error instanceof Error ? error.message : 'Failed to list Remote Ollama models',
    };
    return {
      worker: await updateRemoteOllamaWorkerStatus(id, status),
      status,
      models: [],
    };
  }
}
