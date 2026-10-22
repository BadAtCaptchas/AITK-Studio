import type { JobConfig } from '../types';
import type { TrainingMigrationDraft } from '../domain/trainingModelMigration';
import { isEditableTrainingConfig } from './trainingValidation';

export type TrainingDraft = { config: JobConfig; gpuIDs: string | null; workerID: string; migration?: TrainingMigrationDraft };
export type StoredTrainingDraft = { draft: TrainingDraft; baseline: string };
type DraftStorage = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;
const PREFIX = 'aitk:training-draft:v1:';
const MAX_BYTES = 2_000_000;
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);

// Raw YAML can contain integration credentials. Never serialize these or unlocked
// dataset payloads, including inside model kwargs or a migration's source config.
const privateField = /secret|password|credential|authorization|apikey|privatekey|keyb64|datasetkey|encryptionkey|rawkey|unwrappedkey|catalog|plaintext|signature|securitytoken|^(?:sig|(?:api|hf|access|refresh|bearer|auth|huggingface|openrouter)?token)$/i;
const isPrivateField = (key: string) => privateField.test(key.replace(/[_-]/g, ''));
function sanitize(value: unknown, depth = 0): unknown {
  if (depth > 60) throw new Error('Draft is too deeply nested');
  if (typeof value === 'string' && /^https?:\/\//i.test(value)) {
    try {
      const url = new URL(value);
      url.username = ''; url.password = '';
      for (const key of [...url.searchParams.keys()]) if (isPrivateField(key)) url.searchParams.delete(key);
      return url.toString();
    } catch { return value; }
  }
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return value;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (Array.isArray(value)) return value
    .filter(item => !(record(item) && typeof item.path === 'string' && 'before' in item && 'after' in item && item.path.split(/[.[\]]/).some(isPrivateField)))
    .map(item => sanitize(item, depth + 1));
  if (!record(value)) return undefined;
  return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !['__proto__', 'prototype', 'constructor'].includes(key) && !isPrivateField(key))
    .map(([key, item]) => [key, sanitize(item, depth + 1)]));
}

function validDraft(value: unknown): value is TrainingDraft {
  if (!record(value) || !isEditableTrainingConfig(value.config) || typeof value.workerID !== 'string' ||
      !(value.gpuIDs === null || typeof value.gpuIDs === 'string')) return false;
  if (value.migration !== undefined) {
    const migration = value.migration;
    if (!record(migration) || !isEditableTrainingConfig(migration.source) || !Array.isArray(migration.changes) ||
        !migration.changes.every(change => record(change) && ['path', 'reason', 'before', 'after'].every(key => typeof change[key] === 'string'))) return false;
  }
  return true;
}

export function trainingDraftStorage(): DraftStorage | undefined {
  try { return typeof window === 'undefined' ? undefined : window.sessionStorage; } catch { return undefined; }
}

export function readStoredTrainingDraft(key: string, storage = trainingDraftStorage()): StoredTrainingDraft | undefined {
  try {
    const raw = storage?.getItem(PREFIX + key);
    if (!raw || raw.length > MAX_BYTES) return undefined;
    const value: unknown = sanitize(JSON.parse(raw));
    if (!record(value) || value.version !== 1 || !validDraft(value.draft) || !validDraft(value.baseline)) return undefined;
    return { draft: value.draft, baseline: JSON.stringify(value.baseline) };
  } catch { return undefined; }
}

export function saveStoredTrainingDraft(key: string, value: StoredTrainingDraft, storage = trainingDraftStorage()): boolean {
  try {
    const draft = sanitize(value.draft);
    const baseline: unknown = sanitize(JSON.parse(value.baseline));
    if (!storage || !validDraft(draft) || !validDraft(baseline)) return false;
    const raw = JSON.stringify({ version: 1, draft, baseline });
    if (raw.length > MAX_BYTES) return false;
    storage.setItem(PREFIX + key, raw);
    return true;
  } catch { return false; }
}

export function removeStoredTrainingDraft(key: string, storage = trainingDraftStorage()): void {
  try { storage?.removeItem(PREFIX + key); } catch { /* Browser storage may be disabled. */ }
}
