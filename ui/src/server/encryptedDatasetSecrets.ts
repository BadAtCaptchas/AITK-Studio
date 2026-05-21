import { db } from './db';
import {
  getKeyForRequiredDataset,
  normalizeEncryptedKeyMap,
  resolveConfigPath,
} from './encryptedDatasets';
import type { EncryptedDatasetStartKey } from '../types';

export const ENCRYPTED_DATASET_SECRET_SETTING_PREFIX = 'ENCRYPTED_DATASET_DURABLE_KEYS:';

type DurableEncryptedDatasetKeysPayload = {
  version: 1;
  createdAt: string;
  keys: EncryptedDatasetStartKey[];
};

function durableSettingKey(jobID: string) {
  if (!/^[a-zA-Z0-9_-]+$/.test(jobID)) {
    throw new Error('Invalid job ID');
  }
  return `${ENCRYPTED_DATASET_SECRET_SETTING_PREFIX}${jobID}`;
}

export function isEncryptedDatasetSecretSettingKey(key: string) {
  return key.startsWith(ENCRYPTED_DATASET_SECRET_SETTING_PREFIX);
}

function normalizeDurableKeys(keys: unknown): EncryptedDatasetStartKey[] {
  if (!Array.isArray(keys)) return [];
  const deduped = new Map<string, EncryptedDatasetStartKey>();

  keys.forEach(key => {
    if (!key || typeof key !== 'object') return;
    const datasetPath = (key as EncryptedDatasetStartKey).datasetPath;
    const keyB64 = (key as EncryptedDatasetStartKey).keyB64;
    if (typeof datasetPath !== 'string' || typeof keyB64 !== 'string') return;
    if (!datasetPath.trim() || datasetPath.length > 4096) return;
    if (!/^[A-Za-z0-9+/=]+$/.test(keyB64) || keyB64.length > 2048) return;
    const normalizedPath = resolveConfigPath(datasetPath);
    deduped.set(normalizedPath.toLowerCase(), { datasetPath: normalizedPath, keyB64 });
  });

  return Array.from(deduped.values());
}

export async function getDurableEncryptedDatasetKeys(jobID: string): Promise<EncryptedDatasetStartKey[]> {
  const row = await db.settings.get(durableSettingKey(jobID));
  if (!row?.value) return [];

  try {
    const parsed = JSON.parse(row.value) as DurableEncryptedDatasetKeysPayload | EncryptedDatasetStartKey[];
    const keys = Array.isArray(parsed) ? parsed : parsed?.keys;
    return normalizeDurableKeys(keys);
  } catch {
    return [];
  }
}

export async function storeDurableEncryptedDatasetKeys(
  jobID: string,
  keys: EncryptedDatasetStartKey[],
): Promise<EncryptedDatasetStartKey[]> {
  const existing = await getDurableEncryptedDatasetKeys(jobID);
  const merged = normalizeDurableKeys([...existing, ...keys]);
  if (merged.length === 0) {
    await clearDurableEncryptedDatasetKeys(jobID);
    return [];
  }

  const payload: DurableEncryptedDatasetKeysPayload = {
    version: 1,
    createdAt: new Date().toISOString(),
    keys: merged,
  };
  await db.settings.upsert(durableSettingKey(jobID), JSON.stringify(payload));
  return merged;
}

export async function clearDurableEncryptedDatasetKeys(jobID: string): Promise<void> {
  await db.settings.delete(durableSettingKey(jobID));
}

export async function getEncryptedKeyCoverage(
  jobID: string,
  requiredDatasets: { path: string; name: string }[],
  requestKeys?: EncryptedDatasetStartKey[] | null,
) {
  const durableKeys = await getDurableEncryptedDatasetKeys(jobID);
  const combinedKeys = normalizeDurableKeys([...(durableKeys || []), ...(requestKeys || [])]);
  const keyMap = normalizeEncryptedKeyMap(combinedKeys);
  const missingDatasets = requiredDatasets.filter(dataset => !getKeyForRequiredDataset(keyMap, dataset));

  return {
    durableKeys,
    combinedKeys,
    missingDatasets,
  };
}
