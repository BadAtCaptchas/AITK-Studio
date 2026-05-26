import crypto from 'crypto';
import { db } from './db';
import {
  getKeyForRequiredDataset,
  normalizeEncryptedKeyMap,
  resolveConfigPath,
} from './encryptedDatasets';
import type { EncryptedDatasetStartKey } from '../types';

export const ENCRYPTED_DATASET_SECRET_SETTING_PREFIX = 'ENCRYPTED_DATASET_DURABLE_KEYS:';
export const DURABLE_DATASET_KEY_SECRET_ENV = 'AITK_DURABLE_DATASET_KEY_SECRET';

const DURABLE_PAYLOAD_VERSION = 2;
const AES_GCM_NONCE_BYTES = 12;
const AES_GCM_AUTH_TAG_BYTES = 16;
const WRAPPING_SECRET_MIN_LENGTH = 32;
const HKDF_SALT = 'aitk durable encrypted dataset key salt v2';
const HKDF_INFO = 'aitk durable encrypted dataset key wrapping v2';

type WrappedDurableEncryptedDatasetKey = {
  datasetPath: string;
  nonceB64: string;
  ciphertextB64: string;
  tagB64: string;
};

type DurableEncryptedDatasetKeysPayloadV2 = {
  version: typeof DURABLE_PAYLOAD_VERSION;
  createdAt: string;
  keys: WrappedDurableEncryptedDatasetKey[];
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

export class DurableEncryptedDatasetKeySecretError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'DurableEncryptedDatasetKeySecretError';
  }
}

export function isDurableEncryptedDatasetKeySecretError(
  error: unknown,
): error is DurableEncryptedDatasetKeySecretError {
  return error instanceof DurableEncryptedDatasetKeySecretError;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function normalizeWrappingSecret() {
  const secret = process.env[DURABLE_DATASET_KEY_SECRET_ENV]?.trim() || '';
  if (
    !secret ||
    secret.length < WRAPPING_SECRET_MIN_LENGTH ||
    /^password$/i.test(secret) ||
    /^change_?me$/i.test(secret) ||
    /^\{\{\s*RUNPOD_SECRET_/i.test(secret)
  ) {
    throw new DurableEncryptedDatasetKeySecretError(
      `${DURABLE_DATASET_KEY_SECRET_ENV} must be set to a real secret of at least ${WRAPPING_SECRET_MIN_LENGTH} characters before durable encrypted dataset resume can be used.`,
    );
  }
  return secret;
}

function durableWrappingKey() {
  return Buffer.from(
    crypto.hkdfSync(
      'sha256',
      Buffer.from(normalizeWrappingSecret(), 'utf8'),
      Buffer.from(HKDF_SALT, 'utf8'),
      Buffer.from(HKDF_INFO, 'utf8'),
      32,
    ),
  );
}

function base64ToBuffer(value: unknown, fieldName: string) {
  if (typeof value !== 'string' || !/^[A-Za-z0-9+/]+={0,2}$/.test(value)) {
    throw new Error(`Invalid durable encrypted dataset ${fieldName}`);
  }
  const decoded = Buffer.from(value, 'base64');
  if (decoded.length === 0) {
    throw new Error(`Invalid durable encrypted dataset ${fieldName}`);
  }
  return decoded;
}

function durableKeyAad(jobID: string, datasetPath: string) {
  return Buffer.from(JSON.stringify({ version: DURABLE_PAYLOAD_VERSION, jobID, datasetPath }), 'utf8');
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

function hasPlainDurableKeyMaterial(keys: unknown) {
  return Array.isArray(keys) && keys.some(key => isRecord(key) && typeof key.keyB64 === 'string');
}

export function isLegacyDurableEncryptedDatasetKeysPayload(value: unknown) {
  if (Array.isArray(value)) {
    return hasPlainDurableKeyMaterial(value);
  }
  if (!isRecord(value)) return false;
  return value.version === 1 || hasPlainDurableKeyMaterial(value.keys);
}

function wrapDurableKey(jobID: string, key: EncryptedDatasetStartKey, wrappingKey: Buffer) {
  const nonce = crypto.randomBytes(AES_GCM_NONCE_BYTES);
  const cipher = crypto.createCipheriv('aes-256-gcm', wrappingKey, nonce, {
    authTagLength: AES_GCM_AUTH_TAG_BYTES,
  });
  cipher.setAAD(durableKeyAad(jobID, key.datasetPath));
  const ciphertext = Buffer.concat([cipher.update(key.keyB64, 'utf8'), cipher.final()]);
  const tag = cipher.getAuthTag();

  return {
    datasetPath: key.datasetPath,
    nonceB64: nonce.toString('base64'),
    ciphertextB64: ciphertext.toString('base64'),
    tagB64: tag.toString('base64'),
  };
}

function unwrapDurableKey(
  jobID: string,
  wrapped: WrappedDurableEncryptedDatasetKey,
  wrappingKey: Buffer,
): EncryptedDatasetStartKey {
  const nonce = base64ToBuffer(wrapped.nonceB64, 'nonce');
  const ciphertext = base64ToBuffer(wrapped.ciphertextB64, 'ciphertext');
  const tag = base64ToBuffer(wrapped.tagB64, 'auth tag');

  if (nonce.length !== AES_GCM_NONCE_BYTES || tag.length !== AES_GCM_AUTH_TAG_BYTES) {
    throw new Error('Invalid durable encrypted dataset key payload');
  }

  const decipher = crypto.createDecipheriv('aes-256-gcm', wrappingKey, nonce, {
    authTagLength: AES_GCM_AUTH_TAG_BYTES,
  });
  decipher.setAAD(durableKeyAad(jobID, wrapped.datasetPath));
  decipher.setAuthTag(tag);
  const keyB64 = Buffer.concat([decipher.update(ciphertext), decipher.final()]).toString('utf8');
  return { datasetPath: wrapped.datasetPath, keyB64 };
}

export function wrapDurableEncryptedDatasetKeysPayload(
  jobID: string,
  keys: EncryptedDatasetStartKey[],
  createdAt: Date = new Date(),
): DurableEncryptedDatasetKeysPayloadV2 {
  durableSettingKey(jobID);
  const normalized = normalizeDurableKeys(keys);
  const wrappingKey = durableWrappingKey();
  return {
    version: DURABLE_PAYLOAD_VERSION,
    createdAt: createdAt.toISOString(),
    keys: normalized.map(key => wrapDurableKey(jobID, key, wrappingKey)),
  };
}

export function unwrapDurableEncryptedDatasetKeysPayload(
  jobID: string,
  value: unknown,
): EncryptedDatasetStartKey[] {
  durableSettingKey(jobID);
  if (isLegacyDurableEncryptedDatasetKeysPayload(value)) return [];
  if (!isRecord(value) || value.version !== DURABLE_PAYLOAD_VERSION || !Array.isArray(value.keys)) return [];

  const wrappingKey = durableWrappingKey();
  return normalizeDurableKeys(
    value.keys.map(key => {
      if (!isRecord(key) || typeof key.datasetPath !== 'string') {
        throw new Error('Invalid durable encrypted dataset key payload');
      }
      return unwrapDurableKey(jobID, key as WrappedDurableEncryptedDatasetKey, wrappingKey);
    }),
  );
}

export async function getDurableEncryptedDatasetKeys(jobID: string): Promise<EncryptedDatasetStartKey[]> {
  const row = await db.settings.get(durableSettingKey(jobID));
  if (!row?.value) return [];

  try {
    const parsed = JSON.parse(row.value);
    if (isLegacyDurableEncryptedDatasetKeysPayload(parsed)) {
      await clearDurableEncryptedDatasetKeys(jobID);
      return [];
    }
    return unwrapDurableEncryptedDatasetKeysPayload(jobID, parsed);
  } catch (error) {
    if (isDurableEncryptedDatasetKeySecretError(error)) {
      throw error;
    }
    await clearDurableEncryptedDatasetKeys(jobID);
    return [];
  }
}

export async function storeDurableEncryptedDatasetKeys(
  jobID: string,
  keys: EncryptedDatasetStartKey[],
): Promise<EncryptedDatasetStartKey[]> {
  normalizeWrappingSecret();
  const existing = await getDurableEncryptedDatasetKeys(jobID);
  const merged = normalizeDurableKeys([...existing, ...keys]);
  if (merged.length === 0) {
    await clearDurableEncryptedDatasetKeys(jobID);
    return [];
  }

  const payload = wrapDurableEncryptedDatasetKeysPayload(jobID, merged);
  await db.settings.upsert(durableSettingKey(jobID), JSON.stringify(payload));
  return merged;
}

export async function clearDurableEncryptedDatasetKeys(jobID: string): Promise<void> {
  await db.settings.delete(durableSettingKey(jobID));
}

export async function purgeLegacyDurableEncryptedDatasetKeys(): Promise<number> {
  const settings = await db.settings.list();
  let purged = 0;

  await Promise.all(
    settings.map(async setting => {
      if (!isEncryptedDatasetSecretSettingKey(setting.key)) return;
      try {
        if (!isLegacyDurableEncryptedDatasetKeysPayload(JSON.parse(setting.value))) return;
      } catch {
        return;
      }
      await db.settings.delete(setting.key);
      purged += 1;
    }),
  );

  return purged;
}

export async function getEncryptedKeyCoverage(
  jobID: string,
  requiredDatasets: { path: string; name: string }[],
  requestKeys?: EncryptedDatasetStartKey[] | null,
) {
  let durableKeys: EncryptedDatasetStartKey[] = [];
  try {
    durableKeys = await getDurableEncryptedDatasetKeys(jobID);
  } catch (error) {
    if (!isDurableEncryptedDatasetKeySecretError(error)) {
      throw error;
    }
  }
  const combinedKeys = normalizeDurableKeys([...(durableKeys || []), ...(requestKeys || [])]);
  const keyMap = normalizeEncryptedKeyMap(combinedKeys);
  const missingDatasets = requiredDatasets.filter(dataset => !getKeyForRequiredDataset(keyMap, dataset));

  return {
    durableKeys,
    combinedKeys,
    missingDatasets,
  };
}
