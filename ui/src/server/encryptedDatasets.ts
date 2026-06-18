import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import crypto from 'crypto';
import { TOOLKIT_ROOT } from '../paths';
import type { DatasetSummary, EncryptedDatasetManifest, EncryptedDatasetStartKey } from '../types';
import { DATASET_CAPTION_SIDECAR_EXTENSIONS } from './captionFiles';
import { isDatasetRootCaptionEntry } from './datasetRootCaption';

export const ENCRYPTED_DATASET_MANIFEST = '.aitk_encrypted_dataset.json';
const CATALOG_AAD = Buffer.from('aitk-encrypted-catalog:v1', 'utf8');
const AES_GCM_AUTH_TAG_BYTES = 16;
const AES_GCM_NONCE_BYTES = 12;

const DATASET_CONFIG_FIELDS = [
  'folder_path',
  'dataset_path',
  'control_path',
  'control_path_1',
  'control_path_2',
  'control_path_3',
  'mask_path',
  'unconditional_path',
  'inpaint_path',
  'clip_image_path',
];

const DATASET_MEDIA_EXTENSIONS = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
  '.jxl',
  '.mp4',
  '.avi',
  '.mov',
  '.mkv',
  '.wmv',
  '.m4v',
  '.flv',
  '.mp3',
  '.wav',
  '.flac',
  '.ogg',
]);

const DATASET_CAPTION_EXTENSIONS = DATASET_CAPTION_SIDECAR_EXTENSIONS;
const DETECTED_CAPTION_MIN_MEDIA_COVERAGE = 0.5;
const DETECTED_CAPTION_MIN_CAPTION_SHARE = 0.8;

export type DatasetCaptionSummary = {
  itemCount: number;
  captionedItemCount: number;
  missingCaptionCount: number;
  detectedCaptionExt: string | null;
  captionExtensionCounts: Record<string, number>;
};

function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}

function safeIsFile(filePath: string) {
  try {
    return fs.statSync(filePath).isFile();
  } catch {
    return false;
  }
}

function normalizeCaptionExtension(extension: string) {
  return extension.replace(/^\.+/, '').toLowerCase();
}

function captionSidecars(mediaPath: string, datasetFolder: string) {
  const parsed = path.parse(mediaPath);
  return DATASET_CAPTION_EXTENSIONS.filter(captionExt => {
    const captionFileName = `${parsed.name}${captionExt}`;
    if (isDatasetRootCaptionEntry(datasetFolder, parsed.dir, captionFileName)) return false;
    return safeIsFile(path.join(parsed.dir, captionFileName));
  });
}

function detectDominantCaptionExt(summary: Pick<DatasetCaptionSummary, 'itemCount' | 'captionedItemCount' | 'captionExtensionCounts'>) {
  let bestExtension = '';
  let bestCount = 0;

  for (const extension of DATASET_CAPTION_EXTENSIONS) {
    const count = summary.captionExtensionCounts[extension] || 0;
    if (count > bestCount) {
      bestExtension = extension;
      bestCount = count;
    }
  }

  if (summary.itemCount === 0 || summary.captionedItemCount === 0 || bestCount === 0) return null;
  if (bestCount / summary.itemCount < DETECTED_CAPTION_MIN_MEDIA_COVERAGE) return null;
  if (bestCount / summary.captionedItemCount < DETECTED_CAPTION_MIN_CAPTION_SHARE) return null;

  return normalizeCaptionExtension(bestExtension);
}

export function summarizePlainDatasetCaptions(datasetFolder: string): DatasetCaptionSummary {
  const summary: DatasetCaptionSummary = {
    itemCount: 0,
    captionedItemCount: 0,
    missingCaptionCount: 0,
    detectedCaptionExt: null,
    captionExtensionCounts: {},
  };
  const stack = [datasetFolder];

  while (stack.length > 0) {
    const current = stack.pop() as string;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue;
      const entryPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        if (entry.name !== '_controls') stack.push(entryPath);
        continue;
      }
      if (!entry.isFile() || !DATASET_MEDIA_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) continue;

      summary.itemCount += 1;
      const sidecars = captionSidecars(entryPath, datasetFolder);
      if (sidecars.length > 0) {
        summary.captionedItemCount += 1;
        sidecars.forEach(extension => {
          summary.captionExtensionCounts[extension] = (summary.captionExtensionCounts[extension] || 0) + 1;
        });
      } else {
        summary.missingCaptionCount += 1;
      }
    }
  }

  summary.detectedCaptionExt = detectDominantCaptionExt(summary);
  return summary;
}

export function cleanDatasetName(name: string) {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

export function encryptedManifestPath(datasetFolder: string) {
  return path.join(datasetFolder, ENCRYPTED_DATASET_MANIFEST);
}

export function isEncryptedDatasetFolder(datasetFolder: string) {
  return fs.existsSync(encryptedManifestPath(datasetFolder));
}

export function resolveConfigPath(value: string) {
  return path.isAbsolute(value) ? path.resolve(value) : path.resolve(TOOLKIT_ROOT, value);
}

export function resolveDatasetFolder(datasetsRoot: string, datasetName: string) {
  if (typeof datasetName !== 'string' || !datasetName.trim() || /[\\/]/.test(datasetName) || datasetName.startsWith('.')) {
    throw new Error('Invalid dataset name');
  }
  const root = path.resolve(datasetsRoot);
  const folder = path.resolve(root, datasetName);
  if (!isPathInside(root, folder) || folder === root) {
    throw new Error('Invalid dataset name');
  }
  return folder;
}

export function safeEncryptedObjectPath(objectPath: string) {
  const normalized = objectPath.replace(/\\/g, '/');
  if (!/^objects\/[a-zA-Z0-9_-]+(?:\.caption)?\.bin$/.test(normalized)) {
    throw new Error('Invalid encrypted object path');
  }
  return normalized;
}

export function resolveEncryptedObjectPath(datasetFolder: string, objectPath: string) {
  const safePath = safeEncryptedObjectPath(objectPath);
  const resolved = path.resolve(datasetFolder, ...safePath.split('/'));
  if (!isPathInside(path.resolve(datasetFolder, 'objects'), resolved)) {
    throw new Error('Invalid encrypted object path');
  }
  return resolved;
}

export function validateEncryptedManifest(value: unknown): EncryptedDatasetManifest {
  const manifest = value as EncryptedDatasetManifest;
  if (manifest?.format !== 'aitk-encrypted-dataset' || manifest?.version !== 1) {
    throw new Error('Invalid encrypted dataset manifest');
  }
  if (manifest.crypto?.algorithm !== 'AES-256-GCM') {
    throw new Error('Unsupported encrypted dataset algorithm');
  }
  const kdfType = manifest.crypto?.kdf?.type;
  if (kdfType !== 'PBKDF2-SHA256' && kdfType !== 'KEYFILE-SHA256' && kdfType !== 'WEBAUTHN-PRF') {
    throw new Error('Unsupported encrypted dataset KDF');
  }
  if (kdfType === 'PBKDF2-SHA256') {
    const kdf = manifest.crypto.kdf;
    if (!kdf.salt || !Number.isFinite(kdf.iterations) || kdf.iterations < 100_000 || kdf.keyLength !== 32) {
      throw new Error('Invalid encrypted dataset password KDF header');
    }
  }
  if (kdfType === 'WEBAUTHN-PRF') {
    const kdf = manifest.crypto.kdf as any;
    if (kdf.keyLength !== 32 || typeof kdf.rpId !== 'string' || !kdf.rpId.trim()) {
      throw new Error('Invalid encrypted dataset WebAuthn PRF header');
    }
    if (!Array.isArray(kdf.credentials) || kdf.credentials.length === 0) {
      throw new Error('Encrypted dataset WebAuthn PRF credential is missing');
    }
    kdf.credentials.forEach((credential: any) => {
      if (!credential || !isBase64Url(credential.id) || !isBase64(credential.saltB64)) {
        throw new Error('Invalid encrypted dataset WebAuthn PRF credential');
      }
      if (
        credential.transports != null &&
        (!Array.isArray(credential.transports) || credential.transports.some((item: unknown) => typeof item !== 'string'))
      ) {
        throw new Error('Invalid encrypted dataset WebAuthn PRF transports');
      }
      if (
        credential.wrappedKey?.algorithm !== 'AES-256-GCM' ||
        !isBase64(credential.wrappedKey?.nonce) ||
        !isBase64(credential.wrappedKey?.data)
      ) {
        throw new Error('Invalid encrypted dataset WebAuthn PRF wrapped key');
      }
    });
  }
  if (!manifest.catalog?.nonce || !manifest.catalog?.data) {
    throw new Error('Invalid encrypted dataset catalog');
  }
  return manifest;
}

export async function readEncryptedManifest(datasetFolder: string) {
  const text = await fsp.readFile(encryptedManifestPath(datasetFolder), 'utf-8');
  return validateEncryptedManifest(JSON.parse(text));
}

function base64ToBuffer(value: string, fieldName: string) {
  if (typeof value !== 'string' || !/^[A-Za-z0-9+/]+={0,2}$/.test(value)) {
    throw new Error(`Invalid encrypted dataset ${fieldName}`);
  }
  return Buffer.from(value, 'base64');
}

function isBase64(value: unknown) {
  return typeof value === 'string' && /^[A-Za-z0-9+/]+={0,2}$/.test(value);
}

function isBase64Url(value: unknown) {
  return typeof value === 'string' && /^[A-Za-z0-9_-]+$/.test(value);
}

export function validateEncryptedCatalogKey(manifest: EncryptedDatasetManifest, keyB64: string) {
  validateEncryptedManifest(manifest);
  const key = base64ToBuffer(keyB64, 'key');
  if (key.length !== 32) {
    throw new Error('Encrypted dataset key must be 32 bytes');
  }

  const nonce = base64ToBuffer(manifest.catalog.nonce, 'catalog nonce');
  if (nonce.length !== AES_GCM_NONCE_BYTES) {
    throw new Error('Invalid encrypted dataset catalog nonce');
  }

  const encrypted = base64ToBuffer(manifest.catalog.data, 'catalog data');
  if (encrypted.length <= AES_GCM_AUTH_TAG_BYTES) {
    throw new Error('Invalid encrypted dataset catalog data');
  }

  const ciphertext = encrypted.subarray(0, encrypted.length - AES_GCM_AUTH_TAG_BYTES);
  const tag = encrypted.subarray(encrypted.length - AES_GCM_AUTH_TAG_BYTES);
  const decipher = crypto.createDecipheriv('aes-256-gcm', key, nonce, {
    authTagLength: AES_GCM_AUTH_TAG_BYTES,
  });
  decipher.setAAD(CATALOG_AAD);
  decipher.setAuthTag(tag);
  const plaintext = Buffer.concat([decipher.update(ciphertext), decipher.final()]);
  const catalog = JSON.parse(plaintext.toString('utf8'));
  if (catalog?.version !== 1 || !Array.isArray(catalog.items)) {
    throw new Error('Invalid encrypted dataset catalog');
  }
  return true;
}

export async function validateEncryptedDatasetStartKey(
  requiredDataset: { path: string; name: string },
  keyB64: string,
) {
  const manifest = await readEncryptedManifest(requiredDataset.path);
  return validateEncryptedCatalogKey(manifest, keyB64);
}

export async function writeEncryptedManifest(datasetFolder: string, manifest: EncryptedDatasetManifest) {
  validateEncryptedManifest(manifest);
  await fsp.writeFile(encryptedManifestPath(datasetFolder), JSON.stringify(manifest, null, 2), 'utf-8');
}

export async function listDatasetSummaries(datasetsRoot: string): Promise<DatasetSummary[]> {
  if (!fs.existsSync(datasetsRoot)) {
    await fsp.mkdir(datasetsRoot, { recursive: true });
  }
  const entries = await fsp.readdir(datasetsRoot, { withFileTypes: true });
  return entries
    .filter(entry => entry.isDirectory() && !entry.name.startsWith('.'))
    .map(entry => {
      const datasetFolder = path.join(datasetsRoot, entry.name);
      const encrypted = isEncryptedDatasetFolder(datasetFolder);
      if (encrypted) {
        return {
          name: entry.name,
          encrypted,
          itemCount: null,
          captionedItemCount: null,
          missingCaptionCount: null,
          detectedCaptionExt: null,
          source: 'local' as const,
          worker_id: 'local',
          worker_name: 'Local',
          ref: `aitk-dataset://local/${encodeURIComponent(entry.name)}`,
          path: datasetFolder,
        };
      }

      const captionSummary = summarizePlainDatasetCaptions(datasetFolder);
      return {
        name: entry.name,
        encrypted,
        itemCount: captionSummary.itemCount,
        captionedItemCount: captionSummary.captionedItemCount,
        missingCaptionCount: captionSummary.missingCaptionCount,
        detectedCaptionExt: captionSummary.detectedCaptionExt,
        source: 'local' as const,
        worker_id: 'local',
        worker_name: 'Local',
        ref: `aitk-dataset://local/${encodeURIComponent(entry.name)}`,
        path: datasetFolder,
      };
    })
    .sort((a, b) => a.name.localeCompare(b.name));
}

export function findEncryptedDatasetRoot(filePath: string, datasetsRoot: string) {
  const root = path.resolve(datasetsRoot);
  let current = path.resolve(filePath);
  if (!isPathInside(root, current)) return null;
  if (fs.existsSync(current) && fs.statSync(current).isFile()) {
    current = path.dirname(current);
  }

  while (isPathInside(root, current)) {
    if (isEncryptedDatasetFolder(current)) return current;
    const next = path.dirname(current);
    if (next === current) break;
    current = next;
  }
  return null;
}

function collectConfigPathValues(jobConfig: any) {
  const values: string[] = [];
  const processes = Array.isArray(jobConfig?.config?.process) ? jobConfig.config.process : [];

  processes.forEach((processConfig: any) => {
    const datasets = Array.isArray(processConfig?.datasets) ? processConfig.datasets : [];
    datasets.forEach((dataset: any) => {
      DATASET_CONFIG_FIELDS.forEach(field => {
        const value = dataset?.[field];
        if (typeof value === 'string' && value.trim()) {
          values.push(value);
        } else if (Array.isArray(value)) {
          value.forEach(item => {
            if (typeof item === 'string' && item.trim()) values.push(item);
          });
        }
      });
    });

    const captionPath = processConfig?.caption?.path_to_caption;
    if (typeof captionPath === 'string' && captionPath.trim()) {
      values.push(captionPath);
    }
  });

  return values;
}

export async function getEncryptedDatasetsForJobConfig(jobConfig: any) {
  const required = new Map<string, { path: string; name: string }>();
  for (const value of collectConfigPathValues(jobConfig)) {
    if (/^[a-z][a-z0-9+.-]*:\/\//i.test(value)) continue;
    const resolved = resolveConfigPath(value);
    const manifestTarget = fs.existsSync(resolved) && fs.statSync(resolved).isFile() ? path.dirname(resolved) : resolved;
    if (fs.existsSync(encryptedManifestPath(manifestTarget))) {
      const realPath = await fsp.realpath(manifestTarget).catch(() => path.resolve(manifestTarget));
      required.set(realPath, { path: realPath, name: path.basename(realPath) });
    }
  }
  return Array.from(required.values());
}

export function normalizeEncryptedKeyMap(keys: EncryptedDatasetStartKey[] | undefined | null) {
  const map = new Map<string, string>();
  if (!Array.isArray(keys)) return map;

  keys.forEach(key => {
    if (!key || typeof key.datasetPath !== 'string' || typeof key.keyB64 !== 'string') return;
    if (!/^[A-Za-z0-9+/=]+$/.test(key.keyB64) || key.keyB64.length > 2048) return;
    const normalizedPath = resolveConfigPath(key.datasetPath);
    map.set(path.resolve(normalizedPath).toLowerCase(), key.keyB64);
    map.set(path.basename(normalizedPath).toLowerCase(), key.keyB64);
  });

  return map;
}

export function getKeyForRequiredDataset(
  keyMap: Map<string, string>,
  requiredDataset: { path: string; name: string },
) {
  return keyMap.get(path.resolve(requiredDataset.path).toLowerCase()) || keyMap.get(requiredDataset.name.toLowerCase()) || null;
}
