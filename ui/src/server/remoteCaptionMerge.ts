import crypto from 'crypto';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import {
  isEncryptedDatasetFolder,
  readEncryptedManifest,
  resolveEncryptedObjectPath,
  writeEncryptedManifest,
} from './encryptedDatasets';
import type { EncryptedDatasetCatalog, EncryptedDatasetManifest } from '../types';

const CATALOG_AAD = Buffer.from('aitk-encrypted-catalog:v1', 'utf8');
const AES_GCM_NONCE_BYTES = 12;
const AES_GCM_AUTH_TAG_BYTES = 16;
const DATASET_CACHE_DIR_NAMES = new Set(['_latent_cache', '_clip_vision_cache', '_t_e_cache']);

function nowIso() {
  return new Date().toISOString();
}

function captionExtSuffix(captionExtension: string) {
  const normalized = captionExtension.trim().replace(/^\.+/, '').toLowerCase() || 'txt';
  return `.${normalized}`;
}

function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}

function shouldIncludeDatasetPath(relativePath: string) {
  const normalized = relativePath.replace(/\\/g, '/');
  if (!normalized) return true;
  return !normalized.split('/').some(segment => DATASET_CACHE_DIR_NAMES.has(segment));
}

async function listFilesRecursive(root: string) {
  const files: string[] = [];

  async function walk(current: string) {
    const entries = await fsp.readdir(current, { withFileTypes: true });
    for (const entry of entries) {
      const absolutePath = path.join(current, entry.name);
      const relativePath = path.relative(root, absolutePath);
      if (!shouldIncludeDatasetPath(relativePath)) continue;
      if (entry.isDirectory()) {
        await walk(absolutePath);
      } else if (entry.isFile()) {
        files.push(relativePath);
      }
    }
  }

  if (fs.existsSync(root)) {
    await walk(root);
  }
  return files;
}

function base64ToBuffer(value: string, fieldName: string) {
  if (typeof value !== 'string' || !/^[A-Za-z0-9+/]+={0,2}$/.test(value)) {
    throw new Error(`Invalid encrypted dataset ${fieldName}`);
  }
  return Buffer.from(value, 'base64');
}

function decryptPayload(payload: { nonce: string; data: string }, key: Buffer, aad: Buffer) {
  const nonce = base64ToBuffer(payload.nonce, 'nonce');
  const encrypted = base64ToBuffer(payload.data, 'data');
  if (nonce.length !== AES_GCM_NONCE_BYTES || encrypted.length <= AES_GCM_AUTH_TAG_BYTES) {
    throw new Error('Invalid encrypted dataset payload');
  }
  const ciphertext = encrypted.subarray(0, encrypted.length - AES_GCM_AUTH_TAG_BYTES);
  const tag = encrypted.subarray(encrypted.length - AES_GCM_AUTH_TAG_BYTES);
  const decipher = crypto.createDecipheriv('aes-256-gcm', key, nonce, {
    authTagLength: AES_GCM_AUTH_TAG_BYTES,
  });
  decipher.setAAD(aad);
  decipher.setAuthTag(tag);
  return Buffer.concat([decipher.update(ciphertext), decipher.final()]);
}

function encryptPayload(plaintext: Buffer, key: Buffer, aad: Buffer) {
  const nonce = crypto.randomBytes(AES_GCM_NONCE_BYTES);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, nonce, {
    authTagLength: AES_GCM_AUTH_TAG_BYTES,
  });
  cipher.setAAD(aad);
  const ciphertext = Buffer.concat([cipher.update(plaintext), cipher.final()]);
  const tag = cipher.getAuthTag();
  return {
    nonce: nonce.toString('base64'),
    data: Buffer.concat([ciphertext, tag]).toString('base64'),
  };
}

function decryptCatalog(manifest: EncryptedDatasetManifest, keyB64: string): EncryptedDatasetCatalog {
  const key = base64ToBuffer(keyB64, 'key');
  if (key.length !== 32) throw new Error('Encrypted dataset key must be 32 bytes');
  const catalog = JSON.parse(decryptPayload(manifest.catalog, key, CATALOG_AAD).toString('utf8'));
  if (catalog?.version !== 1 || !Array.isArray(catalog.items)) {
    throw new Error('Invalid encrypted dataset catalog');
  }
  return catalog as EncryptedDatasetCatalog;
}

function encryptCatalog(catalog: EncryptedDatasetCatalog, manifest: EncryptedDatasetManifest, keyB64: string) {
  const key = base64ToBuffer(keyB64, 'key');
  if (key.length !== 32) throw new Error('Encrypted dataset key must be 32 bytes');
  return {
    ...manifest,
    catalog: encryptPayload(Buffer.from(JSON.stringify(catalog), 'utf8'), key, CATALOG_AAD),
  };
}

export async function mergePlainCaptionDataset(
  sourceDatasetPath: string,
  targetDatasetPath: string,
  options: { captionExtension: string; recaption: boolean },
) {
  const sourceRoot = path.resolve(sourceDatasetPath);
  const targetRoot = path.resolve(targetDatasetPath);
  const suffix = captionExtSuffix(options.captionExtension);
  const files = await listFilesRecursive(sourceRoot);
  let copied = 0;
  let skipped = 0;

  for (const relativePath of files) {
    if (!relativePath.toLowerCase().endsWith(suffix)) continue;
    const sourcePath = path.resolve(sourceRoot, relativePath);
    const targetPath = path.resolve(targetRoot, ...relativePath.replace(/\\/g, '/').split('/'));
    if (!isPathInside(targetRoot, targetPath)) continue;
    if (!options.recaption && fs.existsSync(targetPath)) {
      skipped += 1;
      continue;
    }
    await fsp.mkdir(path.dirname(targetPath), { recursive: true });
    await fsp.copyFile(sourcePath, targetPath);
    copied += 1;
  }

  return { copied, skipped };
}

export async function mergeEncryptedCaptionDataset(
  sourceDatasetPath: string,
  targetDatasetPath: string,
  options: { keyB64: string; recaption: boolean },
) {
  if (!isEncryptedDatasetFolder(sourceDatasetPath) || !isEncryptedDatasetFolder(targetDatasetPath)) {
    throw new Error('Encrypted caption merge requires encrypted source and target datasets');
  }

  const sourceManifest = await readEncryptedManifest(sourceDatasetPath);
  const targetManifest = await readEncryptedManifest(targetDatasetPath);
  const sourceCatalog = decryptCatalog(sourceManifest, options.keyB64);
  const targetCatalog = decryptCatalog(targetManifest, options.keyB64);
  const targetItemsById = new Map(targetCatalog.items.map(item => [item.id, item]));
  let copied = 0;
  let skipped = 0;

  for (const sourceItem of sourceCatalog.items) {
    if (!sourceItem.captionObjectPath) continue;
    const targetItem = targetItemsById.get(sourceItem.id);
    if (!targetItem) continue;
    if (!options.recaption && targetItem.captionObjectPath) {
      skipped += 1;
      continue;
    }

    const sourceObjectPath = resolveEncryptedObjectPath(sourceDatasetPath, sourceItem.captionObjectPath);
    const targetObjectPath = resolveEncryptedObjectPath(targetDatasetPath, sourceItem.captionObjectPath);
    if (!fs.existsSync(sourceObjectPath)) continue;
    await fsp.mkdir(path.dirname(targetObjectPath), { recursive: true });
    await fsp.copyFile(sourceObjectPath, targetObjectPath);
    targetItem.captionObjectPath = sourceItem.captionObjectPath;
    targetItem.updatedAt = sourceItem.updatedAt || nowIso();
    copied += 1;
  }

  const updatedManifest = encryptCatalog(targetCatalog, targetManifest, options.keyB64);
  await writeEncryptedManifest(targetDatasetPath, updatedManifest);
  return { copied, skipped };
}
