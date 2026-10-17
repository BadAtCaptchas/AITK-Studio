import crypto from 'crypto';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import {
  isEncryptedDatasetFolder,
  readEncryptedManifest,
  resolveEncryptedObjectPath,
} from './encryptedDatasets';
import {
  hasMatchingTargetMediaFile,
  isPathInside,
  normalizeRemoteCaptionExtension,
} from './remoteCaptionSecurity';
import type { EncryptedDatasetCatalog, EncryptedDatasetManifest } from '../types';

const CATALOG_AAD = Buffer.from('aitk-encrypted-catalog:v1', 'utf8');
const AES_GCM_NONCE_BYTES = 12;
const AES_GCM_AUTH_TAG_BYTES = 16;
const DATASET_CACHE_DIR_NAMES = new Set(['_latent_cache', '_clip_vision_cache', '_t_e_cache']);

function nowIso() {
  return new Date().toISOString();
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

export async function mergePlainCaptionDataset(
  sourceDatasetPath: string,
  targetDatasetPath: string,
  options: { captionExtension: string; recaption: boolean },
) {
  const sourceRoot = path.resolve(sourceDatasetPath);
  const targetRoot = path.resolve(targetDatasetPath);
  const suffix = normalizeRemoteCaptionExtension(options.captionExtension);
  const files = await listFilesRecursive(sourceRoot);
  let copied = 0;
  let skipped = 0;

  for (const relativePath of files) {
    if (!hasMatchingTargetMediaFile(targetRoot, relativePath, suffix)) continue;
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
  _sourceDatasetPath: string, _targetDatasetPath: string, _options: { keyB64: string; recaption: boolean },
): Promise<{ copied: number; skipped: number }> {
  throw new Error('ENCRYPTED_DATASET_MIGRATION_REQUIRED: Legacy encrypted caption results require dataset migration.');
}
