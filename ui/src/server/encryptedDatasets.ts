import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { TOOLKIT_ROOT } from '../paths';
import type { DatasetSummary, EncryptedDatasetManifest, EncryptedDatasetStartKey } from '../types';

export const ENCRYPTED_DATASET_MANIFEST = '.aitk_encrypted_dataset.json';

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

function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
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
  if (kdfType !== 'PBKDF2-SHA256' && kdfType !== 'KEYFILE-SHA256') {
    throw new Error('Unsupported encrypted dataset KDF');
  }
  if (kdfType === 'PBKDF2-SHA256') {
    const kdf = manifest.crypto.kdf;
    if (!kdf.salt || !Number.isFinite(kdf.iterations) || kdf.iterations < 100_000 || kdf.keyLength !== 32) {
      throw new Error('Invalid encrypted dataset password KDF header');
    }
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
      return {
        name: entry.name,
        encrypted,
        itemCount: encrypted ? null : undefined,
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
