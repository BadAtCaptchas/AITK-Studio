import crypto from 'crypto';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import type {
  DatasetSummary,
  EncryptedDatasetCatalog,
  EncryptedDatasetItem,
  EncryptedDatasetManifest,
} from '../types';
import {
  cleanDatasetName,
  encryptedManifestPath,
  isEncryptedDatasetFolder,
  readEncryptedManifest,
  resolveDatasetFolder,
  resolveEncryptedObjectPath,
  validateEncryptedCatalogKey,
  validateEncryptedManifest,
  writeEncryptedManifest,
} from './encryptedDatasets';

const AES_GCM_AUTH_TAG_BYTES = 16;
const AES_GCM_NONCE_BYTES = 12;
const CATALOG_AAD = 'aitk-encrypted-catalog:v1';

const MEDIA_EXTENSIONS = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
  '.gif',
  '.bmp',
  '.mp4',
  '.avi',
  '.mov',
  '.mkv',
  '.wmv',
  '.m4v',
  '.flv',
  '.webm',
  '.mp3',
  '.wav',
  '.flac',
  '.ogg',
  '.m4a',
  '.aac',
]);

const SKIPPED_DIR_NAMES = new Set(['_controls', '_latent_cache', '_clip_vision_cache', '_t_e_cache']);

type EncryptedPayload = {
  nonce: string;
  data: string;
};

export type DatasetCombineKey = {
  datasetName?: string;
  datasetPath?: string;
  keyB64: string;
};

export type DatasetCombineRequest = {
  sourceDatasets: string[];
  outputName: string;
  outputEncrypted?: boolean;
  encryptedDatasetKeys?: DatasetCombineKey[];
  outputEncryptedManifest?: EncryptedDatasetManifest;
  outputKeyB64?: string;
};

export type DatasetCombineResult = {
  dataset: DatasetSummary;
  sourceCount: number;
  itemCount: number;
  renamed: boolean;
};

export function datasetCombineRequestHasKeyMaterial(request: Partial<DatasetCombineRequest> | null | undefined) {
  return (
    (Array.isArray(request?.encryptedDatasetKeys) &&
      request.encryptedDatasetKeys.some(key => typeof key?.keyB64 === 'string' && key.keyB64.length > 0)) ||
    (typeof request?.outputKeyB64 === 'string' && request.outputKeyB64.length > 0)
  );
}

type SourceContext = {
  name: string;
  folder: string;
  encrypted: boolean;
  keyB64?: string;
  manifest?: EncryptedDatasetManifest;
  catalog?: EncryptedDatasetCatalog;
};

type PlainMediaSource = {
  type: 'plain';
  mediaPath: string;
  mediaName: string;
  captionPath?: string;
};

type EncryptedMediaSource = {
  type: 'encrypted';
  datasetFolder: string;
  keyB64: string;
  item: EncryptedDatasetItem;
};

type MediaSource = PlainMediaSource | EncryptedMediaSource;

type AllocatedName = {
  mediaName: string;
  captionName: string;
};

class DatasetCombineError extends Error {
  status: number;

  constructor(message: string, status = 400) {
    super(message);
    this.name = 'DatasetCombineError';
    this.status = status;
  }
}

export { DatasetCombineError };

function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}

function base64ToBuffer(value: string, fieldName: string) {
  if (typeof value !== 'string' || !/^[A-Za-z0-9+/]+={0,2}$/.test(value)) {
    throw new DatasetCombineError(`Invalid encrypted dataset ${fieldName}`);
  }
  const buffer = Buffer.from(value, 'base64');
  if (buffer.length !== 32) {
    throw new DatasetCombineError(`Encrypted dataset ${fieldName} must be 32 bytes`);
  }
  return buffer;
}

function decryptPayload(payload: EncryptedPayload, keyB64: string, aad: string) {
  const key = base64ToBuffer(keyB64, 'key');
  const nonce = Buffer.from(payload.nonce, 'base64');
  const encrypted = Buffer.from(payload.data, 'base64');
  if (nonce.length !== AES_GCM_NONCE_BYTES || encrypted.length <= AES_GCM_AUTH_TAG_BYTES) {
    throw new DatasetCombineError('Invalid encrypted dataset payload');
  }

  const ciphertext = encrypted.subarray(0, encrypted.length - AES_GCM_AUTH_TAG_BYTES);
  const tag = encrypted.subarray(encrypted.length - AES_GCM_AUTH_TAG_BYTES);
  const decipher = crypto.createDecipheriv('aes-256-gcm', key, nonce, {
    authTagLength: AES_GCM_AUTH_TAG_BYTES,
  });
  decipher.setAAD(Buffer.from(aad, 'utf8'));
  decipher.setAuthTag(tag);
  return Buffer.concat([decipher.update(ciphertext), decipher.final()]);
}

function encryptPayload(data: Buffer, keyB64: string, aad: string): EncryptedPayload {
  const key = base64ToBuffer(keyB64, 'key');
  const nonce = crypto.randomBytes(AES_GCM_NONCE_BYTES);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, nonce, {
    authTagLength: AES_GCM_AUTH_TAG_BYTES,
  });
  cipher.setAAD(Buffer.from(aad, 'utf8'));
  const ciphertext = Buffer.concat([cipher.update(data), cipher.final(), cipher.getAuthTag()]);
  return {
    nonce: nonce.toString('base64'),
    data: ciphertext.toString('base64'),
  };
}

function decryptCatalog(manifest: EncryptedDatasetManifest, keyB64: string): EncryptedDatasetCatalog {
  const plaintext = decryptPayload(manifest.catalog, keyB64, CATALOG_AAD);
  const catalog = JSON.parse(plaintext.toString('utf8')) as EncryptedDatasetCatalog;
  if (catalog?.version !== 1 || !Array.isArray(catalog.items)) {
    throw new DatasetCombineError('Invalid encrypted dataset catalog');
  }
  return catalog;
}

function objectAad(objectPath: string) {
  return `aitk-encrypted-object:${objectPath.replace(/\\/g, '/')}`;
}

function randomId(bytes = 16) {
  return crypto.randomBytes(bytes).toString('base64url');
}

function mediaObjectPath(itemId: string) {
  return `objects/${itemId}.bin`;
}

function captionObjectPath(itemId: string) {
  return `objects/${itemId}.caption.bin`;
}

function mimeTypeForExtension(ext: string) {
  const normalized = ext.toLowerCase();
  const map: Record<string, string> = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp',
    '.mp4': 'video/mp4',
    '.avi': 'video/x-msvideo',
    '.mov': 'video/quicktime',
    '.mkv': 'video/x-matroska',
    '.wmv': 'video/x-ms-wmv',
    '.m4v': 'video/x-m4v',
    '.flv': 'video/x-flv',
    '.webm': 'video/webm',
    '.mp3': 'audio/mpeg',
    '.wav': 'audio/wav',
    '.flac': 'audio/flac',
    '.ogg': 'audio/ogg',
    '.m4a': 'audio/mp4',
    '.aac': 'audio/aac',
  };
  return map[normalized] || 'application/octet-stream';
}

function mediaKindForExtension(ext: string): EncryptedDatasetItem['mediaKind'] {
  const normalized = ext.toLowerCase();
  if (['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.webm'].includes(normalized)) {
    return 'video';
  }
  if (['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'].includes(normalized)) {
    return 'audio';
  }
  return 'image';
}

function sanitizeFileName(fileName: string, fallbackExt = '.bin') {
  const parsed = path.parse(path.basename(fileName || ''));
  const ext = parsed.ext && /^\.[a-zA-Z0-9_-]+$/.test(parsed.ext) ? parsed.ext.toLowerCase() : fallbackExt;
  const stem = (parsed.name || 'item')
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, '_')
    .replace(/\s+/g, '_')
    .replace(/^\.+/, '')
    .slice(0, 100);
  return `${stem || 'item'}${ext}`;
}

class UniqueNameAllocator {
  private usedStems = new Set<string>();

  allocate(fileName: string, fallbackExt = '.bin'): AllocatedName {
    const safeName = sanitizeFileName(fileName, fallbackExt);
    const parsed = path.parse(safeName);
    const baseStem = parsed.name || 'item';
    const ext = parsed.ext || fallbackExt;
    let candidateStem = baseStem;
    let suffix = 2;

    while (this.usedStems.has(candidateStem.toLowerCase())) {
      candidateStem = `${baseStem}_${suffix}`;
      suffix += 1;
    }

    this.usedStems.add(candidateStem.toLowerCase());
    return {
      mediaName: `${candidateStem}${ext}`,
      captionName: `${candidateStem}.txt`,
    };
  }
}

function normalizeSourceNames(sourceDatasets: unknown) {
  if (!Array.isArray(sourceDatasets)) {
    throw new DatasetCombineError('sourceDatasets must be an array');
  }

  const names = sourceDatasets
    .map(value => (typeof value === 'string' ? value.trim() : ''))
    .filter(Boolean);
  const unique = Array.from(new Set(names));
  if (unique.length < 2) {
    throw new DatasetCombineError('Select at least two datasets to combine');
  }
  return unique;
}

function normalizeKeyLookup(keys: DatasetCombineKey[] | undefined, datasetsRoot: string) {
  const keyMap = new Map<string, string>();
  if (!Array.isArray(keys)) return keyMap;

  for (const key of keys) {
    if (!key || typeof key.keyB64 !== 'string') continue;
    if (typeof key.datasetName === 'string' && key.datasetName.trim()) {
      keyMap.set(key.datasetName.trim().toLowerCase(), key.keyB64);
    }
    if (typeof key.datasetPath === 'string' && key.datasetPath.trim()) {
      const resolved = path.isAbsolute(key.datasetPath)
        ? path.resolve(key.datasetPath)
        : path.resolve(datasetsRoot, key.datasetPath);
      keyMap.set(resolved.toLowerCase(), key.keyB64);
      keyMap.set(path.basename(resolved).toLowerCase(), key.keyB64);
    }
  }

  return keyMap;
}

async function resolveSources(datasetsRoot: string, sourceNames: string[], keyMap: Map<string, string>) {
  const root = path.resolve(datasetsRoot);
  const sources: SourceContext[] = [];

  for (const name of sourceNames) {
    const folder = resolveDatasetFolder(root, name);
    if (!fs.existsSync(folder) || !fs.statSync(folder).isDirectory()) {
      throw new DatasetCombineError(`Dataset not found: ${name}`, 404);
    }

    const encrypted = isEncryptedDatasetFolder(folder);
    if (!encrypted) {
      sources.push({ name, folder, encrypted: false });
      continue;
    }

    const manifest = await readEncryptedManifest(folder);
    const keyB64 = keyMap.get(name.toLowerCase()) || keyMap.get(path.resolve(folder).toLowerCase());
    if (!keyB64) {
      throw new DatasetCombineError(`Encrypted dataset key is required for "${name}"`, 403);
    }
    validateEncryptedCatalogKey(manifest, keyB64);
    sources.push({
      name,
      folder,
      encrypted: true,
      keyB64,
      manifest,
      catalog: decryptCatalog(manifest, keyB64),
    });
  }

  return sources;
}

async function listPlainMediaSources(datasetFolder: string): Promise<PlainMediaSource[]> {
  const root = path.resolve(datasetFolder);
  const results: PlainMediaSource[] = [];

  async function walk(current: string) {
    const entries = await fsp.readdir(current, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));

    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue;
      const absolutePath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        if (SKIPPED_DIR_NAMES.has(entry.name)) continue;
        await walk(absolutePath);
        continue;
      }
      if (!entry.isFile()) continue;
      const ext = path.extname(entry.name).toLowerCase();
      if (!MEDIA_EXTENSIONS.has(ext)) continue;

      const parsed = path.parse(absolutePath);
      const captionPath = path.join(parsed.dir, `${parsed.name}.txt`);
      results.push({
        type: 'plain',
        mediaPath: absolutePath,
        mediaName: entry.name,
        captionPath:
          isPathInside(root, captionPath) && fs.existsSync(captionPath) && fs.statSync(captionPath).isFile()
            ? captionPath
            : undefined,
      });
    }
  }

  await walk(root);
  return results;
}

async function collectMediaSources(sources: SourceContext[]): Promise<MediaSource[]> {
  const mediaSources: MediaSource[] = [];

  for (const source of sources) {
    if (source.encrypted) {
      const keyB64 = source.keyB64 as string;
      const catalog = source.catalog as EncryptedDatasetCatalog;
      for (const item of catalog.items) {
        mediaSources.push({
          type: 'encrypted',
          datasetFolder: source.folder,
          keyB64,
          item,
        });
      }
      continue;
    }
    mediaSources.push(...(await listPlainMediaSources(source.folder)));
  }

  return mediaSources;
}

function encryptedObjectBytes(datasetFolder: string, objectPath: string, keyB64: string) {
  const resolved = resolveEncryptedObjectPath(datasetFolder, objectPath);
  const payload = JSON.parse(fs.readFileSync(resolved, 'utf-8')) as EncryptedPayload;
  return decryptPayload(payload, keyB64, objectAad(objectPath));
}

function encryptedCaptionBytes(source: EncryptedMediaSource) {
  if (!source.item.captionObjectPath) return null;
  return encryptedObjectBytes(source.datasetFolder, source.item.captionObjectPath, source.keyB64);
}

async function mediaBytes(source: MediaSource) {
  if (source.type === 'plain') {
    return fsp.readFile(source.mediaPath);
  }
  return encryptedObjectBytes(source.datasetFolder, source.item.objectPath, source.keyB64);
}

async function captionBytes(source: MediaSource) {
  if (source.type === 'plain') {
    if (!source.captionPath) return null;
    return fsp.readFile(source.captionPath);
  }
  return encryptedCaptionBytes(source);
}

function sourceFileName(source: MediaSource) {
  if (source.type === 'plain') return source.mediaName;
  return source.item.name || `${source.item.id}${source.item.extension || '.bin'}`;
}

function sourceExtension(source: MediaSource) {
  if (source.type === 'plain') return path.extname(source.mediaName).toLowerCase() || '.bin';
  return source.item.extension || path.extname(source.item.name || '') || '.bin';
}

async function writePlainOutput(outputFolder: string, mediaSources: MediaSource[]) {
  const allocator = new UniqueNameAllocator();
  let itemCount = 0;

  for (const source of mediaSources) {
    const names = allocator.allocate(sourceFileName(source), sourceExtension(source));
    const targetMediaPath = path.join(outputFolder, names.mediaName);
    const targetCaptionPath = path.join(outputFolder, names.captionName);

    if (source.type === 'plain') {
      await fsp.copyFile(source.mediaPath, targetMediaPath, fs.constants.COPYFILE_EXCL);
    } else {
      await fsp.writeFile(targetMediaPath, await mediaBytes(source), { flag: 'wx' });
    }

    const caption = await captionBytes(source);
    if (caption !== null) {
      await fsp.writeFile(targetCaptionPath, caption, { flag: 'wx' });
    }
    itemCount += 1;
  }

  return itemCount;
}

async function writeEncryptedObject(outputFolder: string, objectPath: string, data: Buffer, keyB64: string) {
  const payload = encryptPayload(data, keyB64, objectAad(objectPath));
  const targetPath = path.join(outputFolder, ...objectPath.split('/'));
  await fsp.mkdir(path.dirname(targetPath), { recursive: true });
  await fsp.writeFile(targetPath, JSON.stringify(payload), { flag: 'wx' });
}

function copyEncryptedMetadata(source: MediaSource, item: EncryptedDatasetItem) {
  if (source.type !== 'encrypted') return item;
  return {
    ...item,
    width: source.item.width,
    height: source.item.height,
    durationMs: source.item.durationMs,
  };
}

async function writeEncryptedOutput(
  outputFolder: string,
  mediaSources: MediaSource[],
  manifest: EncryptedDatasetManifest,
  keyB64: string,
) {
  const allocator = new UniqueNameAllocator();
  const now = new Date().toISOString();
  const items: EncryptedDatasetItem[] = [];

  await fsp.mkdir(path.join(outputFolder, 'objects'), { recursive: true });
  for (const source of mediaSources) {
    const names = allocator.allocate(sourceFileName(source), sourceExtension(source));
    const ext = path.extname(names.mediaName).toLowerCase() || sourceExtension(source);
    const itemId = randomId();
    const objectPath = mediaObjectPath(itemId);
    const media = await mediaBytes(source);

    await writeEncryptedObject(outputFolder, objectPath, media, keyB64);

    let item: EncryptedDatasetItem = {
      id: itemId,
      name: names.mediaName,
      extension: ext,
      mimeType: source.type === 'encrypted' ? source.item.mimeType : mimeTypeForExtension(ext),
      mediaKind: source.type === 'encrypted' ? source.item.mediaKind : mediaKindForExtension(ext),
      objectPath,
      size: media.length,
      createdAt: now,
      updatedAt: now,
    };
    item = copyEncryptedMetadata(source, item);

    const caption = await captionBytes(source);
    if (caption !== null) {
      const captionPath = captionObjectPath(randomId());
      await writeEncryptedObject(outputFolder, captionPath, caption, keyB64);
      item.captionObjectPath = captionPath;
    }

    items.push(item);
  }

  const catalog: EncryptedDatasetCatalog = {
    version: 1,
    items,
  };
  const nextManifest: EncryptedDatasetManifest = {
    ...manifest,
    catalog: encryptPayload(Buffer.from(JSON.stringify(catalog), 'utf8'), keyB64, CATALOG_AAD),
  };
  await writeEncryptedManifest(outputFolder, nextManifest);
  return items.length;
}

function nextDatasetPath(datasetsRoot: string, requestedName: string) {
  const outputName = cleanDatasetName(requestedName || '');
  if (!outputName) {
    throw new DatasetCombineError('Output dataset name is required');
  }

  const root = path.resolve(datasetsRoot);
  let candidateName = outputName;
  let candidatePath = resolveDatasetFolder(root, candidateName);
  let suffix = 2;
  while (fs.existsSync(candidatePath)) {
    candidateName = `${outputName}_${suffix}`;
    candidatePath = resolveDatasetFolder(root, candidateName);
    suffix += 1;
  }
  return { requestedName: outputName, finalName: candidateName, finalPath: candidatePath };
}

function validateOutputEncryption(request: DatasetCombineRequest) {
  if (!request.outputEncrypted) return null;
  if (!request.outputEncryptedManifest || typeof request.outputKeyB64 !== 'string') {
    throw new DatasetCombineError('Encrypted output requires a manifest and key');
  }
  const manifest = validateEncryptedManifest(request.outputEncryptedManifest);
  validateEncryptedCatalogKey(manifest, request.outputKeyB64);
  return { manifest, keyB64: request.outputKeyB64 };
}

export async function combineDatasets(datasetsRoot: string, request: DatasetCombineRequest): Promise<DatasetCombineResult> {
  const root = path.resolve(datasetsRoot);
  await fsp.mkdir(root, { recursive: true });

  const sourceNames = normalizeSourceNames(request.sourceDatasets);
  const keyMap = normalizeKeyLookup(request.encryptedDatasetKeys, root);
  const outputEncryption = validateOutputEncryption(request);
  const sources = await resolveSources(root, sourceNames, keyMap);
  const mediaSources = await collectMediaSources(sources);
  if (mediaSources.length === 0) {
    throw new DatasetCombineError('No supported media files were found in the selected datasets', 404);
  }

  const { requestedName, finalName, finalPath } = nextDatasetPath(root, request.outputName);
  const tempPath = path.join(root, `.aitk-dataset-combine-${Date.now()}-${randomId(8)}`);

  try {
    await fsp.mkdir(tempPath, { recursive: false });
    let itemCount = 0;
    if (outputEncryption) {
      itemCount = await writeEncryptedOutput(tempPath, mediaSources, outputEncryption.manifest, outputEncryption.keyB64);
    } else {
      itemCount = await writePlainOutput(tempPath, mediaSources);
    }

    if (fs.existsSync(finalPath)) {
      throw new DatasetCombineError('Output dataset already exists', 409);
    }
    await fsp.rename(tempPath, finalPath);

    return {
      dataset: {
        name: finalName,
        encrypted: request.outputEncrypted === true,
        itemCount,
        source: 'local',
        worker_id: 'local',
        worker_name: 'Local',
        ref: `aitk-dataset://local/${encodeURIComponent(finalName)}`,
        path: finalPath,
      },
      sourceCount: sources.length,
      itemCount,
      renamed: requestedName !== finalName,
    };
  } catch (error) {
    await fsp.rm(tempPath, { recursive: true, force: true }).catch(() => undefined);
    if (error instanceof DatasetCombineError) throw error;
    throw error;
  }
}

export function isDatasetCombineError(error: unknown): error is DatasetCombineError {
  return error instanceof DatasetCombineError;
}

export function hasPlaintextEncryptedOutputFiles(datasetFolder: string) {
  if (!fs.existsSync(encryptedManifestPath(datasetFolder))) return false;
  const entries = fs.readdirSync(datasetFolder).filter(entry => !entry.startsWith('.'));
  return entries.some(entry => entry !== 'objects');
}
