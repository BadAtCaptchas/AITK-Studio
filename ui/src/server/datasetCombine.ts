import crypto from 'crypto';
import { DATASET_MIGRATION_MESSAGE, DATASET_MIGRATION_REQUIRED } from './datasetDecryption';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { copyLayeredImage, listLayeredImages } from './layeredImages';
import type { LayeredImageManifest } from '../domain/layeredImages';
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
} from './encryptedDatasets';

const AES_GCM_AUTH_TAG_BYTES = 16;
const AES_GCM_NONCE_BYTES = 12;
const CATALOG_AAD = 'aitk-encrypted-catalog:v1';

const MEDIA_EXTENSIONS = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
  '.jxl',
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

type SourceContext = { name: string; folder: string };

type PlainMediaSource = {
  type: 'plain';
  mediaPath: string;
  mediaName: string;
  captionPath?: string;
  datasetFolder: string;
  layered?: LayeredImageManifest;
};

type MediaSource = PlainMediaSource;

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

function randomId(bytes = 16) {
  return crypto.randomBytes(bytes).toString('base64url');
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

async function resolveSources(datasetsRoot: string, sourceNames: string[]): Promise<SourceContext[]> {
  const sources: SourceContext[] = [];
  for (const name of sourceNames) {
    const folder = resolveDatasetFolder(datasetsRoot, name);
    if (!fs.existsSync(folder) || !fs.statSync(folder).isDirectory()) throw new DatasetCombineError(`Dataset not found: ${name}`, 404);
    if (isEncryptedDatasetFolder(folder)) throw new DatasetCombineError(DATASET_MIGRATION_REQUIRED, 410);
    sources.push({ name, folder });
  }
  return sources;
}

async function listPlainMediaSources(datasetFolder: string): Promise<PlainMediaSource[]> {
  const root = path.resolve(datasetFolder);
  const results: PlainMediaSource[] = [];
  const groups = new Map((await listLayeredImages(root)).map(group => [path.resolve(root, group.composite), group]));

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
        datasetFolder: root,
        layered: groups.get(absolutePath),
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
  const items: MediaSource[] = [];
  for (const source of sources) items.push(...await listPlainMediaSources(source.folder));
  return items;
}

async function writePlainOutput(outputFolder: string, mediaSources: MediaSource[]) {
  const allocator = new UniqueNameAllocator();
  let itemCount = 0;

  for (const source of mediaSources) {
    const names = allocator.allocate(source.mediaName, path.extname(source.mediaName).toLowerCase() || '.bin');
    if (source.type === 'plain' && source.layered) {
      await copyLayeredImage(source.datasetFolder, source.layered, outputFolder, names.mediaName);
      itemCount += 1;
      continue;
    }
    const targetMediaPath = path.join(outputFolder, names.mediaName);
    const targetCaptionPath = path.join(outputFolder, names.captionName);

    await fsp.copyFile(source.mediaPath, targetMediaPath, fs.constants.COPYFILE_EXCL);
    const caption = source.captionPath ? await fsp.readFile(source.captionPath) : null;
    if (caption !== null) {
      await fsp.writeFile(targetCaptionPath, caption, { flag: 'wx' });
    }
    itemCount += 1;
  }

  return itemCount;
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

export async function combineDatasets(datasetsRoot: string, request: DatasetCombineRequest): Promise<DatasetCombineResult> {
  if (request.outputEncrypted || request.outputEncryptedManifest || request.outputKeyB64 || request.sourceDatasets.some(name => isEncryptedDatasetFolder(resolveDatasetFolder(datasetsRoot, name)))) throw new DatasetCombineError(DATASET_MIGRATION_REQUIRED + ': ' + DATASET_MIGRATION_MESSAGE, 410);
  const root = path.resolve(datasetsRoot);
  await fsp.mkdir(root, { recursive: true });

  const sourceNames = normalizeSourceNames(request.sourceDatasets);
  const sources = await resolveSources(root, sourceNames);
  const mediaSources = await collectMediaSources(sources);
  if (mediaSources.length === 0) {
    throw new DatasetCombineError('No supported media files were found in the selected datasets', 404);
  }

  const { requestedName, finalName, finalPath } = nextDatasetPath(root, request.outputName);
  const tempPath = path.join(root, `.aitk-dataset-combine-${Date.now()}-${randomId(8)}`);

  try {
    await fsp.mkdir(tempPath, { recursive: false });
    let itemCount = 0;
    itemCount = await writePlainOutput(tempPath, mediaSources);

    if (fs.existsSync(finalPath)) {
      throw new DatasetCombineError('Output dataset already exists', 409);
    }
    await fsp.rename(tempPath, finalPath);

    return {
      dataset: {
        name: finalName,
        encrypted: false,
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
