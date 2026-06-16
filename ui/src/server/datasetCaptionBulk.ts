import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import {
  DATASET_CAPTION_SIDECAR_EXTENSIONS,
  DATASET_TEXT_CAPTION_EXTENSIONS,
  captionSidecarPath,
  deleteCaptionSidecars,
  findExistingCaptionSidecar,
  readCaptionSidecar,
} from './captionFiles';
import { cleanDatasetName, isEncryptedDatasetFolder, resolveDatasetFolder } from './encryptedDatasets';
import {
  captionMatchesKeywords,
  parseCaptionKeywordQuery,
  removeCaptionKeywords,
  type CaptionKeywordMatchMode,
} from '../utils/captionKeywordSearch';
import { parseRemoteDatasetAssetRef } from '../utils/remoteDatasetRefs';

export type DatasetCaptionBulkAction = 'delete' | 'move' | 'remove_words';

export type DatasetCaptionBulkRequest = {
  datasetName: string;
  action: DatasetCaptionBulkAction;
  imgPaths: string[];
  query: string;
  matchMode?: CaptionKeywordMatchMode;
  destinationName?: string;
};

export type DatasetCaptionBulkResult = {
  action: DatasetCaptionBulkAction;
  found: number;
  affected: number;
  deleted?: number;
  moved?: number;
  updated?: number;
  removedWords?: number;
  destinationName?: string;
  updatedCaptions?: Record<string, string>;
  removedPaths?: string[];
};

export class DatasetCaptionBulkError extends Error {
  status: number;

  constructor(message: string, status = 400) {
    super(message);
    this.name = 'DatasetCaptionBulkError';
    this.status = status;
  }
}

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
  ...DATASET_TEXT_CAPTION_EXTENSIONS,
]);

type ResolvedMedia = {
  path: string;
  caption: string;
};

function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative !== '' && !relative.startsWith('..') && !path.isAbsolute(relative);
}

function normalizeMatchMode(value: unknown): CaptionKeywordMatchMode {
  return value === 'partial' ? 'partial' : 'whole-word';
}

function ensureAction(value: unknown): DatasetCaptionBulkAction {
  if (value === 'delete' || value === 'move' || value === 'remove_words') return value;
  throw new DatasetCaptionBulkError('Invalid bulk caption action');
}

function ensureImgPaths(value: unknown) {
  if (!Array.isArray(value)) throw new DatasetCaptionBulkError('imgPaths must be an array');
  const paths = value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0);
  if (paths.length === 0) throw new DatasetCaptionBulkError('No images were selected');
  return Array.from(new Set(paths));
}

function resolveMediaPaths(datasetFolder: string, imgPaths: string[]) {
  const resolvedFolder = path.resolve(datasetFolder);
  return imgPaths.map(imgPath => {
    const resolved = path.resolve(imgPath);
    if (!isPathInside(resolvedFolder, resolved)) {
      throw new DatasetCaptionBulkError('Invalid image path');
    }
    const ext = path.extname(resolved).toLowerCase();
    if (!MEDIA_EXTENSIONS.has(ext)) {
      throw new DatasetCaptionBulkError('Unsupported dataset item path');
    }
    if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
      throw new DatasetCaptionBulkError('Dataset item does not exist', 404);
    }
    return resolved;
  });
}

function existingCaptionSidecars(mediaPath: string) {
  if (DATASET_TEXT_CAPTION_EXTENSIONS.includes(path.extname(mediaPath).toLowerCase())) return [];

  return DATASET_CAPTION_SIDECAR_EXTENSIONS.flatMap(extension => {
    const candidate = captionSidecarPath(mediaPath, extension);
    return fs.existsSync(candidate) && fs.statSync(candidate).isFile() ? [candidate] : [];
  });
}

function uniqueDatasetDestination(datasetsRoot: string, requestedName: string) {
  const baseName = cleanDatasetName(requestedName || '');
  if (!baseName) throw new DatasetCaptionBulkError('Destination dataset name is required');
  let finalName = baseName;
  let finalPath = resolveDatasetFolder(datasetsRoot, finalName);
  let counter = 2;
  while (fs.existsSync(finalPath)) {
    finalName = `${baseName}_${counter}`;
    finalPath = resolveDatasetFolder(datasetsRoot, finalName);
    counter += 1;
  }
  return { finalName, finalPath };
}

function uniqueDestinationMediaPath(destinationFolder: string, mediaPath: string, usedStems: Set<string>) {
  const parsed = path.parse(path.basename(mediaPath));
  const safeStem = (parsed.name || 'item')
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, '_')
    .replace(/\s+/g, '_')
    .replace(/^\.+/, '')
    .slice(0, 100) || 'item';
  const ext = parsed.ext.toLowerCase() || '.bin';
  let stem = safeStem;
  let counter = 2;
  while (usedStems.has(stem.toLowerCase()) || fs.existsSync(path.join(destinationFolder, `${stem}${ext}`))) {
    stem = `${safeStem}_${counter}`;
    counter += 1;
  }
  usedStems.add(stem.toLowerCase());
  return path.join(destinationFolder, `${stem}${ext}`);
}

function defaultMoveName(datasetName: string) {
  return `${datasetName}_matches`;
}

function matchedMedia(mediaPaths: string[], terms: string[], matchMode: CaptionKeywordMatchMode) {
  const matched: ResolvedMedia[] = [];
  mediaPaths.forEach(mediaPath => {
    const caption = readCaptionSidecar(mediaPath);
    if (captionMatchesKeywords(caption, terms, matchMode)) {
      matched.push({ path: mediaPath, caption });
    }
  });
  return matched;
}

async function performDelete(matched: ResolvedMedia[]) {
  let deleted = 0;
  const removedPaths: string[] = [];
  for (const item of matched) {
    await fsp.unlink(item.path);
    if (!DATASET_TEXT_CAPTION_EXTENSIONS.includes(path.extname(item.path).toLowerCase())) {
      deleteCaptionSidecars(item.path);
    }
    deleted += 1;
    removedPaths.push(item.path);
  }
  return { deleted, removedPaths };
}

async function performMove(datasetsRoot: string, request: DatasetCaptionBulkRequest, matched: ResolvedMedia[]) {
  const destination = uniqueDatasetDestination(
    datasetsRoot,
    request.destinationName?.trim() || defaultMoveName(request.datasetName),
  );
  await fsp.mkdir(destination.finalPath, { recursive: false });

  const usedStems = new Set<string>();
  let moved = 0;
  const removedPaths: string[] = [];
  for (const item of matched) {
    const targetMediaPath = uniqueDestinationMediaPath(destination.finalPath, item.path, usedStems);
    const sidecars = existingCaptionSidecars(item.path);
    await fsp.rename(item.path, targetMediaPath);
    for (const sidecar of sidecars) {
      const ext = path.extname(sidecar).toLowerCase();
      await fsp.rename(sidecar, captionSidecarPath(targetMediaPath, ext));
    }
    moved += 1;
    removedPaths.push(item.path);
  }

  return { moved, removedPaths, destinationName: destination.finalName };
}

async function performRemoveWords(
  matched: ResolvedMedia[],
  terms: string[],
  matchMode: CaptionKeywordMatchMode,
) {
  let updated = 0;
  let removedWords = 0;
  const updatedCaptions: Record<string, string> = {};

  for (const item of matched) {
    const isTextFile = DATASET_TEXT_CAPTION_EXTENSIONS.includes(path.extname(item.path).toLowerCase());
    const captionPath = isTextFile ? item.path : findExistingCaptionSidecar(item.path);
    if (!captionPath) continue;
    const result = removeCaptionKeywords(item.caption, terms, matchMode);
    if (!result.changed) continue;
    await fsp.writeFile(captionPath, result.caption);
    updated += 1;
    removedWords += result.removedCount;
    updatedCaptions[item.path] = result.caption;
  }

  return { updated, removedWords, updatedCaptions };
}

export async function performPlainDatasetCaptionBulkAction(
  datasetsRoot: string,
  rawRequest: DatasetCaptionBulkRequest,
): Promise<DatasetCaptionBulkResult> {
  const request = {
    ...rawRequest,
    action: ensureAction(rawRequest.action),
    imgPaths: ensureImgPaths(rawRequest.imgPaths),
    matchMode: normalizeMatchMode(rawRequest.matchMode),
  };
  const terms = parseCaptionKeywordQuery(request.query || '');
  if (terms.length === 0) throw new DatasetCaptionBulkError('Enter at least one keyword');

  const datasetFolder = resolveDatasetFolder(datasetsRoot, request.datasetName);
  if (!fs.existsSync(datasetFolder) || !fs.statSync(datasetFolder).isDirectory()) {
    throw new DatasetCaptionBulkError('Dataset not found', 404);
  }
  if (isEncryptedDatasetFolder(datasetFolder)) {
    throw new DatasetCaptionBulkError('Encrypted datasets are updated through the encrypted dataset API', 403);
  }

  const mediaPaths = resolveMediaPaths(datasetFolder, request.imgPaths);
  const matched = matchedMedia(mediaPaths, terms, request.matchMode);

  if (request.action === 'delete') {
    const deleted = await performDelete(matched);
    return {
      action: request.action,
      found: matched.length,
      affected: deleted.deleted,
      deleted: deleted.deleted,
      removedPaths: deleted.removedPaths,
    };
  }

  if (request.action === 'move') {
    const moved = await performMove(datasetsRoot, request, matched);
    return {
      action: request.action,
      found: matched.length,
      affected: moved.moved,
      moved: moved.moved,
      destinationName: moved.destinationName,
      removedPaths: moved.removedPaths,
    };
  }

  const updated = await performRemoveWords(matched, terms, request.matchMode);
  return {
    action: request.action,
    found: matched.length,
    affected: updated.updated,
    updated: updated.updated,
    removedWords: updated.removedWords,
    updatedCaptions: updated.updatedCaptions,
  };
}

export function decodeRemoteCaptionBulkPaths(imgPaths: string[], workerID: string) {
  const refByRemotePath: Record<string, string> = {};
  const remotePaths = imgPaths.map(imgPath => {
    const remoteAsset = parseRemoteDatasetAssetRef(imgPath);
    if (!remoteAsset) return imgPath;
    if (remoteAsset.workerID !== workerID) {
      throw new DatasetCaptionBulkError('Remote image path belongs to a different worker');
    }
    refByRemotePath[remoteAsset.path] = imgPath;
    return remoteAsset.path;
  });
  return { remotePaths, refByRemotePath };
}

export function mapRemoteCaptionBulkResult(
  result: DatasetCaptionBulkResult,
  refByRemotePath: Record<string, string>,
): DatasetCaptionBulkResult {
  const updatedCaptions = result.updatedCaptions
    ? Object.fromEntries(
        Object.entries(result.updatedCaptions).map(([remotePath, caption]) => [
          refByRemotePath[remotePath] || remotePath,
          caption,
        ]),
      )
    : undefined;
  const removedPaths = result.removedPaths?.map(remotePath => refByRemotePath[remotePath] || remotePath);
  return {
    ...result,
    ...(updatedCaptions ? { updatedCaptions } : {}),
    ...(removedPaths ? { removedPaths } : {}),
  };
}

export function isDatasetCaptionBulkError(error: unknown): error is DatasetCaptionBulkError {
  return error instanceof DatasetCaptionBulkError;
}
