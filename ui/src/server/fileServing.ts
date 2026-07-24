import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import type { Stats } from 'fs';
import { findEncryptedDatasetRoot } from './encryptedDatasets';
import { isPathWithinRoot } from './pathContainment';
import {
  normalizeProjectAssetPath,
  verifyProjectAssetSignature,
  type ProjectAssetDisposition,
} from './projectAssetUrls';
import { isRegisteredProjectPath } from './projectMediaSecurity';
import { getProjectRoots, isPathInside, resolveProject } from './projects';
import { flushCache as flushSettingsCache, getDataRoot, getDatasetsRoot, getTrainingFolder } from './settings';
import { isRequestAuthenticated } from '../utils/authSession';
import { parseRemoteDatasetAssetRef } from '../utils/remoteDatasetRefs';

export const FILE_STREAM_BUFFER_BYTES = 4 * 1024 * 1024;
export const PRIVATE_MEDIA_CACHE_CONTROL = 'private, no-cache, must-revalidate';

export type ByteRange = {
  start: number;
  end: number;
};

export type FileResponseResolution =
  | {
      kind: 'proxy';
    }
  | {
      kind: 'response';
      status: number;
      headers: Record<string, string>;
      body: string | null;
      file?: {
        path: string;
        start?: number;
        end?: number;
      };
    };

export type ProjectAssetRequest = {
  projectID: string;
  routePath: string;
  searchParams: URLSearchParams;
  headers: Headers;
};

const DOWNLOAD_CONTENT_TYPES: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.jxl': 'image/jxl',
  '.svg': 'image/svg+xml',
  '.bmp': 'image/bmp',
  '.safetensors': 'application/octet-stream',
  '.zip': 'application/zip',
  '.mp4': 'video/mp4',
  '.avi': 'video/x-msvideo',
  '.mov': 'video/quicktime',
  '.mkv': 'video/x-matroska',
  '.wmv': 'video/x-ms-wmv',
  '.m4v': 'video/x-m4v',
  '.flv': 'video/x-flv',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.flac': 'audio/flac',
  '.ogg': 'audio/ogg',
};

const MEDIA_CONTENT_TYPES: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.jxl': 'image/jxl',
  '.bmp': 'image/bmp',
  '.mp4': 'video/mp4',
  '.avi': 'video/x-msvideo',
  '.mov': 'video/quicktime',
  '.mkv': 'video/x-matroska',
  '.wmv': 'video/x-ms-wmv',
  '.m4v': 'video/x-m4v',
  '.flv': 'video/x-flv',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.flac': 'audio/flac',
  '.ogg': 'audio/ogg',
};

const PROJECT_ASSET_CONTENT_TYPES: Record<string, string> = {
  ...MEDIA_CONTENT_TYPES,
  '.avif': 'image/avif',
  '.webm': 'video/webm',
  '.m4a': 'audio/mp4',
  '.txt': 'text/plain; charset=utf-8',
  '.json': 'application/json',
  '.yaml': 'text/yaml; charset=utf-8',
  '.yml': 'text/yaml; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.safetensors': 'application/octet-stream',
  '.ckpt': 'application/octet-stream',
  '.pt': 'application/octet-stream',
  '.pth': 'application/octet-stream',
  '.bin': 'application/octet-stream',
  '.gguf': 'application/octet-stream',
};

let storageSettingsCacheExpiresAt = 0;

function refreshStorageSettingsCachePeriodically() {
  const now = Date.now();
  if (now < storageSettingsCacheExpiresAt) return;
  flushSettingsCache();
  storageSettingsCacheExpiresAt = now + 10_000;
}

function response(
  status: number,
  body: string | null,
  headers: Record<string, string> = {},
): FileResponseResolution {
  return { kind: 'response', status, body, headers };
}

function weakEtag(stat: Stats) {
  return `W/"${stat.ino.toString(36)}-${stat.size.toString(36)}-${stat.mtimeMs.toString(36)}"`;
}

function cleanDispositionFilename(filePath: string) {
  return path.basename(filePath).replace(/[\r\n"]/g, '_');
}

async function realpathIfExists(filePath: string) {
  return fsp.realpath(filePath).catch(() => null);
}

async function resolveExistingDir(dir: string) {
  if (!dir) return null;
  return realpathIfExists(path.resolve(dir));
}

export function parseSingleByteRange(value: string | null | undefined, size: number): ByteRange | null | 'invalid' {
  if (!value) return null;
  if (!Number.isSafeInteger(size) || size < 0) return 'invalid';

  const match = /^bytes=(\d*)-(\d*)$/i.exec(value.trim());
  if (!match || (!match[1] && !match[2]) || size === 0) return 'invalid';

  let start: number;
  let end: number;
  if (!match[1]) {
    const suffix = Number(match[2]);
    if (!Number.isSafeInteger(suffix) || suffix <= 0) return 'invalid';
    start = Math.max(0, size - suffix);
    end = size - 1;
  } else {
    start = Number(match[1]);
    end = match[2] ? Number(match[2]) : size - 1;
  }

  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(end) ||
    start < 0 ||
    end < start ||
    start >= size
  ) {
    return 'invalid';
  }

  return { start, end: Math.min(end, size - 1) };
}

function resolveFileResponse(
  filePath: string,
  stat: Stats,
  requestHeaders: Headers,
  baseHeaders: Record<string, string>,
): FileResponseResolution {
  const etag = weakEtag(stat);
  const headers = { ...baseHeaders, ETag: etag };
  if (requestHeaders.get('if-none-match') === etag) {
    return response(304, null, headers);
  }

  const requestedRange = parseSingleByteRange(requestHeaders.get('range'), stat.size);
  if (requestedRange === 'invalid') {
    return response(416, null, {
      ...headers,
      'Content-Range': `bytes */${stat.size}`,
    });
  }

  if (requestedRange) {
    return {
      kind: 'response',
      status: 206,
      body: null,
      headers: {
        ...headers,
        'Content-Range': `bytes ${requestedRange.start}-${requestedRange.end}/${stat.size}`,
        'Content-Length': String(requestedRange.end - requestedRange.start + 1),
      },
      file: {
        path: filePath,
        start: requestedRange.start,
        end: requestedRange.end,
      },
    };
  }

  return {
    kind: 'response',
    status: 200,
    body: null,
    headers: {
      ...headers,
      'Content-Length': String(stat.size),
    },
    file: { path: filePath },
  };
}

export async function isAcceleratedApiRequestAuthenticated(headers: Headers) {
  return isRequestAuthenticated({ headers }, process.env.AI_TOOLKIT_AUTH);
}

export function unauthorizedFileResponse(): FileResponseResolution {
  return response(401, JSON.stringify({ error: 'Unauthorized' }), {
    'Content-Type': 'application/json',
  });
}

export async function resolveDownloadFileRequest(
  decodedFilePath: string,
  requestHeaders: Headers,
): Promise<FileResponseResolution> {
  refreshStorageSettingsCachePeriodically();
  const [datasetRoot, trainingRoot] = await Promise.all([getDatasetsRoot(), getTrainingFolder()]);
  const allowedRoots = (
    await Promise.all([datasetRoot, trainingRoot].filter(Boolean).map(root => resolveExistingDir(root)))
  ).filter((root): root is string => root !== null);

  const canonicalPath = await realpathIfExists(decodedFilePath);
  if (!canonicalPath) return response(404, 'File not found');
  if (await isRegisteredProjectPath(canonicalPath)) {
    return response(403, 'Project files require a signed project-relative URL');
  }
  if (!allowedRoots.some(root => isPathWithinRoot(root, canonicalPath))) {
    return response(403, 'Access denied');
  }

  const stat = await fsp.stat(canonicalPath).catch(() => null);
  if (!stat || !stat.isFile()) return response(400, 'Not a file');

  const contentType = DOWNLOAD_CONTENT_TYPES[path.extname(canonicalPath).toLowerCase()];
  if (!contentType) return response(403, 'File type not allowed');

  const filename = cleanDispositionFilename(canonicalPath);
  return resolveFileResponse(canonicalPath, stat, requestHeaders, {
    'Content-Type': contentType,
    'Accept-Ranges': 'bytes',
    'Cache-Control': PRIVATE_MEDIA_CACHE_CONTROL,
    'Content-Disposition': `attachment; filename="${encodeURIComponent(filename)}"`,
    'X-Content-Type-Options': 'nosniff',
  });
}

export async function resolveLocalMediaRequest(
  requestedValue: string,
  requestHeaders: Headers,
): Promise<FileResponseResolution> {
  if (parseRemoteDatasetAssetRef(requestedValue)) return { kind: 'proxy' };

  refreshStorageSettingsCachePeriodically();
  const filepath = path.resolve(requestedValue);
  const [datasetRoot, trainingRoot, dataRoot] = await Promise.all([
    getDatasetsRoot(),
    getTrainingFolder(),
    getDataRoot(),
  ]);
  const [canonicalDatasetRoot, canonicalTrainingRoot, canonicalDataRoot] = await Promise.all(
    [datasetRoot, trainingRoot, dataRoot].map(resolveExistingDir),
  );
  const generalAllowedRoots = [canonicalDatasetRoot, canonicalDataRoot].filter(
    (root): root is string => root !== null,
  );

  const canonicalPath = await realpathIfExists(filepath);
  if (!canonicalPath) return response(404, 'File not found');
  if (await isRegisteredProjectPath(canonicalPath)) {
    return response(403, 'Project media requires a signed project-relative URL');
  }

  const contentType = MEDIA_CONTENT_TYPES[path.extname(canonicalPath).toLowerCase()];
  if (!contentType) return response(415, 'Unsupported media type');

  const trainingRelativePath =
    canonicalTrainingRoot && isPathWithinRoot(canonicalTrainingRoot, canonicalPath)
      ? path.relative(canonicalTrainingRoot, canonicalPath)
      : null;
  const isGeneralMedia =
    trainingRelativePath === null &&
    generalAllowedRoots.some(root => isPathWithinRoot(root, canonicalPath));
  const isTrainingSample =
    trainingRelativePath !== null && trainingRelativePath.split(path.sep).includes('samples');
  if (!isGeneralMedia && !isTrainingSample) return response(403, 'Access denied');

  if (canonicalDatasetRoot && findEncryptedDatasetRoot(canonicalPath, canonicalDatasetRoot)) {
    return response(403, 'Encrypted dataset objects are not served through this route');
  }

  const stat = await fsp.stat(canonicalPath).catch(() => null);
  if (!stat || !stat.isFile()) return response(404, 'File not found');

  return resolveFileResponse(canonicalPath, stat, requestHeaders, {
    'Content-Type': contentType,
    'Accept-Ranges': 'bytes',
    'Cache-Control': PRIVATE_MEDIA_CACHE_CONTROL,
    'X-Content-Type-Options': 'nosniff',
  });
}

export async function resolveProjectAssetFileRequest(
  request: ProjectAssetRequest,
): Promise<FileResponseResolution> {
  refreshStorageSettingsCachePeriodically();

  let queryPath: string;
  let routePath: string;
  try {
    queryPath = normalizeProjectAssetPath(request.searchParams.get('path') || '');
    routePath = normalizeProjectAssetPath(request.routePath);
  } catch {
    return response(403, 'Invalid or expired project asset link');
  }

  const queryProjectID = request.searchParams.get('project_id') || '';
  const dispositionValue = request.searchParams.get('disposition');
  const disposition: ProjectAssetDisposition =
    dispositionValue === 'attachment' ? 'attachment' : 'inline';
  const expires = Number(request.searchParams.get('expires') || '');
  const signature = request.searchParams.get('sig') || '';
  if (
    queryProjectID !== request.projectID ||
    queryPath !== routePath ||
    (dispositionValue !== 'inline' && dispositionValue !== 'attachment') ||
    !verifyProjectAssetSignature({
      projectID: request.projectID,
      relativePath: queryPath,
      disposition,
      expires,
      signature,
    })
  ) {
    return response(403, 'Invalid or expired project asset link');
  }

  try {
    const project = await resolveProject(request.projectID, { intent: 'read' });
    const roots = await getProjectRoots(project);
    const root = await fsp.realpath(roots.root);
    const candidate = path.resolve(root, ...queryPath.split('/'));
    const [target, lstat] = await Promise.all([
      fsp.realpath(candidate).catch(() => null),
      fsp.lstat(candidate).catch(() => null),
    ]);
    if (!target || !lstat || lstat.isSymbolicLink() || !isPathInside(root, target) || target === root) {
      return response(404, 'Project asset not found');
    }

    const stat = await fsp.stat(target).catch(() => null);
    if (!stat || !stat.isFile()) return response(404, 'Project asset not found');

    const contentType =
      PROJECT_ASSET_CONTENT_TYPES[path.extname(target).toLowerCase()] || 'application/octet-stream';
    const filename = cleanDispositionFilename(target);
    return resolveFileResponse(target, stat, request.headers, {
      'Content-Type': contentType,
      'Accept-Ranges': 'bytes',
      'Cache-Control': 'private, max-age=300',
      'Content-Disposition': `${disposition}; filename="${filename}"; filename*=UTF-8''${encodeURIComponent(filename)}`,
      'X-Content-Type-Options': 'nosniff',
    });
  } catch {
    return response(404, 'Project asset not found');
  }
}
