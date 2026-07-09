import { createReadStream, createWriteStream } from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { pipeline } from 'stream/promises';
import {
  cleanupStagedUpload,
  moveStagedUpload,
  streamRequestToStagingFile,
  UploadTooLargeError,
} from './streamedUpload';

const MAX_UPLOAD_CHUNK_AGE_MS = 24 * 60 * 60 * 1000;
const MAX_ARCHIVE_UPLOAD_CHUNK_BYTES = 64 * 1024 * 1024;
const MAX_ARCHIVE_UPLOAD_CHUNKS = 8_192;

type ArchiveUploadRequest = Request & { nextUrl: URL };

export type ArchiveUploadMode = 'chunk' | 'complete' | 'status' | null;
export type ArchiveUploadImportStatus<T = unknown> = {
  uploadID: string;
  status: 'importing' | 'completed' | 'failed';
  result: T | null;
  error: string | null;
  createdAt: string;
  updatedAt: string;
};

type ArchiveUploadImportStatusStore = Map<string, ArchiveUploadImportStatus>;

declare global {
  var __archiveUploadImportStatusStore: ArchiveUploadImportStatusStore | undefined;
}

const archiveUploadImportStatusStore =
  globalThis.__archiveUploadImportStatusStore ?? new Map<string, ArchiveUploadImportStatus>();

if (!globalThis.__archiveUploadImportStatusStore) {
  globalThis.__archiveUploadImportStatusStore = archiveUploadImportStatusStore;
}

export function archiveUploadMode(request: ArchiveUploadRequest): ArchiveUploadMode {
  const mode = request.nextUrl.searchParams.get('aitk_upload');
  return mode === 'chunk' || mode === 'complete' || mode === 'status' ? mode : null;
}

export function readArchiveUploadID(request: ArchiveUploadRequest) {
  return validateArchiveUploadID(request.nextUrl.searchParams.get('uploadID') || '');
}

export function readArchiveUploadChunksTotal(request: ArchiveUploadRequest) {
  const total = readSafeInteger(request.nextUrl.searchParams.get('chunksTotal') || '', 'chunksTotal', 1);
  if (total > MAX_ARCHIVE_UPLOAD_CHUNKS) {
    throw new Error('Invalid archive upload chunksTotal');
  }
  return total;
}

export function readArchiveUploadFileBytes(request: ArchiveUploadRequest, maxBytes: number) {
  const rawFileBytes = request.nextUrl.searchParams.get('fileBytes');
  if (rawFileBytes === null || rawFileBytes === '') return null;
  const fileBytes = readSafeInteger(rawFileBytes, 'fileBytes', 0);
  if (fileBytes > maxBytes) {
    throw new UploadTooLargeError(`Archive upload must be ${Math.floor(maxBytes / (1024 ** 3))} GB or smaller`);
  }
  return fileBytes;
}

export function createArchiveUploadImportStatus(uploadID: string) {
  validateArchiveUploadID(uploadID);
  const now = new Date().toISOString();
  const status: ArchiveUploadImportStatus = {
    uploadID,
    status: 'importing',
    result: null,
    error: null,
    createdAt: now,
    updatedAt: now,
  };
  archiveUploadImportStatusStore.set(uploadID, status);
  return cloneArchiveUploadImportStatus(status);
}

export function updateArchiveUploadImportStatus<T>(
  uploadID: string,
  patch: Pick<ArchiveUploadImportStatus<T>, 'status'> &
    Partial<Pick<ArchiveUploadImportStatus<T>, 'result' | 'error'>>,
) {
  validateArchiveUploadID(uploadID);
  const existing = archiveUploadImportStatusStore.get(uploadID);
  const now = new Date().toISOString();
  const updated: ArchiveUploadImportStatus<T> = {
    uploadID,
    status: patch.status,
    result: patch.result ?? (existing?.result as T | null) ?? null,
    error: patch.error ?? null,
    createdAt: existing?.createdAt ?? now,
    updatedAt: now,
  };
  archiveUploadImportStatusStore.set(uploadID, updated as ArchiveUploadImportStatus);
  return cloneArchiveUploadImportStatus(updated);
}

export function getArchiveUploadImportStatus<T = unknown>(uploadID: string) {
  validateArchiveUploadID(uploadID);
  const status = archiveUploadImportStatusStore.get(uploadID) as ArchiveUploadImportStatus<T> | undefined;
  return status ? cloneArchiveUploadImportStatus(status) : null;
}

function cloneArchiveUploadImportStatus<T>(status: ArchiveUploadImportStatus<T>) {
  return { ...status };
}

function readArchiveUploadChunkIndex(request: ArchiveUploadRequest) {
  return readSafeInteger(request.nextUrl.searchParams.get('chunkIndex') || '', 'chunkIndex', 0);
}

function validateArchiveUploadID(value: string) {
  if (!/^[a-zA-Z0-9_-]{8,120}$/.test(value)) {
    throw new Error('Invalid archive upload ID');
  }
  return value;
}

function readSafeInteger(value: string, label: string, min: number) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < min) {
    throw new Error(`Invalid archive upload ${label}`);
  }
  return parsed;
}

function uploadRootForID(uploadRoot: string, uploadID: string) {
  return path.join(uploadRoot, validateArchiveUploadID(uploadID));
}

function chunkPathForIndex(uploadRoot: string, uploadID: string, chunkIndex: number) {
  return path.join(uploadRootForID(uploadRoot, uploadID), 'chunks', `${chunkIndex}.part`);
}

export async function cleanupOldArchiveUploadChunks(uploadRoot: string) {
  const now = Date.now();
  const entries = await fsp.readdir(uploadRoot, { withFileTypes: true }).catch(() => []);
  await Promise.all(
    entries
      .filter(entry => entry.isDirectory())
      .map(async entry => {
        const targetPath = path.join(uploadRoot, entry.name);
        const stat = await fsp.stat(targetPath).catch(() => null);
        if (!stat || now - stat.mtimeMs <= MAX_UPLOAD_CHUNK_AGE_MS) return;
        await fsp.rm(targetPath, { recursive: true, force: true }).catch(() => undefined);
      }),
  );
}

export async function saveArchiveUploadChunk(
  request: ArchiveUploadRequest,
  uploadRoot: string,
  options: { maxArchiveBytes?: number } = {},
) {
  const uploadID = readArchiveUploadID(request);
  const chunkIndex = readArchiveUploadChunkIndex(request);
  const chunksTotal = readArchiveUploadChunksTotal(request);
  const fileBytes =
    options.maxArchiveBytes === undefined
      ? null
      : readArchiveUploadFileBytes(request, options.maxArchiveBytes);
  if (chunkIndex >= chunksTotal) {
    throw new Error('Invalid archive upload chunkIndex');
  }
  if (!request.body) {
    throw new Error('file is required');
  }

  const chunkPath = chunkPathForIndex(uploadRoot, uploadID, chunkIndex);
  let stagingPath: string | null = null;
  try {
    const staged = await streamRequestToStagingFile(request, path.dirname(chunkPath), {
      maxBytes: MAX_ARCHIVE_UPLOAD_CHUNK_BYTES,
      prefix: `archive-chunk-${chunkIndex}`,
    });
    stagingPath = staged.stagingPath;
    await moveStagedUpload(stagingPath, chunkPath);
    stagingPath = null;
  } finally {
    await cleanupStagedUpload(stagingPath);
  }

  return { uploadID, chunkIndex, chunksTotal, ...(fileBytes === null ? {} : { fileBytes }) };
}

export async function assembleArchiveUploadChunks(
  uploadRoot: string,
  uploadID: string,
  chunksTotal: number,
  outputPath: string,
  options: { maxBytes?: number; expectedBytes?: number } = {},
) {
  validateArchiveUploadID(uploadID);
  if (!Number.isSafeInteger(chunksTotal) || chunksTotal < 1 || chunksTotal > MAX_ARCHIVE_UPLOAD_CHUNKS) {
    throw new Error('Invalid archive upload chunksTotal');
  }

  const chunks: string[] = [];
  let totalBytes = 0;
  for (let index = 0; index < chunksTotal; index += 1) {
    const chunkPath = chunkPathForIndex(uploadRoot, uploadID, index);
    const stat = await fsp.stat(chunkPath).catch(() => null);
    if (!stat?.isFile() || stat.size > MAX_ARCHIVE_UPLOAD_CHUNK_BYTES) {
      throw new Error(`Invalid archive upload chunk ${index + 1} of ${chunksTotal}`);
    }
    totalBytes += stat.size;
    if (!Number.isSafeInteger(totalBytes)) {
      throw new UploadTooLargeError('Archive upload is too large');
    }
    if (options.maxBytes !== undefined && totalBytes > options.maxBytes) {
      throw new UploadTooLargeError(`Archive upload must be ${Math.floor(options.maxBytes / (1024 ** 3))} GB or smaller`);
    }
    chunks.push(chunkPath);
  }

  if (options.expectedBytes !== undefined && totalBytes !== options.expectedBytes) {
    throw new Error('Invalid archive upload fileBytes');
  }

  await fsp.mkdir(path.dirname(outputPath), { recursive: true });
  for (let index = 0; index < chunks.length; index += 1) {
    await pipeline(
      createReadStream(chunks[index]),
      createWriteStream(outputPath, { flags: index === 0 ? 'w' : 'a' }),
    );
  }
}
