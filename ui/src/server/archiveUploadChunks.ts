import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import type { NextRequest } from 'next/server';

const MAX_UPLOAD_CHUNK_AGE_MS = 24 * 60 * 60 * 1000;

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
  // eslint-disable-next-line no-var
  var __archiveUploadImportStatusStore: ArchiveUploadImportStatusStore | undefined;
}

const archiveUploadImportStatusStore =
  globalThis.__archiveUploadImportStatusStore ?? new Map<string, ArchiveUploadImportStatus>();

if (!globalThis.__archiveUploadImportStatusStore) {
  globalThis.__archiveUploadImportStatusStore = archiveUploadImportStatusStore;
}

export function archiveUploadMode(request: NextRequest): ArchiveUploadMode {
  const mode = request.nextUrl.searchParams.get('aitk_upload');
  return mode === 'chunk' || mode === 'complete' || mode === 'status' ? mode : null;
}

export function readArchiveUploadID(request: NextRequest) {
  return validateArchiveUploadID(request.nextUrl.searchParams.get('uploadID') || '');
}

export function readArchiveUploadChunksTotal(request: NextRequest) {
  return readSafeInteger(request.nextUrl.searchParams.get('chunksTotal') || '', 'chunksTotal', 1);
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

function readArchiveUploadChunkIndex(request: NextRequest) {
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

export async function saveArchiveUploadChunk(request: NextRequest, uploadRoot: string) {
  const uploadID = readArchiveUploadID(request);
  const chunkIndex = readArchiveUploadChunkIndex(request);
  const chunksTotal = readArchiveUploadChunksTotal(request);
  if (chunkIndex >= chunksTotal) {
    throw new Error('Invalid archive upload chunkIndex');
  }
  if (!request.body) {
    throw new Error('file is required');
  }

  const chunkPath = chunkPathForIndex(uploadRoot, uploadID, chunkIndex);
  await fsp.mkdir(path.dirname(chunkPath), { recursive: true });
  await pipeline(
    Readable.fromWeb(request.body as any),
    fs.createWriteStream(chunkPath),
  );

  return { uploadID, chunkIndex, chunksTotal };
}

export async function assembleArchiveUploadChunks(
  uploadRoot: string,
  uploadID: string,
  chunksTotal: number,
  outputPath: string,
) {
  validateArchiveUploadID(uploadID);
  if (!Number.isSafeInteger(chunksTotal) || chunksTotal < 1) {
    throw new Error('Invalid archive upload chunksTotal');
  }

  await fsp.mkdir(path.dirname(outputPath), { recursive: true });
  await fsp.writeFile(outputPath, Buffer.alloc(0));
  for (let index = 0; index < chunksTotal; index += 1) {
    const chunkPath = chunkPathForIndex(uploadRoot, uploadID, index);
    const chunk = await fsp.readFile(chunkPath).catch(() => null);
    if (!chunk) {
      throw new Error(`Missing archive upload chunk ${index + 1} of ${chunksTotal}`);
    }
    await fsp.appendFile(outputPath, chunk);
  }
}
