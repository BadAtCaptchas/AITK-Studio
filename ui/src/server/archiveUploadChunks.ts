import { receiveManifestChunk, assembleManifestChunks, cleanupExpiredUploads } from './archiveUploadManifest';
import { UploadTooLargeError } from './streamedUpload';
const MAX_ARCHIVE_UPLOAD_CHUNKS = 8192;
type ArchiveUploadRequest = Request & { nextUrl: URL };
export type ArchiveUploadMode = 'chunk' | 'complete' | 'status' | null;

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

export async function cleanupOldArchiveUploadChunks(uploadRoot: string) {
  await cleanupExpiredUploads(uploadRoot);
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

  return receiveManifestChunk(request, uploadRoot, { uploadID, chunkIndex, chunksTotal,
    expectedBytes: fileBytes, limit: options.maxArchiveBytes ?? 64 * 1024 ** 3, purpose: request.nextUrl.pathname });
}

export async function assembleArchiveUploadChunks(
  uploadRoot: string,
  uploadID: string,
  chunksTotal: number,
  outputPath: string,
  options: { maxBytes?: number; expectedBytes?: number } = {},
) {
  return assembleManifestChunks(uploadRoot, uploadID, chunksTotal, outputPath, options);
}
