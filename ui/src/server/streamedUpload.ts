import { randomUUID } from 'crypto';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { Readable, Transform } from 'stream';
import { pipeline } from 'stream/promises';
import type { ReadableStream as NodeReadableStream } from 'stream/web';

export class UploadTooLargeError extends Error {
  status = 413;

  constructor(message = 'Upload is too large') {
    super(message);
    this.name = 'UploadTooLargeError';
  }
}

export class InvalidUploadError extends Error {
  status = 400;

  constructor(message: string) {
    super(message);
    this.name = 'InvalidUploadError';
  }
}

export function decodedUploadHeader(request: Request, name: string, maxLength = 1_024) {
  const rawValue = request.headers.get(name);
  if (!rawValue) return '';
  let value = rawValue;
  try {
    value = decodeURIComponent(rawValue);
  } catch {
    throw new InvalidUploadError(`Invalid ${name} header`);
  }
  if (value.length > maxLength) {
    throw new InvalidUploadError(`${name} header is too long`);
  }
  return value;
}

export function assertDeclaredUploadSize(request: Request, maxBytes: number) {
  const rawLength = request.headers.get('content-length');
  if (!rawLength) return;
  const contentLength = Number(rawLength);
  if (!Number.isSafeInteger(contentLength) || contentLength < 0) {
    throw new InvalidUploadError('Invalid Content-Length header');
  }
  if (contentLength > maxBytes) {
    throw new UploadTooLargeError();
  }
}

export async function streamRequestToStagingFile(
  request: Request,
  directory: string,
  options: { maxBytes: number; prefix?: string },
) {
  if (!request.body) {
    throw new InvalidUploadError('Upload body is required');
  }
  assertDeclaredUploadSize(request, options.maxBytes);
  await fsp.mkdir(directory, { recursive: true });

  const prefix = (options.prefix || 'upload').replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 48) || 'upload';
  const stagingPath = path.join(directory, `.${prefix}-${randomUUID()}.tmp`);
  let bytesWritten = 0;
  const limiter = new Transform({
    transform(chunk: Buffer, _encoding, callback) {
      bytesWritten += chunk.length;
      if (bytesWritten > options.maxBytes) {
        callback(new UploadTooLargeError());
        return;
      }
      callback(null, chunk);
    },
  });

  try {
    await pipeline(
      Readable.fromWeb(request.body as unknown as NodeReadableStream),
      limiter,
      fs.createWriteStream(stagingPath, { flags: 'wx' }),
    );
    return { stagingPath, bytesWritten };
  } catch (error) {
    await fsp.rm(stagingPath, { force: true }).catch(() => undefined);
    throw error;
  }
}

export async function moveStagedUpload(stagingPath: string, destinationPath: string) {
  await fsp.mkdir(path.dirname(destinationPath), { recursive: true });
  await fsp.rename(stagingPath, destinationPath);
}

function errorCode(error: unknown) {
  return error && typeof error === 'object' && 'code' in error && typeof error.code === 'string'
    ? error.code
    : null;
}

async function copyStagedUploadNoReplace(stagingPath: string, destinationPath: string) {
  let copied = false;
  try {
    await fsp.copyFile(stagingPath, destinationPath, fs.constants.COPYFILE_EXCL);
    copied = true;
    await fsp.unlink(stagingPath);
  } catch (error) {
    if (copied) {
      await fsp.rm(destinationPath, { force: true }).catch(() => undefined);
    }
    throw error;
  }
}

/** Commit a staged file without replacing a file created by another upload. */
export async function moveStagedUploadNoReplace(stagingPath: string, destinationPath: string) {
  await fsp.mkdir(path.dirname(destinationPath), { recursive: true });
  try {
    // A hard link is an atomic, same-filesystem no-replace move once the
    // staging name is removed.
    await fsp.link(stagingPath, destinationPath);
  } catch (error) {
    if (!['EPERM', 'ENOTSUP', 'EOPNOTSUPP'].includes(errorCode(error) || '')) {
      throw error;
    }
    // Some mounted filesystems do not support hard links. COPYFILE_EXCL keeps
    // collision handling safe on those filesystems as well.
    await copyStagedUploadNoReplace(stagingPath, destinationPath);
    return;
  }

  try {
    await fsp.unlink(stagingPath);
  } catch (error) {
    await fsp.unlink(destinationPath).catch(() => undefined);
    throw error;
  }
}

export async function cleanupStagedUpload(stagingPath: string | null | undefined) {
  if (!stagingPath) return;
  await fsp.rm(stagingPath, { force: true }).catch(() => undefined);
}
