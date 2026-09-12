import { createHash, createHmac } from 'crypto';
import { createReadStream, createWriteStream } from 'fs';
import fs from 'fs/promises';
import path from 'path';
import { Transform } from 'stream';
import { pipeline } from 'stream/promises';
import { isRecord } from './commandInput';
import { db } from './db';
import { withProcessLease } from './processLease';
import {
  cleanupStagedUpload,
  moveStagedUploadNoReplace,
  streamRequestToStagingFile,
  UploadTooLargeError,
} from './streamedUpload';

export const MAX_CHUNK_BYTES = 64 * 1024 * 1024;
const MAX_CONCURRENT_UPLOADS = 8;
const RESERVE_BYTES = 2 * 1024 ** 3;
const MAX_STAGED_BYTES = 128 * 1024 ** 3;
type Chunk = { bytes: number; sha256: string };
export type UploadManifest = {
  version: 1;
  uploadID: string;
  root: string;
  purpose: string;
  principal: string;
  chunksTotal: number;
  expectedBytes: number | null;
  limit: number;
  receivedBytes: number;
  reservedBytes: number;
  chunks: Record<string, Chunk>;
  state: 'receiving' | 'finalizing' | 'assembled' | 'importing' | 'completed' | 'failed';
  createdAt: number;
  lastActivity: number;
  expiresAt: number;
  assembledPath?: string;
};
export class UploadConflictError extends Error {
  readonly status = 409;
}
export function uploadRecordKey(root: string, id: string): string {
  if (!/^[a-zA-Z0-9_-]{8,120}$/.test(id)) throw new UploadConflictError('Invalid archive upload ID');
  return `upload:${createHash('sha256').update(path.resolve(root)).digest('hex')}:${id}`;
}
function principal(): string {
  return createHmac('sha256', process.env.AI_TOOLKIT_AUTH || 'loopback-local')
    .update('aitk-upload-admin-v1')
    .digest('hex');
}
function manifest(value: unknown): UploadManifest {
  if (!isRecord(value) || value.version !== 1 || typeof value.root !== 'string')
    throw new UploadConflictError('Invalid persisted upload manifest');
  if (
    !isRecord(value) ||
    typeof value.uploadID !== 'string' ||
    !/^[a-zA-Z0-9_-]{8,120}$/.test(value.uploadID) ||
    typeof value.principal !== 'string' ||
    !/^[a-f0-9]{64}$/.test(value.principal) ||
    typeof value.purpose !== 'string' ||
    !isRecord(value.chunks) ||
    !['receiving', 'finalizing', 'assembled', 'importing', 'completed', 'failed'].includes(String(value.state))
  )
    throw new UploadConflictError('Invalid persisted upload identity');
  for (const field of [
    'chunksTotal',
    'limit',
    'receivedBytes',
    'reservedBytes',
    'createdAt',
    'lastActivity',
    'expiresAt',
  ]) {
    if (typeof value[field] !== 'number' || !Number.isSafeInteger(value[field]) || value[field] < 0)
      throw new UploadConflictError('Invalid persisted upload accounting');
  }
  if (
    Number(value.chunksTotal) < 1 ||
    Number(value.chunksTotal) > 8192 ||
    Number(value.limit) > MAX_STAGED_BYTES ||
    (value.expectedBytes !== null &&
      (typeof value.expectedBytes !== 'number' ||
        !Number.isSafeInteger(value.expectedBytes) ||
        value.expectedBytes < 0 ||
        value.expectedBytes > Number(value.limit)))
  )
    throw new UploadConflictError('Invalid persisted upload limits');
  let total = 0;
  for (const [index, chunk] of Object.entries(value.chunks)) {
    if (
      !/^(0|[1-9][0-9]*)$/.test(index) ||
      Number(index) >= Number(value.chunksTotal) ||
      !isRecord(chunk) ||
      typeof chunk.bytes !== 'number' ||
      !Number.isSafeInteger(chunk.bytes) ||
      chunk.bytes < 0 ||
      chunk.bytes > MAX_CHUNK_BYTES ||
      typeof chunk.sha256 !== 'string' ||
      !/^[a-f0-9]{64}$/.test(chunk.sha256)
    )
      throw new UploadConflictError('Invalid persisted chunk');
    total += chunk.bytes;
  }
  if (total !== value.receivedBytes || total > Number(value.limit))
    throw new UploadConflictError('Invalid persisted upload total');
  return value as UploadManifest;
}
/** All staging descendants are private real directories, never symlink aliases. */
export async function ensureUploadDirectory(root: string, id: string, chunks = false): Promise<string> {
  uploadRecordKey(root, id);
  let directory = path.resolve(root);
  await fs.mkdir(directory, { recursive: true });
  // Check ancestors explicitly: Windows short names are valid directory aliases,
  // while symlinks and junctions must still be rejected throughout staging.
  for (let ancestor = directory; ; ancestor = path.dirname(ancestor)) {
    const stat = await fs.lstat(ancestor);
    if (!stat.isDirectory() || stat.isSymbolicLink())
      throw new UploadConflictError('Upload staging must use canonical private directories');
    if (ancestor === path.dirname(ancestor)) break;
  }
  directory = await fs.realpath(directory);
  for (const component of ['', id, ...(chunks ? ['chunks'] : [])]) {
    if (component) directory = path.join(directory, component);
    await fs.mkdir(directory, { recursive: true });
    const stat = await fs.lstat(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink() || path.resolve(await fs.realpath(directory)) !== directory)
      throw new UploadConflictError('Upload staging must use canonical private directories');
  }
  // Preserve the caller's spelling for manifest identity and destination checks.
  return path.resolve(root, id, ...(chunks ? ['chunks'] : []));
}
export async function getUploadManifest(root: string, id: string): Promise<UploadManifest | null> {
  const row = await db.runtime.get(uploadRecordKey(root, id));
  if (!row) return null;
  const value = manifest(row.value);
  if (value.root !== path.resolve(root) || value.uploadID !== id || value.principal !== principal())
    throw new UploadConflictError('Upload identity does not match');
  return value;
}
async function put(root: string, id: string, value: UploadManifest): Promise<void> {
  const key = uploadRecordKey(root, id),
    current = await db.runtime.get(key);
  const now = Date.now();
  if (
    !(await db.runtime.compareAndSwap(key, current?.version ?? null, {
      ...value,
      lastActivity: now,
      expiresAt: now + 24 * 60 * 60 * 1000,
    }))
  )
    throw new UploadConflictError('Upload changed concurrently');
}
async function checkFreeSpace(root: string, needed: number) {
  await fs.mkdir(root, { recursive: true });
  const stat = await fs.statfs(root);
  if (stat.bavail * stat.bsize < needed + RESERVE_BYTES)
    throw new UploadTooLargeError(
      'Insufficient free disk space; 2 GiB must remain available for training and database writes',
    );
}
async function reserve(root: string, value: UploadManifest, bytes: number, creating = false) {
  await withProcessLease('upload-budget', async () => {
    const links = await db.runtime.list('upload-active:', 100);
    const active: UploadManifest[] = [];
    for (const link of links) {
      const row = typeof link.value === 'string' ? await db.runtime.get(link.value) : null;
      if (row && !['completed', 'failed'].includes(manifest(row.value).state)) active.push(manifest(row.value));
      else await db.runtime.delete(link.key, link.version);
    }
    if (creating && active.length >= MAX_CONCURRENT_UPLOADS)
      throw new UploadConflictError('Too many active uploads; finish or cancel an existing upload');
    const consumed = active.reduce((total, item) => total + item.receivedBytes + item.reservedBytes, 0);
    if (consumed + bytes > MAX_STAGED_BYTES) throw new UploadTooLargeError('Combined staged upload quota exceeded');
    await checkFreeSpace(root, bytes + active.reduce((total, item) => total + item.reservedBytes, 0));
    value.reservedBytes = bytes;
    const key = uploadRecordKey(root, value.uploadID);
    // Link first: a crash before the manifest write leaves a safely removable empty link.
    await db.runtime.compareAndSwap('upload-active:' + key, null, key);
    await put(root, value.uploadID, value);
  });
}
async function digestFile(filename: string): Promise<Chunk> {
  const hash = createHash('sha256');
  let bytes = 0;
  for await (const raw of createReadStream(filename)) {
    const chunk = Buffer.isBuffer(raw) ? raw : Buffer.from(raw);
    hash.update(chunk);
    bytes += chunk.length;
  }
  return { bytes, sha256: hash.digest('hex') };
}

export async function receiveManifestChunk(
  request: Request,
  root: string,
  input: {
    uploadID: string;
    chunkIndex: number;
    chunksTotal: number;
    expectedBytes: number | null;
    limit: number;
    purpose: string;
  },
) {
  const { uploadID, chunkIndex, chunksTotal, expectedBytes, limit, purpose } = input;
  if (
    chunkIndex < 0 ||
    chunkIndex >= chunksTotal ||
    !Number.isSafeInteger(chunkIndex) ||
    !Number.isSafeInteger(chunksTotal) ||
    chunksTotal > 8192
  )
    throw new UploadConflictError('Invalid archive upload chunk index');
  return withProcessLease(uploadRecordKey(root, uploadID), async lease => {
    let current = await getUploadManifest(root, uploadID);
    const creating = !current;
    current ??= {
      version: 1,
      uploadID,
      root: path.resolve(root),
      purpose,
      principal: principal(),
      chunksTotal,
      expectedBytes,
      limit,
      receivedBytes: 0,
      reservedBytes: 0,
      chunks: {},
      state: 'receiving',
      createdAt: Date.now(),
      lastActivity: Date.now(),
      expiresAt: Date.now() + 86400000,
    };
    if (
      current.state !== 'receiving' ||
      current.purpose !== purpose ||
      current.chunksTotal !== chunksTotal ||
      current.expectedBytes !== expectedBytes ||
      current.limit !== limit
    )
      throw new UploadConflictError('Upload manifest is frozen or its parameters conflict');
    const previous = current.chunks[String(chunkIndex)];
    const maxBytes = Math.min(
      MAX_CHUNK_BYTES,
      limit - current.receivedBytes + (previous?.bytes || 0),
      (expectedBytes ?? limit) - current.receivedBytes + (previous?.bytes || 0),
    );
    if (maxBytes < 0) throw new UploadTooLargeError('Archive intake quota exceeded');
    await reserve(root, current, maxBytes, creating);
    const directory = await ensureUploadDirectory(root, uploadID, true);
    let staged: string | null = null;
    try {
      const result = await streamRequestToStagingFile(request, directory, { maxBytes, prefix: `chunk-${chunkIndex}` });
      staged = result.stagingPath;
      const digest = await digestFile(staged);
      if (previous) {
        if (previous.sha256 !== digest.sha256 || previous.bytes !== digest.bytes)
          throw new UploadConflictError('A different chunk already occupies this index');
      } else {
        const destination = path.join(directory, `${chunkIndex}.part`);
        await lease.assertOwned();
        try {
          await moveStagedUploadNoReplace(staged, destination);
          staged = null;
        } catch (error) {
          if (!error || typeof error !== 'object' || !('code' in error) || error.code !== 'EEXIST') throw error;
          // A process may have died after publishing a chunk but before recording it.
          const stored = await digestFile(destination);
          if (stored.sha256 !== digest.sha256 || stored.bytes !== digest.bytes)
            throw new UploadConflictError('Unacknowledged chunk conflicts with retry');
        }
        current.chunks[String(chunkIndex)] = digest;
        current.receivedBytes += digest.bytes;
      }
      return {
        uploadID,
        chunkIndex,
        chunksTotal,
        receivedBytes: current.receivedBytes,
        ...(expectedBytes === null ? {} : { fileBytes: expectedBytes }),
      };
    } finally {
      await cleanupStagedUpload(staged);
      current.reservedBytes = 0;
      await put(root, uploadID, current);
    }
  });
}

export async function assembleManifestChunks(
  root: string,
  id: string,
  count: number,
  output: string,
  options: { maxBytes?: number; expectedBytes?: number },
) {
  return withProcessLease(uploadRecordKey(root, id), async lease => {
    const directory = await ensureUploadDirectory(root, id);
    if (path.resolve(output) !== path.join(directory, 'upload.zip'))
      throw new UploadConflictError('Invalid assembled upload destination');
    const current = await getUploadManifest(root, id);
    if (!current) throw new UploadConflictError('This upload has no manifest; upload the archive again');
    if (current.chunksTotal !== count || Object.keys(current.chunks).length !== count)
      throw new UploadConflictError('Archive upload is incomplete');
    if (
      (options.expectedBytes !== undefined && current.receivedBytes !== options.expectedBytes) ||
      (current.expectedBytes !== null && current.receivedBytes !== current.expectedBytes)
    )
      throw new UploadConflictError('Invalid archive upload fileBytes');
    if (current.receivedBytes > (options.maxBytes ?? current.limit)) throw new UploadTooLargeError();
    if (current.state === 'assembled' && current.assembledPath === output) return;
    if (!['receiving', 'finalizing'].includes(current.state))
      throw new UploadConflictError('Upload is already being imported');
    current.state = 'finalizing';
    await reserve(root, current, current.receivedBytes);
    const staging = `${output}.assembling`;
    try {
      await fs.mkdir(path.dirname(output), { recursive: true });
      for (let index = 0; index < count; index++) {
        const expected = current.chunks[String(index)],
          hash = createHash('sha256');
        let bytes = 0;
        const verify = new Transform({
          transform(chunk: Buffer, _encoding, callback) {
            bytes += chunk.length;
            hash.update(chunk);
            callback(
              bytes > expected.bytes ? new UploadConflictError('Chunk changed during finalization') : null,
              chunk,
            );
          },
        });
        await pipeline(
          createReadStream(path.join(root, id, 'chunks', `${index}.part`)),
          verify,
          createWriteStream(staging, { flags: index === 0 ? 'w' : 'a' }),
        );
        if (bytes !== expected.bytes || hash.digest('hex') !== expected.sha256)
          throw new UploadConflictError('Chunk checksum mismatch');
      }
      await lease.assertOwned();
      await fs.rename(staging, output);
      current.state = 'assembled';
      current.assembledPath = output;
    } catch (error) {
      await fs.rm(staging, { force: true }).catch(() => undefined);
      current.state = 'receiving';
      throw error;
    } finally {
      current.reservedBytes = current.state === 'assembled' ? current.receivedBytes : 0;
      await put(root, id, current);
    }
  });
}

export async function setUploadState(root: string, id: string, state: UploadManifest['state']): Promise<void> {
  await withProcessLease(uploadRecordKey(root, id), async () => {
    const current = await getUploadManifest(root, id);
    if (current) {
      current.state = state;
      await put(root, id, current);
      if (['completed', 'failed'].includes(state))
        await db.runtime.delete('upload-active:' + uploadRecordKey(root, id));
    }
  });
}

export async function cleanupExpiredUploads(root: string): Promise<void> {
  const prefix = uploadRecordKey(root, 'placeholder').replace(/placeholder$/, '');
  const cursorKey = 'upload-cleanup-cursor:' + prefix;
  const cursor = await db.runtime.get(cursorKey);
  const records = await db.runtime.list(prefix, 100, typeof cursor?.value === 'string' ? cursor.value : '');
  await db.runtime.compareAndSwap(
    cursorKey,
    cursor?.version ?? null,
    records.length === 100 ? records.at(-1)!.key : '',
  );
  for (const row of records) {
    const current = manifest(row.value);
    if (current.expiresAt > Date.now() || current.state === 'importing') continue;
    await withProcessLease(row.key, async () => {
      const fresh = await getUploadManifest(root, current.uploadID);
      if (!fresh || fresh.expiresAt > Date.now() || fresh.state === 'importing') return;
      const directory = await ensureUploadDirectory(root, fresh.uploadID);
      await fs.rm(directory, { recursive: true, force: true });
      const latest = await db.runtime.get(row.key);
      if (latest) await db.runtime.delete(row.key, latest.version);
      await db.runtime.delete('upload-active:' + row.key);
    }).catch(() => undefined);
  }
}
