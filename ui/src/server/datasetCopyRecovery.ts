import { createHash, randomUUID } from 'crypto';
import fs from 'fs/promises';
import path from 'path';
import { db } from './db';
import { CommandInputError, isRecord } from './commandInput';
import { withProcessLease, LeaseBusyError } from './processLease';
import { publishDatasetDirectory } from './publishDatasetDirectory';
import { assertPlainDatasetPath } from './plainDatasetCommand';
import { isPathInside } from './remoteCaptionSecurity';

type DirectoryIdentity = { device: string; inode: string; birth: string };
type CopyStage = { name: string; wrapper: DirectoryIdentity; payload: DirectoryIdentity };
type CopyReceipt = {
  version: 1;
  digest: string;
  owner: string;
  name: string;
  state: 'reserved' | 'copying' | 'ready' | 'completed';
  stage: CopyStage | null;
};
type PreparedCopy = { sourcePath: string; copyTo: (destination: string) => Promise<void> };
type CopyRecoveryRequest = {
  idempotencyKey: string;
  datasetPath: string;
  sourceDatasetsRoot: string;
  destinationDatasetsRoot: string;
  requestedName: string;
  suffix: string;
  preferredName: (sourcePath: string) => string;
  prepare: () => Promise<PreparedCopy>;
};

const hash = (value: string) => createHash('sha256').update(value).digest('hex');
const ownerMarker = '.aitk-copy-owner.json';
function fail(message: string, code = 'COPY_OWNERSHIP_CONFLICT', status = 409): never {
  throw new CommandInputError(message, status, code);
}
export function validateDatasetCopyKey(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== 'string' || !/^[a-zA-Z0-9_-]{8,128}$/.test(value))
    throw new CommandInputError('Invalid idempotencyKey');
  return value;
}
function isName(value: unknown): value is string {
  return typeof value === 'string' && !!value && value !== '.' && !value.includes('..') && !/[\\/<>:"|?*\x00-\x1f]/.test(value);
}
function isIdentity(value: unknown): value is DirectoryIdentity {
  return isRecord(value) && ['device', 'inode', 'birth'].every(key => typeof value[key] === 'string' && /^\d+$/.test(value[key]));
}
function readReceipt(value: unknown): CopyReceipt {
  if (!isRecord(value) || value.version !== 1 || typeof value.digest !== 'string' || !/^[a-f0-9]{64}$/.test(value.digest) ||
      typeof value.owner !== 'string' || !/^[a-f0-9-]{36}$/.test(value.owner) || !isName(value.name) ||
      typeof value.state !== 'string' || !['reserved', 'copying', 'ready', 'completed'].includes(value.state)) fail('Invalid dataset copy receipt');
  if (value.stage !== null && (!isRecord(value.stage) || typeof value.stage.name !== 'string' ||
      !/^\.aitk-copy-[a-f0-9-]{36}-[A-Za-z0-9]+$/.test(value.stage.name) ||
      !isIdentity(value.stage.wrapper) || !isIdentity(value.stage.payload))) fail('Invalid dataset copy staging receipt');
  if (value.state !== 'reserved' && value.stage === null) fail('Dataset copy receipt is incomplete');
  return value as CopyReceipt;
}
function child(root: string, name: string): string {
  const result = path.resolve(root, name);
  if (path.dirname(result) !== root || !isPathInside(root, result) || result === root) fail('Invalid dataset copy path');
  return result;
}
async function directoryIdentity(filename: string): Promise<DirectoryIdentity | null> {
  const stat = await fs.lstat(filename, { bigint: true }).catch((error: unknown) => {
    if (isRecord(error) && error.code === 'ENOENT') return null;
    throw error;
  });
  if (!stat) return null;
  if (!stat.isDirectory() || stat.isSymbolicLink()) fail('Dataset copy directory was replaced');
  if (stat.ino === BigInt(0)) fail('Filesystem cannot establish dataset copy ownership');
  if (await fs.realpath(filename) !== filename) fail('Dataset copy path no longer identifies its original directory');
  return { device: String(stat.dev), inode: String(stat.ino), birth: String(stat.birthtimeNs) };
}
const sameIdentity = (left: DirectoryIdentity | null, right: DirectoryIdentity) =>
  left !== null && left.device === right.device && left.inode === right.inode && left.birth === right.birth;

async function assertStage(root: string, receipt: CopyReceipt, requirePayload = true): Promise<string> {
  if (!receipt.stage) fail('Dataset copy staging is missing');
  const wrapper = child(root, receipt.stage.name);
  if (!sameIdentity(await directoryIdentity(wrapper), receipt.stage.wrapper)) fail('Dataset copy staging ownership changed');
  const marker = child(wrapper, ownerMarker);
  const stat = await fs.lstat(marker);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 1024) fail('Invalid dataset copy staging owner');
  const value: unknown = JSON.parse(await fs.readFile(marker, 'utf8'));
  if (!isRecord(value) || value.owner !== receipt.owner || value.digest !== receipt.digest) fail('Dataset copy staging owner changed');
  const payload = child(wrapper, 'payload');
  const identity = await directoryIdentity(payload);
  if ((requirePayload || identity !== null) && !sameIdentity(identity, receipt.stage.payload)) fail('Dataset copy payload ownership changed');
  return wrapper;
}
async function removeOwnedStage(root: string, receipt: CopyReceipt): Promise<void> {
  if (!receipt.stage) return;
  const wrapper = await assertStage(root, receipt, false);
  // Refuse nested aliases before recursive cleanup, including junctions on Windows.
  const verify = async (directory: string): Promise<void> => {
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
      const filename = path.join(directory, entry.name);
      const stat = await fs.lstat(filename);
      if (stat.isSymbolicLink() || !isPathInside(wrapper, await fs.realpath(filename))) fail('Dataset copy staging contains an unsafe entry');
      if (stat.isDirectory()) await verify(filename);
    }
  };
  await verify(wrapper);
  await fs.rm(wrapper, { recursive: true });
}

/** A receipt stores only request/ownership digests and filesystem identities, never dataset contents. */
export async function recoverDatasetCopy(request: CopyRecoveryRequest): Promise<{ name: string; path: string }> {
  validateDatasetCopyKey(request.idempotencyKey);
  const sourceRoot = await fs.realpath(path.resolve(request.sourceDatasetsRoot));
  const requestedRoot = path.resolve(request.destinationDatasetsRoot);
  await fs.mkdir(requestedRoot, { recursive: true });
  const root = await fs.realpath(requestedRoot);
  const digest = hash(JSON.stringify({
    source: path.resolve(request.datasetPath), sourceRoot, root,
    name: request.requestedName, suffix: request.suffix,
  }));
  const key = `dataset-copy-command:${hash(request.idempotencyKey)}`;
  return withProcessLease(key, async lease => {
    let row = await db.runtime.get(key);
    let receipt = row ? readReceipt(row.value) : null;
    if (receipt && receipt.digest !== digest) fail('Dataset copy key belongs to different input', 'COPY_KEY_CONFLICT');
    let prepared: PreparedCopy | undefined;
    const save = async (next: CopyReceipt): Promise<void> => {
      await lease.assertOwned();
      if (!await db.runtime.compareAndSwap(key, row?.version ?? null, next)) fail('Dataset copy command is busy; retry', 'COPY_BUSY');
      row = await db.runtime.get(key);
      if (!row) fail('Dataset copy receipt disappeared');
      receipt = next;
    };
    if (!receipt) {
      prepared = await request.prepare();
      if (isPathInside(prepared.sourcePath, root)) throw new CommandInputError('Dataset copy source and destination cannot contain one another');
      const owner = randomUUID();
      await withProcessLease(`dataset-copy-root:${hash(root)}`, async rootLease => {
        const base = request.preferredName(prepared!.sourcePath);
        for (let counter = 1; counter <= 10000; counter++) {
          const name = counter === 1 ? base : `${base}_${counter}`;
          if (!isName(name)) throw new CommandInputError('Invalid dataset copy destination');
          const destination = child(root, name);
          if (isPathInside(destination, prepared!.sourcePath)) continue;
          if (await fs.lstat(destination).catch((error: unknown) => {
            if (isRecord(error) && error.code === 'ENOENT') return null;
            throw error;
          })) continue;
          const slot = `dataset-copy-destination:${hash(process.platform === 'win32' ? destination.toLowerCase() : destination)}`;
          await rootLease.assertOwned();
          if (!await db.runtime.compareAndSwap(slot, null, owner)) continue;
          await save({ version: 1, digest, owner, name, state: 'reserved', stage: null });
          return;
        }
        fail('No dataset copy destination is available', 'COPY_DESTINATION_CONFLICT');
      }, { waitMs: 5000 });
    }
    if (!receipt) fail('Dataset copy receipt is missing');
    // Re-read after durable writes; asynchronous callbacks cannot narrow this variable.
    let current = readReceipt(row!.value);
    const destination = child(root, current.name);
    const result = { name: current.name, path: path.join(requestedRoot, current.name) };
    const existing = await directoryIdentity(destination);
    if (existing) {
      if (!['ready', 'completed'].includes(current.state) || !current.stage || !sameIdentity(existing, current.stage.payload))
        fail('Dataset copy destination is occupied by another directory', 'COPY_DESTINATION_CONFLICT');
      await assertPlainDatasetPath(destination, root);
      if (current.state !== 'completed') await save({ ...current, state: 'completed' });
      await removeOwnedStage(root, current).catch(() => undefined);
      return result;
    }
    if (current.state === 'completed') fail('The completed dataset copy was deleted', 'COPY_RESULT_MISSING', 410);
    if (current.state !== 'ready') {
      prepared ??= await request.prepare();
      if (isPathInside(prepared.sourcePath, destination) || isPathInside(destination, prepared.sourcePath))
        throw new CommandInputError('Dataset copy source and destination cannot contain one another');
      // A retry starts in a fresh owned wrapper. A crash before its receipt write
      // can leave only an empty staging artifact, never an ambiguous publication.
      const wrapper = await fs.mkdtemp(path.join(root, `.aitk-copy-${current.owner}-`));
      const payload = child(wrapper, 'payload');
      await fs.mkdir(payload);
      await fs.writeFile(child(wrapper, ownerMarker), JSON.stringify({ owner: current.owner, digest }), { flag: 'wx' });
      const wrapperIdentity = await directoryIdentity(wrapper), payloadIdentity = await directoryIdentity(payload);
      if (!wrapperIdentity || !payloadIdentity) fail('Dataset copy staging disappeared');
      const prior = current;
      current = { ...current, state: 'copying', stage: { name: path.basename(wrapper), wrapper: wrapperIdentity, payload: payloadIdentity } };
      await save(current);
      if (prior.stage) await removeOwnedStage(root, prior).catch(() => undefined);
      await prepared.copyTo(payload);
      await assertStage(root, current);
      current = { ...current, state: 'ready' };
      await save(current);
    }
    const wrapper = await assertStage(root, current).catch((error: unknown) => {
      if (isRecord(error) && error.code === 'ENOENT') fail('Dataset copy publication is missing', 'COPY_RESULT_MISSING', 410);
      throw error;
    });
    await lease.assertOwned();
    if (await fs.realpath(requestedRoot) !== root) fail('Dataset copy root changed');
    await publishDatasetDirectory(child(wrapper, 'payload'), destination);
    if (!current.stage || !sameIdentity(await directoryIdentity(destination), current.stage.payload)) fail('Dataset copy publication ownership changed');
    await save({ ...current, state: 'completed' });
    await removeOwnedStage(root, current).catch(() => undefined);
    return result;
  }, { waitMs: 5000 }).catch((error: unknown) => {
    if (error instanceof LeaseBusyError) fail('Dataset copy is in progress; retry the same command', 'COPY_BUSY');
    throw error;
  });
}
