import fs from 'fs/promises';
import path from 'path';
import { createHash } from 'crypto';
import { db } from './db';
import { withProcessLease } from './processLease';
import { isPathWithinRoot } from './pathContainment';

export class DatasetUploadCollision extends Error {
  readonly status = 409;
  constructor() { super('An image or caption with this name already exists. Rename the media and captions together before uploading.'); }
}

/** Serialize publication by canonical stem across requests/processes, not by extension.
 * Only members of the same client-planned group may add sidecars to newly uploaded media.
 * Claims are durable so a delayed request cannot attach itself to another batch.
 */
export async function publishDatasetUpload<T>(root: string, target: string, groupID: string, publish: () => Promise<T>) {
  await fs.mkdir(path.dirname(target), { recursive: true });
  const [canonicalRoot, directory] = await Promise.all([fs.realpath(root), fs.realpath(path.dirname(target))]);
  if (!isPathWithinRoot(canonicalRoot, directory)) throw new Error('Upload path escapes dataset');
  const name = path.basename(target);
  const stem = name.slice(0, name.length - path.extname(name).length).toLowerCase();
  const key = 'dataset-upload:' + createHash('sha256').update(path.join(directory, stem)).digest('hex');
  return withProcessLease(key, async () => {
    const row = await db.runtime.get(key);
    const sameGroup = row?.value && typeof row.value === 'object' && 'groupID' in row.value && row.value.groupID === groupID;
    const names = await fs.readdir(directory);
    const occupied = names.some(entry => entry.slice(0, entry.length - path.extname(entry).length).toLowerCase() === stem);
    if (names.some(entry => entry.toLowerCase() === name.toLowerCase()) || (occupied && !sameGroup)) throw new DatasetUploadCollision();
    if (!sameGroup && !(await db.runtime.compareAndSwap(key, row?.version ?? null, { groupID }))) throw new DatasetUploadCollision();
    return publish();
  });
}
