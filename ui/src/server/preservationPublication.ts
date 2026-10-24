import fs from 'fs/promises';
import path from 'path';
import { isRecord } from './commandInput';
import { validSha256 } from '../domain/loraContinuation';
import { hashContinuationAsset } from './loraContinuation';

export type PreservationReceipt = { path: string; name: string; source_sha256: string; count: number };

/** An idempotent retry acknowledges the original, intact snapshot. */
export async function readPreservationPublication(destination: string, requestID: string,
  jobID: string, samples: string[]): Promise<PreservationReceipt | null> {
  const existing = await fs.lstat(destination).catch((error: unknown) => {
    if (isRecord(error) && error.code === 'ENOENT') return null;
    throw error;
  });
  if (!existing) return null;
  if (!existing.isDirectory() || existing.isSymbolicLink()) throw new Error('Dataset name is already in use.');
  const manifestPath = await fs.realpath(path.join(destination, '.aitk-preservation.json'));
  if (path.dirname(manifestPath) !== destination || (await fs.stat(manifestPath)).size > 4 * 1024 ** 2)
    throw new Error('Invalid existing preservation manifest.');
  const manifest: unknown = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  if (!isRecord(manifest) || manifest.version !== 1 || manifest.request_id !== requestID || manifest.generation_job !== jobID ||
      JSON.stringify(manifest.selected_samples) !== JSON.stringify(samples) || !validSha256(manifest.source_sha256) ||
      !Array.isArray(manifest.examples) || manifest.examples.length !== samples.length)
    throw new Error('Dataset name is already in use.');
  for (const [index, example] of manifest.examples.entries()) {
    const stem = String(index).padStart(4, '0');
    if (!isRecord(example) || example.image !== stem + path.extname(samples[index]).toLowerCase() || example.caption !== stem + '.txt')
      throw new Error('Invalid existing preservation example.');
    for (const [field, limit] of [['image', 64 * 1024 ** 2], ['caption', 512 * 1024]] as const) {
      const file = await fs.realpath(path.join(destination, String(example[field])));
      const stat = await fs.stat(file);
      if (path.dirname(file) !== destination || !stat.isFile() || stat.size > limit ||
          !validSha256(example[field + '_sha256']) || await hashContinuationAsset(file) !== example[field + '_sha256'])
        throw new Error('Published preservation dataset changed; use a new dataset name.');
    }
  }
  return { path: destination, name: path.basename(destination), source_sha256: manifest.source_sha256, count: manifest.examples.length };
}
