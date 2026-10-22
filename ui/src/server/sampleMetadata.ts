import fs from 'fs/promises';
import path from 'path';
import { parseSampleMetadata, type SampleMetadata } from '../utils/sampleMetadata';

export async function readSampleMetadata(mediaPath: string): Promise<SampleMetadata | null> {
  const sidecar = mediaPath + '.metadata.json';
  try {
    const stat = await fs.lstat(sidecar);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 512 * 1024) return null;
    const root = await fs.realpath(path.dirname(mediaPath));
    if (path.dirname(await fs.realpath(sidecar)) !== root) return null;
    const value: unknown = JSON.parse(await fs.readFile(sidecar, 'utf8'));
    return parseSampleMetadata(value);
  } catch { return null; }
}

export async function deleteSampleMetadata(mediaPath: string): Promise<void> {
  // Unlinking the sidecar itself never follows a symbolic link or deletes its target.
  await fs.unlink(mediaPath + '.metadata.json').catch((error: unknown) => {
    if (!(error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT')) throw error;
  });
}
