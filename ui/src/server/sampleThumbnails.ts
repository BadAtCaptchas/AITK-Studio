import fs from 'fs';
import path from 'path';
import { isPathWithinRoot } from './pathContainment';

async function realpathIfExists(filepath: string) {
  return fs.promises.realpath(path.resolve(filepath)).catch(() => null);
}

export async function resolveSampleThumbnail(root: string, filename: string) {
  if (filename !== path.basename(filename) || filename.includes('/') || filename.includes('\\')) {
    return null;
  }

  const canonicalRoot = await realpathIfExists(root);
  if (!canonicalRoot) return null;
  const thumbnailsRoot = await realpathIfExists(path.join(canonicalRoot, '.thumbs'));
  if (!thumbnailsRoot || !isPathWithinRoot(canonicalRoot, thumbnailsRoot)) return null;

  const thumbnailPath = await realpathIfExists(path.join(thumbnailsRoot, `${filename}.jpg`));
  if (!thumbnailPath || !isPathWithinRoot(thumbnailsRoot, thumbnailPath)) return null;

  const stat = await fs.promises.stat(thumbnailPath).catch(() => null);
  if (!stat || !stat.isFile()) return null;

  return { path: thumbnailPath, stat, contentType: 'image/jpeg' as const };
}
