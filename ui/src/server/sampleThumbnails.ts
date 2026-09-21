import fs from 'fs';
import path from 'path';
import { isPathWithinRoot } from './pathContainment';

async function realpathIfExists(filepath: string) {
  return fs.promises.realpath(path.resolve(filepath)).catch(() => null);
}

type SampleThumbnail = {
  path: string;
  stat: fs.Stats;
  contentType: 'image/png' | 'image/jpeg';
};

async function resolveSampleThumbnails(root: string, filename: string): Promise<SampleThumbnail[]> {
  if (filename !== path.basename(filename) || filename.includes('/') || filename.includes('\\')) {
    return [];
  }

  const canonicalRoot = await realpathIfExists(root);
  if (!canonicalRoot) return [];
  const thumbnailsRoot = await realpathIfExists(path.join(canonicalRoot, '.thumbs'));
  if (!thumbnailsRoot || !isPathWithinRoot(canonicalRoot, thumbnailsRoot)) return [];

  const thumbnails: SampleThumbnail[] = [];
  for (const [extension, contentType] of [['.png', 'image/png'], ['.jpg', 'image/jpeg']] as const) {
    const thumbnailPath = await realpathIfExists(path.join(thumbnailsRoot, filename + extension));
    if (!thumbnailPath || !isPathWithinRoot(thumbnailsRoot, thumbnailPath)) continue;

    const stat = await fs.promises.stat(thumbnailPath).catch(() => null);
    if (!stat || !stat.isFile()) continue;
    thumbnails.push({ path: thumbnailPath, stat, contentType });
  }
  return thumbnails;
}

export async function resolveSampleThumbnail(root: string, filename: string): Promise<SampleThumbnail | null> {
  return (await resolveSampleThumbnails(root, filename))[0] ?? null;
}

export async function removeSampleThumbnails(root: string, filename: string): Promise<void> {
  const thumbnails = await resolveSampleThumbnails(root, filename);
  await Promise.all(thumbnails.map(thumbnail => fs.promises.unlink(thumbnail.path).catch(() => undefined)));
}
