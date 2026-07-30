import fs from 'fs';
import path from 'path';

function isPathInsideRoot(root: string, filepath: string) {
  const relativePath = path.relative(root, filepath);
  return relativePath === '' || (!relativePath.startsWith('..') && !path.isAbsolute(relativePath));
}

async function realpathIfExists(filepath: string) {
  return fs.promises.realpath(path.resolve(filepath)).catch(() => null);
}

export async function resolveSampleThumbnail(root: string, filename: string) {
  if (filename !== path.basename(filename) || filename.includes('/') || filename.includes('\\')) {
    return null;
  }

  const thumbnailsRoot = await realpathIfExists(path.join(root, '.thumbs'));
  if (!thumbnailsRoot || !isPathInsideRoot(root, thumbnailsRoot)) return null;

  const thumbnailPath = await realpathIfExists(path.join(thumbnailsRoot, `${filename}.jpg`));
  if (!thumbnailPath || !isPathInsideRoot(thumbnailsRoot, thumbnailPath)) return null;

  const stat = await fs.promises.stat(thumbnailPath).catch(() => null);
  if (!stat || !stat.isFile()) return null;

  return { path: thumbnailPath, stat, contentType: 'image/jpeg' as const };
}
