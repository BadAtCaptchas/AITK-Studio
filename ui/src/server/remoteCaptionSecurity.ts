import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';

export const REMOTE_CAPTION_MEDIA_EXTENSIONS = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
  '.bmp',
  '.gif',
  '.mp4',
  '.avi',
  '.mov',
  '.mkv',
  '.wmv',
  '.m4v',
  '.flv',
  '.mp3',
  '.wav',
  '.flac',
  '.ogg',
]);

const REMOTE_CAPTION_TEXT_EXTENSIONS = new Set(['.txt', '.caption', '.json', '.sdxl', '.md']);

export function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}

export function normalizeRemoteCaptionExtension(captionExtension: string) {
  const normalized = captionExtension.trim().replace(/^\.+/, '').toLowerCase() || 'txt';
  const suffix = `.${normalized}`;
  if (!REMOTE_CAPTION_TEXT_EXTENSIONS.has(suffix)) {
    throw new Error('Remote caption results must use a supported text caption extension');
  }
  return suffix;
}

export async function realpathForPath(pathname: string) {
  return fsp.realpath(pathname).catch(() => path.resolve(pathname));
}

export async function resolveDatasetDirectoryInsideRoot(datasetPath: string, datasetsRoot: string) {
  const stat = await fsp.stat(datasetPath);
  if (!stat.isDirectory()) {
    throw new Error('Caption dataset not found');
  }

  const [realDatasetPath, realDatasetsRoot] = await Promise.all([
    fsp.realpath(datasetPath),
    realpathForPath(datasetsRoot),
  ]);

  if (!isPathInside(realDatasetsRoot, realDatasetPath)) {
    throw new Error('Remote caption dataset must be inside the configured datasets folder');
  }

  return realDatasetPath;
}

export function hasMatchingTargetMediaFile(targetRoot: string, captionRelativePath: string, captionSuffix: string) {
  if (!captionRelativePath.toLowerCase().endsWith(captionSuffix)) return false;
  const withoutCaptionExt = captionRelativePath.slice(0, -captionSuffix.length);
  for (const mediaExt of REMOTE_CAPTION_MEDIA_EXTENSIONS) {
    const mediaPath = path.resolve(targetRoot, ...`${withoutCaptionExt}${mediaExt}`.replace(/\\/g, '/').split('/'));
    if (!isPathInside(targetRoot, mediaPath)) continue;
    try {
      if (fs.statSync(mediaPath).isFile()) return true;
    } catch {
      // Try the next supported media extension.
    }
  }
  return false;
}
