import fs from 'fs';
import path from 'path';
import { DATASET_TEXT_CAPTION_EXTENSIONS } from './captionFiles';
import { isDatasetRootCaptionEntry } from './datasetRootCaption';

export const DATASET_MEDIA_EXTENSIONS = [
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
  '.jxl',
  '.gif',
  '.bmp',
  '.mp4',
  '.avi',
  '.mov',
  '.mkv',
  '.wmv',
  '.m4v',
  '.flv',
  '.webm',
  '.mp3',
  '.wav',
  '.flac',
  '.ogg',
  '.m4a',
  '.aac',
];

/**
 * Recursively finds all editable dataset items in a directory and its subdirectories.
 */
export function findDatasetItemsRecursively(dir: string, datasetRoot = dir): string[] {
  const mediaStems = new Set<string>();
  const candidateTextFiles: string[] = [];
  let results: string[] = [];

  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const name = entry.name;
    if (name.startsWith('.')) continue;
    const itemPath = path.join(dir, name);

    if (entry.isDirectory()) {
      if (name === '_controls') continue;
      results = results.concat(findDatasetItemsRecursively(itemPath, datasetRoot));
    } else if (entry.isFile()) {
      if (isDatasetRootCaptionEntry(datasetRoot, dir, name)) continue;
      const ext = path.extname(name).toLowerCase();
      if (DATASET_MEDIA_EXTENSIONS.includes(ext)) {
        mediaStems.add(itemPath.slice(0, -ext.length).toLowerCase());
        results.push(itemPath);
      } else if (DATASET_TEXT_CAPTION_EXTENSIONS.includes(ext)) {
        candidateTextFiles.push(itemPath);
      }
    }
  }

  for (const textPath of candidateTextFiles) {
    const ext = path.extname(textPath).toLowerCase();
    const stem = textPath.slice(0, -ext.length).toLowerCase();
    if (!mediaStems.has(stem)) {
      results.push(textPath);
    }
  }

  return results;
}

/**
 * Async variant for route handlers that need to list large datasets without
 * blocking other requests while walking subdirectories.
 */
export async function findDatasetItemsRecursivelyAsync(dir: string, datasetRoot = dir): Promise<string[]> {
  const mediaStems = new Set<string>();
  const candidateTextFiles: string[] = [];
  let results: string[] = [];

  const entries = await fs.promises.readdir(dir, { withFileTypes: true });
  const subdirs: string[] = [];

  for (const entry of entries) {
    const name = entry.name;
    if (name.startsWith('.')) continue;
    const itemPath = path.join(dir, name);

    if (entry.isDirectory()) {
      if (name === '_controls') continue;
      subdirs.push(itemPath);
    } else if (entry.isFile()) {
      if (isDatasetRootCaptionEntry(datasetRoot, dir, name)) continue;
      const ext = path.extname(name).toLowerCase();
      if (DATASET_MEDIA_EXTENSIONS.includes(ext)) {
        mediaStems.add(itemPath.slice(0, -ext.length).toLowerCase());
        results.push(itemPath);
      } else if (DATASET_TEXT_CAPTION_EXTENSIONS.includes(ext)) {
        candidateTextFiles.push(itemPath);
      }
    }
  }

  const nestedResults = await Promise.all(subdirs.map(subdir => findDatasetItemsRecursivelyAsync(subdir, datasetRoot)));
  for (const nested of nestedResults) {
    results = results.concat(nested);
  }

  for (const textPath of candidateTextFiles) {
    const ext = path.extname(textPath).toLowerCase();
    const stem = textPath.slice(0, -ext.length).toLowerCase();
    if (!mediaStems.has(stem)) {
      results.push(textPath);
    }
  }

  return results;
}
