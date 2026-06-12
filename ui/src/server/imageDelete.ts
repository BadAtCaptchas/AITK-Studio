import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { DATASET_TEXT_CAPTION_EXTENSIONS, deleteCaptionSidecars } from './captionFiles';
import { findEncryptedDatasetRoot } from './encryptedDatasets';

export type ImageDeleteItemResult = {
  imgPath: string;
  deleted: boolean;
  skipped?: boolean;
  error?: string;
};

export type ImageDeleteBulkResult = {
  success: boolean;
  requested: number;
  deleted: number;
  skipped: number;
  failed: number;
  removedPaths: string[];
  results: ImageDeleteItemResult[];
};

export class ImageDeleteError extends Error {
  status: number;

  constructor(message: string, status = 400) {
    super(message);
    this.name = 'ImageDeleteError';
    this.status = status;
  }
}

const MEDIA_EXTENSIONS = new Set([
  '.jpg',
  '.jpeg',
  '.png',
  '.bmp',
  '.gif',
  '.tiff',
  '.webp',
  '.mp4',
  '.mp3',
  '.wav',
  '.flac',
  '.ogg',
  ...DATASET_TEXT_CAPTION_EXTENSIONS,
]);

function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative !== '' && !relative.startsWith('..') && !path.isAbsolute(relative);
}

function ensureImagePaths(value: unknown) {
  if (!Array.isArray(value)) throw new ImageDeleteError('imgPaths must be an array');
  const paths = value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0);
  if (paths.length === 0) throw new ImageDeleteError('No images were selected');
  return Array.from(new Set(paths));
}

function validatePlainImagePaths(imgPaths: string[], datasetsRoot: string, trainingRoot: string) {
  const allowedRoots = [datasetsRoot, trainingRoot].map(root => path.resolve(root));
  const resolvedDatasetsRoot = path.resolve(datasetsRoot);

  return imgPaths.map(imgPath => {
    const normalized = path.resolve(imgPath);
    const isWithinAllowedRoot = allowedRoots.some(root => isPathInside(root, normalized));
    if (!isWithinAllowedRoot) {
      throw new ImageDeleteError('Invalid image path');
    }

    if (findEncryptedDatasetRoot(normalized, resolvedDatasetsRoot)) {
      throw new ImageDeleteError('Encrypted dataset objects must be deleted through the encrypted dataset API', 403);
    }

    const extension = path.extname(normalized).toLowerCase();
    if (!MEDIA_EXTENSIONS.has(extension)) {
      throw new ImageDeleteError('Unsupported media path');
    }

    return { imgPath, normalized };
  });
}

export async function deletePlainImagePaths(
  rawImgPaths: unknown,
  datasetsRoot: string,
  trainingRoot: string,
): Promise<ImageDeleteBulkResult> {
  const imgPaths = ensureImagePaths(rawImgPaths);
  const validated = validatePlainImagePaths(imgPaths, datasetsRoot, trainingRoot);
  const results: ImageDeleteItemResult[] = [];
  const removedPaths: string[] = [];

  for (const item of validated) {
    try {
      if (!fs.existsSync(item.normalized)) {
        results.push({ imgPath: item.imgPath, deleted: false, skipped: true });
        continue;
      }
      await fsp.unlink(item.normalized);
      const extension = path.extname(item.normalized).toLowerCase();
      if (!DATASET_TEXT_CAPTION_EXTENSIONS.includes(extension)) {
        deleteCaptionSidecars(item.normalized);
      }
      removedPaths.push(item.imgPath);
      results.push({ imgPath: item.imgPath, deleted: true });
    } catch (error) {
      results.push({
        imgPath: item.imgPath,
        deleted: false,
        error: error instanceof Error ? error.message : 'Failed to delete image',
      });
    }
  }

  const deleted = results.filter(result => result.deleted).length;
  const skipped = results.filter(result => result.skipped).length;
  const failed = results.filter(result => result.error).length;

  return {
    success: failed === 0,
    requested: validated.length,
    deleted,
    skipped,
    failed,
    removedPaths,
    results,
  };
}

export function isImageDeleteError(error: unknown): error is ImageDeleteError {
  return error instanceof ImageDeleteError;
}
