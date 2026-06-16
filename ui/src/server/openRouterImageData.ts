import fs from 'fs/promises';
import path from 'path';
import { findEncryptedDatasetRoot } from '@/server/encryptedDatasets';
import { getRemoteWorker, remoteFetch } from '@/server/remoteClient';
import { isPathInside, realpathForPath } from '@/server/remoteCaptionSecurity';
import { getDatasetsRoot } from '@/server/settings';
import { parseRemoteDatasetAssetRef } from '@/utils/remoteDatasetRefs';

const MAX_ENCRYPTED_UPLOAD_BYTES = 32 * 1024 * 1024;

const IMAGE_MIME_BY_EXT: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
  '.jxl': 'image/jxl',
  '.gif': 'image/gif',
  '.bmp': 'image/bmp',
};

const ALLOWED_UPLOAD_MIMES = new Set(Object.values(IMAGE_MIME_BY_EXT));

function dataUrlFromBytes(bytes: Buffer, mimeType: string) {
  return `data:${mimeType};base64,${bytes.toString('base64')}`;
}

function imageMimeForPath(filepath: string) {
  return IMAGE_MIME_BY_EXT[path.extname(filepath).toLowerCase()] || null;
}

export function boolFromValue(value: unknown) {
  return value === true || value === 'true' || value === '1';
}

export function positiveNumberFromValue(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}

export function nonNegativeIntegerFromValue(value: unknown) {
  const number = Number(value);
  return Number.isInteger(number) && number >= 0 ? number : null;
}

export async function plainOpenRouterImageDataUrl(imgPath: unknown, featureName: string) {
  if (typeof imgPath !== 'string' || !imgPath.trim()) {
    throw new Error('imgPath is required.');
  }

  const remoteAsset = parseRemoteDatasetAssetRef(imgPath);
  if (remoteAsset) {
    if (remoteAsset.type !== 'img') {
      throw new Error(`${featureName} works on image assets only.`);
    }
    const worker = await getRemoteWorker(remoteAsset.workerID);
    const remoteResponse = await remoteFetch(worker, `/api/img/${encodeURIComponent(remoteAsset.path)}`);
    const responseMime = remoteResponse.headers.get('content-type')?.split(';', 1)[0]?.trim().toLowerCase() || '';
    const mimeType = ALLOWED_UPLOAD_MIMES.has(responseMime) ? responseMime : imageMimeForPath(remoteAsset.path);
    if (!mimeType) {
      throw new Error(`${featureName} supports JPG, PNG, WebP, JPEG XL, GIF, and BMP images.`);
    }
    return dataUrlFromBytes(Buffer.from(await remoteResponse.arrayBuffer()), mimeType);
  }

  const datasetsRoot = await getDatasetsRoot();
  const resolvedImagePath = path.resolve(imgPath);
  const [realDatasetsRoot, realImagePath] = await Promise.all([
    realpathForPath(datasetsRoot),
    realpathForPath(resolvedImagePath),
  ]);

  if (!isPathInside(realDatasetsRoot, realImagePath)) {
    throw new Error('Image path must be inside the datasets folder.');
  }
  if (findEncryptedDatasetRoot(realImagePath, realDatasetsRoot)) {
    throw new Error('Encrypted images must be sent from the unlocked studio after confirmation.');
  }

  const stat = await fs.stat(realImagePath).catch(() => null);
  if (!stat?.isFile()) {
    throw new Error('Image file was not found.');
  }

  const mimeType = imageMimeForPath(realImagePath);
  if (!mimeType) {
    throw new Error(`${featureName} supports JPG, PNG, WebP, JPEG XL, GIF, and BMP images.`);
  }

  return dataUrlFromBytes(await fs.readFile(realImagePath), mimeType);
}

export async function encryptedOpenRouterUploadImageDataUrl(formData: FormData, featureName: string) {
  if (!boolFromValue(formData.get('encryptedConfirmed'))) {
    throw new Error('Encrypted image upload requires confirmation.');
  }

  const image = formData.get('image');
  if (!(image instanceof Blob)) {
    throw new Error('Encrypted image upload requires an image file.');
  }
  if (image.size <= 0) {
    throw new Error('Encrypted image upload is empty.');
  }
  if (image.size > MAX_ENCRYPTED_UPLOAD_BYTES) {
    throw new Error('Encrypted image upload is too large.');
  }

  const mimeType = image.type || 'image/jpeg';
  if (!ALLOWED_UPLOAD_MIMES.has(mimeType)) {
    throw new Error(`${featureName} supports JPG, PNG, WebP, JPEG XL, GIF, and BMP images.`);
  }

  return dataUrlFromBytes(Buffer.from(await image.arrayBuffer()), mimeType);
}
