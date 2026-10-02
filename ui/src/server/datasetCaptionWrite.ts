import fsp from 'fs/promises';
import path from 'path';
import { isLayeredImageAssetPath } from '../domain/layeredImages';
import { resolveCaptionWritePathAsync } from './captionFiles';
import { DatasetMutationPathError, resolveDatasetMutationPath } from './datasetMutationPaths';
import { findEncryptedDatasetRoot } from './encryptedDatasets';

export class DatasetCaptionWriteError extends Error {
  constructor(message: string, readonly status = 400) {
    super(message);
    this.name = 'DatasetCaptionWriteError';
  }
}

function assertPlainCaptionPath(target: string, root: string): void {
  if (isLayeredImageAssetPath(path.relative(root, target))) {
    throw new DatasetCaptionWriteError('Edit layer captions in Layered documents');
  }
  if (findEncryptedDatasetRoot(target, root)) {
    throw new DatasetCaptionWriteError('Encrypted captions must be saved through the encrypted dataset API', 403);
  }
}

export async function writePlainDatasetCaption(
  datasetsRoot: string,
  imgPath: string,
  caption: string,
): Promise<{ success: true; captioned_at: string }> {
  try {
    const resolvedImagePath = path.resolve(imgPath);
    assertPlainCaptionPath(resolvedImagePath, path.resolve(datasetsRoot));
    const image = await resolveDatasetMutationPath(datasetsRoot, resolvedImagePath);
    assertPlainCaptionPath(image.canonicalPath, image.canonicalRoot);
    if (!(await fsp.stat(image.canonicalPath)).isFile()) {
      throw new DatasetCaptionWriteError('Image does not exist', 404);
    }
    const requestedCaption = await resolveCaptionWritePathAsync(image.entryPath, caption);
    const target = await resolveDatasetMutationPath(image.canonicalRoot, requestedCaption, { allowMissingLeaf: true });
    assertPlainCaptionPath(target.entryPath, target.canonicalRoot);
    assertPlainCaptionPath(target.canonicalPath, target.canonicalRoot);
    await fsp.writeFile(target.canonicalPath, caption);
    return { success: true, captioned_at: (await fsp.stat(target.canonicalPath)).mtime.toISOString() };
  } catch (error) {
    if (error instanceof DatasetMutationPathError) throw new DatasetCaptionWriteError('Invalid image or caption path');
    if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') {
      throw new DatasetCaptionWriteError('Image does not exist', 404);
    }
    throw error;
  }
}
