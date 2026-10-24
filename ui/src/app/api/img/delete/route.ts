import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';
import { findEncryptedDatasetRoot } from '@/server/encryptedDatasets';
import { getRemoteWorker, remoteJson } from '@/server/remoteClient';
import { deleteCaptionSidecarsAsync } from '@/server/captionFiles';
import { parseRemoteDatasetAssetRef } from '@/utils/remoteDatasetRefs';
import { isRequestAuthenticated } from '@/utils/authSession';
import { deleteLayeredImageForPath } from '@/server/layeredImages';
import { deleteSampleLayers } from '@/server/sampleLayers';
import { deleteSampleMetadata } from '@/server/sampleMetadata';
import { isLayeredImageAssetPath } from '@/domain/layeredImages';

async function postCommand(request: Request) {
  try {
    const tokenToUse = process.env.AI_TOOLKIT_AUTH || null;
    if (!(await isRequestAuthenticated(request, tokenToUse))) {
      return Response.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = assertGlobalPayload(await readJsonCommand(request));
    const { imgPath } = body;
    if (typeof imgPath !== 'string') {
      return Response.json({ error: 'Invalid image path' }, { status: 400 });
    }

    const remoteAsset = parseRemoteDatasetAssetRef(imgPath);
    if (remoteAsset) {
      const worker = await getRemoteWorker(remoteAsset.workerID);
      return Response.json(
        await remoteJson(worker, '/api/img/delete', {
          method: 'POST',
          body: JSON.stringify({ imgPath: remoteAsset.path }),
        }),
      );
    }

    const datasetsPath = await getDatasetsRoot();
    const trainingPath = await getTrainingFolder();
    const normalizedImgPath = path.resolve(imgPath);
    if (isLayeredImageAssetPath(normalizedImgPath)) return Response.json({ error: 'Delete the composite to remove its layers' }, { status: 400 });
    const allowedRoots = [datasetsPath, trainingPath].map(root => path.resolve(root));
    const isWithinAllowedRoot = allowedRoots.some(root => {
      const rel = path.relative(root, normalizedImgPath);
      return rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel);
    });

    // make sure the dataset path is in the image path
    if (!isWithinAllowedRoot) {
      return Response.json({ error: 'Invalid image path' }, { status: 400 });
    }

    if (findEncryptedDatasetRoot(normalizedImgPath, datasetsPath)) {
      return Response.json(
        { error: 'Encrypted dataset objects must be deleted through the encrypted dataset API' },
        { status: 403 },
      );
    }

    // make sure it is an image
    if (!/\.(jpg|jpeg|png|bmp|gif|tiff|webp|jxl|mp4|mp3|wav|flac|ogg)$/i.test(normalizedImgPath.toLowerCase())) {
      return Response.json({ error: 'Not an image' }, { status: 400 });
    }

    // if img doesnt exist, ignore
    try {
      await fs.promises.access(normalizedImgPath);
    } catch {
      return Response.json({ success: true });
    }

    // delete it and return success
    if (!await deleteLayeredImageForPath(normalizedImgPath, datasetsPath)) {
      await deleteSampleLayers(normalizedImgPath);
      await deleteSampleMetadata(normalizedImgPath);
      await fs.promises.unlink(normalizedImgPath);
    }

    await deleteCaptionSidecarsAsync(normalizedImgPath);

    return Response.json({ success: true });
  } catch (error) {
    return Response.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}

export const POST = withCommandBoundary(postCommand);
