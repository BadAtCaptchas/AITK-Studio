import fsp from 'fs/promises';
import { resolveDatasetScope } from '@/server/datasetScope';
import { resolveDatasetFolder, isEncryptedDatasetFolder } from '@/server/encryptedDatasets';
import { resolveDatasetDirectoryInsideRoot } from '@/server/remoteCaptionSecurity';
import {
  listLayeredImages,
  safeLayeredFile,
  layeredRevision,
  saveLayeredCaptions,
  LayeredImageError,
} from '@/server/layeredImages';
import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { makeSignedRemoteDatasetAssetRef } from '@/server/remoteDatasetAssetAccess';
import { getMediaUrl } from '@/utils/media';
import type { LayeredImageDocument } from '@/domain/layeredImages';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

async function postCommand(request: Request) {
  try {
    const body = await readJsonCommand(request);
    if (typeof body.datasetName !== 'string') throw new LayeredImageError('Dataset name is required');
    const workerID = typeof body.worker_id === 'string' ? body.worker_id : 'local';
    if (!isLocalWorker(workerID)) {
      const response: unknown = await remoteJson<unknown>(await getRemoteWorker(workerID), '/api/datasets/layered', {
        method: 'POST',
        body: JSON.stringify({ ...body, worker_id: 'local' }),
      });
      if (!response || typeof response !== 'object' || !('documents' in response) || !Array.isArray(response.documents))
        throw new LayeredImageError('Invalid remote layer response', 502);
      const documents = response.documents.map((document: unknown) => {
        if (
          !document ||
          typeof document !== 'object' ||
          !('compositePath' in document) ||
          typeof document.compositePath !== 'string' ||
          !('layerPaths' in document) ||
          !Array.isArray(document.layerPaths) ||
          document.layerPaths.some((value: unknown) => typeof value !== 'string')
        )
          throw new LayeredImageError('Invalid remote layer assets', 502);
        return {
          ...document,
          compositeUrl: getMediaUrl(makeSignedRemoteDatasetAssetRef(workerID, 'img', document.compositePath)),
          layerUrls: document.layerPaths.map((value: string) =>
            getMediaUrl(makeSignedRemoteDatasetAssetRef(workerID, 'img', value)),
          ),
        };
      });
      return Response.json({ documents });
    }
    const { datasetsRoot } = await resolveDatasetScope();
    const root = await resolveDatasetDirectoryInsideRoot(
      resolveDatasetFolder(datasetsRoot, body.datasetName),
      datasetsRoot,
    );
    if (isEncryptedDatasetFolder(root))
      throw new LayeredImageError('Layered groups are not supported in encrypted datasets', 403);
    if (body.action === 'save') {
      if (typeof body.id !== 'string' || !/^[A-Za-z0-9_-]{8,128}$/.test(body.id))
        throw new LayeredImageError('Invalid document ID');
      await saveLayeredCaptions(root, body.id, body.revision, body.caption, body.layers);
    } else if (body.action !== undefined && body.action !== 'list')
      throw new LayeredImageError('Unknown layered document action');
    const documents = await Promise.all(
      (await listLayeredImages(root)).map(async manifest => {
        const [compositePath, captionPath, manifestPath] = await Promise.all([
          safeLayeredFile(root, manifest.composite),
          safeLayeredFile(root, manifest.caption),
          safeLayeredFile(root, `.layers/${manifest.id}/manifest.json`),
        ]);
        const [captionText, manifestText] = await Promise.all([
          fsp.readFile(captionPath, 'utf8'),
          fsp.readFile(manifestPath, 'utf8'),
        ]);
        const layerPaths = await Promise.all(manifest.layers.map(layer => safeLayeredFile(root, layer.path)));
        return {
          manifest,
          captionText,
          revision: layeredRevision(manifestText, captionText),
          compositePath,
          layerPaths,
          compositeUrl: getMediaUrl(compositePath),
          layerUrls: layerPaths.map(value => getMediaUrl(value)),
        } satisfies LayeredImageDocument & { compositePath: string; layerPaths: string[] };
      }),
    );
    return Response.json({ documents });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : 'Layered document operation failed' },
      { status: error instanceof LayeredImageError ? error.status : 400 },
    );
  }
}
export const POST = withCommandBoundary(postCommand);
