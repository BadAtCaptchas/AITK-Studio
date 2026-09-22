import fsp from 'fs/promises';
import os from 'os';
import path from 'path';
import { resolveDatasetScope } from '@/server/datasetScope';
import { resolveDatasetFolder, isEncryptedDatasetFolder } from '@/server/encryptedDatasets';
import { resolveDatasetDirectoryInsideRoot } from '@/server/remoteCaptionSecurity';
import { decodedUploadHeader, streamRequestToStagingFile } from '@/server/streamedUpload';
import { importLayeredDocument, MAX_LAYERED_UPLOAD_BYTES } from '@/server/layeredImport';
import { LayeredImageError } from '@/server/layeredImages';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  let stagingDirectory: string | undefined;
  try {
    if ((request.headers.get('content-type') || '').split(';')[0] !== 'application/octet-stream')
      throw new LayeredImageError('Stream a PSD or ORA file as application/octet-stream', 415);
    const datasetName = decodedUploadHeader(request, 'x-aitk-dataset-name', 256);
    const filename = decodedUploadHeader(request, 'x-aitk-file-name', 512);
    const workerID = decodedUploadHeader(request, 'x-aitk-worker-id', 256) || 'local';
    if (!datasetName || !filename) throw new LayeredImageError('Dataset and file names are required');
    if (!isLocalWorker(workerID)) {
      const headers = new Headers({
        'content-type': 'application/octet-stream',
        'x-aitk-dataset-name': encodeURIComponent(datasetName),
        'x-aitk-file-name': encodeURIComponent(filename),
      });
      const length = request.headers.get('content-length');
      if (length) headers.set('content-length', length);
      return Response.json(
        await remoteJson<unknown>(await getRemoteWorker(workerID), '/api/datasets/import-layered', {
          method: 'POST',
          headers,
          body: request.body,
          duplex: 'half',
          signal: request.signal,
          timeoutMs: 6 * 60 * 1000,
          headerTimeoutMs: 6 * 60 * 1000,
        } as RequestInit & { duplex: 'half'; timeoutMs: number; headerTimeoutMs: number }),
      );
    }
    const { datasetsRoot } = await resolveDatasetScope();
    const folder = await resolveDatasetDirectoryInsideRoot(
      resolveDatasetFolder(datasetsRoot, datasetName),
      datasetsRoot,
    );
    if (isEncryptedDatasetFolder(folder))
      throw new LayeredImageError('Import layered documents into an unencrypted dataset', 403);
    stagingDirectory = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-layer-import-'));
    const staged = await streamRequestToStagingFile(request, stagingDirectory, {
      maxBytes: MAX_LAYERED_UPLOAD_BYTES,
      prefix: 'layered',
    });
    return Response.json(await importLayeredDocument(staged.stagingPath, folder, filename, request.signal));
  } catch (error) {
    const status =
      error && typeof error === 'object' && 'status' in error && typeof error.status === 'number' ? error.status : 400;
    return Response.json({ error: error instanceof Error ? error.message : 'Layered import failed' }, { status });
  } finally {
    if (stagingDirectory) await fsp.rm(stagingDirectory, { recursive: true, force: true });
  }
}
