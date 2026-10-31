import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import fs from 'fs';
import zlib from 'zlib';
import { promisify } from 'util';
import { isEncryptedDatasetFolder, readEncryptedManifest, resolveDatasetFolder } from '@/server/encryptedDatasets';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { makeSignedRemoteDatasetAssetRef } from '@/server/remoteDatasetAssetAccess';
import { listDatasetImages } from '@/server/datasetListing';
import { resolveDatasetDirectoryInsideRoot } from '@/server/remoteCaptionSecurity';
import { DatasetScopeError, resolveDatasetScope } from '@/server/datasetScope';

const brotliCompress = promisify(zlib.brotliCompress);
const gzipCompress = promisify(zlib.gzip);

async function jsonResponse(request: Request, payload: unknown) {
  const json = JSON.stringify(payload);
  const acceptEncoding = request.headers.get('accept-encoding') ?? '';

  if (/\bbr\b/.test(acceptEncoding)) {
    const body = await brotliCompress(json);
    return new Response(new Uint8Array(body), {
      headers: { 'Content-Type': 'application/json', 'Content-Encoding': 'br' },
    });
  }

  if (/\bgzip\b/.test(acceptEncoding)) {
    const body = await gzipCompress(json);
    return new Response(new Uint8Array(body), {
      headers: { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip' },
    });
  }

  return new Response(json, { headers: { 'Content-Type': 'application/json' } });
}

async function postCommand(request: Request) {
  const body = assertGlobalPayload(await readJsonCommand(request));
  const { datasetName } = body;
  if (typeof datasetName !== 'string' || !datasetName) return Response.json({ error: 'datasetName is required' }, { status: 400 });
  const workerID = typeof body?.worker_id === 'string' ? body.worker_id : 'local';

  const compact = body?.compact === true;
  const limit = body.limit;
  const offset = body.offset ?? 0;
  if (limit !== undefined && (typeof limit !== 'number' || !Number.isSafeInteger(limit) || limit < 1 || limit > 1000) ||
      typeof offset !== 'number' || !Number.isSafeInteger(offset) || offset < 0 || offset > 10_000_000) {
    return Response.json({ error: 'Invalid listing pagination' }, { status: 400 });
  }

  try {
  } catch (error: any) {
    return Response.json({ error: error.message }, { status: error.status || 400 });
  }

  if (!isLocalWorker(workerID)) {
    try {
    } catch (error: any) {
      return Response.json({ error: error.message }, { status: error.status || 400 });
    }
    const worker = await getRemoteWorker(workerID);
    const data = await remoteJson<Record<string, unknown>>(worker, '/api/datasets/listImages', {
      method: 'POST',
      body: JSON.stringify({ datasetName, compact: true, ...(limit !== undefined ? { limit, offset } : {}) }),
    });
    const responseData: Record<string, unknown> = { ...data };
    const remoteRoot = typeof data?.root === 'string' ? data.root : '';
    if (Array.isArray(data?.images)) {
      responseData.images = data.images.map(image => {
        const imageRecord = image && typeof image === 'object' ? (image as Record<string, unknown>) : {};
        const rawPath =
          typeof imageRecord.img_path === 'string' ? `${remoteRoot}${imageRecord.img_path}` : imageRecord.img_path;
        return {
          ...imageRecord,
          img_path: typeof rawPath === 'string' ? makeSignedRemoteDatasetAssetRef(workerID, 'img', rawPath) : rawPath,
        };
      });
    }
    delete responseData.root;
    return jsonResponse(request, responseData);
  }

  let datasetFolder: string;
  let datasetsPath: string;
  try {
    const scope = await resolveDatasetScope();
    datasetsPath = scope.datasetsRoot;
    datasetFolder = resolveDatasetFolder(datasetsPath, datasetName);
  } catch (error: any) {
    if (error instanceof DatasetScopeError) {
      return Response.json({ error: error.message }, { status: error.status });
    }
    return Response.json({ error: error?.message || 'Invalid dataset name' }, { status: 400 });
  }

  try {
    try {
      await fs.promises.access(datasetFolder);
    } catch {
      return Response.json({ error: `Folder '${datasetName}' not found` }, { status: 404 });
    }

    const canonicalFolder = await resolveDatasetDirectoryInsideRoot(datasetFolder, datasetsPath);
    if (isEncryptedDatasetFolder(canonicalFolder)) {
      return jsonResponse(request, {
        encrypted: true,
        manifest: await readEncryptedManifest(canonicalFolder),
        images: [],
      });
    }

    return jsonResponse(request, await listDatasetImages(canonicalFolder, {
      responseRoot: datasetFolder, compact, ...(typeof limit === 'number' ? { limit, offset } : {}),
    }));
  } catch (error) {
    console.error('Error finding images:', error);
    return Response.json({ error: 'Failed to process request' }, { status: 500 });
  }
}

export const POST = withCommandBoundary(postCommand);
