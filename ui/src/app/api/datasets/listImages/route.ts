import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import { promisify } from 'util';
import { isEncryptedDatasetFolder, readEncryptedManifest, resolveDatasetFolder } from '@/server/encryptedDatasets';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { makeSignedRemoteDatasetAssetRef } from '@/server/remoteDatasetAssetAccess';
import { findDatasetItemsRecursivelyAsync } from '@/server/datasetImages';
import { findExistingCaptionSidecarAsync, isTextCaptionFilePath } from '@/server/captionFiles';
import { assertProjectScopeEnabled, DatasetScopeError, rejectRemoteProjectScope, resolveDatasetScope } from '@/server/datasetScope';

const brotliCompress = promisify(zlib.brotliCompress);
const gzipCompress = promisify(zlib.gzip);

type DatasetImageListEntry = {
  img_path: string;
  added_at: string | null;
  captioned_at: string | null;
  size_bytes: number;
};

function dateToIso(date: Date | undefined) {
  if (!date) return null;
  const ms = date.getTime();
  return typeof ms === 'number' && Number.isFinite(ms) && ms > 0 ? date.toISOString() : null;
}

function addedAtForStat(stat: fs.Stats) {
  return dateToIso(stat.birthtime) || dateToIso(stat.ctime) || dateToIso(stat.mtime);
}

async function captionedAtForItem(itemPath: string) {
  try {
    const captionPath = isTextCaptionFilePath(itemPath) ? itemPath : await findExistingCaptionSidecarAsync(itemPath);
    if (!captionPath) return null;
    const stat = await fs.promises.stat(captionPath);
    return stat.isFile() ? dateToIso(stat.mtime) : null;
  } catch {
    return null;
  }
}

async function imageEntry(imgPath: string, root: string | null): Promise<DatasetImageListEntry> {
  const stat = await fs.promises.stat(imgPath);
  return {
    img_path: root && imgPath.startsWith(root) ? imgPath.slice(root.length) : imgPath,
    added_at: addedAtForStat(stat),
    captioned_at: await captionedAtForItem(imgPath),
    size_bytes: stat.size,
  };
}

async function jsonResponse(request: Request, payload: unknown) {
  const json = JSON.stringify(payload);
  const acceptEncoding = request.headers.get('accept-encoding') ?? '';

  if (/\bbr\b/.test(acceptEncoding)) {
    const body = await brotliCompress(json);
    return new NextResponse(new Uint8Array(body), {
      headers: { 'Content-Type': 'application/json', 'Content-Encoding': 'br' },
    });
  }

  if (/\bgzip\b/.test(acceptEncoding)) {
    const body = await gzipCompress(json);
    return new NextResponse(new Uint8Array(body), {
      headers: { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip' },
    });
  }

  return new NextResponse(json, { headers: { 'Content-Type': 'application/json' } });
}

export async function POST(request: Request) {
  const body = await request.json();
  const { datasetName } = body;
  const workerID = typeof body?.worker_id === 'string' ? body.worker_id : 'local';
  const projectID = body?.project_id;
  const compact = body?.compact === true;

  try {
    await assertProjectScopeEnabled(projectID);
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: error.status || 400 });
  }

  if (!isLocalWorker(workerID)) {
    try {
      rejectRemoteProjectScope(workerID, projectID);
    } catch (error: any) {
      return NextResponse.json({ error: error.message }, { status: error.status || 400 });
    }
    const worker = await getRemoteWorker(workerID);
    const data = await remoteJson<Record<string, unknown>>(worker, '/api/datasets/listImages', {
      method: 'POST',
      body: JSON.stringify({ datasetName, compact: true }),
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
    const scope = await resolveDatasetScope(projectID);
    datasetsPath = scope.datasetsRoot;
    datasetFolder = resolveDatasetFolder(datasetsPath, datasetName);
  } catch (error: any) {
    if (error instanceof DatasetScopeError) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    return NextResponse.json({ error: error?.message || 'Invalid dataset name' }, { status: 400 });
  }

  try {
    try {
      await fs.promises.access(datasetFolder);
    } catch {
      return NextResponse.json({ error: `Folder '${datasetName}' not found` }, { status: 404 });
    }

    if (isEncryptedDatasetFolder(datasetFolder)) {
      return jsonResponse(request, {
        encrypted: true,
        manifest: await readEncryptedManifest(datasetFolder),
        images: [],
      });
    }

    const imageFiles = await findDatasetItemsRecursivelyAsync(datasetFolder);
    imageFiles.sort((a, b) => a.localeCompare(b));
    const root = datasetFolder + path.sep;
    const images = await Promise.all(imageFiles.map(imgPath => imageEntry(imgPath, compact ? root : null)));

    return jsonResponse(request, compact ? { root, images } : { images });
  } catch (error) {
    console.error('Error finding images:', error);
    return NextResponse.json({ error: 'Failed to process request' }, { status: 500 });
  }
}
