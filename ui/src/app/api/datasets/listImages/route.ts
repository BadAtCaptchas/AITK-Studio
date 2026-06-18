import { NextResponse } from 'next/server';
import fs from 'fs';
import { isEncryptedDatasetFolder, readEncryptedManifest, resolveDatasetFolder } from '@/server/encryptedDatasets';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { makeSignedRemoteDatasetAssetRef } from '@/server/remoteDatasetAssetAccess';
import { findDatasetItemsRecursively } from '@/server/datasetImages';
import { assertProjectScopeEnabled, DatasetScopeError, rejectRemoteProjectScope, resolveDatasetScope } from '@/server/datasetScope';

export async function POST(request: Request) {
  const body = await request.json();
  const { datasetName } = body;
  const workerID = typeof body?.worker_id === 'string' ? body.worker_id : 'local';
  const projectID = body?.project_id;

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
    const data: any = await remoteJson(worker, '/api/datasets/listImages', {
      method: 'POST',
      body: JSON.stringify({ datasetName }),
    });
    if (Array.isArray(data?.images)) {
      data.images = data.images.map((image: any) => ({
        ...image,
        img_path:
          typeof image?.img_path === 'string'
            ? makeSignedRemoteDatasetAssetRef(workerID, 'img', image.img_path)
            : image?.img_path,
      }));
    }
    return NextResponse.json(data);
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
    if (!fs.existsSync(datasetFolder)) {
      return NextResponse.json({ error: `Folder '${datasetName}' not found` }, { status: 404 });
    }

    if (isEncryptedDatasetFolder(datasetFolder)) {
      return NextResponse.json({
        encrypted: true,
        manifest: await readEncryptedManifest(datasetFolder),
        images: [],
      });
    }

    const imageFiles = findDatasetItemsRecursively(datasetFolder);
    imageFiles.sort((a, b) => a.localeCompare(b));

    return NextResponse.json({
      images: imageFiles.map(imgPath => ({ img_path: imgPath })),
    });
  } catch (error) {
    console.error('Error finding images:', error);
    return NextResponse.json({ error: 'Failed to process request' }, { status: 500 });
  }
}
