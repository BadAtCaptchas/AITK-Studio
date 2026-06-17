import { NextRequest, NextResponse } from 'next/server';
import fsp from 'fs/promises';
import path from 'path';
import {
  isEncryptedDatasetFolder,
  resolveDatasetFolder,
  resolveEncryptedObjectPath,
  safeEncryptedObjectPath,
  validateEncryptedManifest,
  writeEncryptedManifest,
} from '@/server/encryptedDatasets';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { rejectRemoteProjectScope, resolveDatasetScope } from '@/server/datasetScope';

type EncryptedObjectUpdate = {
  objectPath: string;
  dataBase64: string;
};

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { datasetName, manifest, objects, deleteObjects, worker_id, project_id } = body;
    if (typeof datasetName !== 'string') {
      return NextResponse.json({ error: 'Dataset name is required' }, { status: 400 });
    }

    if (!isLocalWorker(worker_id)) {
      rejectRemoteProjectScope(worker_id, project_id);
      const worker = await getRemoteWorker(worker_id);
      return NextResponse.json(
        await remoteJson(worker, '/api/datasets/encrypted/update', {
          method: 'POST',
          body: JSON.stringify({ datasetName, manifest, objects, deleteObjects }),
        }),
      );
    }

    const { datasetsRoot } = await resolveDatasetScope(project_id);
    const datasetFolder = resolveDatasetFolder(datasetsRoot, datasetName);
    if (!isEncryptedDatasetFolder(datasetFolder)) {
      return NextResponse.json({ error: 'Encrypted dataset not found' }, { status: 404 });
    }

    const nextManifest = validateEncryptedManifest(manifest);
    await fsp.mkdir(path.join(datasetFolder, 'objects'), { recursive: true });

    if (Array.isArray(objects)) {
      for (const object of objects as EncryptedObjectUpdate[]) {
        if (typeof object?.objectPath !== 'string' || typeof object?.dataBase64 !== 'string') {
          return NextResponse.json({ error: 'Invalid encrypted object update' }, { status: 400 });
        }
        const resolvedObjectPath = resolveEncryptedObjectPath(datasetFolder, object.objectPath);
        await fsp.writeFile(resolvedObjectPath, Buffer.from(object.dataBase64, 'base64'));
      }
    }

    if (Array.isArray(deleteObjects)) {
      for (const objectPath of deleteObjects) {
        if (typeof objectPath !== 'string') continue;
        safeEncryptedObjectPath(objectPath);
        const resolvedObjectPath = resolveEncryptedObjectPath(datasetFolder, objectPath);
        await fsp.rm(resolvedObjectPath, { force: true });
      }
    }

    await writeEncryptedManifest(datasetFolder, nextManifest);
    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Failed to update encrypted dataset' }, { status: 400 });
  }
}
