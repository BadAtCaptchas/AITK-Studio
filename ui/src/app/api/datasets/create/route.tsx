import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import {
  cleanDatasetName,
  resolveDatasetFolder,
  validateEncryptedManifest,
  writeEncryptedManifest,
} from '@/server/encryptedDatasets';
import { resolveDatasetScope } from '@/server/datasetScope';

export async function POST(request: Request) {
  try {
    const body = assertGlobalPayload(await request.json());
    const workerID = typeof body?.worker_id === 'string' ? body.worker_id : 'local';

    if (!isLocalWorker(workerID)) {
      const worker = await getRemoteWorker(workerID);
      const { worker_id, ...remoteBody } = body;
      const remoteResult = await remoteJson<any>(worker, '/api/datasets/create', {
        method: 'POST',
        body: JSON.stringify(remoteBody),
      });
      return NextResponse.json({
        ...remoteResult,
        worker_id: worker.id,
        worker_name: worker.name,
      });
    }

    let { name } = body;
    const { encrypted, encryptedManifest } = body;
    const { datasetsRoot } = await resolveDatasetScope();
    name = cleanDatasetName(name || '');
    if (!name) {
      return NextResponse.json({ error: 'Dataset name is required' }, { status: 400 });
    }

    const datasetPath = resolveDatasetFolder(datasetsRoot, name);

    // if folder doesnt exist, create it
    if (!fs.existsSync(datasetPath)) {
      fs.mkdirSync(datasetPath);
    } else if (fs.readdirSync(datasetPath).length > 0) {
      return NextResponse.json({ error: 'Dataset already exists' }, { status: 409 });
    }

    if (encrypted) {
      const manifest = validateEncryptedManifest(encryptedManifest);
      fs.mkdirSync(path.join(datasetPath, 'objects'), { recursive: true });
      await writeEncryptedManifest(datasetPath, manifest);
    }

    return NextResponse.json({ success: true, name: name, encrypted: !!encrypted });
  } catch (error: any) {
    return NextResponse.json(
      { error: error?.message || 'Failed to create dataset' },
      { status: typeof error?.status === 'number' ? error.status : 500 },
    );
  }
}
