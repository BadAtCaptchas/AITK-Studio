import { DATASET_MIGRATION_MESSAGE, DATASET_MIGRATION_REQUIRED } from '@/server/datasetDecryption';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import fs from 'fs';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { cleanDatasetName, resolveDatasetFolder } from '@/server/encryptedDatasets';
import { resolveDatasetScope } from '@/server/datasetScope';

export async function POST(request: Request) {
  try {
    const body = assertGlobalPayload(await request.json());
    if (body.encrypted || body.encryptedManifest)
      return Response.json({ error: DATASET_MIGRATION_MESSAGE, code: DATASET_MIGRATION_REQUIRED }, { status: 410 });
    const workerID = typeof body?.worker_id === 'string' ? body.worker_id : 'local';

    if (!isLocalWorker(workerID)) {
      const worker = await getRemoteWorker(workerID);
      const { worker_id, ...remoteBody } = body;
      const remoteResult = await remoteJson<any>(worker, '/api/datasets/create', {
        method: 'POST',
        body: JSON.stringify(remoteBody),
      });
      return Response.json({
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
      return Response.json({ error: 'Dataset name is required' }, { status: 400 });
    }

    const datasetPath = resolveDatasetFolder(datasetsRoot, name);

    // if folder doesnt exist, create it
    if (!fs.existsSync(datasetPath)) {
      fs.mkdirSync(datasetPath);
    } else if (fs.readdirSync(datasetPath).length > 0) {
      return Response.json({ error: 'Dataset already exists' }, { status: 409 });
    }

    return Response.json({ success: true, name: name, encrypted: !!encrypted });
  } catch (error: any) {
    return Response.json(
      { error: error?.message || 'Failed to create dataset' },
      { status: typeof error?.status === 'number' ? error.status : 500 },
    );
  }
}
