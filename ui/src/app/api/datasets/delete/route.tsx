import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { NextResponse } from 'next/server';
import { deleteDatasetFolder, DatasetDeleteError } from '@/server/datasetDelete';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { resolveDatasetScope } from '@/server/datasetScope';

export async function POST(request: Request) {
  try {
    const body = assertGlobalPayload(await request.json());
    const { name } = body;
    const workerID = typeof body?.worker_id === 'string' ? body.worker_id : 'local';

    if (!isLocalWorker(workerID)) {
      const worker = await getRemoteWorker(workerID);
      return NextResponse.json(
        await remoteJson(worker, '/api/datasets/delete', {
          method: 'POST',
          body: JSON.stringify({ name }),
        }),
      );
    }

    const { datasetsRoot: datasetsPath } = await resolveDatasetScope();
    const result = await deleteDatasetFolder(datasetsPath, name);
    return NextResponse.json({ success: result.success });
  } catch (error: any) {
    const status =
      error instanceof DatasetDeleteError ? error.status : typeof error?.status === 'number' ? error.status : 500;
    return NextResponse.json({ error: error?.message || 'Failed to delete dataset' }, { status });
  }
}
