import { NextResponse } from 'next/server';
import { getDatasetsRoot } from '@/server/settings';
import { renameDatasetFolder, DatasetRenameError } from '@/server/datasetRename';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { renameSecureCaptionSystemPrompt } from '@/server/secureCaptionSettings';
import { makeRemoteDatasetRef } from '@/utils/remoteDatasetRefs';

export const runtime = 'nodejs';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const workerID = typeof body?.worker_id === 'string' ? body.worker_id : 'local';
    const oldName = body?.oldName ?? body?.name;
    const newName = body?.newName;

    if (!isLocalWorker(workerID)) {
      const worker = await getRemoteWorker(workerID);
      const remoteResult = await remoteJson<any>(worker, '/api/datasets/rename', {
        method: 'POST',
        body: JSON.stringify({ name: oldName, newName }),
      });
      const { path: _remotePath, dataset, ...safeRemoteResult } = remoteResult || {};
      return NextResponse.json({
        ...safeRemoteResult,
        worker_id: worker.id,
        worker_name: worker.name,
        dataset: dataset
          ? {
              ...dataset,
              source: 'remote',
              worker_id: worker.id,
              worker_name: worker.name,
              ref: makeRemoteDatasetRef(worker.id, dataset.name),
              path: undefined,
            }
          : dataset,
      });
    }

    const datasetsRoot = await getDatasetsRoot();
    const result = await renameDatasetFolder(datasetsRoot, oldName, newName);
    await renameSecureCaptionSystemPrompt(result.oldName, result.name);
    return NextResponse.json(result);
  } catch (error: any) {
    const status =
      error instanceof DatasetRenameError
        ? error.status
        : typeof error?.status === 'number'
          ? error.status
          : 500;
    return NextResponse.json(
      { error: error?.message || 'Failed to rename dataset' },
      { status },
    );
  }
}
