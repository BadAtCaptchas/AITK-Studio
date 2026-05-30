import { NextRequest, NextResponse } from 'next/server';
import { getDatasetsRoot } from '@/server/settings';
import { combineDatasets, isDatasetCombineError, type DatasetCombineRequest } from '@/server/datasetCombine';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import type { DatasetSummary } from '@/types';
import { makeRemoteDatasetRef } from '@/utils/remoteDatasetRefs';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function decorateRemoteDataset(worker: { id: string; name: string }, dataset: DatasetSummary): DatasetSummary {
  return {
    ...dataset,
    source: 'remote',
    worker_id: worker.id,
    worker_name: worker.name,
    ref: makeRemoteDatasetRef(worker.id, dataset.name),
    path: undefined,
  };
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as DatasetCombineRequest & { worker_id?: string };
    const workerID = typeof body?.worker_id === 'string' ? body.worker_id : 'local';

    if (!isLocalWorker(workerID)) {
      const worker = await getRemoteWorker(workerID);
      const remoteBody = { ...body, worker_id: 'local' };
      const remoteResult = await remoteJson<any>(worker, '/api/datasets/combine', {
        method: 'POST',
        body: JSON.stringify(remoteBody),
      });
      return NextResponse.json({
        ...remoteResult,
        dataset: remoteResult?.dataset ? decorateRemoteDataset(worker, remoteResult.dataset) : remoteResult?.dataset,
      });
    }

    const datasetsRoot = await getDatasetsRoot();
    const result = await combineDatasets(datasetsRoot, body);
    return NextResponse.json(result);
  } catch (error) {
    console.error('Dataset combine error:', error);
    if (isDatasetCombineError(error)) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to combine datasets' },
      { status: 500 },
    );
  }
}
