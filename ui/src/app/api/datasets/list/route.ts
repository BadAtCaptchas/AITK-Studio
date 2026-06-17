import { NextResponse } from 'next/server';
import { db } from '@/server/db';
import { listDatasetSummaries } from '@/server/encryptedDatasets';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import type { DatasetSummary } from '@/types';
import { makeRemoteDatasetRef } from '@/utils/remoteDatasetRefs';
import { rejectRemoteProjectScope, resolveDatasetScope } from '@/server/datasetScope';

function decorateRemoteDatasets(worker: { id: string; name: string }, datasets: DatasetSummary[]) {
  return datasets.map(dataset => ({
    ...dataset,
    source: 'remote' as const,
    worker_id: worker.id,
    worker_name: worker.name,
    ref: makeRemoteDatasetRef(worker.id, dataset.name),
    path: undefined,
  }));
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const workerID = searchParams.get('worker_id') || 'local';
    const includeRemote = searchParams.get('include_remote') === '1';
    const projectID = searchParams.get('project_id');
    rejectRemoteProjectScope(workerID, projectID);

    if (!isLocalWorker(workerID)) {
      const worker = await getRemoteWorker(workerID);
      const remoteDatasets = await remoteJson<DatasetSummary[]>(worker, '/api/datasets/list');
      return NextResponse.json({
        datasets: decorateRemoteDatasets(worker, Array.isArray(remoteDatasets) ? remoteDatasets : []),
        errors: [],
      });
    }

    const { datasetsRoot, project } = await resolveDatasetScope(projectID);
    const localDatasets = await listDatasetSummaries(datasetsRoot);
    if (!includeRemote || project) {
      return NextResponse.json(localDatasets);
    }

    const errors: Array<{ worker_id: string; worker_name: string; error: string }> = [];
    const workers = await db.workerNodes.list({ enabled: true });
    const remoteResults = await Promise.all(
      workers.map(async workerRecord => {
        try {
          const worker = await getRemoteWorker(workerRecord.id);
          const remoteDatasets = await remoteJson<DatasetSummary[]>(worker, '/api/datasets/list');
          return decorateRemoteDatasets(worker, Array.isArray(remoteDatasets) ? remoteDatasets : []);
        } catch (error) {
          errors.push({
            worker_id: workerRecord.id,
            worker_name: workerRecord.name,
            error: error instanceof Error ? error.message : 'Failed to fetch remote datasets',
          });
          return [];
        }
      }),
    );

    return NextResponse.json({
      datasets: [...localDatasets, ...remoteResults.flat()],
      errors,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error?.message || 'Failed to fetch datasets' },
      { status: typeof error?.status === 'number' ? error.status : 500 },
    );
  }
}
