import { NextResponse } from 'next/server';
import { db } from '@/server/db';
import { listDatasetSummaries } from '@/server/encryptedDatasets';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import type { DatasetSummary } from '@/types';
import { makeRemoteDatasetRef } from '@/utils/remoteDatasetRefs';
import { resolveDatasetScope } from '@/server/datasetScope';
import { isLegacyScopedRecord } from '@/utils/obsoleteWorkspaceGuard';

function decorateRemoteDatasets(worker: { id: string; name: string }, datasets: DatasetSummary[]) {
  return datasets
    .filter(dataset => !isLegacyScopedRecord(dataset))
    .map(dataset => ({
      ...dataset,
      source: 'remote' as const,
      worker_id: worker.id,
      worker_name: worker.name,
      ref: makeRemoteDatasetRef(worker.id, dataset.name),
      path: undefined,
    }));
}

function decorateScopedDatasets(datasets: DatasetSummary[]) {
  return datasets
    .filter(dataset => !isLegacyScopedRecord(dataset))
    .map(dataset => ({
      ...dataset,
      source: dataset.source || ('local' as const),
      worker_id: dataset.worker_id || 'local',
      worker_name: dataset.worker_name || 'Local',
      ref: dataset.ref || `aitk-dataset://local/${encodeURIComponent(dataset.name)}`,
    }));
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const workerID = searchParams.get('worker_id') || 'local';
    const includeRemote = searchParams.get('include_remote') === '1';
    if (!isLocalWorker(workerID)) {
      const worker = await getRemoteWorker(workerID);
      const remoteDatasets = await remoteJson<DatasetSummary[]>(worker, '/api/datasets/list');
      return NextResponse.json({
        datasets: decorateRemoteDatasets(worker, Array.isArray(remoteDatasets) ? remoteDatasets : []),
        errors: [],
      });
    }

    const { datasetsRoot } = await resolveDatasetScope();
    const localDatasets = decorateScopedDatasets(await listDatasetSummaries(datasetsRoot, { createIfMissing: false }));

    const errors: Array<{ worker_id: string; worker_name: string; error: string }> = [];
    const remoteResults = includeRemote
      ? await Promise.all(
          (await db.workerNodes.list({ enabled: true })).map(async workerRecord => {
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
        )
      : [];

    if (!includeRemote) return NextResponse.json(localDatasets);

    return NextResponse.json({
      datasets: [...localDatasets, ...remoteResults.flat()],
      errors,
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        error: error?.message || 'Failed to fetch datasets',
        ...(typeof error?.code === 'string' ? { code: error.code } : {}),
        ...(error?.details === undefined ? {} : { details: error.details }),
      },
      { status: typeof error?.status === 'number' ? error.status : 500 },
    );
  }
}
