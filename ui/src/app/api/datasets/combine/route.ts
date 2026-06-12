import { NextRequest, NextResponse } from 'next/server';
import { getDatasetsRoot } from '@/server/settings';
import {
  combineDatasets,
  datasetCombineRequestHasKeyMaterial,
  isDatasetCombineError,
  type DatasetCombineRequest,
} from '@/server/datasetCombine';
import { getRemoteWorker, isLocalWorker, remoteJson, withoutRemoteRedirects } from '@/server/remoteClient';
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
      const hasKeyMaterial = datasetCombineRequestHasKeyMaterial(body);
      if (
        hasKeyMaterial &&
        !worker.base_url.toLowerCase().startsWith('https://') &&
        process.env.AITK_ALLOW_INSECURE_REMOTE_ENCRYPTED_DATASETS !== '1'
      ) {
        return NextResponse.json(
          { error: 'Remote encrypted dataset combine requires an HTTPS worker URL.' },
          { status: 400 },
        );
      }

      const remoteBody = { ...body, worker_id: 'local' };
      const remoteInit: RequestInit = {
        method: 'POST',
        body: JSON.stringify(remoteBody),
      };
      const remoteResult = await remoteJson<any>(
        worker,
        '/api/datasets/combine',
        hasKeyMaterial ? withoutRemoteRedirects(remoteInit) : remoteInit,
      );
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
