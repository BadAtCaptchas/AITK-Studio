import { NextRequest, NextResponse } from 'next/server';
import {
  decodeRemoteCaptionBulkPaths,
  isDatasetCaptionBulkError,
  mapRemoteCaptionBulkResult,
  performPlainDatasetCaptionBulkAction,
  type DatasetCaptionBulkRequest,
  type DatasetCaptionBulkResult,
} from '@/server/datasetCaptionBulk';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { assertProjectScopeEnabled, DatasetScopeError, rejectRemoteProjectScope, resolveDatasetScope } from '@/server/datasetScope';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as DatasetCaptionBulkRequest & { worker_id?: string; project_id?: string };
    const workerID = typeof body?.worker_id === 'string' ? body.worker_id : 'local';
    await assertProjectScopeEnabled(body.project_id);

    if (!isLocalWorker(workerID)) {
      rejectRemoteProjectScope(workerID, body.project_id);
      const worker = await getRemoteWorker(workerID);
      const imgPaths = Array.isArray(body.imgPaths) ? body.imgPaths : [];
      const decoded = decodeRemoteCaptionBulkPaths(imgPaths, workerID);
      const remoteResult = await remoteJson<DatasetCaptionBulkResult>(worker, '/api/datasets/caption-bulk', {
        method: 'POST',
        body: JSON.stringify({
          ...body,
          worker_id: 'local',
          imgPaths: decoded.remotePaths,
        }),
      });
      return NextResponse.json(mapRemoteCaptionBulkResult(remoteResult, decoded.refByRemotePath));
    }

    const { datasetsRoot } = await resolveDatasetScope(body.project_id);
    const result = await performPlainDatasetCaptionBulkAction(datasetsRoot, body);
    return NextResponse.json(result);
  } catch (error) {
    if (isDatasetCaptionBulkError(error)) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    if (error instanceof DatasetScopeError) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to run bulk caption action' },
      { status: 500 },
    );
  }
}
