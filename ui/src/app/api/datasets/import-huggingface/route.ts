import { NextRequest, NextResponse } from 'next/server';
import {
  decorateRemoteHfDatasetImportResult,
  importHfDataset,
  normalizeHfDatasetImportRequest,
  previewHfDatasetImport,
  type HfDatasetImportRequest,
} from '@/server/hfDatasetImport';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { assertProjectScopeEnabled, DatasetScopeError, rejectRemoteProjectScope, resolveDatasetScope } from '@/server/datasetScope';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const normalized = normalizeHfDatasetImportRequest(body) as HfDatasetImportRequest;
    const workerID = normalized.worker_id || 'local';
    const projectID = (body as any)?.project_id;
    await assertProjectScopeEnabled(projectID);

    if (!isLocalWorker(workerID)) {
      rejectRemoteProjectScope(workerID, projectID);
      const worker = await getRemoteWorker(workerID);
      const remoteBody = { ...normalized, worker_id: 'local' };
      const remoteResult = await remoteJson<any>(worker, '/api/datasets/import-huggingface', {
        method: 'POST',
        body: JSON.stringify(remoteBody),
      });
      if (normalized.action === 'import' && remoteResult?.dataset) {
        return NextResponse.json(decorateRemoteHfDatasetImportResult(worker, remoteResult));
      }
      return NextResponse.json(remoteResult);
    }

    if (normalized.action === 'preview') {
      return NextResponse.json(await previewHfDatasetImport(normalized));
    }

    const { datasetsRoot } = await resolveDatasetScope(projectID);
    return NextResponse.json(await importHfDataset(datasetsRoot, normalized));
  } catch (error) {
    console.error('Hugging Face dataset import error:', error);
    if (error instanceof DatasetScopeError) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to import Hugging Face dataset' },
      { status: 400 },
    );
  }
}
