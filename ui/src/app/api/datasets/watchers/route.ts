import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { NextRequest, NextResponse } from 'next/server';
import {
  deleteDatasetWatcher,
  getDatasetWatcherStatuses,
  listDatasetWatchers,
  readWatcherSourceRootCaption,
  runDatasetWatcherOnce,
  saveDatasetWatcher,
} from '@/server/datasetWatchers';
import { DatasetScopeError } from '@/server/datasetScope';
import { isLocalWorker } from '@/server/remoteClient';

export const runtime = 'nodejs';

function rejectRemoteWorker(workerID: unknown) {
  const normalizedWorkerID = typeof workerID === 'string' && workerID.trim() ? workerID.trim() : 'local';

  if (!isLocalWorker(normalizedWorkerID)) {
    throw new Error('Dataset watch folders are only available on the local worker.');
  }
}

function errorResponse(error: unknown, fallback = 'Dataset watcher request failed') {
  const known = error as { status?: unknown; code?: unknown; details?: unknown };
  const status = typeof known?.status === 'number' ? known.status : 400;
  return NextResponse.json(
    {
      error: error instanceof Error ? error.message : fallback,
      ...(typeof known?.code === 'string' ? { code: known.code } : {}),
      ...(known?.details === undefined ? {} : { details: known.details }),
    },
    { status },
  );
}

export async function GET(request: NextRequest) {
  try {
    const params = request.nextUrl.searchParams;
    const datasetName = params.get('datasetName') || undefined;

    const workerID = params.get('worker_id') || 'local';
    rejectRemoteWorker(workerID);

    if (params.get('action') === 'root-caption') {
      const sourcePath = params.get('sourcePath') || '';
      if (!sourcePath.trim()) return NextResponse.json({ found: false, systemPrompt: '' });
      return NextResponse.json(await readWatcherSourceRootCaption(sourcePath));
    }

    const watchers = await listDatasetWatchers({ datasetName });
    return NextResponse.json({
      watchers,
      statuses: await getDatasetWatcherStatuses(watchers.map(watcher => watcher.id)),
    });
  } catch (error) {
    return errorResponse(error);
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = assertGlobalPayload(await request.json());

    rejectRemoteWorker(body?.worker_id);

    if (body?.action === 'run') {
      const id = typeof body?.id === 'string' ? body.id : '';
      const watcher = (await listDatasetWatchers({})).find(item => item.id === id);
      if (!watcher) return NextResponse.json({ error: 'Watcher not found' }, { status: 404 });
      return NextResponse.json({ result: await runDatasetWatcherOnce(watcher, { stableMs: 0 }) });
    }

    const watcher = await saveDatasetWatcher({
      ...body,
    });
    return NextResponse.json({
      watcher,
      statuses: await getDatasetWatcherStatuses([watcher.id]),
    });
  } catch (error) {
    return errorResponse(error);
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const body = assertGlobalPayload(await request.json());

    rejectRemoteWorker(body?.worker_id);
    const watcher = await saveDatasetWatcher({
      ...body,
    });
    return NextResponse.json({
      watcher,
      statuses: await getDatasetWatcherStatuses([watcher.id]),
    });
  } catch (error) {
    return errorResponse(error);
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const params = request.nextUrl.searchParams;

    let workerID: unknown = params.get('worker_id');
    rejectRemoteWorker(workerID);
    let id = params.get('id') || '';
    if (!id) {
      const body = await request.json().catch(() => null);
      id = typeof body?.id === 'string' ? body.id : '';

      workerID = body?.worker_id;
      rejectRemoteWorker(workerID);
    }
    const scopedWatcher = (await listDatasetWatchers({})).find(item => item.id === id);
    if (!scopedWatcher) return NextResponse.json({ error: 'Watcher not found' }, { status: 404 });
    const deleted = await deleteDatasetWatcher(id);
    if (!deleted) return NextResponse.json({ error: 'Watcher not found' }, { status: 404 });
    return NextResponse.json({ deleted });
  } catch (error) {
    return errorResponse(error);
  }
}
