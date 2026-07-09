import { NextRequest, NextResponse } from 'next/server';
import {
  deleteDatasetWatcher,
  getDatasetWatcherStatuses,
  listDatasetWatchers,
  readWatcherSourceRootCaption,
  runDatasetWatcherOnce,
  saveDatasetWatcher,
} from '@/server/datasetWatchers';
import { DatasetScopeError, rejectRemoteProjectScope } from '@/server/datasetScope';
import { isLocalWorker } from '@/server/remoteClient';

export const runtime = 'nodejs';

function projectIDFromValue(value: unknown, supplied: boolean) {
  if (!supplied || value === null || value === undefined) return null;
  if (typeof value !== 'string' || !value.trim()) {
    throw new DatasetScopeError('project_id must be a project UUID or slug', 400, 'PROJECT_INVALID_INPUT');
  }
  return value.trim();
}

function projectIDFromBody(body: unknown) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return null;
  const record = body as Record<string, unknown>;
  const hasProjectID = Object.prototype.hasOwnProperty.call(record, 'projectID');
  const hasProjectIDSnakeCase = Object.prototype.hasOwnProperty.call(record, 'project_id');
  if (hasProjectID && hasProjectIDSnakeCase && record.projectID !== record.project_id) {
    throw new DatasetScopeError('projectID and project_id must identify the same project', 400, 'PROJECT_INVALID_INPUT');
  }
  const supplied = hasProjectID || hasProjectIDSnakeCase;
  return projectIDFromValue(hasProjectID ? record.projectID : record.project_id, supplied);
}

function rejectRemoteWorker(workerID: unknown, projectID: unknown) {
  const normalizedWorkerID = typeof workerID === 'string' && workerID.trim() ? workerID.trim() : 'local';
  rejectRemoteProjectScope(normalizedWorkerID, projectID);
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
    const projectID = projectIDFromValue(params.get('project_id'), params.has('project_id'));
    const workerID = params.get('worker_id') || 'local';
    rejectRemoteWorker(workerID, projectID);

    if (params.get('action') === 'root-caption') {
      const sourcePath = params.get('sourcePath') || '';
      if (!sourcePath.trim()) return NextResponse.json({ found: false, systemPrompt: '' });
      return NextResponse.json(await readWatcherSourceRootCaption(sourcePath, projectID));
    }

    const watchers = await listDatasetWatchers({ datasetName, projectID }, { intent: 'read' });
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
    const body = await request.json();
    const projectID = projectIDFromBody(body);
    rejectRemoteWorker(body?.worker_id, projectID);

    if (body?.action === 'run') {
      const id = typeof body?.id === 'string' ? body.id : '';
      const watcher = (await listDatasetWatchers({ projectID })).find(item => item.id === id);
      if (!watcher) return NextResponse.json({ error: 'Watcher not found' }, { status: 404 });
      return NextResponse.json({ result: await runDatasetWatcherOnce(watcher, { stableMs: 0 }) });
    }

    const watcher = await saveDatasetWatcher({
      ...body,
      projectID,
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
    const body = await request.json();
    const projectID = projectIDFromBody(body);
    rejectRemoteWorker(body?.worker_id, projectID);
    const watcher = await saveDatasetWatcher({
      ...body,
      projectID,
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
    let projectID = projectIDFromValue(params.get('project_id'), params.has('project_id'));
    let workerID: unknown = params.get('worker_id');
    rejectRemoteWorker(workerID, projectID);
    let id = params.get('id') || '';
    if (!id) {
      const body = await request.json().catch(() => null);
      id = typeof body?.id === 'string' ? body.id : '';
      projectID = projectIDFromBody(body);
      workerID = body?.worker_id;
      rejectRemoteWorker(workerID, projectID);
    }
    const scopedWatcher = (await listDatasetWatchers({ projectID })).find(item => item.id === id);
    if (!scopedWatcher) return NextResponse.json({ error: 'Watcher not found' }, { status: 404 });
    const deleted = await deleteDatasetWatcher(id);
    if (!deleted) return NextResponse.json({ error: 'Watcher not found' }, { status: 404 });
    return NextResponse.json({ deleted });
  } catch (error) {
    return errorResponse(error);
  }
}
