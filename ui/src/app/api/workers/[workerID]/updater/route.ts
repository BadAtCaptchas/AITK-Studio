import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { db, type WorkerNodeRecord } from '@/server/db';
import { getRemoteWorker, RemoteClientError, remoteJson } from '@/server/remoteClient';
import type { RepoUpdateRequestAction, RepoUpdateStatus } from '@/server/updater';


function toPublicWorker(worker: WorkerNodeRecord) {
  const { api_token: _apiToken, ...publicWorker } = worker;
  return publicWorker;
}

function parseAction(value: unknown): RepoUpdateRequestAction {
  return value === 'apply' || value === 'restart' ? value : 'check';
}

function remoteUpdaterError(error: unknown) {
  if (error instanceof RemoteClientError) {
    const unavailableMessage =
      error.status === 404
        ? 'Remote worker does not expose the updater API yet. Update that worker manually once, then remote updates will be available.'
        : error.message;

    return Response.json(
      {
        error: unavailableMessage,
        remoteStatus: error.status,
        remoteBody: error.body,
      },
      { status: 502 },
    );
  }

  return Response.json(
    { error: error instanceof Error ? error.message : 'Failed to contact remote worker updater' },
    { status: 500 },
  );
}

export async function GET(_request: Request, { params }: { params: Promise<{ workerID: string }> }) {
  const { workerID } = await params;
  const existing = await db.workerNodes.findById(workerID);
  if (!existing) return Response.json({ error: 'Worker not found' }, { status: 404 });

  try {
    const worker = await getRemoteWorker(workerID);
    const status = await remoteJson<RepoUpdateStatus>(worker, '/api/updater');
    return Response.json({ worker: toPublicWorker(worker), status });
  } catch (error) {
    return remoteUpdaterError(error);
  }
}

async function postCommand(request: Request, { params }: { params: Promise<{ workerID: string }> }) {
  const { workerID } = await params;
  const existing = await db.workerNodes.findById(workerID);
  if (!existing) return Response.json({ error: 'Worker not found' }, { status: 404 });

  try {
    const body = await readJsonCommand(request);
    const action = parseAction(body?.action);
    const worker = await getRemoteWorker(workerID);
    const result = await remoteJson(worker, '/api/updater', {
      method: 'POST',
      body: JSON.stringify({ action }),
    });
    return Response.json({ worker: toPublicWorker(worker), result });
  } catch (error) {
    return remoteUpdaterError(error);
  }
}

export const POST = withCommandBoundary(postCommand);
