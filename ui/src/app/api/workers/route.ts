import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { db, type WorkerNodeRecord } from '@/server/db';
import { normalizeWorkerBaseUrl } from '@/server/remoteClient';


function toPublicWorker(worker: WorkerNodeRecord) {
  const { api_token: _apiToken, ...publicWorker } = worker;
  return publicWorker;
}

function asString(value: unknown) {
  return typeof value === 'string' ? value.trim() : '';
}

export async function GET() {
  const workers = await db.workerNodes.list();
  return Response.json({ workers: workers.map(toPublicWorker) });
}

async function postCommand(request: Request) {
  try {
    const body = assertGlobalPayload(await readJsonCommand(request));
    const id = asString(body.id);
    const name = asString(body.name);
    const baseUrl = normalizeWorkerBaseUrl(asString(body.base_url));
    const apiToken = asString(body.api_token);
    const enabled = body.enabled !== false;
    const offlineBypassEnabled = body.offline_bypass_enabled === true;

    if (!name) return Response.json({ error: 'Worker name is required' }, { status: 400 });

    if (id) {
      const patch: any = { name, base_url: baseUrl, enabled, offline_bypass_enabled: offlineBypassEnabled };
      if (apiToken) patch.api_token = apiToken;
      const worker = await db.workerNodes.update(id, patch);
      return Response.json(toPublicWorker(worker));
    }

    if (!apiToken) return Response.json({ error: 'API token is required for new workers' }, { status: 400 });
    const worker = await db.workerNodes.create({
      name,
      base_url: baseUrl,
      api_token: apiToken,
      enabled,
      offline_bypass_enabled: offlineBypassEnabled,
    });
    return Response.json(toPublicWorker(worker));
  } catch (error: any) {
    if (error.code === 'P2002') {
      return Response.json({ error: 'Worker name already exists' }, { status: 409 });
    }
    return Response.json(
      { error: error instanceof Error ? error.message : 'Failed to save worker' },
      { status: 500 },
    );
  }
}

export const POST = withCommandBoundary(postCommand);
