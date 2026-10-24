import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import {
  listRemoteOllamaWorkers,
  saveRemoteOllamaWorker,
  toPublicRemoteOllamaWorker,
} from '@/server/remoteOllamaWorkers';


export async function GET() {
  const workers = await listRemoteOllamaWorkers();
  return Response.json({ workers: workers.map(toPublicRemoteOllamaWorker) });
}

async function postCommand(request: Request) {
  try {
    const body = assertGlobalPayload(await readJsonCommand(request));
    const worker = await saveRemoteOllamaWorker(body || {});
    return Response.json(toPublicRemoteOllamaWorker(worker));
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Failed to save Remote Ollama worker';
    return Response.json({ error: message }, { status: /already exists/i.test(message) ? 409 : 400 });
  }
}

export const POST = withCommandBoundary(postCommand);
