import { getOllamaStatus, listOllamaModels } from '@/server/ollama';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';


export async function GET(request: Request) {
  const workerID = new URL(request.url).searchParams.get('worker_id') || 'local';

  try {
    if (!isLocalWorker(workerID)) {
      const worker = await getRemoteWorker(workerID);
      return Response.json(await remoteJson(worker, '/api/secure-caption/ollama/models'));
    }

    const models = await listOllamaModels();
    return Response.json({
      ok: true,
      status: await getOllamaStatus(),
      models,
    });
  } catch (error) {
    return Response.json(
      {
        ok: false,
        status: await getOllamaStatus().catch(() => null),
        models: [],
        error: error instanceof Error ? error.message : 'Failed to list Ollama models',
      },
      { status: 502 },
    );
  }
}
