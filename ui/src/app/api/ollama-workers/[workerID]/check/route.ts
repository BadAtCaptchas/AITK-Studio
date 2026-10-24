import { checkRemoteOllamaWorker, toPublicRemoteOllamaWorker } from '@/server/remoteOllamaWorkers';


export async function POST(_request: Request, { params }: { params: Promise<{ workerID: string }> }) {
  const { workerID } = await params;
  try {
    const { worker, status } = await checkRemoteOllamaWorker(workerID);
    return Response.json({ worker: toPublicRemoteOllamaWorker(worker), status });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : 'Remote Ollama health check failed' },
      { status: 502 },
    );
  }
}
