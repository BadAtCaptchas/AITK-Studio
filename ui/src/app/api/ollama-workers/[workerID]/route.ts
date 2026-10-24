import { deleteRemoteOllamaWorker, toPublicRemoteOllamaWorker } from '@/server/remoteOllamaWorkers';


export async function DELETE(_request: Request, { params }: { params: Promise<{ workerID: string }> }) {
  const { workerID } = await params;
  try {
    const worker = await deleteRemoteOllamaWorker(workerID);
    if (!worker) {
      return Response.json({ error: 'Remote Ollama worker not found' }, { status: 404 });
    }
    return Response.json({ worker: toPublicRemoteOllamaWorker(worker) });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : 'Failed to delete Remote Ollama worker' },
      { status: 400 },
    );
  }
}
