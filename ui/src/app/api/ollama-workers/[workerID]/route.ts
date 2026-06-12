import { NextResponse } from 'next/server';
import { deleteRemoteOllamaWorker, toPublicRemoteOllamaWorker } from '@/server/remoteOllamaWorkers';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function DELETE(_request: Request, { params }: { params: Promise<{ workerID: string }> }) {
  const { workerID } = await params;
  try {
    const worker = await deleteRemoteOllamaWorker(workerID);
    if (!worker) {
      return NextResponse.json({ error: 'Remote Ollama worker not found' }, { status: 404 });
    }
    return NextResponse.json({ worker: toPublicRemoteOllamaWorker(worker) });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to delete Remote Ollama worker' },
      { status: 400 },
    );
  }
}
