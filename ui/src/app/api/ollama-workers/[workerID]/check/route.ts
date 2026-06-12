import { NextResponse } from 'next/server';
import { checkRemoteOllamaWorker, toPublicRemoteOllamaWorker } from '@/server/remoteOllamaWorkers';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(_request: Request, { params }: { params: Promise<{ workerID: string }> }) {
  const { workerID } = await params;
  try {
    const { worker, status } = await checkRemoteOllamaWorker(workerID);
    return NextResponse.json({ worker: toPublicRemoteOllamaWorker(worker), status });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Remote Ollama health check failed' },
      { status: 502 },
    );
  }
}
