import { NextResponse } from 'next/server';
import {
  listRemoteOllamaWorkerModels,
  toPublicRemoteOllamaWorker,
} from '@/server/remoteOllamaWorkers';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(_request: Request, { params }: { params: Promise<{ workerID: string }> }) {
  const { workerID } = await params;
  try {
    const { worker, status, models } = await listRemoteOllamaWorkerModels(workerID);
    return NextResponse.json(
      {
        ok: status.ok,
        worker: toPublicRemoteOllamaWorker(worker),
        status,
        models,
        error: status.error,
      },
      { status: status.ok ? 200 : 502 },
    );
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        models: [],
        error: error instanceof Error ? error.message : 'Failed to list Remote Ollama models',
      },
      { status: 502 },
    );
  }
}
