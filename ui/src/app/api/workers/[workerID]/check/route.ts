import { NextResponse } from 'next/server';
import { db, type WorkerNodeRecord } from '@/server/db';
import { fetchWorkerGpu, fetchWorkerHealth, getRemoteWorker } from '@/server/remoteClient';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function toPublicWorker(worker: WorkerNodeRecord) {
  const { api_token: _apiToken, ...publicWorker } = worker;
  return publicWorker;
}

export async function POST(_request: Request, { params }: { params: Promise<{ workerID: string }> }) {
  const { workerID } = await params;

  try {
    const worker = await getRemoteWorker(workerID);
    const [health, gpu] = await Promise.all([fetchWorkerHealth(worker), fetchWorkerGpu(worker).catch(error => ({ error }))]);
    const gpus = 'gpus' in gpu ? gpu.gpus : [];
    const ollama = health.ollama as { ok?: boolean; error?: string | null; modelCount?: number } | undefined;
    const ollamaError = ollama && ollama.ok === false ? `Ollama: ${ollama.error || 'unavailable'}` : null;
    const updated = await db.workerNodes.update(workerID, {
      last_status: health.ok ? 'online' : 'error',
      last_error: health.ok ? ollamaError : 'Worker health check failed',
      last_checked_at: new Date(),
      capabilities: JSON.stringify({ health, hasGpuApi: 'gpus' in gpu }),
      gpus: JSON.stringify(gpus),
    });
    return NextResponse.json({ worker: toPublicWorker(updated), health, gpu });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Worker health check failed';
    const existing = await db.workerNodes.findById(workerID);
    if (existing) {
      const updated = await db.workerNodes.update(workerID, {
        last_status: 'error',
        last_error: message,
        last_checked_at: new Date(),
      });
      return NextResponse.json({ worker: toPublicWorker(updated), error: message }, { status: 502 });
    }
    return NextResponse.json({ error: message }, { status: 404 });
  }
}
