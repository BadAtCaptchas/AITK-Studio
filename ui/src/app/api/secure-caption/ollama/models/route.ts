import { NextRequest, NextResponse } from 'next/server';
import { getOllamaStatus, listOllamaModels } from '@/server/ollama';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  const workerID = request.nextUrl.searchParams.get('worker_id') || 'local';

  try {
    if (!isLocalWorker(workerID)) {
      const worker = await getRemoteWorker(workerID);
      return NextResponse.json(await remoteJson(worker, '/api/secure-caption/ollama/models'));
    }

    const models = await listOllamaModels();
    return NextResponse.json({
      ok: true,
      status: await getOllamaStatus(),
      models,
    });
  } catch (error) {
    return NextResponse.json(
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
