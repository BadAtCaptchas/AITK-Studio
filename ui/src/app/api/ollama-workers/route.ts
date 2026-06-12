import { NextRequest, NextResponse } from 'next/server';
import {
  listRemoteOllamaWorkers,
  saveRemoteOllamaWorker,
  toPublicRemoteOllamaWorker,
} from '@/server/remoteOllamaWorkers';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  const workers = await listRemoteOllamaWorkers();
  return NextResponse.json({ workers: workers.map(toPublicRemoteOllamaWorker) });
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const worker = await saveRemoteOllamaWorker(body || {});
    return NextResponse.json(toPublicRemoteOllamaWorker(worker));
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Failed to save Remote Ollama worker';
    return NextResponse.json(
      { error: message },
      { status: /already exists/i.test(message) ? 409 : 400 },
    );
  }
}
