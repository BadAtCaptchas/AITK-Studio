import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
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

async function postCommand(request: NextRequest) {
  try {
    const body = assertGlobalPayload(await readJsonCommand(request));
    const worker = await saveRemoteOllamaWorker(body || {});
    return NextResponse.json(toPublicRemoteOllamaWorker(worker));
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Failed to save Remote Ollama worker';
    return NextResponse.json({ error: message }, { status: /already exists/i.test(message) ? 409 : 400 });
  }
}

export const POST = withCommandBoundary(postCommand);
