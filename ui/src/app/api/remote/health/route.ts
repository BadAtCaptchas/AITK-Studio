import { NextResponse } from 'next/server';
import { getCloudflaredStatus } from '@/server/cloudflared';
import { getOllamaStatus } from '@/server/ollama';
import { getExecutionHealth } from '@/server/executionHealth';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  return NextResponse.json({
    ok: true,
    app: 'ai-toolkit',
    timestamp: new Date().toISOString(),
    ...await getExecutionHealth(),
    cloudflared: await getCloudflaredStatus(),
    ollama: await getOllamaStatus(),
  });
}
