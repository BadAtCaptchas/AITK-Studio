import { NextResponse } from 'next/server';
import { getCloudflaredStatus } from '@/server/cloudflared';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  return NextResponse.json({
    ok: true,
    app: 'ai-toolkit',
    timestamp: new Date().toISOString(),
    cloudflared: await getCloudflaredStatus(),
  });
}
