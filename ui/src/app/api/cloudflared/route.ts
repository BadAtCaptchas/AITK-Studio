import { NextRequest, NextResponse } from 'next/server';
import { downloadCloudflared, getCloudflaredStatus, startCloudflared, stopCloudflared } from '@/server/cloudflared';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  return NextResponse.json(await getCloudflaredStatus());
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => ({}));
    return NextResponse.json(await startCloudflared({ autoDownload: Boolean(body?.autoDownload) }));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to start cloudflared' },
      { status: 400 },
    );
  }
}

export async function PUT() {
  try {
    const download = await downloadCloudflared();
    return NextResponse.json({
      download,
      status: await getCloudflaredStatus(),
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to download cloudflared' },
      { status: 400 },
    );
  }
}

export async function DELETE() {
  return NextResponse.json(await stopCloudflared());
}
