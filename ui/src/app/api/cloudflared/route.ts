import { NextRequest, NextResponse } from 'next/server';
import { getCloudflaredStatus, startCloudflared, stopCloudflared } from '@/server/cloudflared';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  return NextResponse.json(await getCloudflaredStatus());
}

export async function POST(_request: NextRequest) {
  try {
    return NextResponse.json(await startCloudflared());
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to start cloudflared' },
      { status: 400 },
    );
  }
}

export async function DELETE() {
  return NextResponse.json(await stopCloudflared());
}
