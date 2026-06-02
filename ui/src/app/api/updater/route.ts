import { NextRequest, NextResponse } from 'next/server';
import { getRepoUpdateStatus, requestRepoUpdateCheck } from '@/server/updater';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  return NextResponse.json(await getRepoUpdateStatus());
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => ({}));
    const action = body?.action === 'apply' || body?.action === 'restart' ? body.action : 'check';
    return NextResponse.json(await requestRepoUpdateCheck(action));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to request update check' },
      { status: 500 },
    );
  }
}
