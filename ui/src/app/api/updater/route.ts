import { NextResponse } from 'next/server';
import { getRepoUpdateStatus, requestRepoUpdateCheck } from '@/server/updater';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  return NextResponse.json(await getRepoUpdateStatus());
}

export async function POST() {
  try {
    return NextResponse.json(await requestRepoUpdateCheck());
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to request update check' },
      { status: 500 },
    );
  }
}
