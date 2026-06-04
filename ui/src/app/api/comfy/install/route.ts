import { NextResponse } from 'next/server';
import { getComfyManagedInstallStatus, startComfyManagedInstall } from '@/server/comfyManagedInstall';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  return NextResponse.json(await getComfyManagedInstallStatus());
}

export async function POST() {
  try {
    return NextResponse.json(await startComfyManagedInstall());
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to start managed ComfyUI install' },
      { status: 400 },
    );
  }
}
