import { NextResponse } from 'next/server';
import { getProjectSyncCapabilities, projectSyncWorkerError } from '@/server/projectSyncWorker';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    return NextResponse.json(await getProjectSyncCapabilities());
  } catch (error) {
    const response = projectSyncWorkerError(error, 'Failed to read project sync capabilities');
    return NextResponse.json(response.body, { status: response.status });
  }
}
