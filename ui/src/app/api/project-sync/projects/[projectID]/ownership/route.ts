import { NextResponse } from 'next/server';
import { getProjectOwnership, projectSyncWorkerError } from '@/server/projectSyncWorker';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(_request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  try {
    const { projectID } = await params;
    return NextResponse.json(await getProjectOwnership(decodeURIComponent(projectID)));
  } catch (error) {
    const response = projectSyncWorkerError(error, 'Failed to read project ownership');
    return NextResponse.json(response.body, { status: response.status });
  }
}
