import { NextResponse } from 'next/server';
import { projectSyncApiError, removeProjectWorkerReplica } from '@/server/projectSync';
import { ensureProjectApiAccess } from '../../../projectApi';

export const runtime = 'nodejs';

export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ projectID: string; workerID: string }> },
) {
  const denied = ensureProjectApiAccess(request);
  if (denied) return denied;
  try {
    const { projectID, workerID } = await params;
    return NextResponse.json(
      await removeProjectWorkerReplica(decodeURIComponent(projectID), decodeURIComponent(workerID)),
    );
  } catch (error) {
    const response = projectSyncApiError(error, 'Failed to remove project replica');
    return NextResponse.json(response.body, { status: response.status });
  }
}
