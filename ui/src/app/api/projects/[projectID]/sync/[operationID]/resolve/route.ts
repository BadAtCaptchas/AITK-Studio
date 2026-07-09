import { NextResponse } from 'next/server';
import {
  projectSyncApiError,
  resolveProjectSyncConflicts,
  runProjectSyncOperation,
} from '@/server/projectSync';
import { ensureProjectApiAccess, readJsonObject } from '../../../../projectApi';

export const runtime = 'nodejs';

type SyncParams = { projectID: string; operationID: string };

export async function POST(request: Request, { params }: { params: Promise<SyncParams> }) {
  const denied = ensureProjectApiAccess(request);
  if (denied) return denied;
  try {
    const { projectID, operationID } = await params;
    const body = await readJsonObject(request);
    if (!body.resolutions || typeof body.resolutions !== 'object' || Array.isArray(body.resolutions)) {
      return NextResponse.json({ error: 'resolutions must be a path-to-resolution object' }, { status: 400 });
    }
    let operation = await resolveProjectSyncConflicts(
      decodeURIComponent(projectID),
      decodeURIComponent(operationID),
      body.resolutions as Record<string, unknown>,
    );
    if (body.run_now !== false) operation = await runProjectSyncOperation(operation.id);
    return NextResponse.json({ operation }, { status: operation.status === 'completed' ? 200 : 202 });
  } catch (error) {
    const response = projectSyncApiError(error, 'Failed to resolve project sync conflicts');
    return NextResponse.json(response.body, { status: response.status });
  }
}
