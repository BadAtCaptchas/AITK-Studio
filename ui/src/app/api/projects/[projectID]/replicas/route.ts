import { NextResponse } from 'next/server';
import { linkProjectWorker, listProjectReplicas, projectSyncApiError } from '@/server/projectSync';
import { ensureProjectApiAccess, readJsonObject } from '../../projectApi';

export const runtime = 'nodejs';

export async function GET(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const denied = ensureProjectApiAccess(request);
  if (denied) return denied;
  try {
    const { projectID } = await params;
    return NextResponse.json(await listProjectReplicas(decodeURIComponent(projectID)));
  } catch (error) {
    const response = projectSyncApiError(error, 'Failed to list project replicas');
    return NextResponse.json(response.body, { status: response.status });
  }
}

export async function POST(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const denied = ensureProjectApiAccess(request);
  if (denied) return denied;
  try {
    const { projectID } = await params;
    const body = await readJsonObject(request);
    if (typeof body.worker_id !== 'string' || !body.worker_id.trim()) {
      return NextResponse.json({ error: 'worker_id is required', code: 'PROJECT_SYNC_WORKER_REQUIRED' }, { status: 400 });
    }
    const replica = await linkProjectWorker(decodeURIComponent(projectID), body.worker_id.trim());
    return NextResponse.json({ replica }, { status: 201 });
  } catch (error) {
    const response = projectSyncApiError(error, 'Failed to link project replica');
    return NextResponse.json(response.body, { status: response.status });
  }
}
