import { NextResponse } from 'next/server';
import { db } from '@/server/db';
import { projectSyncApiError, runProjectSyncOperation } from '@/server/projectSync';
import { resolveProject } from '@/server/projects';
import { ensureProjectApiAccess } from '../../../projectApi';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type SyncParams = { projectID: string; operationID: string };

export async function GET(request: Request, { params }: { params: Promise<SyncParams> }) {
  const denied = ensureProjectApiAccess(request);
  if (denied) return denied;
  try {
    const { projectID, operationID } = await params;
    const project = await resolveProject(decodeURIComponent(projectID), { intent: 'read' });
    const operation = await db.projectSyncOperations.findById(decodeURIComponent(operationID));
    if (!operation || operation.project_id !== project.id) {
      return NextResponse.json({ error: 'Project sync operation not found' }, { status: 404 });
    }
    return NextResponse.json({ operation });
  } catch (error) {
    const response = projectSyncApiError(error, 'Failed to load project sync operation');
    return NextResponse.json(response.body, { status: response.status });
  }
}

export async function POST(request: Request, { params }: { params: Promise<SyncParams> }) {
  const denied = ensureProjectApiAccess(request);
  if (denied) return denied;
  try {
    const { projectID, operationID } = await params;
    const project = await resolveProject(decodeURIComponent(projectID), { intent: 'write' });
    const existing = await db.projectSyncOperations.findById(decodeURIComponent(operationID));
    if (!existing || existing.project_id !== project.id) {
      return NextResponse.json({ error: 'Project sync operation not found' }, { status: 404 });
    }
    const operation = await runProjectSyncOperation(existing.id);
    return NextResponse.json({ operation }, { status: operation.status === 'completed' ? 200 : 202 });
  } catch (error) {
    const response = projectSyncApiError(error, 'Failed to resume project sync operation');
    return NextResponse.json(response.body, { status: response.status });
  }
}
