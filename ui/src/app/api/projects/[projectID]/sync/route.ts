import { NextResponse } from 'next/server';
import { db } from '@/server/db';
import {
  createProjectSyncOperation,
  projectSyncApiError,
  runProjectSyncOperation,
} from '@/server/projectSync';
import { PROJECT_SYNC_PROFILES, type ProjectSyncProfileName } from '@/server/projectSyncProtocol';
import { resolveProject } from '@/server/projects';
import { ensureProjectApiAccess, readJsonObject } from '../../projectApi';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const denied = await ensureProjectApiAccess(request);
  if (denied) return denied;
  try {
    const { projectID } = await params;
    const project = await resolveProject(decodeURIComponent(projectID), { intent: 'read' });
    return NextResponse.json({ operations: await db.projectSyncOperations.list({ project_id: project.id }) });
  } catch (error) {
    const response = projectSyncApiError(error, 'Failed to list project sync operations');
    return NextResponse.json(response.body, { status: response.status });
  }
}

export async function POST(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const denied = await ensureProjectApiAccess(request);
  if (denied) return denied;
  try {
    const { projectID } = await params;
    const body = await readJsonObject(request);
    if (typeof body.worker_id !== 'string' || !body.worker_id.trim()) {
      return NextResponse.json({ error: 'worker_id is required', code: 'PROJECT_SYNC_WORKER_REQUIRED' }, { status: 400 });
    }
    if (typeof body.profile !== 'string' || !(PROJECT_SYNC_PROFILES as readonly string[]).includes(body.profile)) {
      return NextResponse.json({ error: 'profile must be full, launch, or results' }, { status: 400 });
    }
    let operation = await createProjectSyncOperation(
      decodeURIComponent(projectID),
      body.worker_id.trim(),
      body.profile as ProjectSyncProfileName,
    );
    if (body.run_now !== false) operation = await runProjectSyncOperation(operation.id);
    return NextResponse.json({ operation }, { status: operation.status === 'completed' ? 200 : 202 });
  } catch (error) {
    const response = projectSyncApiError(error, 'Failed to start project sync');
    return NextResponse.json(response.body, { status: response.status });
  }
}
