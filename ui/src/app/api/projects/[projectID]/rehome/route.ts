import { NextResponse } from 'next/server';
import { projectSyncApiError, rehomeProject } from '@/server/projectSync';
import { ensureProjectApiAccess, readJsonObject } from '../../projectApi';

export const runtime = 'nodejs';

export async function POST(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const denied = ensureProjectApiAccess(request);
  if (denied) return denied;
  try {
    const { projectID } = await params;
    const body = await readJsonObject(request);
    if (typeof body.worker_id !== 'string' || !body.worker_id.trim()) {
      return NextResponse.json({ error: 'worker_id is required' }, { status: 400 });
    }
    if (!Number.isSafeInteger(body.expected_revision)) {
      return NextResponse.json({ error: 'expected_revision is required' }, { status: 400 });
    }
    return NextResponse.json(
      await rehomeProject(decodeURIComponent(projectID), body.worker_id.trim(), Number(body.expected_revision)),
    );
  } catch (error) {
    const response = projectSyncApiError(error, 'Failed to rehome project');
    return NextResponse.json(response.body, { status: response.status });
  }
}
