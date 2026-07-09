import { NextResponse } from 'next/server';
import { assertExecutionReplica, removeExecutionReplica, projectSyncWorkerError } from '@/server/projectSyncWorker';

export const runtime = 'nodejs';

export async function GET(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  try {
    const { projectID } = await params;
    const expectedHome = new URL(request.url).searchParams.get('home_instance_id');
    const project = await assertExecutionReplica(decodeURIComponent(projectID), expectedHome);
    return NextResponse.json({ project });
  } catch (error) {
    const response = projectSyncWorkerError(error, 'Failed to discover project replica');
    return NextResponse.json(response.body, { status: response.status });
  }
}

export async function DELETE(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  try {
    const { projectID } = await params;
    const body: unknown = await request.json();
    const expectedHome =
      body && typeof body === 'object' && !Array.isArray(body)
        ? (body as Record<string, unknown>).home_instance_id
        : null;
    if (typeof expectedHome !== 'string' || !expectedHome.trim()) {
      return NextResponse.json({ error: 'home_instance_id is required', code: 'PROJECT_SYNC_IDENTITY_INVALID' }, { status: 400 });
    }
    return NextResponse.json(await removeExecutionReplica(decodeURIComponent(projectID), expectedHome));
  } catch (error) {
    const response = projectSyncWorkerError(error, 'Failed to remove project replica');
    return NextResponse.json(response.body, { status: response.status });
  }
}
