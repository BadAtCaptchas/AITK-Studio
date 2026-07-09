import { NextResponse } from 'next/server';
import { linkProjectJobReplica, projectSyncWorkerError } from '@/server/projectSyncWorker';
import { ProjectSyncProtocolError } from '@/server/projectSyncProtocol';

export const runtime = 'nodejs';

export async function POST(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  try {
    const { projectID } = await params;
    const body: unknown = await request.json();
    if (!body || typeof body !== 'object' || Array.isArray(body)) {
      throw new ProjectSyncProtocolError('Request body must be an object');
    }
    const value = body as Record<string, unknown>;
    if (typeof value.home_instance_id !== 'string' || !value.home_instance_id.trim()) {
      throw new ProjectSyncProtocolError('home_instance_id is required');
    }
    return NextResponse.json(
      await linkProjectJobReplica(decodeURIComponent(projectID), value.home_instance_id, value),
      { status: 201 },
    );
  } catch (error) {
    const response = projectSyncWorkerError(error, 'Failed to link project job replica');
    return NextResponse.json(response.body, { status: response.status });
  }
}
