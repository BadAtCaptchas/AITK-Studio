import { NextResponse } from 'next/server';
import { linkProjectReplica, projectSyncWorkerError } from '@/server/projectSyncWorker';
import { ProjectSyncProtocolError } from '@/server/projectSyncProtocol';

export const runtime = 'nodejs';

export async function POST(request: Request) {
  try {
    const body: unknown = await request.json();
    if (!body || typeof body !== 'object' || Array.isArray(body)) {
      throw new ProjectSyncProtocolError('Request body must be an object');
    }
    return NextResponse.json(await linkProjectReplica(body as Record<string, unknown>), { status: 201 });
  } catch (error) {
    const response = projectSyncWorkerError(error, 'Failed to link project replica');
    return NextResponse.json(response.body, { status: response.status });
  }
}
