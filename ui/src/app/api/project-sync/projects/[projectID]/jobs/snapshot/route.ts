import { NextResponse } from 'next/server';
import {
  importProjectJobSnapshot,
  projectSyncWorkerError,
  readImportedProjectJobSnapshot,
} from '@/server/projectSyncWorker';
import { ProjectSyncProtocolError } from '@/server/projectSyncProtocol';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  try {
    const { projectID } = await params;
    const homeInstanceID = new URL(request.url).searchParams.get('home_instance_id');
    if (!homeInstanceID) throw new ProjectSyncProtocolError('home_instance_id is required');
    return NextResponse.json(
      await readImportedProjectJobSnapshot(decodeURIComponent(projectID), homeInstanceID),
    );
  } catch (error) {
    const response = projectSyncWorkerError(error, 'Failed to read project job snapshot');
    return NextResponse.json(response.body, { status: response.status });
  }
}

export async function POST(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  try {
    const { projectID } = await params;
    const body: unknown = await request.json();
    if (!body || typeof body !== 'object' || Array.isArray(body)) {
      throw new ProjectSyncProtocolError('Request body must be an object');
    }
    const homeInstanceID = (body as Record<string, unknown>).home_instance_id;
    if (typeof homeInstanceID !== 'string' || !homeInstanceID.trim()) {
      throw new ProjectSyncProtocolError('home_instance_id is required');
    }
    return NextResponse.json(
      await importProjectJobSnapshot(decodeURIComponent(projectID), homeInstanceID, body),
    );
  } catch (error) {
    const response = projectSyncWorkerError(error, 'Failed to import project job snapshot');
    return NextResponse.json(response.body, { status: response.status });
  }
}
