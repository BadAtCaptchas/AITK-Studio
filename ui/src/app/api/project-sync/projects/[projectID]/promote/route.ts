import { NextResponse } from 'next/server';
import {
  cancelExecutionReplicaPromotion,
  commitExecutionReplicaPromotion,
  prepareExecutionReplicaPromotion,
  projectSyncWorkerError,
} from '@/server/projectSyncWorker';
import { ProjectSyncProtocolError } from '@/server/projectSyncProtocol';

export const runtime = 'nodejs';

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
    const action = (body as Record<string, unknown>).action;
    if (action === 'prepare') {
      return NextResponse.json(await prepareExecutionReplicaPromotion(decodeURIComponent(projectID), homeInstanceID));
    }
    if (action === 'commit') {
      return NextResponse.json(
        await commitExecutionReplicaPromotion(
          decodeURIComponent(projectID),
          homeInstanceID,
          (body as Record<string, unknown>).token,
        ),
      );
    }
    if (action === 'cancel') {
      return NextResponse.json(
        await cancelExecutionReplicaPromotion(
          decodeURIComponent(projectID),
          homeInstanceID,
          (body as Record<string, unknown>).token,
        ),
      );
    }
    throw new ProjectSyncProtocolError('action must be prepare, commit, or cancel');
  } catch (error) {
    const response = projectSyncWorkerError(error, 'Failed to promote project replica');
    return NextResponse.json(response.body, { status: response.status });
  }
}
