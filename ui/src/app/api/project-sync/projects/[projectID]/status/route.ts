import { NextResponse } from 'next/server';
import { assertExecutionReplica, projectSyncWorkerError } from '@/server/projectSyncWorker';
import { getProjectSyncChunkReceipt, ProjectSyncProtocolError } from '@/server/projectSyncProtocol';
import { getProjectRoots } from '@/server/projects';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  try {
    const { projectID } = await params;
    const body: unknown = await request.json();
    if (!body || typeof body !== 'object' || Array.isArray(body)) {
      throw new ProjectSyncProtocolError('Request body must be an object');
    }
    const value = body as Record<string, unknown>;
    const project = await assertExecutionReplica(
      decodeURIComponent(projectID),
      typeof value.home_instance_id === 'string' ? value.home_instance_id : null,
    );
    if (!Array.isArray(value.files) || value.files.length > 10_000) {
      throw new ProjectSyncProtocolError('files must be an array of no more than 10,000 entries');
    }
    const roots = await getProjectRoots(project);
    const receipts = await Promise.all(
      value.files.map(async file => {
        if (!file || typeof file !== 'object' || Array.isArray(file)) {
          throw new ProjectSyncProtocolError('Invalid status file entry');
        }
        const item = file as Record<string, unknown>;
        if (typeof item.sha256 !== 'string' || !Number.isSafeInteger(item.size) || Number(item.size) < 0) {
          throw new ProjectSyncProtocolError('Invalid status file entry');
        }
        return getProjectSyncChunkReceipt(roots.root, item.sha256, Number(item.size));
      }),
    );
    return NextResponse.json({ project_id: project.id, receipts });
  } catch (error) {
    const response = projectSyncWorkerError(error, 'Failed to inspect project sync status');
    return NextResponse.json(response.body, { status: response.status });
  }
}
