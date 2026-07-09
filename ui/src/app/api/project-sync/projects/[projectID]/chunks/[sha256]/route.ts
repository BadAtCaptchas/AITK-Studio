import { NextResponse } from 'next/server';
import { assertExecutionReplica, projectSyncWorkerError } from '@/server/projectSyncWorker';
import {
  appendProjectSyncChunk,
  getProjectSyncChunkReceipt,
  PROJECT_SYNC_CHUNK_BYTES,
  ProjectSyncProtocolError,
} from '@/server/projectSyncProtocol';
import { getProjectRoots } from '@/server/projects';

export const runtime = 'nodejs';

type ChunkParams = { projectID: string; sha256: string };

function positiveInteger(value: string | null, label: string, allowZero = false) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < (allowZero ? 0 : 1)) {
    throw new ProjectSyncProtocolError(`${label} must be a safe integer`);
  }
  return parsed;
}

export async function GET(request: Request, { params }: { params: Promise<ChunkParams> }) {
  try {
    const { projectID, sha256 } = await params;
    const url = new URL(request.url);
    const project = await assertExecutionReplica(decodeURIComponent(projectID), url.searchParams.get('home_instance_id'));
    const total = positiveInteger(url.searchParams.get('total'), 'total', true);
    const roots = await getProjectRoots(project);
    return NextResponse.json(await getProjectSyncChunkReceipt(roots.root, sha256, total));
  } catch (error) {
    const response = projectSyncWorkerError(error, 'Failed to inspect project sync chunk');
    return NextResponse.json(response.body, { status: response.status });
  }
}

export async function PUT(request: Request, { params }: { params: Promise<ChunkParams> }) {
  try {
    const { projectID, sha256 } = await params;
    const url = new URL(request.url);
    const project = await assertExecutionReplica(decodeURIComponent(projectID), url.searchParams.get('home_instance_id'));
    const total = positiveInteger(url.searchParams.get('total'), 'total', true);
    const offset = positiveInteger(url.searchParams.get('offset'), 'offset', true);
    const declaredLength = Number(request.headers.get('content-length') || 0);
    if (declaredLength > PROJECT_SYNC_CHUNK_BYTES) {
      throw new ProjectSyncProtocolError('Chunk exceeds the 8 MiB protocol limit', {
        status: 413,
        code: 'PROJECT_SYNC_CHUNK_TOO_LARGE',
      });
    }
    const bytes = new Uint8Array(await request.arrayBuffer());
    const roots = await getProjectRoots(project);
    return NextResponse.json(await appendProjectSyncChunk(roots.root, sha256, total, offset, bytes));
  } catch (error) {
    const response = projectSyncWorkerError(error, 'Failed to receive project sync chunk');
    return NextResponse.json(response.body, { status: response.status });
  }
}
