import fs from 'fs';
import fsp from 'fs/promises';
import { Readable } from 'stream';
import { NextResponse } from 'next/server';
import { assertExecutionReplica, projectSyncWorkerError } from '@/server/projectSyncWorker';
import {
  parseHttpByteRange,
  assertProjectSyncPathContained,
  PROJECT_SYNC_PROFILES,
  ProjectSyncProtocolError,
  resolveProjectSyncPath,
  type ProjectSyncProfileName,
} from '@/server/projectSyncProtocol';
import { getProjectRoots } from '@/server/projects';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type BlobParams = { projectID: string; filePath: string[] };

export async function GET(request: Request, { params }: { params: Promise<BlobParams> }) {
  try {
    const { projectID, filePath } = await params;
    const url = new URL(request.url);
    const profileValue = url.searchParams.get('profile') || 'full';
    if (!(PROJECT_SYNC_PROFILES as readonly string[]).includes(profileValue)) {
      throw new ProjectSyncProtocolError('Unsupported project sync profile');
    }
    const project = await assertExecutionReplica(decodeURIComponent(projectID), url.searchParams.get('home_instance_id'));
    const roots = await getProjectRoots(project);
    const target = resolveProjectSyncPath(roots.root, filePath.map(decodeURIComponent).join('/'), profileValue as ProjectSyncProfileName);
    await assertProjectSyncPathContained(roots.root, target);
    const stat = await fsp.lstat(target).catch(() => null);
    if (!stat || !stat.isFile() || stat.isSymbolicLink()) {
      return NextResponse.json({ error: 'Blob not found', code: 'PROJECT_SYNC_BLOB_NOT_FOUND' }, { status: 404 });
    }
    const range = parseHttpByteRange(request.headers.get('range'), stat.size);
    const stream = fs.createReadStream(target, { start: range.start, end: range.end });
    if (request.signal.aborted) stream.destroy();
    else request.signal.addEventListener('abort', () => stream.destroy(), { once: true });
    const headers = new Headers({
      'Accept-Ranges': 'bytes',
      'Content-Length': String(range.end - range.start + 1),
      'Content-Type': 'application/octet-stream',
      'Cache-Control': 'private, no-store',
    });
    if (range.partial) headers.set('Content-Range', `bytes ${range.start}-${range.end}/${stat.size}`);
    return new NextResponse(Readable.toWeb(stream) as unknown as BodyInit, {
      status: range.partial ? 206 : 200,
      headers,
    });
  } catch (error) {
    const response = projectSyncWorkerError(error, 'Failed to read project sync blob');
    return NextResponse.json(response.body, { status: response.status });
  }
}
