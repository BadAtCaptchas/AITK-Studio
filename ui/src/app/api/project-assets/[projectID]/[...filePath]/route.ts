import fs from 'fs';
import { Readable } from 'stream';
import { NextResponse } from 'next/server';
import {
  FILE_STREAM_BUFFER_BYTES,
  isAcceleratedApiRequestAuthenticated,
  resolveProjectAssetFileRequest,
  unauthorizedFileResponse,
  type FileResponseResolution,
} from '@/server/fileServing';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function toNextResponse(request: Request, resolution: FileResponseResolution) {
  if (resolution.kind === 'proxy') {
    return new NextResponse('Internal Server Error', { status: 500 });
  }
  if (!resolution.file || request.method === 'HEAD') {
    return new NextResponse(resolution.body, {
      status: resolution.status,
      headers: resolution.headers,
    });
  }

  const stream = fs.createReadStream(resolution.file.path, {
    ...(resolution.file.start !== undefined ? { start: resolution.file.start } : {}),
    ...(resolution.file.end !== undefined ? { end: resolution.file.end } : {}),
    highWaterMark: FILE_STREAM_BUFFER_BYTES,
  });
  const onAbort = () => stream.destroy();
  if (request.signal.aborted) {
    stream.destroy();
  } else {
    request.signal.addEventListener('abort', onAbort, { once: true });
  }
  stream.once('close', () => request.signal.removeEventListener('abort', onAbort));

  return new NextResponse(Readable.toWeb(stream) as ReadableStream, {
    status: resolution.status,
    headers: resolution.headers,
  });
}

async function serve(
  request: Request,
  { params }: { params: Promise<{ projectID: string; filePath: string[] }> },
) {
  try {
    if (request.signal.aborted) return new NextResponse(null, { status: 499 });

    const { projectID: encodedProjectID, filePath } = await params;
    const projectID = decodeURIComponent(encodedProjectID);
    const resolution = await resolveProjectAssetFileRequest({
      projectID,
      routePath: filePath.map(decodeURIComponent).join('/'),
      searchParams: new URL(request.url).searchParams,
      headers: request.headers,
    });

    if (
      resolution.kind === 'response' &&
      resolution.status === 403 &&
      process.env.AI_TOOLKIT_AUTH &&
      !(await isAcceleratedApiRequestAuthenticated(request.headers))
    ) {
      return toNextResponse(request, unauthorizedFileResponse());
    }
    return toNextResponse(request, resolution);
  } catch {
    return new NextResponse('Project asset not found', { status: 404 });
  }
}

export const GET = serve;
export const HEAD = serve;
