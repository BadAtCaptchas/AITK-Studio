import fs from 'fs';
import { Readable } from 'stream';
import { NextRequest, NextResponse } from 'next/server';
import {
  FILE_STREAM_BUFFER_BYTES,
  isAcceleratedApiRequestAuthenticated,
  resolveDownloadFileRequest,
  unauthorizedFileResponse,
  type FileResponseResolution,
} from '@/server/fileServing';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function toNextResponse(request: NextRequest, resolution: FileResponseResolution) {
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
  request: NextRequest,
  { params }: { params: Promise<{ filePath: string[] }> },
) {
  try {
    if (!(await isAcceleratedApiRequestAuthenticated(request.headers))) {
      return toNextResponse(request, unauthorizedFileResponse());
    }
    if (request.signal.aborted) return new NextResponse(null, { status: 499 });

    const { filePath } = await params;
    const decodedFilePath = decodeURIComponent(filePath.join('/'));
    return toNextResponse(
      request,
      await resolveDownloadFileRequest(decodedFilePath, request.headers),
    );
  } catch (error) {
    console.error('Error serving file:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}

export const GET = serve;
export const HEAD = serve;
