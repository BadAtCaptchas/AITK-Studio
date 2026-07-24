import fs from 'fs';
import { Readable } from 'stream';
import { NextRequest, NextResponse } from 'next/server';
import {
  FILE_STREAM_BUFFER_BYTES,
  PRIVATE_MEDIA_CACHE_CONTROL,
  isAcceleratedApiRequestAuthenticated,
  resolveLocalMediaRequest,
  unauthorizedFileResponse,
  type FileResponseResolution,
} from '@/server/fileServing';
import { getRemoteWorker, remoteProxyFetch } from '@/server/remoteClient';
import { isRemoteDatasetAssetRequestAuthorized } from '@/server/remoteDatasetAssetAccess';
import { parseRemoteDatasetAssetRef } from '@/utils/remoteDatasetRefs';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type ImageRouteParams = {
  imagePath: string[];
};

function getRequestedValue(request: NextRequest, imagePath: string[]) {
  const routePrefix = '/api/img/';
  const pathname = request.nextUrl?.pathname;
  const rawPath =
    pathname && pathname.startsWith(routePrefix)
      ? pathname.slice(routePrefix.length)
      : imagePath.join('/');
  return decodeURIComponent(rawPath);
}

function remoteAssetPath(remotePath: string) {
  return `/api/img/${encodeURIComponent(remotePath)}`;
}

function copyResponseHeaders(source: Response) {
  const headers = new Headers();
  for (const name of [
    'content-type',
    'content-length',
    'content-range',
    'accept-ranges',
    'cache-control',
    'content-disposition',
    'x-content-type-options',
    'etag',
  ]) {
    const value = source.headers.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
}

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
  { params }: { params: Promise<ImageRouteParams> },
) {
  try {
    if (!(await isAcceleratedApiRequestAuthenticated(request.headers))) {
      return toNextResponse(request, unauthorizedFileResponse());
    }
    if (request.signal.aborted) return new NextResponse(null, { status: 499 });

    const { imagePath } = await params;
    const requestedValue = getRequestedValue(request, imagePath);
    const remoteAsset = parseRemoteDatasetAssetRef(requestedValue);
    if (remoteAsset) {
      if (
        !(await isRemoteDatasetAssetRequestAuthorized(
          request.headers,
          remoteAsset.workerID,
          remoteAsset.path,
          remoteAsset.expires,
          remoteAsset.signature,
        ))
      ) {
        return new NextResponse('Unauthorized', { status: 401 });
      }
      const worker = await getRemoteWorker(remoteAsset.workerID);
      const remoteResponse = await remoteProxyFetch(
        worker,
        remoteAssetPath(remoteAsset.path),
        request.headers,
        request.method === 'HEAD' ? 'HEAD' : 'GET',
      );
      if ((remoteResponse.headers.get('content-type') || '').toLowerCase().includes('image/svg+xml')) {
        return new NextResponse('Unsupported media type', { status: 415 });
      }
      const headers = copyResponseHeaders(remoteResponse);
      headers.set('Cache-Control', PRIVATE_MEDIA_CACHE_CONTROL);
      headers.set('X-Content-Type-Options', 'nosniff');
      return new NextResponse(request.method === 'HEAD' ? null : remoteResponse.body, {
        status: remoteResponse.status,
        headers,
      });
    }

    const resolution = await resolveLocalMediaRequest(requestedValue, request.headers);
    return toNextResponse(request, resolution);
  } catch (error) {
    console.error('Error serving image:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}

export const GET = serve;
export const HEAD = serve;
