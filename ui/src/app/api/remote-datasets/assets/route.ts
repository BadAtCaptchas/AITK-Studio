import { NextRequest, NextResponse } from 'next/server';
import { getRemoteWorker, remoteProxyFetch } from '@/server/remoteClient';
import { isRemoteDatasetAssetRequestAuthorized } from '@/server/remoteDatasetAssetAccess';
import type { RemoteDatasetAssetType } from '@/utils/remoteDatasetRefs';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function remoteAssetPath(type: RemoteDatasetAssetType, remotePath: string) {
  const encoded = encodeURIComponent(remotePath);
  if (type === 'file') return `/api/files/${encoded}`;
  if (type === 'audio-art') return `/api/audio/art/${encoded}`;
  return `/api/img/${encoded}`;
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

export async function GET(request: NextRequest) {
  const workerID = request.nextUrl.searchParams.get('worker_id') || '';
  const remotePath = request.nextUrl.searchParams.get('path') || '';
  const type = (request.nextUrl.searchParams.get('type') || 'img') as RemoteDatasetAssetType;

  if (!workerID || !remotePath) {
    return new NextResponse('Missing remote dataset asset parameters', { status: 400 });
  }
  if (type !== 'img' && type !== 'file' && type !== 'audio-art') {
    return new NextResponse('Invalid remote dataset asset type', { status: 400 });
  }
  if (
    !isRemoteDatasetAssetRequestAuthorized(
      request.headers,
      workerID,
      remotePath,
      request.nextUrl.searchParams.get('expires'),
      request.nextUrl.searchParams.get('sig'),
    )
  ) {
    return new NextResponse('Unauthorized', { status: 401 });
  }

  try {
    const worker = await getRemoteWorker(workerID);
    const remoteResponse = await remoteProxyFetch(worker, remoteAssetPath(type, remotePath), request.headers);
    return new NextResponse(remoteResponse.body, {
      status: remoteResponse.status,
      headers: copyResponseHeaders(remoteResponse),
    });
  } catch (error) {
    console.error('Remote dataset asset proxy failed:', error);
    return new NextResponse('Remote dataset asset unavailable', { status: 502 });
  }
}
