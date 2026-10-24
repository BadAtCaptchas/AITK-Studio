import { getRemoteWorker, remoteProxyFetch } from '@/server/remoteClient';
import { isRemoteDatasetAssetRequestAuthorized } from '@/server/remoteDatasetAssetAccess';
import type { RemoteDatasetAssetType } from '@/utils/remoteDatasetRefs';


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
    'last-modified',
  ]) {
    const value = source.headers.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
}

export async function GET(request: Request) {
  const workerID = new URL(request.url).searchParams.get('worker_id') || '';
  const remotePath = new URL(request.url).searchParams.get('path') || '';
  const type = (new URL(request.url).searchParams.get('type') || 'img') as RemoteDatasetAssetType;

  if (!workerID || !remotePath) {
    return new Response('Missing remote dataset asset parameters', { status: 400 });
  }
  if (type !== 'img' && type !== 'file' && type !== 'audio-art') {
    return new Response('Invalid remote dataset asset type', { status: 400 });
  }
  if (
    !(await isRemoteDatasetAssetRequestAuthorized(
      request.headers,
      workerID,
      remotePath,
      new URL(request.url).searchParams.get('expires'),
      new URL(request.url).searchParams.get('sig'),
      type,
      request.method,
    ))
  ) {
    return new Response('Unauthorized', { status: 401 });
  }

  try {
    const worker = await getRemoteWorker(workerID);
    const remoteResponse = await remoteProxyFetch(worker, remoteAssetPath(type, remotePath), request.headers, request.method === 'HEAD' ? 'HEAD' : 'GET', request.signal);
    return new Response(request.method === 'HEAD' ? null : remoteResponse.body, {
      status: remoteResponse.status,
      headers: copyResponseHeaders(remoteResponse),
    });
  } catch (error) {
    console.error('Remote dataset asset proxy failed:', error);
    return new Response('Remote dataset asset unavailable', { status: 502 });
  }
}

export const HEAD = GET;
