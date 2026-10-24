import { fileResponse } from '@/server/http/fileResponse';
import {
  PRIVATE_MEDIA_CACHE_CONTROL,
  isAcceleratedApiRequestAuthenticated,
  resolveLocalMediaRequest,
  unauthorizedFileResponse,
} from '@/server/fileServing';
import { getRemoteWorker, remoteProxyFetch } from '@/server/remoteClient';
import { isRemoteDatasetAssetRequestAuthorized } from '@/server/remoteDatasetAssetAccess';
import { parseRemoteDatasetAssetRef } from '@/utils/remoteDatasetRefs';


type ImageRouteParams = {
  imagePath: string[];
};

function getRequestedValue(request: Request, imagePath: string[]) {
  const routePrefix = '/api/img/';
  const pathname = new URL(request.url)?.pathname;
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
    'last-modified',
  ]) {
    const value = source.headers.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
}


async function serve(
  request: Request,
  { params }: { params: Promise<ImageRouteParams> },
) {
  try {
    if (!(await isAcceleratedApiRequestAuthenticated(request.headers))) {
      return fileResponse(request, unauthorizedFileResponse());
    }
    if (request.signal.aborted) return new Response(null, { status: 499 });

    const { imagePath } = await params;
    const requestedValue = getRequestedValue(request, imagePath);
    const remoteAsset = parseRemoteDatasetAssetRef(requestedValue);
    if (remoteAsset) {
      if (remoteAsset.type !== 'img') return new Response('Invalid image representation', { status: 400 });
      if (
        !(await isRemoteDatasetAssetRequestAuthorized(
          request.headers,
          remoteAsset.workerID,
          remoteAsset.path,
          remoteAsset.expires,
          remoteAsset.signature,
          remoteAsset.type,
          request.method,
        ))
      ) {
        return new Response('Unauthorized', { status: 401 });
      }
      const worker = await getRemoteWorker(remoteAsset.workerID);
      const remoteResponse = await remoteProxyFetch(
        worker,
        remoteAssetPath(remoteAsset.path),
        request.headers,
        request.method === 'HEAD' ? 'HEAD' : 'GET',
        request.signal,
      );
      if ((remoteResponse.headers.get('content-type') || '').toLowerCase().includes('image/svg+xml')) {
        return new Response('Unsupported media type', { status: 415 });
      }
      const headers = copyResponseHeaders(remoteResponse);
      headers.set('Cache-Control', PRIVATE_MEDIA_CACHE_CONTROL);
      headers.set('X-Content-Type-Options', 'nosniff');
      return new Response(request.method === 'HEAD' ? null : remoteResponse.body, {
        status: remoteResponse.status,
        headers,
      });
    }

    const resolution = await resolveLocalMediaRequest(requestedValue, request.headers);
    return fileResponse(request, resolution);
  } catch (error) {
    console.error('Error serving image:', error);
    return new Response('Internal Server Error', { status: 500 });
  }
}

export const GET = serve;
export const HEAD = serve;
