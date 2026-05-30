// middleware.ts (at the root of your project)
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Direct media/download URLs are loaded by <img>, <audio>, <video>, and links,
// so they cannot attach the localStorage bearer token used by apiClient.
const publicReadRoutePrefixes = ['/api/img/', '/api/files/', '/api/remote-assets'];
const publicReadMethods = new Set(['GET', 'HEAD', 'OPTIONS']);
const remoteDatasetAssetsRoute = '/api/remote-datasets/assets';
const remoteDatasetAssetSignatureContext = 'remote-dataset-asset-v1';

function isRemoteDatasetAssetType(type: string) {
  return type === 'img' || type === 'file' || type === 'audio-art';
}

function remoteDatasetAssetSignaturePayload(workerID: string, remotePath: string, expires: number) {
  return [remoteDatasetAssetSignatureContext, workerID, remotePath, String(expires)].join('\n');
}

function base64Url(bytes: Uint8Array) {
  const binary = String.fromCharCode(...bytes);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function constantTimeEqual(left: string, right: string) {
  if (left.length !== right.length) return false;
  let diff = 0;
  for (let i = 0; i < left.length; i++) {
    diff |= left.charCodeAt(i) ^ right.charCodeAt(i);
  }
  return diff === 0;
}

async function signedRemoteDatasetAssetRequest(searchParams: URLSearchParams, secret: string) {
  const workerID = searchParams.get('worker_id') || '';
  const remotePath = searchParams.get('path') || '';
  const type = searchParams.get('type') || 'img';
  const expires = Number(searchParams.get('expires') || '');
  const signature = searchParams.get('sig') || '';

  if (
    !workerID ||
    !remotePath ||
    !isRemoteDatasetAssetType(type) ||
    !signature ||
    !Number.isSafeInteger(expires) ||
    expires <= Date.now()
  ) {
    return false;
  }

  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const digest = await crypto.subtle.sign(
    'HMAC',
    key,
    new TextEncoder().encode(remoteDatasetAssetSignaturePayload(workerID, remotePath, expires)),
  );
  return constantTimeEqual(signature, base64Url(new Uint8Array(digest)));
}

export async function middleware(request: NextRequest) {
  // check env var for AI_TOOLKIT_AUTH, if not set, approve all requests
  // if it is set make sure bearer token matches
  const tokenToUse = process.env.AI_TOOLKIT_AUTH || null;
  if (!tokenToUse) {
    return NextResponse.next();
  }

  // Get the token from the headers
  const token = request.headers.get('Authorization')?.split(' ')[1];

  const { pathname } = request.nextUrl;

  // allow public read routes to pass through
  if (publicReadMethods.has(request.method) && publicReadRoutePrefixes.some(route => pathname.startsWith(route))) {
    return NextResponse.next();
  }
  if (
    publicReadMethods.has(request.method) &&
    pathname === remoteDatasetAssetsRoute &&
    (await signedRemoteDatasetAssetRequest(request.nextUrl.searchParams, tokenToUse))
  ) {
    return NextResponse.next();
  }

  // Check if the route should be protected
  // This will apply to all API routes that start with /api/
  if (pathname.startsWith('/api/')) {
    if (!token || token !== tokenToUse) {
      // Return a JSON response with 401 Unauthorized
      return new NextResponse(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // For authorized users, continue
    return NextResponse.next();
  }

  // For non-API routes, just continue
  return NextResponse.next();
}

// Configure which paths this middleware will run on
export const config = {
  matcher: [
    // Apply to all API routes
    '/api/:path*',
  ],
};
