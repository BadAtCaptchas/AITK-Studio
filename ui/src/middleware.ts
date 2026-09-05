// middleware.ts (at the root of your project)
import { hasObsoleteWorkspaceScope, hasObsoleteWorkspaceHeaders } from '@/utils/obsoleteWorkspaceGuard';
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { isRequestAuthenticated } from '@/utils/authSession';

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
  if (hasObsoleteWorkspaceScope(request.nextUrl.searchParams) || hasObsoleteWorkspaceHeaders(request.headers)) {
    return NextResponse.json(
      { error: 'Project workspaces have been removed.', code: 'PROJECT_WORKSPACES_REMOVED' },
      { status: 400 },
    );
  }
  if (request.headers.get('content-type')?.includes('application/json')) {
    const payload: unknown = await request
      .clone()
      .json()
      .catch(() => null);
    if (hasObsoleteWorkspaceScope(payload)) {
      return NextResponse.json(
        { error: 'Project workspaces have been removed.', code: 'PROJECT_WORKSPACES_REMOVED' },
        { status: 400 },
      );
    }
  }
  const tokenToUse = process.env.AI_TOOLKIT_AUTH || null;
  if (!tokenToUse) {
    return NextResponse.next();
  }

  const { pathname } = request.nextUrl;

  // The auth endpoint validates login credentials itself and must be reachable
  // before a session exists. GET and DELETE are also validated in the handler.
  if (pathname === '/api/auth') {
    return NextResponse.next();
  }

  if (
    publicReadMethods.has(request.method) &&
    pathname === remoteDatasetAssetsRoute &&
    (await signedRemoteDatasetAssetRequest(request.nextUrl.searchParams, tokenToUse))
  ) {
    return NextResponse.next();
  }

  if (pathname.startsWith('/api/')) {
    if (!(await isRequestAuthenticated(request, tokenToUse))) {
      return new NextResponse(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    return NextResponse.next();
  }

  return NextResponse.next();
}

// Configure which paths this middleware will run on
export const config = {
  matcher: [
    // Apply to all API routes
    '/api/:path*',
  ],
};
