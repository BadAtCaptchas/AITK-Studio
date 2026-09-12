// middleware.ts (at the root of your project)
import { hasObsoleteWorkspaceScope, hasObsoleteWorkspaceHeaders } from '@/utils/obsoleteWorkspaceGuard';
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { isRequestAuthenticated } from '@/utils/authSession';
import { isRemoteDatasetAssetSignatureValid } from '@/server/remoteDatasetAssetAccess';
import { browserRequestError } from '@/server/ingressPolicy';
import { internalOperationAuthorized } from '@/server/operationKeys';

const publicReadMethods = new Set(['GET', 'HEAD']);
const remoteDatasetAssetsRoute = '/api/remote-datasets/assets';

function signedRemoteDatasetAssetRequest(params: URLSearchParams, method: string) {
  const type = params.get('type') || 'img';
  if (type !== 'img' && type !== 'file' && type !== 'audio-art') return false;
  return isRemoteDatasetAssetSignatureValid(params.get('worker_id') || '', params.get('path') || '',
    params.get('expires'), params.get('sig'), type, method);
}

export async function middleware(request: NextRequest) {
  const ingressError = browserRequestError(request.headers, request.url, request.method);
  if (ingressError) return NextResponse.json({ error: ingressError }, { status: 403 });
  if (hasObsoleteWorkspaceScope(request.nextUrl.searchParams) || hasObsoleteWorkspaceHeaders(request.headers)) {
    return NextResponse.json(
      { error: 'Project workspaces have been removed.', code: 'PROJECT_WORKSPACES_REMOVED' },
      { status: 400 },
    );
  }

  const tokenToUse = process.env.AI_TOOLKIT_AUTH || null;
  const { pathname } = request.nextUrl;
  if (pathname.startsWith('/api/internal/operation-keys/')) {
    return internalOperationAuthorized(request.headers) ? NextResponse.next() : NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  if (tokenToUse && pathname.startsWith('/api/')) {
    // The auth endpoint validates login credentials itself and must be reachable
    // before a session exists. GET and DELETE are also validated in the handler.
    if (pathname === '/api/auth') {
      return NextResponse.next();
    }

    if (
      publicReadMethods.has(request.method) &&
      pathname === remoteDatasetAssetsRoute &&
      (await signedRemoteDatasetAssetRequest(request.nextUrl.searchParams, request.method))
    ) {
      return NextResponse.next();
    }

    if (!(await isRequestAuthenticated(request, tokenToUse))) {
      return new NextResponse(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      });
    }
  }

  return NextResponse.next();
}

// Configure which paths this middleware will run on
export const config = {
  runtime: 'nodejs',
  matcher: [
    // Apply to all API routes
    '/api/:path*',
  ],
};
