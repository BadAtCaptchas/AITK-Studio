import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { readJsonCommand, commandError, withCommandBoundary } from '@/server/commandInput';
import { allowLogin } from '@/server/loginLimiter';
import { NextRequest, NextResponse } from 'next/server';
import {
  AUTH_SESSION_COOKIE_NAME,
  constantTimeStringEqual,
  createAuthSessionValue,
  isRequestAuthenticated,
  isSecureRequest,
  revokeAuthSession,
  revokeAllAuthSessions,
} from '@/utils/authSession';

function cookieOptions(request: NextRequest) {
  return {
    httpOnly: true,
    sameSite: 'strict' as const,
    secure: isSecureRequest(request),
    path: '/',
  };
}

function noStore(response: NextResponse) {
  response.headers.set('Cache-Control', 'no-store');
  return response;
}

export async function GET(request: NextRequest) {
  const expectedToken = process.env.AI_TOOLKIT_AUTH;
  if (!(await isRequestAuthenticated(request, expectedToken))) {
    return noStore(NextResponse.json({ error: 'Unauthorized' }, { status: 401 }));
  }
  return noStore(NextResponse.json({ isAuthenticated: true }));
}

async function postCommand(request: NextRequest) {
  const expectedToken = process.env.AI_TOOLKIT_AUTH;
  if (!expectedToken) {
    return noStore(NextResponse.json({ isAuthenticated: true }));
  }
  if (!(await allowLogin(request.headers))) return noStore(NextResponse.json({ error: 'Too many login attempts. Retry in a minute.' }, { status: 429, headers: { 'Retry-After': '60' } }));

  let body: unknown;
  try {
    body = await readJsonCommand(request, { maxBytes: 4096 });
  } catch (error) {
    return noStore(NextResponse.json({ error: 'Invalid request body' }, { status: commandError(error)?.status ?? 400 }));
  }

  const tokenValue =
    body && typeof body === 'object' && !Array.isArray(body) ? (body as Record<string, unknown>).token : null;
  const token = typeof tokenValue === 'string' ? tokenValue : '';
  if (!token || !constantTimeStringEqual(token, expectedToken)) {
    return noStore(NextResponse.json({ error: 'Unauthorized' }, { status: 401 }));
  }

  const response = NextResponse.json({ isAuthenticated: true });
  response.cookies.set(AUTH_SESSION_COOKIE_NAME, await createAuthSessionValue(expectedToken), cookieOptions(request));
  return noStore(response);
}

async function deleteCommand(request: NextRequest) {
  const secret = process.env.AI_TOOLKIT_AUTH;
  if (secret) {
    if (!(await isRequestAuthenticated(request, secret))) return noStore(NextResponse.json({ error: 'Unauthorized' }, { status: 401 }));
    if (request.nextUrl.searchParams.get('all') === '1') await revokeAllAuthSessions(secret);
    else await revokeAuthSession(request.headers, secret);
  }
  const response = NextResponse.json({ isAuthenticated: false });
  response.cookies.set(AUTH_SESSION_COOKIE_NAME, '', {
    ...cookieOptions(request),
    expires: new Date(0),
  });
  return noStore(response);
}

export const POST = withCommandBoundary(postCommand);
export const DELETE = withCommandBoundary(deleteCommand);
