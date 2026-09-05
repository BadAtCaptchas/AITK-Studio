import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { NextRequest, NextResponse } from 'next/server';
import {
  AUTH_SESSION_COOKIE_NAME,
  constantTimeStringEqual,
  createAuthSessionValue,
  isRequestAuthenticated,
  isSecureRequest,
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

export async function POST(request: NextRequest) {
  const expectedToken = process.env.AI_TOOLKIT_AUTH;
  if (!expectedToken) {
    return noStore(NextResponse.json({ isAuthenticated: true }));
  }

  let body: unknown;
  try {
    body = assertGlobalPayload(await request.json());
  } catch {
    return noStore(NextResponse.json({ error: 'Invalid request body' }, { status: 400 }));
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

export async function DELETE(request: NextRequest) {
  const response = NextResponse.json({ isAuthenticated: false });
  response.cookies.set(AUTH_SESSION_COOKIE_NAME, '', {
    ...cookieOptions(request),
    expires: new Date(0),
  });
  return noStore(response);
}
