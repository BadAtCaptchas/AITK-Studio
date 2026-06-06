import { NextResponse } from 'next/server';

function getBearerToken(request: Request) {
  const authHeader = request.headers.get('authorization') || '';
  const [scheme, token] = authHeader.split(' ');
  return scheme?.toLowerCase() === 'bearer' ? token : undefined;
}

export function requireApiAuth(request: Request): NextResponse | null {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) {
    return NextResponse.json(
      { error: 'AI_TOOLKIT_AUTH is required before using privileged job APIs' },
      { status: 403 },
    );
  }

  if (getBearerToken(request) !== tokenToUse) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  return null;
}

export function optionalApiAuth(request: Request): NextResponse | null {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) {
    return null;
  }

  if (getBearerToken(request) !== tokenToUse) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  return null;
}
