const AUTH_SESSION_CONTEXT = 'aitk-session-v1';

export const AUTH_SESSION_COOKIE_NAME = 'aitk_session';
export const LEGACY_AUTH_STORAGE_KEY = 'AI_TOOLKIT_AUTH';

function base64Url(bytes: Uint8Array) {
  const binary = String.fromCharCode(...bytes);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export function constantTimeStringEqual(left: string, right: string) {
  if (left.length !== right.length) return false;
  let difference = 0;
  for (let index = 0; index < left.length; index += 1) {
    difference |= left.charCodeAt(index) ^ right.charCodeAt(index);
  }
  return difference === 0;
}

export function getBearerToken(headers: Headers) {
  const authorization = headers.get('authorization') || '';
  return authorization.match(/^Bearer\s+(.+)$/i)?.[1] || null;
}

export function getCookieValue(headers: Headers, name: string) {
  const cookieHeader = headers.get('cookie');
  if (!cookieHeader) return null;

  for (const cookie of cookieHeader.split(';')) {
    const separator = cookie.indexOf('=');
    if (separator < 0) continue;
    if (cookie.slice(0, separator).trim() === name) {
      return cookie.slice(separator + 1).trim() || null;
    }
  }
  return null;
}

export async function createAuthSessionValue(secret: string) {
  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  );
  const digest = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(AUTH_SESSION_CONTEXT));
  return `v1.${base64Url(new Uint8Array(digest))}`;
}

export async function isRequestAuthenticated(
  request: Pick<Request, 'headers'>,
  expectedToken: string | null | undefined,
) {
  if (!expectedToken) return true;

  const bearerToken = getBearerToken(request.headers);
  if (bearerToken && constantTimeStringEqual(bearerToken, expectedToken)) {
    return true;
  }

  const sessionValue = getCookieValue(request.headers, AUTH_SESSION_COOKIE_NAME);
  if (!sessionValue) return false;
  return constantTimeStringEqual(sessionValue, await createAuthSessionValue(expectedToken));
}

export function isSecureRequest(request: Pick<Request, 'headers' | 'url'>) {
  const forwardedProtocol = request.headers.get('x-forwarded-proto')?.split(',')[0]?.trim().toLowerCase();
  if (forwardedProtocol) return forwardedProtocol === 'https';
  return new URL(request.url).protocol === 'https:';
}
