import { createHmac, randomBytes } from 'crypto';
import { db } from '../server/db';
import { AUTH_SESSION_COOKIE_NAME } from './authConstants';
export { AUTH_SESSION_COOKIE_NAME, LEGACY_AUTH_STORAGE_KEY } from './authConstants';

const ABSOLUTE_SESSION_MS = 24 * 60 * 60 * 1000;
const IDLE_SESSION_MS = 2 * 60 * 60 * 1000;
type Session = { issuedAt: number; expiresAt: number; idleExpiresAt: number; generation: string };
function sessionRecord(value: unknown): value is Session {
  return value !== null && typeof value === 'object' && 'issuedAt' in value && typeof value.issuedAt === 'number' &&
    'expiresAt' in value && typeof value.expiresAt === 'number' && 'idleExpiresAt' in value && typeof value.idleExpiresAt === 'number' &&
    'generation' in value && typeof value.generation === 'string';
}
function keyFor(secret: string, value: string): string {
  return `session:${createHmac('sha256', secret).update(`aitk-session-v2:${value}`).digest('hex')}`;
}
async function generation(secret: string): Promise<string> {
  const row = await db.runtime.get(keyFor(secret, 'generation'));
  return row && typeof row.value === 'string' ? row.value : 'initial';
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
  const token = `v2.${randomBytes(32).toString('base64url')}`;
  const now = Date.now();
  const created = await db.runtime.compareAndSwap(keyFor(secret, token), null, {
    issuedAt: now, expiresAt: now + ABSOLUTE_SESSION_MS, idleExpiresAt: now + IDLE_SESSION_MS, generation: await generation(secret),
  } satisfies Session);
  if (!created) throw new Error('Session could not be created; retry');
  return token;
}

export async function revokeAuthSession(headers: Headers, secret: string): Promise<void> {
  const token = getCookieValue(headers, AUTH_SESSION_COOKIE_NAME);
  if (token) await db.runtime.delete(keyFor(secret, token));
}
export async function revokeAllAuthSessions(secret: string): Promise<void> {
  const key = keyFor(secret, 'generation');
  for (let retry = 0; retry < 10; retry++) {
    const row = await db.runtime.get(key);
    if (await db.runtime.compareAndSwap(key, row?.version ?? null, randomBytes(16).toString('hex'))) return;
  }
  throw new Error('Session revocation is busy; retry');
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
  if (!sessionValue || !/^v2\.[A-Za-z0-9_-]{43}$/.test(sessionValue)) return false;
  const key = keyFor(expectedToken, sessionValue);
  for (let retry = 0; retry < 3; retry++) {
    const row = await db.runtime.get(key);
    if (!row || !sessionRecord(row.value)) return false;
    const now = Date.now();
    const session = row.value;
    if (session.expiresAt <= now || session.idleExpiresAt <= now || session.generation !== await generation(expectedToken)) {
      await db.runtime.delete(key, row.version);
      return false;
    }
    if (session.idleExpiresAt - now > IDLE_SESSION_MS - 60_000) return true;
    if (await db.runtime.compareAndSwap(key, row.version, { ...session, idleExpiresAt: Math.min(session.expiresAt, now + IDLE_SESSION_MS) })) return true;
  }
  return false;
}

export function isSecureRequest(request: Pick<Request, 'headers' | 'url'>) {
  const forwardedProtocol = request.headers.get('x-aitk-forwarded-proto');
  if (forwardedProtocol) return forwardedProtocol === 'https';
  return new URL(request.url).protocol === 'https:';
}
