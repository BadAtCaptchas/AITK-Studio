import assert from 'node:assert/strict';
import test from 'node:test';
import { installMemoryRuntime } from './memoryRuntimeFixture.mjs';
installMemoryRuntime();

import {
  AUTH_SESSION_COOKIE_NAME,
  createAuthSessionValue,
  getBearerToken,
  isRequestAuthenticated,
  isSecureRequest,
  revokeAuthSession,
  revokeAllAuthSessions,
} from '../dist/src/utils/authSession.js';

function request(headers = {}, url = 'http://localhost/api/jobs') {
  return new Request(url, { headers });
}

test('central auth accepts exact bearer credentials and rejects malformed values', async () => {
  assert.equal(await isRequestAuthenticated(request({ Authorization: 'Bearer secret-token' }), 'secret-token'), true);
  assert.equal(await isRequestAuthenticated(request({ Authorization: 'Basic secret-token' }), 'secret-token'), false);
  assert.equal(await isRequestAuthenticated(request({ Authorization: 'Bearer wrong-token' }), 'secret-token'), false);
  assert.equal(getBearerToken(new Headers({ Authorization: 'bearer secret-token' })), 'secret-token');
});

test('session credentials are random, opaque, and tied to the configured token', async () => {
  const session = await createAuthSessionValue('secret-token');
  assert.equal(session.includes('secret-token'), false);
  assert.notEqual(session, await createAuthSessionValue('secret-token'));

  const cookieHeader = `other=value; ${AUTH_SESSION_COOKIE_NAME}=${session}`;
  assert.equal(await isRequestAuthenticated(request({ Cookie: cookieHeader }), 'secret-token'), true);
  assert.equal(await isRequestAuthenticated(request({ Cookie: cookieHeader }), 'rotated-token'), false);
});

test('unconfigured auth is open and secure cookie detection ignores untrusted proxy protocol', async () => {
  assert.equal(await isRequestAuthenticated(request(), null), true);
  assert.equal(isSecureRequest(request({}, 'https://localhost/api/auth')), true);
  assert.equal(isSecureRequest(request({ 'X-Forwarded-Proto': 'https' })), false);
  assert.equal(isSecureRequest(request({ 'X-Forwarded-Proto': 'http' }, 'https://localhost/api/auth')), true);
  assert.equal(isSecureRequest(request({ 'x-aitk-forwarded-proto': 'https' })), true);
});

test('logout, logout all, idle expiry and legacy cookies are enforced server-side', async () => {
  const secret = 'revocation-secret';
  const headers = async () => new Headers({ Cookie: `${AUTH_SESSION_COOKIE_NAME}=${await createAuthSessionValue(secret)}` });
  const first = await headers(), second = await headers();
  await revokeAuthSession(first, secret);
  assert.equal(await isRequestAuthenticated({ headers: first }, secret), false);
  assert.equal(await isRequestAuthenticated({ headers: second }, secret), true);
  await revokeAllAuthSessions(secret);
  assert.equal(await isRequestAuthenticated({ headers: second }, secret), false);
  const expiring = await headers(), originalNow = Date.now;
  try {
    const future = originalNow() + 2 * 60 * 60 * 1000 + 1;
    Date.now = () => future;
    assert.equal(await isRequestAuthenticated({ headers: expiring }, secret), false);
  } finally { Date.now = originalNow; }
  assert.equal(await isRequestAuthenticated(request({ Cookie: `${AUTH_SESSION_COOKIE_NAME}=v1.fake` }), secret), false);
});
