import assert from 'node:assert/strict';
import test from 'node:test';

import {
  AUTH_SESSION_COOKIE_NAME,
  createAuthSessionValue,
  getBearerToken,
  isRequestAuthenticated,
  isSecureRequest,
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

test('session credentials are opaque, deterministic, and tied to the configured token', async () => {
  const session = await createAuthSessionValue('secret-token');
  assert.equal(session.includes('secret-token'), false);
  assert.equal(session, await createAuthSessionValue('secret-token'));

  const cookieHeader = `other=value; ${AUTH_SESSION_COOKIE_NAME}=${session}`;
  assert.equal(await isRequestAuthenticated(request({ Cookie: cookieHeader }), 'secret-token'), true);
  assert.equal(await isRequestAuthenticated(request({ Cookie: cookieHeader }), 'rotated-token'), false);
});

test('unconfigured auth is open and secure cookie detection honors proxy protocol', async () => {
  assert.equal(await isRequestAuthenticated(request(), null), true);
  assert.equal(isSecureRequest(request({}, 'https://localhost/api/auth')), true);
  assert.equal(isSecureRequest(request({ 'X-Forwarded-Proto': 'https' })), true);
  assert.equal(isSecureRequest(request({ 'X-Forwarded-Proto': 'http' }, 'https://localhost/api/auth')), false);
});
