import assert from 'node:assert/strict';
import { test } from 'node:test';
import { installMemoryRuntime } from './memoryRuntimeFixture.mjs';
import { readJsonCommand, parseJobStartCommand, withCommandBoundary } from '../dist/src/server/commandInput.js';
import { ingressConfig, browserRequestError, trustedForwardHeaders } from '../dist/src/server/ingressPolicy.js';
import {
  signRemoteDatasetAsset,
  isRemoteDatasetAssetSignatureValid,
} from '../dist/src/server/remoteDatasetAssetAccess.js';
import { allowLogin } from '../dist/src/server/loginLimiter.js';
installMemoryRuntime();

function jsonRequest(body, headers = {}) {
  return new Request('http://localhost/api/command', {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body,
  });
}

test('one bounded parse is reused and invalid commands cannot reach the handler', async () => {
  const req = jsonRequest('{"value":1}');
  assert.equal(await readJsonCommand(req), await readJsonCommand(req));
  let called = 0;
  const handler = withCommandBoundary(async request => {
    called++;
    return Response.json(await readJsonCommand(request));
  });
  for (const [body, status] of [
    ['{', 400],
    ['[]', 400],
    ['{"project_id":"old"}', 400],
    [JSON.stringify({ x: 'x'.repeat(2 * 1024 * 1024) }), 413],
  ]) {
    assert.equal((await handler(jsonRequest(body))).status, status);
  }
  assert.equal(called, 0);
  assert.throws(() => parseJobStartCommand({ background: 'true' }));
  assert.throws(() => parseJobStartCommand({ encryptedDatasetKeys: [{ datasetPath: 'd', keyB64: 'short' }] }));
  assert.throws(() => parseJobStartCommand({ accidentalField: true }));
});

test('streamed byte and complexity limits apply without Content-Length', async () => {
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('{"data":"' + 'x'.repeat(1000) + '"}'));
      controller.close();
    },
  });
  const req = new Request('http://localhost/api/command', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body,
    duplex: 'half',
  });
  await assert.rejects(readJsonCommand(req, { maxBytes: 100 }), error => error.status === 413);
  await assert.rejects(
    readJsonCommand(jsonRequest('{"nested":' + '['.repeat(34) + '0' + ']'.repeat(34) + '}')),
    error => error.status === 413,
  );
});

test('network binding requires explicit mode and authentication', () => {
  assert.equal(ingressConfig({}).host, '127.0.0.1');
  assert.throws(() => ingressConfig({ AITK_BIND_HOST: '0.0.0.0' }));
  assert.throws(() => ingressConfig({ AITK_BIND_HOST: '0.0.0.0', AITK_NETWORK_MODE: '1' }));
  assert.equal(
    ingressConfig({ AITK_BIND_HOST: '0.0.0.0', AITK_NETWORK_MODE: '1', AI_TOOLKIT_AUTH: 'test' }).network,
    true,
  );
});

test('ingress rejects rebinding/cross-site commands and strips spoofed client identity', () => {
  assert.equal(
    browserRequestError(new Headers({ host: 'evil.example' }), 'http://localhost/', 'GET'),
    'Host is not allowed',
  );
  assert.equal(
    browserRequestError(
      new Headers({ host: 'localhost', origin: 'https://evil.example' }),
      'http://localhost/',
      'POST',
    ),
    'Origin is not allowed',
  );
  assert.equal(
    browserRequestError(new Headers({ host: 'localhost', cookie: 'session=x' }), 'http://localhost/', 'POST'),
    'Origin is required for browser commands',
  );
  const forwarded = trustedForwardHeaders({
    headers: {
      host: 'localhost',
      'x-aitk-client-ip': '8.8.8.8',
      'x-forwarded-for': '8.8.4.4',
      'x-forwarded-proto': 'https',
    },
    socket: { remoteAddress: '127.0.0.1' },
  });
  assert.equal(forwarded['x-aitk-client-ip'], '127.0.0.1');
  assert.equal(forwarded['x-aitk-forwarded-proto'], 'http');
  assert.equal(forwarded['x-forwarded-for'], undefined);
});

test('asset capabilities bind representation, read method, worker, path and expiry', () => {
  const original = process.env.AI_TOOLKIT_AUTH;
  process.env.AI_TOOLKIT_AUTH = 'asset-test-key';
  try {
    const signed = signRemoteDatasetAsset('worker', '/root/image.png', 'img');
    const valid = (worker = 'worker', pathname = '/root/image.png', type = 'img', method = 'GET') =>
      isRemoteDatasetAssetSignatureValid(worker, pathname, signed.expires, signed.signature, type, method);
    assert.equal(valid(), true);
    assert.equal(valid('worker', '/root/image.png', 'img', 'HEAD'), true);
    assert.equal(valid('other'), false);
    assert.equal(valid('worker', '/other'), false);
    for (const type of ['file', 'audio-art']) assert.equal(valid('worker', '/root/image.png', type), false);
    for (const method of ['POST', 'OPTIONS']) assert.equal(valid('worker', '/root/image.png', 'img', method), false);
    assert.equal(
      isRemoteDatasetAssetSignatureValid('worker', '/root/image.png', Date.now() - 1, signed.signature, 'img'),
      false,
    );
  } finally {
    if (original === undefined) delete process.env.AI_TOOLKIT_AUTH;
    else process.env.AI_TOOLKIT_AUTH = original;
  }
});

test('login limiter shares a client bucket across calls', async () => {
  const headers = new Headers({ 'x-aitk-client-ip': '192.0.2.1' });
  for (let i = 0; i < 10; i++) assert.equal(await allowLogin(headers), true);
  assert.equal(await allowLogin(headers), false);
});

test('wrong command MIME and stalled bodies fail before handlers run', async () => {
  let invoked = false;
  const handler = withCommandBoundary(async () => {
    invoked = true;
    return Response.json({ ok: true });
  });
  const invalid = new Request('http://localhost/api/jobs', {
    method: 'POST',
    headers: { 'content-type': 'text/plain' },
    body: '{}',
  });
  assert.equal((await handler(invalid)).status, 415);
  assert.equal(invoked, false);
  let canceled = false;
  const body = new ReadableStream({
    pull() {},
    cancel() {
      canceled = true;
    },
  });
  const stalled = new Request('http://localhost/api/jobs', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body,
    duplex: 'half',
  });
  await assert.rejects(readJsonCommand(stalled, { timeoutMs: 20 }), error => error.status === 408);
  assert.equal(canceled, true);
});

test('empty streamed DELETE bodies reach session revocation', async () => {
  let invoked = false;
  const handler = withCommandBoundary(async () => {
    invoked = true;
    return Response.json({ ok: true });
  });
  const request = new Request('http://localhost/api/auth', {
    method: 'DELETE',
    body: new ReadableStream({
      start(controller) {
        controller.close();
      },
    }),
    duplex: 'half',
  });
  assert.equal((await handler(request)).status, 200);
  assert.equal(invoked, true);
});
