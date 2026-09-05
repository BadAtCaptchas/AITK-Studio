import assert from 'node:assert/strict';
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import {
  configureFrontServerTimeouts,
  proxyRetryDecision,
  registerWorkerCrash,
} from '../dist/cron/fileServerPolicy.js';
import { isAcceleratedApiRequestAuthenticated, parseSingleByteRange } from '../dist/src/server/fileServing.js';
import { AUTH_SESSION_COOKIE_NAME, createAuthSessionValue } from '../dist/src/utils/authSession.js';

test('single byte ranges support bounded, open-ended, and suffix requests', () => {
  assert.deepEqual(parseSingleByteRange('bytes=10-19', 100), { start: 10, end: 19 });
  assert.deepEqual(parseSingleByteRange('bytes=90-', 100), { start: 90, end: 99 });
  assert.deepEqual(parseSingleByteRange('bytes=-10', 100), { start: 90, end: 99 });
  assert.deepEqual(parseSingleByteRange('bytes=-1000', 100), { start: 0, end: 99 });
  assert.deepEqual(parseSingleByteRange('bytes=90-1000', 100), { start: 90, end: 99 });
});

test('single byte ranges reject malformed and unsatisfiable requests', () => {
  for (const value of ['items=0-1', 'bytes=-', 'bytes=10-5', 'bytes=100-', 'bytes=-0', 'bytes=0-1,3-4']) {
    assert.equal(parseSingleByteRange(value, 100), 'invalid', value);
  }
  assert.equal(parseSingleByteRange('bytes=0-', 0), 'invalid');
  assert.equal(parseSingleByteRange(null, 100), null);
});

test('accelerated routes accept the same bearer and session authentication as Next middleware', async () => {
  const originalToken = process.env.AI_TOOLKIT_AUTH;
  process.env.AI_TOOLKIT_AUTH = 'file-serving-test-secret';
  try {
    assert.equal(await isAcceleratedApiRequestAuthenticated(new Headers()), false);
    assert.equal(
      await isAcceleratedApiRequestAuthenticated(new Headers({ Authorization: 'Bearer file-serving-test-secret' })),
      true,
    );
    const session = await createAuthSessionValue(process.env.AI_TOOLKIT_AUTH);
    assert.equal(
      await isAcceleratedApiRequestAuthenticated(new Headers({ Cookie: `${AUTH_SESSION_COOKIE_NAME}=${session}` })),
      true,
    );
  } finally {
    if (originalToken === undefined) delete process.env.AI_TOOLKIT_AUTH;
    else process.env.AI_TOOLKIT_AUTH = originalToken;
  }
});

test('front server disables Node request-body timeouts for streamed uploads', async () => {
  const server = http.createServer();
  server.requestTimeout = 1234;

  configureFrontServerTimeouts(server);

  assert.equal(server.requestTimeout, 0);
  const source = await readFile(new URL('../cron/fileServer.ts', import.meta.url), 'utf8');
  assert.match(source, /configureFrontServerTimeouts\(server\)/);
});

test('proxy retry policy only replays safe bodyless requests before a response starts', () => {
  const base = {
    method: 'GET',
    attempt: 0,
    headersSent: false,
    responseDestroyed: false,
  };
  assert.deepEqual(proxyRetryDecision({ ...base, errorCode: 'ECONNREFUSED', reusedSocket: false }), {
    retry: true,
    delayMs: 250,
  });
  for (const errorCode of ['ECONNRESET', 'EPIPE']) {
    assert.deepEqual(proxyRetryDecision({ ...base, errorCode, reusedSocket: true }), { retry: true, delayMs: 0 });
    assert.equal(proxyRetryDecision({ ...base, errorCode, reusedSocket: false }).retry, false);
  }
  assert.equal(
    proxyRetryDecision({ ...base, method: 'POST', errorCode: 'ECONNRESET', reusedSocket: true }).retry,
    false,
  );
  assert.equal(proxyRetryDecision({ ...base, headersSent: true, errorCode: 'EPIPE', reusedSocket: true }).retry, false);
  assert.equal(
    proxyRetryDecision({ ...base, attempt: 120, errorCode: 'ECONNREFUSED', reusedSocket: false }).retry,
    false,
  );
});

test('worker crash policy backs off, expires old crashes, and stops a crash loop', () => {
  const recentCrashes = [];
  const options = {
    windowMs: 30_000,
    limit: 3,
    baseDelayMs: 250,
    maximumDelayMs: 600,
  };

  assert.deepEqual(registerWorkerCrash(recentCrashes, 1_000, options), {
    shouldStop: false,
    restartDelayMs: 250,
  });
  assert.deepEqual(registerWorkerCrash(recentCrashes, 2_000, options), {
    shouldStop: false,
    restartDelayMs: 500,
  });
  assert.deepEqual(registerWorkerCrash(recentCrashes, 3_000, options), {
    shouldStop: false,
    restartDelayMs: 600,
  });
  assert.deepEqual(registerWorkerCrash(recentCrashes, 4_000, options), {
    shouldStop: true,
    restartDelayMs: 600,
  });

  assert.deepEqual(registerWorkerCrash(recentCrashes, 40_001, options), {
    shouldStop: false,
    restartDelayMs: 250,
  });
  assert.deepEqual(recentCrashes, [40_001]);
});

test('front server waits for workers and routes startup failures through cleanup', async () => {
  const source = await readFile(new URL('../cron/fileServer.ts', import.meta.url), 'utf8');

  assert.match(source, /await Promise\.all\(workers\.map\(waitForWorkerReady\)\)/);
  assert.match(source, /process\.send\(\{ type: 'aitk-file-server-ready' \}\)/);
  assert.match(
    source,
    /try \{[\s\S]*await waitForUpstream\(upstreamPort, nextChild\)[\s\S]*\} catch \(error\) \{[\s\S]*shutdown\(1\)/,
  );
  assert.match(source, /registerWorkerCrash\([\s\S]*if \(crashDecision\.shouldStop\)/);
});

test('proxy response errors use pipeline and retry listeners are removed', async () => {
  const source = await readFile(new URL('../cron/fileServer.ts', import.meta.url), 'utf8');

  assert.match(source, /pipeline\(upstreamResponse, res,/);
  assert.match(source, /const removeAbortListeners = \(\) => \{/);
  assert.match(source, /req\.off\('aborted', abortUpstream\)/);
  assert.match(source, /res\.off\('close', abortUpstreamIfIncomplete\)/);
  assert.match(source, /upstreamRequest\.once\('close', removeAbortListeners\)/);
  assert.match(source, /if \(!req\.destroyed && !res\.destroyed && !res\.writableEnded\) \{\s*proxy\(/);
});

test('accelerated file routes refuse obsolete scope before resolving files', async () => {
  const source = await readFile(new URL('../cron/fileServer.ts', import.meta.url), 'utf8');
  assert.match(source, /hasObsoleteWorkspaceScope\(requestUrl.searchParams\)/);
  assert.match(source, /hasObsoleteWorkspaceHeaders\(requestHeaders\(req.headers\)\)/);
  assert.ok(
    source.indexOf('hasObsoleteWorkspaceScope(requestUrl.searchParams)') <
      source.indexOf("requestUrl.pathname.startsWith('/api/files/')"),
  );
});
