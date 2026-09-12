import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const dbModule = require('../dist/src/server/db.js');
const {
  OFFLINE_ALLOWED_HOSTS_ENV,
  OfflineModeError,
  assertUrlAllowedByOfflineMode,
  guardedFetch,
  isLocalPrivateIp,
} = require('../dist/src/server/networkPolicy.js');

const originalWorkerNodes = dbModule.db.workerNodes;
const originalSettings = dbModule.db.settings;
const originalFetch = globalThis.fetch;
const envKeys = ['AITK_OFFLINE_MODE', 'AI_TOOLKIT_OFFLINE_MODE', OFFLINE_ALLOWED_HOSTS_ENV];
const originalEnv = Object.fromEntries(envKeys.map(key => [key, process.env[key]]));

function restoreEnv() {
  for (const key of envKeys) {
    if (originalEnv[key] === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = originalEnv[key];
    }
  }
}

function installPolicyDb({ workers = [], remoteOllamaWorkers = null } = {}) {
  dbModule.db.workerNodes = {
    async list() {
      return workers;
    },
  };
  dbModule.db.settings = {
    async get(key) {
      if (key === 'REMOTE_OLLAMA_WORKERS' && remoteOllamaWorkers != null) {
        return { key, value: JSON.stringify(remoteOllamaWorkers) };
      }
      return null;
    },
  };
}

afterEach(() => {
  dbModule.db.workerNodes = originalWorkerNodes;
  dbModule.db.settings = originalSettings;
  globalThis.fetch = originalFetch;
  restoreEnv();
});

test('local/private IP classifier allows only local and private address space', () => {
  assert.equal(isLocalPrivateIp('127.0.0.1'), true);
  assert.equal(isLocalPrivateIp('10.10.0.5'), true);
  assert.equal(isLocalPrivateIp('172.31.0.5'), true);
  assert.equal(isLocalPrivateIp('192.168.1.5'), true);
  assert.equal(isLocalPrivateIp('::1'), true);
  assert.equal(isLocalPrivateIp('8.8.8.8'), false);
  assert.equal(isLocalPrivateIp('1.1.1.1'), false);
});

test('offline mode blocks public IP URLs and allows local/private URLs', async () => {
  process.env.AITK_OFFLINE_MODE = '1';
  installPolicyDb();

  await assert.doesNotReject(() => assertUrlAllowedByOfflineMode('http://127.0.0.1:8675/api', 'local test'));
  await assert.doesNotReject(() => assertUrlAllowedByOfflineMode('http://192.168.1.50/api', 'private test'));
  await assert.rejects(
    () => assertUrlAllowedByOfflineMode('https://8.8.8.8/api', 'public test'),
    error => error instanceof OfflineModeError && /outside local\/private IP space/.test(error.message),
  );
});

test('offline mode allows exact hostnames from env and worker bypass records', async () => {
  process.env.AITK_OFFLINE_MODE = '1';
  process.env[OFFLINE_ALLOWED_HOSTS_ENV] = 'env-worker.example';
  installPolicyDb({
    workers: [{ base_url: 'https://secure-worker.example/api', offline_bypass_enabled: true }],
    remoteOllamaWorkers: [{ base_url: 'https://ollama-worker.example', offline_bypass_enabled: true }],
  });

  await assert.doesNotReject(() => assertUrlAllowedByOfflineMode('https://env-worker.example/api', 'env worker'));
  await assert.doesNotReject(() =>
    assertUrlAllowedByOfflineMode('https://secure-worker.example/jobs', 'secure worker'),
  );
  await assert.doesNotReject(() =>
    assertUrlAllowedByOfflineMode('https://ollama-worker.example/api/tags', 'Ollama worker'),
  );
});

test('guardedFetch validates redirects before following them in offline mode', async () => {
  process.env.AITK_OFFLINE_MODE = '1';
  installPolicyDb();
  const calls = [];

  globalThis.fetch = async url => {
    calls.push(url instanceof Request ? url.url : String(url));
    return new Response('', {
      status: 302,
      headers: { Location: 'https://8.8.8.8/escaped' },
    });
  };

  await assert.rejects(
    () => guardedFetch('http://127.0.0.1/start', undefined, 'redirect test'),
    error => error instanceof OfflineModeError && /redirect/.test(error.message),
  );
  assert.deepEqual(calls, ['http://127.0.0.1/start']);
});

test('guardedFetch follows allowed redirects in offline mode', async () => {
  process.env.AITK_OFFLINE_MODE = '1';
  installPolicyDb();
  const calls = [];

  globalThis.fetch = async url => {
    calls.push(url instanceof Request ? url.url : String(url));
    if (calls.length === 1) {
      return new Response('', {
        status: 302,
        headers: { Location: 'http://127.0.0.1/final' },
      });
    }
    return new Response('ok', { status: 200 });
  };

  const response = await guardedFetch('http://127.0.0.1/start', undefined, 'redirect test');

  assert.equal(response.status, 200);
  assert.equal(await response.text(), 'ok');
  assert.deepEqual(calls, ['http://127.0.0.1/start', 'http://127.0.0.1/final']);
});

test('offline redirects cannot disclose credentials or replay POST bodies across origins', async () => {
  process.env.AITK_OFFLINE_MODE = '1';
  installPolicyDb();
  for (const status of [302, 303, 307, 308]) {
    let calls = 0;
    globalThis.fetch = async () => {
      calls++;
      return new Response('', { status, headers: { location: 'http://127.0.0.1:19002/target' } });
    };
    await assert.rejects(() => guardedFetch(new Request('http://127.0.0.1:19001/source', {
      method: 'POST', headers: { authorization: 'Bearer synthetic' }, body: 'synthetic body',
    })), /cross-origin/);
    assert.equal(calls, 1);
  }
});

test('Request manual policy and HEAD method survive offline handling', async () => {
  process.env.AITK_OFFLINE_MODE = '1';
  installPolicyDb();
  const seen = [];
  globalThis.fetch = async input => {
    seen.push(input.method);
    return new Response(null, { status: seen.length === 1 ? 303 : 200, headers: { location: '/final' } });
  };
  await guardedFetch(new Request('http://127.0.0.1/source', { method: 'HEAD' }));
  assert.deepEqual(seen, ['HEAD', 'HEAD']);
  seen.length = 0;
  const response = await guardedFetch(new Request('http://127.0.0.1/source', { redirect: 'manual' }));
  assert.equal(response.status, 303);
  assert.equal(seen.length, 1);
});
