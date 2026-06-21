import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { createRequire } from 'node:module';

import {
  RemoteClientError,
  remoteJson,
  resetRemoteBackgroundPollingStateForTests,
  runRemoteBackgroundPoll,
  withoutRemoteRedirects,
} from '../dist/src/server/remoteClient.js';

const require = createRequire(import.meta.url);
const dbModule = require('../dist/src/server/db.js');

const originalFetch = globalThis.fetch;
const originalWorkerNodes = dbModule.db.workerNodes;
const originalSettings = dbModule.db.settings;
const envKeys = ['AITK_OFFLINE_MODE', 'AI_TOOLKIT_OFFLINE_MODE', 'AITK_OFFLINE_ALLOWED_HOSTS'];
const originalEnv = Object.fromEntries(envKeys.map(key => [key, process.env[key]]));

function makeWorker(baseUrl = 'https://worker.example') {
  return {
    id: 'worker-1',
    name: 'Remote One',
    base_url: baseUrl,
    api_token: 'token',
    enabled: true,
    offline_bypass_enabled: false,
    last_status: 'ready',
    last_error: null,
    last_checked_at: null,
    capabilities: '{}',
    gpus: '[]',
    created_at: new Date(),
    updated_at: new Date(),
  };
}

function restoreEnv() {
  for (const key of envKeys) {
    if (originalEnv[key] === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = originalEnv[key];
    }
  }
}

function installRemotePolicyDb(workers = []) {
  dbModule.db.workerNodes = {
    async list() {
      return workers;
    },
    async findById(id) {
      return workers.find(worker => worker.id === id) || null;
    },
  };
  dbModule.db.settings = {
    async get() {
      return null;
    },
  };
}

afterEach(() => {
  globalThis.fetch = originalFetch;
  dbModule.db.workerNodes = originalWorkerNodes;
  dbModule.db.settings = originalSettings;
  resetRemoteBackgroundPollingStateForTests();
  restoreEnv();
});

test('remoteJson can reject a redirect before a secret-bearing POST is replayed', async () => {
  const calls = [];

  globalThis.fetch = async (url, init = {}) => {
    calls.push({
      url: String(url),
      redirect: init.redirect,
      authorization: new Headers(init.headers).get('authorization'),
    });
    return new Response('', {
      status: 307,
      headers: {
        Location: 'http://worker.example/api/datasets/combine',
      },
    });
  };

  try {
    await assert.rejects(
      () =>
        remoteJson(
          makeWorker(),
          '/api/datasets/combine',
          withoutRemoteRedirects({
            method: 'POST',
            body: JSON.stringify({
              encryptedDatasetKeys: [{ datasetName: 'locked', keyB64: 'secret-key' }],
              outputKeyB64: 'secret-output-key',
            }),
          }),
        ),
      error => {
        assert.ok(error instanceof RemoteClientError);
        assert.equal(error.status, 307);
        return true;
      },
    );

    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, 'https://worker.example/api/datasets/combine');
    assert.equal(calls[0].redirect, 'manual');
    assert.equal(calls[0].authorization, 'Bearer token');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('background polling skips an offline non-bypass public worker without running the task', async () => {
  process.env.AITK_OFFLINE_MODE = '1';
  const worker = makeWorker('https://public-worker.example');
  installRemotePolicyDb([worker]);
  let taskCalls = 0;

  const result = await runRemoteBackgroundPoll(worker, 'test discovery', async () => {
    taskCalls += 1;
    return 'should-not-run';
  });

  assert.equal(result.skipped, true);
  assert.equal(taskCalls, 0);
});

test('background polling allows an offline bypass worker', async () => {
  process.env.AITK_OFFLINE_MODE = '1';
  const worker = { ...makeWorker('https://public-worker.example'), offline_bypass_enabled: true };
  installRemotePolicyDb([worker]);
  let taskCalls = 0;

  const result = await runRemoteBackgroundPoll(worker, 'test discovery', async () => {
    taskCalls += 1;
    return 'ok';
  });

  assert.deepEqual(result, { skipped: false, value: 'ok' });
  assert.equal(taskCalls, 1);
});

test('background polling enters cooldown after a transient DNS failure', async () => {
  const worker = makeWorker('https://public-worker.example');
  installRemotePolicyDb([worker]);
  let taskCalls = 0;

  const first = await runRemoteBackgroundPoll(worker, 'test discovery', async () => {
    taskCalls += 1;
    const error = new Error('fetch failed');
    error.cause = { code: 'EAI_AGAIN' };
    throw error;
  });

  const second = await runRemoteBackgroundPoll(worker, 'test discovery', async () => {
    taskCalls += 1;
    return 'should-not-run-during-cooldown';
  });

  assert.equal(first.skipped, true);
  assert.match(first.reason, /EAI_AGAIN/);
  assert.equal(second.skipped, true);
  assert.match(second.reason, /cooling down/);
  assert.equal(taskCalls, 1);
});
