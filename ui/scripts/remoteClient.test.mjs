import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import ts from 'typescript';

import {
  RemoteClientError,
  remoteJson,
  remotePollingJson,
  RemotePollUnavailableError,
  getRemoteBackgroundPollEligibility,
  noteRemoteBackgroundPollFailure,
  resetRemoteBackgroundPollingStateForTests,
  runRemoteBackgroundPoll,
  withoutRemoteRedirects,
} from '../dist/src/server/remoteClient.js';

test('remote JSON deadline includes stalled response bodies and rejects pre-aborted requests', async () => {
  installRemotePolicyDb();
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    return new Response(new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode('{')); } }));
  };
  // Keep the test process alive while the deliberately unref'ed transport timer expires.
  const keepAlive = setTimeout(() => undefined, 1_000);
  try {
    await assert.rejects(remoteJson(makeWorker(), '/stall', { timeoutMs: 25 }), /deadline/);
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(remoteJson(makeWorker(), '/cancel', { timeoutMs: 25, signal: controller.signal }));
    assert.equal(calls, 1);
  } finally { clearTimeout(keepAlive); }
});

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

test('remote metrics route accepts multi-series histories above the generic 8 MiB limit', async () => {
  installRemotePolicyDb();
  const keys = Array.from({ length: 32 }, (_, i) => `train/metric_${i}`);
  const points = Array.from({ length: 4000 }, (_, step) => ({
    step, wall_time: 1792700000.123, value: 0.1234567890123456, value_text: null,
  }));
  const payload = {
    keys,
    keyInfo: keys.map(key => ({ key })),
    series: Object.fromEntries(keys.map(key => [key, {
      key, totalCount: 4000, firstStep: 0, lastStep: 3999,
      latest: points.at(-1), downsampled: false, points,
    }])),
  };
  const body = JSON.stringify(payload);
  assert.ok(Buffer.byteLength(body) > 8 * 1024 * 1024);
  const requests = [];
  globalThis.fetch = async (url, init) => {
    requests.push({ url: String(url), init });
    return new Response(body);
  };
  await assert.rejects(remoteJson(makeWorker(), '/ordinary-json'), /8388608 byte limit/);

  const code = ts.transpileModule(
    fs.readFileSync(new URL('../src/app/api/jobs/[jobID]/metrics/route.ts', import.meta.url), 'utf8'),
    { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true } },
  ).outputText;
  const dependencies = {
    '../../../../../utils/jobIdentity': {},
    'next/server': { NextResponse: Response },
    path,
    '@/server/db': { db: { jobs: { findById: async () => ({ worker_id: 'worker-1', remote_job_id: 'remote/job' }) } } },
    '@/server/trainingPaths': {},
    '@/server/remoteClient': { isLocalWorker: () => false, getRemoteWorker: async () => makeWorker(), remotePollingJson, RemotePollUnavailableError, isRemoteJobMissingError: () => false },
  };
  const loadedModule = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency ${name}`);
    return dependencies[name];
  }, loadedModule, loadedModule.exports);
  const query = 'keys=train%2F*&max_points=4000&since_steps=train%252Fmetric_0%3A-1';
  const response = await loadedModule.exports.GET(new Request(`http://localhost/api/jobs/mirror/metrics?${query}`), {
    params: Promise.resolve({ jobID: 'mirror' }),
  });
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.deepEqual(result.keys, keys);
  for (const key of keys) {
    assert.deepEqual(result.series[key], payload.series[key]);
  }
  assert.equal(requests.at(-1).url, `https://worker.example/api/jobs/remote%2Fjob/metrics?${query}`);
  assert.equal(Object.hasOwn(requests.at(-1).init, 'maxResponseBytes'), false);
});

test('remote JSON custom limit bounds chunked bodies and cancels oversized responses', async () => {
  installRemotePolicyDb();
  let cancelled = false;
  globalThis.fetch = async () => new Response(new ReadableStream({
    pull(controller) { controller.enqueue(new TextEncoder().encode(' '.repeat(1024))); },
    cancel() { cancelled = true; },
  }));
  await assert.rejects(remoteJson(makeWorker(), '/bounded', { maxResponseBytes: 2048 }), /2048 byte limit/);
  assert.equal(cancelled, true);
});

test('remote JSON rejects invalid byte limits before making a request', async () => {
  globalThis.fetch = async () => { assert.fail('Invalid limit must not initiate a request'); };
  for (const maxResponseBytes of [NaN, Infinity, 0, -1, 1.5]) {
    await assert.rejects(remoteJson(makeWorker(), '/invalid', { maxResponseBytes }), /Invalid remote JSON response byte limit/);
  }
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

test('worker outage logs are shared across jobs, quiet during cooldown, and reset after recovery', async t => {
  installRemotePolicyDb();
  const worker = makeWorker();
  let now = 1_000_000;
  t.mock.method(Date, 'now', () => now);
  const warnings = t.mock.method(console, 'warn', () => {});
  const info = t.mock.method(console, 'info', () => {});
  const failure = status => new RemoteClientError(`Worker failed for job-${now}`, status, '<html>Cloudflare Tunnel error</html>');
  noteRemoteBackgroundPollFailure(worker, failure(530), 'job sync one');
  noteRemoteBackgroundPollFailure(worker, failure(502), 'queue sync');
  for (let i = 0; i < 5; i++) {
    now += 5000;
    for (const feature of ['job sync one', 'job sync two', 'queue sync', 'job discovery']) {
      assert.equal((await getRemoteBackgroundPollEligibility(worker, feature)).allowed, false);
    }
  }
  assert.equal(warnings.mock.callCount(), 1);
  assert.equal(info.mock.callCount(), 0);
  now += 6000;
  noteRemoteBackgroundPollFailure(worker, failure(530), 'samples');
  assert.equal(warnings.mock.callCount(), 1, 'retry failures must not restart the log interval');
  now += 300_000;
  noteRemoteBackgroundPollFailure(worker, failure(502), 'metrics');
  assert.equal(warnings.mock.callCount(), 2);
  assert.match(warnings.mock.calls[1].arguments[0], /suppressed/);
  now += 31_000;
  assert.deepEqual(await runRemoteBackgroundPoll(worker, 'samples', async () => 'ok'), { skipped: false, value: 'ok' });
  noteRemoteBackgroundPollFailure(worker, failure(530), 'samples');
  assert.equal(warnings.mock.callCount(), 3, 'a new outage after recovery must be reported');
});

function loadPollingRoute(kind) {
  const dependencies = {
    '../../../../../utils/jobIdentity': {},
    'next/server': { NextResponse: Response }, path,
    '@/server/db': { db: { jobs: { findById: async () => ({ id: 'mirror', worker_id: 'worker-1', remote_job_id: 'remote-job' }) } } },
    '@/server/trainingPaths': {}, '@/server/remoteAssets': {}, '@/server/jobSamples': {}, '@/utils/sampleMetadata': {},
    '@/server/remoteClient': {
      isLocalWorker: () => false, getRemoteWorker: async () => makeWorker(),
      remotePollingJson, RemotePollUnavailableError, isRemoteJobMissingError: () => false,
    },
  };
  const code = ts.transpileModule(
    fs.readFileSync(new URL(`../src/app/api/jobs/[jobID]/${kind}/route.ts`, import.meta.url), 'utf8'),
    { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true } },
  ).outputText;
  const loadedModule = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency ${name}`);
    return dependencies[name];
  }, loadedModule, loadedModule.exports);
  return loadedModule.exports.GET;
}

test('sample and metric routes return failed polls without HTML dumps or repeated network requests', async t => {
  installRemotePolicyDb();
  const warnings = t.mock.method(console, 'warn', () => {});
  const errors = t.mock.method(console, 'error', () => {});
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    return new Response('<html>Cloudflare Tunnel error secret-body</html>', { status: 530 });
  };
  for (const kind of ['samples', 'metrics', 'samples', 'metrics']) {
    const response = await loadPollingRoute(kind)(new Request(`http://localhost/api/jobs/mirror/${kind}`), {
      params: Promise.resolve({ jobID: 'mirror' }),
    });
    assert.equal(response.status, 502, 'outages must not look like empty successful data');
    assert.deepEqual(await response.json(), { error: `Error reading remote ${kind}` });
  }
  assert.equal(calls, 1);
  assert.equal(warnings.mock.callCount(), 1);
  assert.equal(errors.mock.callCount(), 0);
  assert.doesNotMatch(warnings.mock.calls[0].arguments[0], /<html>|secret-body/);
});

test('polling preserves missing-job and authentication errors and does not cool down canceled requests', async t => {
  installRemotePolicyDb();
  const warnings = t.mock.method(console, 'warn', () => {});
  for (const status of [404, 401]) {
    globalThis.fetch = async () => new Response('{"error":"Job not found"}', { status });
    await assert.rejects(remotePollingJson(makeWorker(), '/samples', 'samples'), error => error instanceof RemoteClientError && error.status === status);
    assert.equal((await getRemoteBackgroundPollEligibility(makeWorker())).allowed, true);
  }
  const controller = new AbortController();
  globalThis.fetch = async () => {
    controller.abort();
    throw controller.signal.reason;
  };
  await assert.rejects(remotePollingJson(makeWorker(), '/samples', 'samples', { signal: controller.signal }), { name: 'AbortError' });
  assert.equal((await getRemoteBackgroundPollEligibility(makeWorker())).allowed, true);
  assert.equal(warnings.mock.callCount(), 0);
});
