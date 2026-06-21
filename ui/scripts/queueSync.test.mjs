import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { createRequire } from 'node:module';

import { listQueuesForQueueApi } from '../dist/src/server/queueSync.js';
import { resetRemoteBackgroundPollingStateForTests } from '../dist/src/server/remoteClient.js';

const require = createRequire(import.meta.url);
const dbModule = require('../dist/src/server/db.js');

const originalFetch = globalThis.fetch;
const originalWorkerNodes = dbModule.db.workerNodes;
const originalSettings = dbModule.db.settings;
const originalQueues = dbModule.db.queues;
const envKeys = ['AITK_OFFLINE_MODE', 'AI_TOOLKIT_OFFLINE_MODE', 'AITK_OFFLINE_ALLOWED_HOSTS'];
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

function makeWorker() {
  const now = new Date();
  return {
    id: 'worker-1',
    name: 'Remote One',
    base_url: 'https://public-worker.example',
    api_token: 'token',
    enabled: true,
    offline_bypass_enabled: false,
    last_status: 'ready',
    last_error: null,
    last_checked_at: null,
    capabilities: '{}',
    gpus: '[]',
    created_at: now,
    updated_at: now,
  };
}

afterEach(() => {
  globalThis.fetch = originalFetch;
  dbModule.db.workerNodes = originalWorkerNodes;
  dbModule.db.settings = originalSettings;
  dbModule.db.queues = originalQueues;
  resetRemoteBackgroundPollingStateForTests();
  restoreEnv();
});

test('queue API helper returns local queues when offline mode skips remote queue sync', async () => {
  process.env.AITK_OFFLINE_MODE = '1';
  const worker = makeWorker();
  const queues = [
    { id: 1, worker_id: 'local', gpu_ids: '0', is_running: true },
    { id: 2, worker_id: worker.id, gpu_ids: '0', is_running: false },
  ];
  let fetchCalls = 0;
  let queueWrites = 0;

  globalThis.fetch = async () => {
    fetchCalls += 1;
    throw new Error('fetch must not be called while offline polling is skipped');
  };
  dbModule.db.workerNodes = {
    async list() {
      return [worker];
    },
    async findById(id) {
      return id === worker.id ? worker : null;
    },
  };
  dbModule.db.settings = {
    async get() {
      return null;
    },
  };
  dbModule.db.queues = {
    async list(order) {
      assert.equal(order, 'gpu_ids');
      return queues;
    },
    async findByGpuIds() {
      throw new Error('remote queues should not be read when sync is skipped');
    },
    async update() {
      queueWrites += 1;
      throw new Error('remote queues should not be updated when sync is skipped');
    },
    async create() {
      queueWrites += 1;
      throw new Error('remote queues should not be created when sync is skipped');
    },
  };

  const result = await listQueuesForQueueApi();

  assert.deepEqual(result, queues);
  assert.equal(fetchCalls, 0);
  assert.equal(queueWrites, 0);
});
