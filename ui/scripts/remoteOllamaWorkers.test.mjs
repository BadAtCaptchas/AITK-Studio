import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const dbModule = require('../dist/src/server/db.js');
const {
  REMOTE_OLLAMA_WORKERS_SETTING_KEY,
  checkRemoteOllamaWorker,
  deleteRemoteOllamaWorker,
  listRemoteOllamaWorkerModels,
  listRemoteOllamaWorkers,
  saveRemoteOllamaWorker,
  toPublicRemoteOllamaWorker,
} = require('../dist/src/server/remoteOllamaWorkers.js');

const originalSettings = dbModule.db.settings;
const originalFetch = globalThis.fetch;

afterEach(() => {
  dbModule.db.settings = originalSettings;
  globalThis.fetch = originalFetch;
});

function installSettingsStore() {
  const store = new Map();
  dbModule.db.settings = {
    async get(key) {
      return store.has(key) ? { key, value: store.get(key) } : null;
    },
    async upsert(key, value) {
      store.set(key, value);
      return { key, value };
    },
    async list() {
      return [...store.entries()].map(([key, value]) => ({ key, value }));
    },
    async delete(key) {
      store.delete(key);
    },
  };
  return store;
}

test('Remote Ollama workers are settings-backed and omit private tokens publicly', async () => {
  const store = installSettingsStore();

  const created = await saveRemoteOllamaWorker({
    name: 'DGX Ollama',
    base_url: 'https://ollama.example.com/',
    auth_token: 'secret-token',
  });

  assert.equal(created.base_url, 'https://ollama.example.com');
  assert.equal(created.auth_token, 'secret-token');
  assert.equal(toPublicRemoteOllamaWorker(created).auth_token, undefined);

  const updated = await saveRemoteOllamaWorker({
    id: created.id,
    name: 'DGX Ollama',
    base_url: 'https://ollama.example.com',
    auth_token: '',
    enabled: false,
  });

  assert.equal(updated.auth_token, 'secret-token');
  assert.equal(updated.enabled, false);

  const savedRows = JSON.parse(store.get(REMOTE_OLLAMA_WORKERS_SETTING_KEY));
  assert.equal(savedRows[0].auth_token, 'secret-token');

  const workers = await listRemoteOllamaWorkers();
  assert.equal(workers.length, 1);
  assert.equal(workers[0].id, created.id);

  const deleted = await deleteRemoteOllamaWorker(created.id);
  assert.equal(deleted.id, created.id);
  assert.deepEqual(await listRemoteOllamaWorkers(), []);
});

test('Remote Ollama health check updates status and sends bearer token', async () => {
  installSettingsStore();
  const created = await saveRemoteOllamaWorker({
    name: 'Private Ollama',
    base_url: 'http://ollama.test',
    auth_token: 'remote-token',
  });

  globalThis.fetch = async (url, init = {}) => {
    assert.equal(String(url), 'http://ollama.test/api/tags');
    assert.equal(new Headers(init.headers).get('Authorization'), 'Bearer remote-token');
    return new Response(JSON.stringify({ models: [{ model: 'qwen3.5:35b' }] }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };

  const { worker, status } = await checkRemoteOllamaWorker(created.id);

  assert.equal(status.ok, true);
  assert.equal(status.modelCount, 1);
  assert.equal(worker.last_status, 'online');
  assert.equal(worker.model_count, 1);
  assert.equal(worker.last_error, null);
});

test('Remote Ollama model list failures update saved status', async () => {
  installSettingsStore();
  const created = await saveRemoteOllamaWorker({
    name: 'Unavailable Ollama',
    base_url: 'http://ollama.test',
  });

  globalThis.fetch = async () =>
    new Response(JSON.stringify({ error: 'upstream unavailable' }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' },
    });

  const { worker, status, models } = await listRemoteOllamaWorkerModels(created.id);

  assert.equal(status.ok, false);
  assert.equal(worker.last_status, 'error');
  assert.match(worker.last_error, /upstream unavailable/);
  assert.deepEqual(models, []);
});
