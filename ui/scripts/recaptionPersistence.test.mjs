import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { afterEach, test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  persistedRecaptionQueueEntry,
  purgeLegacyPersistedRecaptionQueues,
  readPersistedRecaptionQueue,
  recaptionQueueStorageKey,
} = require('../dist/src/components/dataset-image-studio/recaption.js');

function installLocalStorage(entries = {}) {
  const store = new Map(Object.entries(entries));
  const localStorage = {
    get length() {
      return store.size;
    },
    key(index) {
      return Array.from(store.keys())[index] ?? null;
    },
    getItem(key) {
      return store.has(key) ? store.get(key) : null;
    },
    setItem(key, value) {
      store.set(String(key), String(value));
    },
    removeItem(key) {
      store.delete(String(key));
    },
  };

  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: { localStorage },
  });

  return store;
}

const settings = {
  provider: 'openrouter',
  model: 'x-ai/grok-4.3',
  outputFormat: 'text',
  prompt: 'Caption the image.',
  systemPrompt: '',
  remoteWorkerId: '',
  maxNewTokens: 256,
};

afterEach(() => {
  delete globalThis.window;
});

test('persisted recaption queue entries omit plaintext captions', () => {
  const persisted = persistedRecaptionQueueEntry(
    {
      id: 'entry-1',
      item: { kind: 'plain', path: 'datasets/private/image.png' },
      key: 'datasets/private/image.png',
      name: 'image.png',
      existingCaption: 'SECRET_DECRYPTED_CAPTION',
      settings,
    },
    'queued',
  );
  const serialized = JSON.stringify(persisted);

  assert.equal(Object.hasOwn(persisted, 'existingCaption'), false);
  assert.equal(serialized.includes('SECRET_DECRYPTED_CAPTION'), false);
  assert.equal(serialized.includes('existingCaption'), false);
});

test('v2 queue snapshots restore metadata and settings only', () => {
  const storageKey = recaptionQueueStorageKey({
    datasetName: 'plain',
    datasetPath: 'E:/datasets/plain',
    workerID: 'local',
  });
  installLocalStorage({
    [storageKey]: JSON.stringify({
      version: 2,
      active: {
        id: 'entry-1',
        key: 'datasets/plain/image.png',
        name: 'image.png',
        existingCaption: 'SECRET_DECRYPTED_CAPTION',
        settings,
        status: 'running',
        updatedAt: '2026-07-06T00:00:00.000Z',
      },
      queue: [
        {
          id: 'entry-2',
          key: 'datasets/plain/next.png',
          name: 'next.png',
          existingCaption: 'ANOTHER_SECRET',
          settings,
          status: 'queued',
          updatedAt: '2026-07-06T00:01:00.000Z',
        },
      ],
      updatedAt: '2026-07-06T00:02:00.000Z',
    }),
  });

  const restored = readPersistedRecaptionQueue(storageKey);

  assert.equal(restored.version, 2);
  assert.equal(restored.active.key, 'datasets/plain/image.png');
  assert.equal(Object.hasOwn(restored.active, 'existingCaption'), false);
  assert.equal(restored.queue[0].key, 'datasets/plain/next.png');
  assert.equal(Object.hasOwn(restored.queue[0], 'existingCaption'), false);
  assert.equal(restored.queue[0].settings.model, settings.model);
});

test('legacy v1 queue snapshots are purged without touching v2 queues', () => {
  const store = installLocalStorage({
    'aitk.datasetEditor.recaptionQueue.v1:local:secret': '{"existingCaption":"SECRET"}',
    'aitk.datasetEditor.recaptionQueue.v2:local:plain': '{"version":2}',
    'aitk.datasetEditor.recaptionSettings.v1:local:plain': '{"model":"x-ai/grok-4.3"}',
  });

  purgeLegacyPersistedRecaptionQueues();

  assert.equal(store.has('aitk.datasetEditor.recaptionQueue.v1:local:secret'), false);
  assert.equal(store.has('aitk.datasetEditor.recaptionQueue.v2:local:plain'), true);
  assert.equal(store.has('aitk.datasetEditor.recaptionSettings.v1:local:plain'), true);
});

test('DatasetImageStudio gates localStorage persistence for encrypted datasets', () => {
  const source = fs.readFileSync(
    path.join(process.cwd(), 'src/components/dataset-image-studio/DatasetImageStudio.tsx'),
    'utf8',
  );

  assert.match(source, /const hasEncryptedItems = useMemo\(\(\) => items\.some\(item => item\.kind === 'encrypted'\), \[items\]\);/);
  assert.match(source, /const canPersistRecaptionState = !hasEncryptedItems;/);
  assert.match(source, /if \(!hasEncryptedItems \|\| typeof window === 'undefined'\) return;[\s\S]*removeItem\(recaptionQueueStorageKeyValue\)[\s\S]*removeItem\(recaptionStorageKey\)/);
  assert.match(source, /if \(!canPersistRecaptionState\) return;[\s\S]*readPersistedRecaptionQueue\(storageKey\)/);
  assert.match(source, /if \(!canPersistRecaptionState\) return;[\s\S]*window\.localStorage\.setItem\(storageKey, JSON\.stringify\(snapshot\)\)/);
  assert.match(source, /if \(!canPersistRecaptionState\) \{[\s\S]*setHasRecaptionSettingsForDataset\(true\);[\s\S]*return;[\s\S]*window\.localStorage\.setItem\(recaptionStorageKey, JSON\.stringify\(settings\)\)/);
});
