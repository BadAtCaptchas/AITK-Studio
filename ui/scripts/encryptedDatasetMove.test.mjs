import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { createRequire, Module } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const apiModulePath = require.resolve('../dist/src/utils/api.js');
const apiClient = { post: async () => { throw new Error('Unexpected network request'); } };
const apiModule = new Module(apiModulePath);
apiModule.exports = { apiClient };
apiModule.loaded = true;
require.cache[apiModulePath] = apiModule;
const { moveEncryptedDatasetItems } = require('../dist/src/utils/encryptedDatasetMove.js');
const { decryptCatalog, importRawAesKey } = require('../dist/src/utils/encryptedDatasets.js');
const { readJsonCommand } = require('../dist/src/server/commandInput.js');
if (!globalThis.crypto) Object.defineProperty(globalThis, 'crypto', { value: crypto.webcrypto });

async function fixture(t, { sizes, workerID = 'local', failUpload, failCommit = false }) {
  const key = await importRawAesKey(Buffer.alloc(32, 7).toString('base64'));
  const items = sizes.map((size, index) => ({
    id: `image-${index}`, name: `image-${index}.png`, extension: '.png', mimeType: 'image/png', mediaKind: 'image',
    objectPath: `objects/image-${index}.bin`, size,
    ...(index === 0 ? { captionObjectPath: 'objects/image-0.caption.bin' } : {}),
    createdAt: '2026-01-01T00:00:00.000Z', updatedAt: '2026-01-01T00:00:00.000Z',
  }));
  const { captionObjectPath: _captionObjectPath, ...remainingFields } = items[0];
  const remaining = { ...remainingFields, id: 'remaining', objectPath: 'objects/remaining.bin' };
  const objects = new Map(items.map((item, index) => [item.objectPath, new Blob([Buffer.alloc(item.size, index + 1)])]));
  objects.set('objects/image-0.caption.bin', new Blob(['encrypted-caption-fixture']));
  const uploaded = new Map();
  const events = [];
  let destinationCatalog;
  let sourceCatalog;
  t.mock.method(apiClient, 'post', async (url, body, config) => {
    if (url === '/api/datasets/upload') {
      assert.ok(body instanceof Blob);
      assert.equal(config.headers['Content-Type'], 'application/octet-stream');
      assert.equal(decodeURIComponent(config.headers['X-AITK-Dataset-Name']), 'destination_2');
      assert.equal(config.headers['X-AITK-Worker-ID'], workerID === 'local' ? undefined : encodeURIComponent(workerID));
      const objectPath = decodeURIComponent(config.headers['X-AITK-Encrypted-Object-Path']);
      assert.equal(decodeURIComponent(config.headers['X-AITK-File-Name']), objectPath.replace(/^objects\//, ''));
      events.push(`upload:${objectPath}`);
      if (objectPath === failUpload) throw new Error('simulated upload failure');
      assert.equal(body, objects.get(objectPath), 'The original encrypted blob must be uploaded without a JSON/base64 conversion');
      uploaded.set(objectPath, body);
      return { data: { success: true } };
    }
    // Apply the real command parser to every JSON call. Large ciphertext must
    // never be hidden inside a command that the server rejects at 2 MiB.
    await readJsonCommand(new Request(`http://localhost${url}`, {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
    }));
    assert.equal(body.worker_id, workerID);
    if (url === '/api/datasets/create') {
      assert.equal(body.name, 'destination');
      assert.deepEqual((await decryptCatalog(body.encryptedManifest, key)).items, []);
      events.push('create');
      return { data: { name: 'destination_2' } };
    }
    if (url === '/api/datasets/encrypted/object') {
      assert.equal(body.datasetName, 'source');
      assert.equal(config.responseType, 'blob');
      events.push(`read:${body.objectPath}`);
      assert.ok(objects.has(body.objectPath));
      return { data: objects.get(body.objectPath) };
    }
    assert.equal(url, '/api/datasets/encrypted/update');
    assert.equal(body.objects, undefined);
    if (body.datasetName === 'destination_2') {
      events.push('destination-commit');
      if (failCommit) throw new Error('simulated destination commit failure');
      assert.equal(uploaded.size, objects.size);
      assert.equal(body.deleteObjects, undefined);
      destinationCatalog = await decryptCatalog(body.manifest, key);
    } else {
      assert.equal(body.datasetName, 'source');
      assert.ok(destinationCatalog, 'The destination must commit before source deletion');
      events.push('source-remove');
      assert.deepEqual(body.deleteObjects.slice().sort(), Array.from(objects.keys()).sort());
      sourceCatalog = await decryptCatalog(body.manifest, key);
    }
    return { data: { success: true } };
  });
  return {
    options: {
      datasetName: 'source', destinationName: 'destination', workerID, key, items,
      catalog: { version: 1, items: [...items, remaining], rootCaption: 'retain source instructions' },
      manifest: {
        format: 'aitk-encrypted-dataset', version: 1,
        crypto: { algorithm: 'AES-256-GCM', kdf: { type: 'KEYFILE-SHA256', keyLength: 32 } },
        catalog: { nonce: '', data: '' },
      },
    },
    events,
    uploaded,
    catalogs: () => ({ destinationCatalog, sourceCatalog }),
  };
}

for (const [label, sizes, workerID] of [
  ['one image larger than 2 MiB', [3 * 1024 * 1024], 'local'],
  ['several images exceeding the aggregate JSON limit on a remote worker', [1024 * 1024, 1024 * 1024, 1024 * 1024], 'worker 1'],
]) {
  test(`encrypted move streams ${label} and commits before removing source objects`, async t => {
    const state = await fixture(t, { sizes, workerID });
    const result = await moveEncryptedDatasetItems(state.options);
    assert.equal(result.createdName, 'destination_2');
    assert.deepEqual(result.nextCatalog.items.map(item => item.id), ['remaining']);
    assert.equal(result.nextCatalog.rootCaption, 'retain source instructions');
    const { sourceCatalog, destinationCatalog } = state.catalogs();
    assert.deepEqual(sourceCatalog, result.nextCatalog);
    assert.deepEqual(destinationCatalog.items.map(item => item.id), state.options.items.map(item => item.id));
    assert.deepEqual(state.events.slice(-2), ['destination-commit', 'source-remove']);
    assert.ok(state.events.indexOf('upload:objects/image-0.bin') < state.events.indexOf('read:objects/image-0.caption.bin'));
    assert.equal(state.options.catalog.items.length, sizes.length + 1, 'Input catalog is not mutated');
  });
}

test('an encrypted object upload failure leaves the source catalog and objects untouched', async t => {
  const state = await fixture(t, { sizes: [1024, 1024], failUpload: 'objects/image-1.bin' });
  await assert.rejects(moveEncryptedDatasetItems(state.options), /simulated upload failure/);
  assert.equal(state.events.includes('destination-commit'), false);
  assert.equal(state.events.includes('source-remove'), false);
  assert.equal(state.options.catalog.items.length, 3);
});

test('a destination catalog commit failure never removes source objects', async t => {
  const state = await fixture(t, { sizes: [3 * 1024 * 1024], failCommit: true });
  await assert.rejects(moveEncryptedDatasetItems(state.options), /simulated destination commit failure/);
  assert.equal(state.events.includes('source-remove'), false);
  assert.equal(state.catalogs().sourceCatalog, undefined);
});
