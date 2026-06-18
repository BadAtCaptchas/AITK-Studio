import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const encryptedDatasets = require('../dist/src/server/encryptedDatasets.js');
const datasetRootCaption = require('../dist/src/server/datasetRootCaption.js');
const datasetImages = require('../dist/src/server/datasetImages.js');
const encryptedDatasetUtils = require('../dist/src/utils/encryptedDatasets.js');
const encryptedObjectMediaCache = require('../dist/src/utils/encryptedObjectMediaCache.js');
const webauthnPrfCrypto = require('../dist/src/utils/webauthnPrfCrypto.js');

const CATALOG_AAD = Buffer.from('aitk-encrypted-catalog:v1', 'utf8');

if (!globalThis.crypto) {
  Object.defineProperty(globalThis, 'crypto', { value: crypto.webcrypto });
}

function encryptedCatalogPayload(key, catalog) {
  const nonce = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, nonce, { authTagLength: 16 });
  cipher.setAAD(CATALOG_AAD);
  const ciphertext = Buffer.concat([
    cipher.update(Buffer.from(JSON.stringify(catalog), 'utf8')),
    cipher.final(),
    cipher.getAuthTag(),
  ]);
  return {
    nonce: nonce.toString('base64'),
    data: ciphertext.toString('base64'),
  };
}

function manifestForKey(key) {
  return {
    format: 'aitk-encrypted-dataset',
    version: 1,
    crypto: {
      algorithm: 'AES-256-GCM',
      kdf: {
        type: 'KEYFILE-SHA256',
        keyLength: 32,
      },
    },
    catalog: encryptedCatalogPayload(key, { version: 1, items: [] }),
  };
}

function webAuthnPrfManifestForKey(key) {
  return {
    format: 'aitk-encrypted-dataset',
    version: 1,
    crypto: {
      algorithm: 'AES-256-GCM',
      kdf: {
        type: 'WEBAUTHN-PRF',
        keyLength: 32,
        rpId: 'localhost',
        credentials: [
          {
            id: 'mockCredentialId',
            label: 'Mock YubiKey',
            transports: ['usb'],
            saltB64: crypto.randomBytes(32).toString('base64'),
            createdAt: '2026-05-30T00:00:00.000Z',
            wrappedKey: {
              algorithm: 'AES-256-GCM',
              nonce: crypto.randomBytes(12).toString('base64'),
              data: crypto.randomBytes(48).toString('base64'),
            },
          },
        ],
        nativeUsb: {
          provider: 'ctap2-hmac-secret',
          status: 'planned',
        },
      },
    },
    catalog: encryptedCatalogPayload(key, { version: 1, items: [] }),
  };
}

test('listDatasetSummaries counts missing captions for plain datasets', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-caption-summary-'));
  const plain = path.join(root, 'plain_dataset');
  const locked = path.join(root, 'locked_dataset');
  await fs.mkdir(path.join(plain, 'nested'), { recursive: true });
  await fs.mkdir(path.join(plain, '_controls'), { recursive: true });
  await fs.mkdir(locked, { recursive: true });

  await fs.writeFile(path.join(plain, 'captioned.png'), 'media');
  await fs.writeFile(path.join(plain, 'captioned.txt'), 'caption');
  await fs.writeFile(path.join(plain, 'json_captioned.webp'), 'media');
  await fs.writeFile(path.join(plain, 'json_captioned.json'), JSON.stringify({ caption: 'json caption' }));
  await fs.writeFile(path.join(plain, 'jxl_captioned.jxl'), 'media');
  await fs.writeFile(path.join(plain, 'jxl_captioned.txt'), 'caption');
  await fs.writeFile(path.join(plain, 'refused.png'), 'media');
  await fs.writeFile(path.join(plain, 'refused.txt'), 'I cannot fulfill this request.');
  await fs.writeFile(path.join(plain, 'nested', 'missing.jpg'), 'media');
  await fs.writeFile(path.join(plain, 'notes.txt'), 'orphan text');
  await fs.writeFile(path.join(plain, '_controls', 'control.png'), 'ignored media');
  await fs.writeFile(
    path.join(locked, encryptedDatasets.ENCRYPTED_DATASET_MANIFEST),
    JSON.stringify(manifestForKey(crypto.randomBytes(32))),
    'utf8',
  );

  const summaries = await encryptedDatasets.listDatasetSummaries(root);
  const plainSummary = summaries.find(dataset => dataset.name === 'plain_dataset');
  const lockedSummary = summaries.find(dataset => dataset.name === 'locked_dataset');

  assert.equal(plainSummary.itemCount, 5);
  assert.equal(plainSummary.captionedItemCount, 3);
  assert.equal(plainSummary.missingCaptionCount, 2);
  assert.equal(plainSummary.detectedCaptionExt, null);
  assert.equal(lockedSummary.encrypted, true);
  assert.equal(lockedSummary.itemCount, null);
  assert.equal(lockedSummary.captionedItemCount, null);
  assert.equal(lockedSummary.missingCaptionCount, null);
  assert.equal(lockedSummary.detectedCaptionExt, null);
});

test('listDatasetSummaries detects clearly JSON-captioned plain datasets', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-json-caption-summary-'));
  const jsonDataset = path.join(root, 'json_dataset');
  await fs.mkdir(jsonDataset, { recursive: true });

  await fs.writeFile(path.join(jsonDataset, 'one.png'), 'media');
  await fs.writeFile(path.join(jsonDataset, 'one.json'), JSON.stringify({ caption: 'one json caption' }));
  await fs.writeFile(path.join(jsonDataset, 'two.webp'), 'media');
  await fs.writeFile(path.join(jsonDataset, 'two.json'), JSON.stringify({ caption: 'two json caption' }));

  const summaries = await encryptedDatasets.listDatasetSummaries(root);
  const jsonSummary = summaries.find(dataset => dataset.name === 'json_dataset');

  assert.equal(jsonSummary.itemCount, 2);
  assert.equal(jsonSummary.captionedItemCount, 2);
  assert.equal(jsonSummary.missingCaptionCount, 0);
  assert.equal(jsonSummary.detectedCaptionExt, 'json');
});

test('dataset root caption helper reads exact root metadata file first', async () => {
  const datasetPath = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-root-caption-'));
  await fs.writeFile(path.join(datasetPath, 'root_caption.TXT'), 'fallback prompt');
  await fs.writeFile(path.join(datasetPath, datasetRootCaption.ROOT_CAPTION_FILE_NAME), 'exact prompt');
  await fs.mkdir(path.join(datasetPath, 'nested'));
  await fs.writeFile(path.join(datasetPath, 'nested', datasetRootCaption.ROOT_CAPTION_FILE_NAME), 'nested prompt');

  assert.equal(datasetRootCaption.isDatasetRootCaptionEntry(datasetPath, datasetPath, 'ROOT_CAPTION.txt'), true);
  assert.equal(
    datasetRootCaption.isDatasetRootCaptionEntry(datasetPath, path.join(datasetPath, 'nested'), 'ROOT_CAPTION.txt'),
    false,
  );
  assert.deepEqual(await datasetRootCaption.readDatasetRootCaption(datasetPath), {
    found: true,
    systemPrompt: 'exact prompt',
  });
});

test('listDatasetSummaries ignores root caption metadata for plain datasets', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-root-caption-summary-'));
  const datasetPath = path.join(root, 'plain_dataset');
  await fs.mkdir(datasetPath, { recursive: true });
  await fs.writeFile(path.join(datasetPath, 'ROOT_CAPTION.txt'), 'system prompt');
  await fs.writeFile(path.join(datasetPath, 'image.png'), 'media');

  const summaries = await encryptedDatasets.listDatasetSummaries(root);
  const summary = summaries.find(dataset => dataset.name === 'plain_dataset');

  assert.equal(summary.itemCount, 1);
  assert.equal(summary.captionedItemCount, 0);
  assert.equal(summary.missingCaptionCount, 1);
});

test('dataset item scanner hides root caption metadata from editable items', async () => {
  const datasetPath = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-root-caption-items-'));
  await fs.mkdir(path.join(datasetPath, 'nested'));
  await fs.writeFile(path.join(datasetPath, 'ROOT_CAPTION.txt'), 'system prompt');
  await fs.writeFile(path.join(datasetPath, 'image.png'), 'media');
  await fs.writeFile(path.join(datasetPath, 'nested', 'ROOT_CAPTION.txt'), 'nested text item');

  const items = datasetImages.findDatasetItemsRecursively(datasetPath).map(item => path.relative(datasetPath, item));

  assert.deepEqual(items.sort(), ['image.png', path.join('nested', 'ROOT_CAPTION.txt')].sort());
});

test('validateEncryptedCatalogKey accepts the matching dataset key', () => {
  const key = crypto.randomBytes(32);
  const manifest = manifestForKey(key);

  assert.equal(encryptedDatasets.validateEncryptedCatalogKey(manifest, key.toString('base64')), true);
});

test('encrypted catalog preserves root caption metadata', async () => {
  const rawKey = crypto.randomBytes(32);
  const key = await encryptedDatasetUtils.importRawAesKey(rawKey.toString('base64'));
  const catalog = {
    version: 1,
    rootCaption: 'Use this as the system prompt.',
    items: [],
  };
  const { manifest } = await encryptedDatasetUtils.encryptCatalog(catalog, key, manifestForKey(rawKey));

  assert.deepEqual(await encryptedDatasetUtils.decryptCatalog(manifest, key), catalog);
});

test('encrypted upload utilities do not pair root caption metadata as an item caption', async () => {
  const files = [
    {
      name: 'ROOT_CAPTION.txt',
      type: 'text/plain',
      text: async () => 'Dataset-wide system prompt',
    },
    {
      name: 'image.png',
      type: 'image/png',
    },
    {
      name: 'image.txt',
      text: async () => 'Image sidecar caption',
    },
  ];

  const pairs = encryptedDatasetUtils.pairMediaAndCaptionFiles(files);

  assert.equal(await encryptedDatasetUtils.readRootCaptionFile(files), 'Dataset-wide system prompt');
  assert.equal(pairs.length, 1);
  assert.equal(pairs[0].file.name, 'image.png');
  assert.equal(pairs[0].captionFile.name, 'image.txt');
});

test('validateEncryptedCatalogKey rejects the wrong dataset key', () => {
  const manifest = manifestForKey(crypto.randomBytes(32));

  assert.throws(
    () => encryptedDatasets.validateEncryptedCatalogKey(manifest, crypto.randomBytes(32).toString('base64')),
    /authenticate|auth|decrypt|Unsupported state|bad decrypt/i,
  );
});

test('validateEncryptedDatasetStartKey reads and validates the manifest on disk', async () => {
  const key = crypto.randomBytes(32);
  const datasetPath = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-encrypted-dataset-'));
  await fs.writeFile(
    path.join(datasetPath, encryptedDatasets.ENCRYPTED_DATASET_MANIFEST),
    JSON.stringify(manifestForKey(key)),
    'utf8',
  );

  assert.equal(
    await encryptedDatasets.validateEncryptedDatasetStartKey(
      { path: datasetPath, name: path.basename(datasetPath) },
      key.toString('base64'),
    ),
    true,
  );
});

test('validateEncryptedCatalogKey accepts a WebAuthn PRF manifest with USB metadata', () => {
  const key = crypto.randomBytes(32);
  const manifest = webAuthnPrfManifestForKey(key);

  assert.equal(encryptedDatasets.validateEncryptedCatalogKey(manifest, key.toString('base64')), true);
});

test('validateEncryptedManifest rejects malformed WebAuthn PRF credentials', () => {
  const key = crypto.randomBytes(32);
  const manifest = webAuthnPrfManifestForKey(key);
  manifest.crypto.kdf.credentials = [];

  assert.throws(() => encryptedDatasets.validateEncryptedManifest(manifest), /credential/i);
});

test('mocked WebAuthn PRF output wraps and unwraps a dataset key', async () => {
  const rawDatasetKey = crypto.randomBytes(32);
  const prfOutput = crypto.randomBytes(32);
  const aad = webauthnPrfCrypto.webAuthnPrfWrappedKeyAad('localhost', 'mockCredentialId', crypto.randomBytes(32).toString('base64'));

  const wrapped = await webauthnPrfCrypto.encryptWithWebAuthnPrfKey(prfOutput, rawDatasetKey, aad);
  const unwrapped = await webauthnPrfCrypto.decryptWithWebAuthnPrfKey(prfOutput, wrapped.nonce, wrapped.data, aad);

  assert.deepEqual(Buffer.from(new Uint8Array(unwrapped)), rawDatasetKey);
});

test('mocked WebAuthn PRF output rejects the wrong security key result', async () => {
  const rawDatasetKey = crypto.randomBytes(32);
  const prfOutput = crypto.randomBytes(32);
  const wrongPrfOutput = crypto.randomBytes(32);
  const aad = webauthnPrfCrypto.webAuthnPrfWrappedKeyAad('localhost', 'mockCredentialId', crypto.randomBytes(32).toString('base64'));

  const wrapped = await webauthnPrfCrypto.encryptWithWebAuthnPrfKey(prfOutput, rawDatasetKey, aad);
  await assert.rejects(
    () => webauthnPrfCrypto.decryptWithWebAuthnPrfKey(wrongPrfOutput, wrapped.nonce, wrapped.data, aad),
    /decrypt|operation|auth|tag/i,
  );
});

function encryptedMediaItem(objectPath, overrides = {}) {
  return {
    objectPath,
    updatedAt: '2026-06-18T00:00:00.000Z',
    mimeType: 'image/png',
    ...overrides,
  };
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

test('encrypted object request bodies keep project scope local-only', () => {
  assert.deepEqual(
    encryptedObjectMediaCache.buildEncryptedObjectRequestBody({
      datasetName: 'locked',
      workerID: 'local',
      projectID: 'project-1',
      objectPath: 'objects/a.bin',
    }),
    {
      datasetName: 'locked',
      objectPath: 'objects/a.bin',
      worker_id: 'local',
      project_id: 'project-1',
    },
  );
  assert.deepEqual(
    encryptedObjectMediaCache.buildEncryptedObjectRequestBody({
      datasetName: 'locked',
      workerID: 'worker-1',
      projectID: 'project-1',
      objectPath: 'objects/a.bin',
    }),
    {
      datasetName: 'locked',
      objectPath: 'objects/a.bin',
      worker_id: 'worker-1',
    },
  );
});

test('encrypted object media cache dedupes concurrent loads for the same object', async () => {
  encryptedObjectMediaCache.clearEncryptedObjectMediaCache();
  let fetches = 0;
  let decrypts = 0;
  let urls = 0;
  const options = {
    datasetName: 'locked',
    workerID: 'worker-1',
    cryptoKey: {},
    item: encryptedMediaItem('objects/a.bin'),
    loadEncryptedObject: async body => {
      fetches += 1;
      assert.equal(body.worker_id, 'worker-1');
      await delay(5);
      return new Blob(['encrypted']);
    },
    decryptEncryptedObject: async () => {
      decrypts += 1;
      return new Uint8Array([1, 2, 3]).buffer;
    },
    createObjectUrl: () => {
      urls += 1;
      return `blob:${urls}`;
    },
    revokeObjectUrl: () => undefined,
  };

  try {
    const [first, second] = await Promise.all([
      encryptedObjectMediaCache.loadEncryptedObjectMediaUrl(options),
      encryptedObjectMediaCache.loadEncryptedObjectMediaUrl(options),
    ]);

    assert.equal(first, second);
    assert.equal(fetches, 1);
    assert.equal(decrypts, 1);
    assert.equal(urls, 1);
  } finally {
    encryptedObjectMediaCache.clearEncryptedObjectMediaCache();
  }
});

test('encrypted object media cache retries after failed loads', async () => {
  encryptedObjectMediaCache.clearEncryptedObjectMediaCache();
  let fetches = 0;
  const options = {
    datasetName: 'locked',
    workerID: 'worker-1',
    cryptoKey: {},
    item: encryptedMediaItem('objects/retry.bin'),
    loadEncryptedObject: async () => {
      fetches += 1;
      if (fetches === 1) throw new Error('temporary remote failure');
      return new Blob(['encrypted']);
    },
    decryptEncryptedObject: async () => new Uint8Array([1]).buffer,
    createObjectUrl: () => 'blob:retry',
    revokeObjectUrl: () => undefined,
  };

  try {
    await assert.rejects(
      () => encryptedObjectMediaCache.loadEncryptedObjectMediaUrl(options),
      /temporary remote failure/,
    );
    assert.equal(await encryptedObjectMediaCache.loadEncryptedObjectMediaUrl(options), 'blob:retry');
    assert.equal(fetches, 2);
  } finally {
    encryptedObjectMediaCache.clearEncryptedObjectMediaCache();
  }
});

test('encrypted object media cache evicts least-recently-used object URLs', async () => {
  encryptedObjectMediaCache.clearEncryptedObjectMediaCache();
  const revoked = [];
  let urlID = 0;
  const base = {
    datasetName: 'locked',
    workerID: 'worker-1',
    cryptoKey: {},
    loadEncryptedObject: async () => new Blob(['encrypted']),
    decryptEncryptedObject: async () => new Uint8Array([1]).buffer,
    createObjectUrl: () => {
      const url = `blob:${urlID}`;
      urlID += 1;
      return url;
    },
    revokeObjectUrl: url => revoked.push(url),
  };

  try {
    for (let index = 0; index < 129; index += 1) {
      await encryptedObjectMediaCache.loadEncryptedObjectMediaUrl({
        ...base,
        item: encryptedMediaItem(`objects/${index}.bin`),
      });
    }

    assert.equal(revoked.length, 1);
    assert.equal(revoked[0], 'blob:0');
  } finally {
    encryptedObjectMediaCache.clearEncryptedObjectMediaCache();
  }
});

test('encrypted object media cache limits parallel remote loads', async () => {
  encryptedObjectMediaCache.clearEncryptedObjectMediaCache();
  let active = 0;
  let maxActive = 0;

  try {
    await Promise.all(
      Array.from({ length: 9 }).map((_, index) =>
        encryptedObjectMediaCache.loadEncryptedObjectMediaUrl({
          datasetName: 'locked',
          workerID: 'worker-1',
          cryptoKey: {},
          item: encryptedMediaItem(`objects/concurrent-${index}.bin`),
          loadEncryptedObject: async () => {
            active += 1;
            maxActive = Math.max(maxActive, active);
            await delay(10);
            active -= 1;
            return new Blob(['encrypted']);
          },
          decryptEncryptedObject: async () => new Uint8Array([1]).buffer,
          createObjectUrl: () => `blob:concurrent-${index}`,
          revokeObjectUrl: () => undefined,
        }),
      ),
    );

    assert.equal(maxActive, 4);
  } finally {
    encryptedObjectMediaCache.clearEncryptedObjectMediaCache();
  }
});

test('encrypted object media cache keys include worker, local project, dataset, crypto key, object, update time, and MIME type', async () => {
  encryptedObjectMediaCache.clearEncryptedObjectMediaCache();
  const bodies = [];
  let urlID = 0;
  const cryptoKeyA = {};
  const cryptoKeyB = {};
  const base = {
    datasetName: 'locked',
    workerID: 'local',
    projectID: 'project-a',
    cryptoKey: cryptoKeyA,
    item: encryptedMediaItem('objects/same.bin'),
    loadEncryptedObject: async body => {
      bodies.push(body);
      return new Blob(['encrypted']);
    },
    decryptEncryptedObject: async () => new Uint8Array([1]).buffer,
    createObjectUrl: () => {
      const url = `blob:key-${urlID}`;
      urlID += 1;
      return url;
    },
    revokeObjectUrl: () => undefined,
  };

  try {
    const first = await encryptedObjectMediaCache.loadEncryptedObjectMediaUrl(base);
    assert.equal(await encryptedObjectMediaCache.loadEncryptedObjectMediaUrl(base), first);
    await encryptedObjectMediaCache.loadEncryptedObjectMediaUrl({ ...base, workerID: 'worker-1', projectID: 'project-a' });
    await encryptedObjectMediaCache.loadEncryptedObjectMediaUrl({ ...base, projectID: 'project-b' });
    await encryptedObjectMediaCache.loadEncryptedObjectMediaUrl({ ...base, datasetName: 'other-locked' });
    await encryptedObjectMediaCache.loadEncryptedObjectMediaUrl({ ...base, cryptoKey: cryptoKeyB });
    await encryptedObjectMediaCache.loadEncryptedObjectMediaUrl({ ...base, item: encryptedMediaItem('objects/changed.bin') });
    await encryptedObjectMediaCache.loadEncryptedObjectMediaUrl({
      ...base,
      item: encryptedMediaItem('objects/same.bin', { updatedAt: '2026-06-18T01:00:00.000Z' }),
    });
    await encryptedObjectMediaCache.loadEncryptedObjectMediaUrl({
      ...base,
      item: encryptedMediaItem('objects/same.bin', { mimeType: 'image/webp' }),
    });

    assert.equal(bodies.length, 8);
    assert.deepEqual(bodies[0], {
      datasetName: 'locked',
      objectPath: 'objects/same.bin',
      worker_id: 'local',
      project_id: 'project-a',
    });
    assert.deepEqual(bodies[1], {
      datasetName: 'locked',
      objectPath: 'objects/same.bin',
      worker_id: 'worker-1',
    });
  } finally {
    encryptedObjectMediaCache.clearEncryptedObjectMediaCache();
  }
});
