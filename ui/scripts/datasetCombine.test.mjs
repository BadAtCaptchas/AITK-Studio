import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  combineDatasets,
  datasetCombineRequestHasKeyMaterial,
  hasPlaintextEncryptedOutputFiles,
} = require('../dist/src/server/datasetCombine.js');
const { ENCRYPTED_DATASET_MANIFEST } = require('../dist/src/server/encryptedDatasets.js');

const CATALOG_AAD = 'aitk-encrypted-catalog:v1';
const AUTH_TAG_BYTES = 16;

test('datasetCombineRequestHasKeyMaterial detects raw keys before remote forwarding', () => {
  assert.equal(datasetCombineRequestHasKeyMaterial(null), false);
  assert.equal(datasetCombineRequestHasKeyMaterial({ sourceDatasets: ['plain'], outputName: 'combined' }), false);
  assert.equal(
    datasetCombineRequestHasKeyMaterial({
      sourceDatasets: ['locked'],
      outputName: 'combined',
      encryptedDatasetKeys: [{ datasetName: 'locked', keyB64: 'source-key' }],
    }),
    true,
  );
  assert.equal(
    datasetCombineRequestHasKeyMaterial({
      sourceDatasets: ['plain'],
      outputName: 'combined',
      outputEncrypted: true,
      outputKeyB64: 'output-key',
    }),
    true,
  );
});

function b64(value) {
  return Buffer.from(value).toString('base64');
}

function encryptPayload(key, data, aad) {
  const nonce = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, nonce, { authTagLength: AUTH_TAG_BYTES });
  cipher.setAAD(Buffer.from(aad, 'utf8'));
  const encrypted = Buffer.concat([cipher.update(Buffer.from(data)), cipher.final(), cipher.getAuthTag()]);
  return {
    nonce: b64(nonce),
    data: b64(encrypted),
  };
}

function decryptPayload(key, payload, aad) {
  const encrypted = Buffer.from(payload.data, 'base64');
  const ciphertext = encrypted.subarray(0, encrypted.length - AUTH_TAG_BYTES);
  const tag = encrypted.subarray(encrypted.length - AUTH_TAG_BYTES);
  const decipher = crypto.createDecipheriv('aes-256-gcm', key, Buffer.from(payload.nonce, 'base64'), {
    authTagLength: AUTH_TAG_BYTES,
  });
  decipher.setAAD(Buffer.from(aad, 'utf8'));
  decipher.setAuthTag(tag);
  return Buffer.concat([decipher.update(ciphertext), decipher.final()]);
}

function manifestForKey(key, catalog = { version: 1, items: [] }) {
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
    catalog: encryptPayload(key, Buffer.from(JSON.stringify(catalog), 'utf8'), CATALOG_AAD),
  };
}

async function makeRoot() {
  return fs.mkdtemp(path.join(os.tmpdir(), 'aitk-combine-test-'));
}

async function writePlainDataset(root, name, files) {
  const folder = path.join(root, name);
  await fs.mkdir(folder, { recursive: true });
  for (const file of files) {
    const filePath = path.join(folder, ...file.path.split('/'));
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, file.data ?? `media:${file.path}`);
    if (file.caption !== undefined) {
      const parsed = path.parse(filePath);
      await fs.writeFile(path.join(parsed.dir, `${parsed.name}.txt`), file.caption);
    }
  }
  return folder;
}

async function writeEncryptedDataset(root, name, key, specs) {
  const folder = path.join(root, name);
  await fs.mkdir(path.join(folder, 'objects'), { recursive: true });
  const items = [];

  for (const spec of specs) {
    const id = spec.id || crypto.randomBytes(8).toString('hex');
    const objectPath = `objects/${id}.bin`;
    await fs.writeFile(
      path.join(folder, objectPath),
      JSON.stringify(encryptPayload(key, Buffer.from(spec.data), `aitk-encrypted-object:${objectPath}`)),
    );

    const item = {
      id,
      name: spec.name,
      extension: path.extname(spec.name).toLowerCase(),
      mimeType: 'image/png',
      mediaKind: 'image',
      objectPath,
      size: Buffer.byteLength(spec.data),
      width: 4,
      height: 3,
      createdAt: '2026-05-30T00:00:00.000Z',
      updatedAt: '2026-05-30T00:00:00.000Z',
    };

    if (spec.caption !== undefined) {
      const captionObjectPath = `objects/${id}.caption.bin`;
      await fs.writeFile(
        path.join(folder, captionObjectPath),
        JSON.stringify(
          encryptPayload(key, Buffer.from(spec.caption), `aitk-encrypted-object:${captionObjectPath}`),
        ),
      );
      item.captionObjectPath = captionObjectPath;
    }
    items.push(item);
  }

  await fs.writeFile(
    path.join(folder, ENCRYPTED_DATASET_MANIFEST),
    JSON.stringify(manifestForKey(key, { version: 1, items }), null, 2),
  );
  return folder;
}

async function readOutputCatalog(datasetFolder, key) {
  const manifest = JSON.parse(await fs.readFile(path.join(datasetFolder, ENCRYPTED_DATASET_MANIFEST), 'utf8'));
  return JSON.parse(decryptPayload(key, manifest.catalog, CATALOG_AAD).toString('utf8'));
}

test('combineDatasets combines plain datasets and renames duplicate stems', async () => {
  const root = await makeRoot();
  await writePlainDataset(root, 'alpha', [
    { path: 'image.png', data: 'alpha-image', caption: 'alpha caption' },
    { path: 'same.jpg', data: 'alpha-same', caption: 'same alpha' },
  ]);
  await writePlainDataset(root, 'beta', [
    { path: 'nested/image.png', data: 'beta-image', caption: 'beta caption' },
    { path: 'same.png', data: 'beta-same', caption: 'same beta' },
  ]);
  await fs.mkdir(path.join(root, 'combined'));

  const result = await combineDatasets(root, {
    sourceDatasets: ['alpha', 'beta'],
    outputName: 'combined',
  });

  assert.equal(result.dataset.name, 'combined_2');
  const output = path.join(root, result.dataset.name);
  assert.deepEqual((await fs.readdir(output)).sort(), [
    'image.png',
    'image.txt',
    'image_2.png',
    'image_2.txt',
    'same.jpg',
    'same.txt',
    'same_2.png',
    'same_2.txt',
  ]);
  assert.equal(await fs.readFile(path.join(output, 'image.txt'), 'utf8'), 'alpha caption');
  assert.equal(await fs.readFile(path.join(output, 'image_2.txt'), 'utf8'), 'beta caption');
  assert.equal(await fs.readFile(path.join(output, 'same_2.txt'), 'utf8'), 'same beta');
});

test('combineDatasets requires valid keys for encrypted sources and can decrypt to plain output', async () => {
  const root = await makeRoot();
  const key = crypto.randomBytes(32);
  await writePlainDataset(root, 'plain', [{ path: 'secret.png', data: 'plain bytes', caption: 'plain caption' }]);
  await writeEncryptedDataset(root, 'locked', key, [
    { id: 'locked1', name: 'secret.png', data: 'encrypted bytes', caption: 'encrypted caption' },
  ]);

  await assert.rejects(
    () => combineDatasets(root, { sourceDatasets: ['plain', 'locked'], outputName: 'missing_key' }),
    /key is required/i,
  );
  await assert.rejects(
    () =>
      combineDatasets(root, {
        sourceDatasets: ['plain', 'locked'],
        outputName: 'wrong_key',
        encryptedDatasetKeys: [{ datasetName: 'locked', keyB64: crypto.randomBytes(32).toString('base64') }],
      }),
    /auth|decrypt|Unsupported state|authenticate/i,
  );

  const result = await combineDatasets(root, {
    sourceDatasets: ['plain', 'locked'],
    outputName: 'plain_out',
    encryptedDatasetKeys: [{ datasetName: 'locked', keyB64: key.toString('base64') }],
  });

  const output = path.join(root, result.dataset.name);
  assert.equal(await fs.readFile(path.join(output, 'secret.png'), 'utf8'), 'plain bytes');
  assert.equal(await fs.readFile(path.join(output, 'secret_2.png'), 'utf8'), 'encrypted bytes');
  assert.equal(await fs.readFile(path.join(output, 'secret_2.txt'), 'utf8'), 'encrypted caption');
});

test('combineDatasets writes encrypted output without plaintext files', async () => {
  const root = await makeRoot();
  const sourceKey = crypto.randomBytes(32);
  const outputKey = crypto.randomBytes(32);
  await writePlainDataset(root, 'plain', [{ path: 'image.png', data: 'plain bytes', caption: 'plain caption' }]);
  await writeEncryptedDataset(root, 'locked', sourceKey, [
    { id: 'locked1', name: 'image.png', data: 'encrypted bytes', caption: 'encrypted caption' },
  ]);

  const result = await combineDatasets(root, {
    sourceDatasets: ['plain', 'locked'],
    outputName: 'encrypted_out',
    outputEncrypted: true,
    encryptedDatasetKeys: [{ datasetName: 'locked', keyB64: sourceKey.toString('base64') }],
    outputEncryptedManifest: manifestForKey(outputKey),
    outputKeyB64: outputKey.toString('base64'),
  });

  const output = path.join(root, result.dataset.name);
  assert.equal(hasPlaintextEncryptedOutputFiles(output), false);
  assert.equal(fsSync.existsSync(path.join(output, 'image.png')), false);

  const catalog = await readOutputCatalog(output, outputKey);
  assert.deepEqual(
    catalog.items.map(item => item.name).sort(),
    ['image.png', 'image_2.png'],
  );

  const second = catalog.items.find(item => item.name === 'image_2.png');
  const objectPayload = JSON.parse(await fs.readFile(path.join(output, second.objectPath), 'utf8'));
  assert.equal(
    decryptPayload(outputKey, objectPayload, `aitk-encrypted-object:${second.objectPath}`).toString('utf8'),
    'encrypted bytes',
  );
  const captionPayload = JSON.parse(await fs.readFile(path.join(output, second.captionObjectPath), 'utf8'));
  assert.equal(
    decryptPayload(outputKey, captionPayload, `aitk-encrypted-object:${second.captionObjectPath}`).toString('utf8'),
    'encrypted caption',
  );
});
