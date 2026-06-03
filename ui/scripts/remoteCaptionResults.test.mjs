import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  mergeEncryptedCaptionDataset,
  mergePlainCaptionDataset,
} = require('../dist/src/server/remoteCaptionMerge.js');
const {
  resolveDatasetDirectoryInsideRoot,
} = require('../dist/src/server/remoteCaptionSecurity.js');

const CATALOG_AAD = Buffer.from('aitk-encrypted-catalog:v1', 'utf8');
const KEY = Buffer.from('11111111111111111111111111111111', 'utf8');
const KEY_B64 = KEY.toString('base64');

function objectAad(objectPath) {
  return Buffer.from(`aitk-encrypted-object:${objectPath.replace(/\\/g, '/')}`, 'utf8');
}

function encryptPayload(plaintext, aad) {
  const nonce = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', KEY, nonce, { authTagLength: 16 });
  cipher.setAAD(aad);
  const ciphertext = Buffer.concat([cipher.update(Buffer.from(plaintext)), cipher.final()]);
  return {
    nonce: nonce.toString('base64'),
    data: Buffer.concat([ciphertext, cipher.getAuthTag()]).toString('base64'),
  };
}

function decryptPayload(payload, aad) {
  const nonce = Buffer.from(payload.nonce, 'base64');
  const encrypted = Buffer.from(payload.data, 'base64');
  const ciphertext = encrypted.subarray(0, encrypted.length - 16);
  const tag = encrypted.subarray(encrypted.length - 16);
  const decipher = crypto.createDecipheriv('aes-256-gcm', KEY, nonce, { authTagLength: 16 });
  decipher.setAAD(aad);
  decipher.setAuthTag(tag);
  return Buffer.concat([decipher.update(ciphertext), decipher.final()]);
}

async function writeEncryptedDataset(root, item, captionText = null) {
  await fs.mkdir(path.join(root, 'objects'), { recursive: true });
  const catalogItem = { ...item };
  if (captionText !== null) {
    catalogItem.captionObjectPath = `objects/${item.id}.caption.bin`;
    await fs.writeFile(
      path.join(root, 'objects', `${item.id}.caption.bin`),
      JSON.stringify(encryptPayload(captionText, objectAad(catalogItem.captionObjectPath))),
    );
  }
  const manifest = {
    format: 'aitk-encrypted-dataset',
    version: 1,
    crypto: {
      algorithm: 'AES-256-GCM',
      kdf: {
        type: 'PBKDF2-SHA256',
        salt: Buffer.from('salt').toString('base64'),
        iterations: 100000,
        keyLength: 32,
      },
    },
    catalog: encryptPayload(JSON.stringify({ version: 1, items: [catalogItem] }), CATALOG_AAD),
  };
  await fs.writeFile(path.join(root, '.aitk_encrypted_dataset.json'), JSON.stringify(manifest, null, 2));
}

test('plain remote caption merge copies caption files only', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-plain-caption-'));
  const source = path.join(root, 'source');
  const target = path.join(root, 'target');
  await fs.mkdir(path.join(source, 'nested'), { recursive: true });
  await fs.mkdir(path.join(target, 'nested'), { recursive: true });
  await fs.writeFile(path.join(source, 'a.jpg'), 'remote image');
  await fs.writeFile(path.join(source, 'a.txt'), 'new caption');
  await fs.writeFile(path.join(source, 'nested', 'b.jpg'), 'remote nested image');
  await fs.writeFile(path.join(source, 'nested', 'b.txt'), 'nested caption');
  await fs.writeFile(path.join(target, 'a.jpg'), 'local image');
  await fs.writeFile(path.join(target, 'a.txt'), 'existing caption');
  await fs.writeFile(path.join(target, 'nested', 'b.jpg'), 'local nested image');

  const first = await mergePlainCaptionDataset(source, target, { captionExtension: 'txt', recaption: false });
  assert.deepEqual(first, { copied: 1, skipped: 1 });
  assert.equal(await fs.readFile(path.join(target, 'a.txt'), 'utf8'), 'existing caption');
  assert.equal(await fs.readFile(path.join(target, 'nested', 'b.txt'), 'utf8'), 'nested caption');
  assert.equal(await fs.readFile(path.join(target, 'a.jpg'), 'utf8'), 'local image');

  const second = await mergePlainCaptionDataset(source, target, { captionExtension: 'txt', recaption: true });
  assert.equal(second.copied, 2);
  assert.equal(await fs.readFile(path.join(target, 'a.txt'), 'utf8'), 'new caption');
});


test('plain remote caption merge rejects non-caption extensions', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-plain-caption-ext-'));
  const source = path.join(root, 'source');
  const target = path.join(root, 'target');
  await fs.mkdir(source, { recursive: true });
  await fs.mkdir(target, { recursive: true });
  await fs.writeFile(path.join(source, 'a.py'), 'print("remote")');
  await fs.writeFile(path.join(target, 'a.jpg'), 'local image');

  await assert.rejects(
    () => mergePlainCaptionDataset(source, target, { captionExtension: 'py', recaption: true }),
    /supported text caption extension/,
  );
});

test('plain remote caption merge only copies sidecars for known target media', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-plain-caption-sidecar-'));
  const source = path.join(root, 'source');
  const target = path.join(root, 'target');
  await fs.mkdir(source, { recursive: true });
  await fs.mkdir(target, { recursive: true });
  await fs.writeFile(path.join(source, 'known.txt'), 'caption');
  await fs.writeFile(path.join(source, 'orphan.txt'), 'orphan caption');
  await fs.writeFile(path.join(target, 'known.jpg'), 'local image');

  const result = await mergePlainCaptionDataset(source, target, { captionExtension: 'txt', recaption: true });
  assert.deepEqual(result, { copied: 1, skipped: 0 });
  assert.equal(await fs.readFile(path.join(target, 'known.txt'), 'utf8'), 'caption');
  await assert.rejects(() => fs.stat(path.join(target, 'orphan.txt')));
});

test('remote caption dataset paths must stay inside the datasets root after realpath', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-caption-root-'));
  const datasetsRoot = path.join(root, 'datasets');
  const dataset = path.join(datasetsRoot, 'cats');
  const outside = path.join(root, 'outside');
  await fs.mkdir(dataset, { recursive: true });
  await fs.mkdir(outside, { recursive: true });
  await fs.symlink(outside, path.join(datasetsRoot, 'outside-link'));

  assert.equal(await resolveDatasetDirectoryInsideRoot(dataset, datasetsRoot), await fs.realpath(dataset));
  await assert.rejects(
    () => resolveDatasetDirectoryInsideRoot(outside, datasetsRoot),
    /inside the configured datasets folder/,
  );
  await assert.rejects(
    () => resolveDatasetDirectoryInsideRoot(path.join(datasetsRoot, 'outside-link'), datasetsRoot),
    /inside the configured datasets folder/,
  );
});

test('encrypted remote caption merge copies caption object and updates catalog', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-encrypted-caption-'));
  const source = path.join(root, 'source');
  const target = path.join(root, 'target');
  const item = {
    id: 'img1',
    name: 'cat.jpg',
    extension: 'jpg',
    mimeType: 'image/jpeg',
    mediaKind: 'image',
    objectPath: 'objects/img1.bin',
    size: 123,
    createdAt: '2026-01-01T00:00:00.000Z',
    updatedAt: '2026-01-01T00:00:00.000Z',
  };
  await writeEncryptedDataset(source, item, 'remote encrypted caption');
  await writeEncryptedDataset(target, item, null);

  const result = await mergeEncryptedCaptionDataset(source, target, { keyB64: KEY_B64, recaption: false });
  assert.deepEqual(result, { copied: 1, skipped: 0 });

  const manifest = JSON.parse(await fs.readFile(path.join(target, '.aitk_encrypted_dataset.json'), 'utf8'));
  const catalog = JSON.parse(decryptPayload(manifest.catalog, CATALOG_AAD).toString('utf8'));
  assert.equal(catalog.items[0].captionObjectPath, 'objects/img1.caption.bin');

  const copiedCaption = JSON.parse(await fs.readFile(path.join(target, 'objects', 'img1.caption.bin'), 'utf8'));
  assert.equal(decryptPayload(copiedCaption, objectAad('objects/img1.caption.bin')).toString('utf8'), 'remote encrypted caption');
});
