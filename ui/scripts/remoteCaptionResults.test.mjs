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
  await fs.writeFile(path.join(source, 'nested', 'b.txt'), 'nested caption');
  await fs.writeFile(path.join(target, 'a.txt'), 'existing caption');

  const first = await mergePlainCaptionDataset(source, target, { captionExtension: 'txt', recaption: false });
  assert.deepEqual(first, { copied: 1, skipped: 1 });
  assert.equal(await fs.readFile(path.join(target, 'a.txt'), 'utf8'), 'existing caption');
  assert.equal(await fs.readFile(path.join(target, 'nested', 'b.txt'), 'utf8'), 'nested caption');
  await assert.rejects(() => fs.stat(path.join(target, 'a.jpg')));

  const second = await mergePlainCaptionDataset(source, target, { captionExtension: 'txt', recaption: true });
  assert.equal(second.copied, 2);
  assert.equal(await fs.readFile(path.join(target, 'a.txt'), 'utf8'), 'new caption');
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
