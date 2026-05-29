import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const encryptedDatasets = require('../dist/src/server/encryptedDatasets.js');

const CATALOG_AAD = Buffer.from('aitk-encrypted-catalog:v1', 'utf8');

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

test('validateEncryptedCatalogKey accepts the matching dataset key', () => {
  const key = crypto.randomBytes(32);
  const manifest = manifestForKey(key);

  assert.equal(encryptedDatasets.validateEncryptedCatalogKey(manifest, key.toString('base64')), true);
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
