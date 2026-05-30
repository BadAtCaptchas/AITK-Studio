import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const encryptedDatasets = require('../dist/src/server/encryptedDatasets.js');
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
