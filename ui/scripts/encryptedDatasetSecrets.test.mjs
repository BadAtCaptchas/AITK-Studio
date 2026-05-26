import assert from 'node:assert/strict';
import path from 'node:path';
import { afterEach, test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const secrets = require('../dist/src/server/encryptedDatasetSecrets.js');
const envKey = secrets.DURABLE_DATASET_KEY_SECRET_ENV;
const originalSecret = process.env[envKey];

function setSecret(value) {
  process.env[envKey] = value;
}

function restoreSecret() {
  if (originalSecret === undefined) {
    delete process.env[envKey];
  } else {
    process.env[envKey] = originalSecret;
  }
}

afterEach(restoreSecret);

test('durable encrypted dataset keys are wrapped without storing raw key material', () => {
  setSecret('durable-test-secret-alpha-0123456789abcdef');
  const datasetPath = path.resolve('datasets/private');
  const keyB64 = Buffer.from('0123456789abcdef0123456789abcdef', 'utf8').toString('base64');

  const payload = secrets.wrapDurableEncryptedDatasetKeysPayload(
    'job_resume_1',
    [{ datasetPath, keyB64 }],
    new Date('2026-01-01T00:00:00.000Z'),
  );
  const serialized = JSON.stringify(payload);

  assert.equal(payload.version, 2);
  assert.equal(serialized.includes('keyB64'), false);
  assert.equal(serialized.includes(keyB64), false);

  const recovered = secrets.unwrapDurableEncryptedDatasetKeysPayload('job_resume_1', payload);
  assert.deepEqual(recovered, [{ datasetPath, keyB64 }]);
});

test('durable encrypted dataset keys require the same wrapping secret', () => {
  setSecret('durable-test-secret-alpha-0123456789abcdef');
  const datasetPath = path.resolve('datasets/private');
  const keyB64 = Buffer.from('fedcba9876543210fedcba9876543210', 'utf8').toString('base64');
  const payload = secrets.wrapDurableEncryptedDatasetKeysPayload('job_resume_2', [{ datasetPath, keyB64 }]);

  setSecret('durable-test-secret-beta-0123456789abcdef');
  assert.throws(
    () => secrets.unwrapDurableEncryptedDatasetKeysPayload('job_resume_2', payload),
    /authenticate|decrypt|unable/i,
  );

  delete process.env[envKey];
  assert.throws(
    () => secrets.unwrapDurableEncryptedDatasetKeysPayload('job_resume_2', payload),
    /AITK_DURABLE_DATASET_KEY_SECRET/,
  );
});

test('legacy plaintext durable key payloads are identified and ignored', () => {
  const datasetPath = path.resolve('datasets/private');
  const keyB64 = Buffer.from('0123456789abcdef0123456789abcdef', 'utf8').toString('base64');
  const legacy = {
    version: 1,
    createdAt: '2026-01-01T00:00:00.000Z',
    keys: [{ datasetPath, keyB64 }],
  };

  assert.equal(secrets.isLegacyDurableEncryptedDatasetKeysPayload(legacy), true);
  assert.deepEqual(secrets.unwrapDurableEncryptedDatasetKeysPayload('job_resume_3', legacy), []);
});
