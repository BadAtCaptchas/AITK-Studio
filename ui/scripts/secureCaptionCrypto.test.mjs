import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const crypto = require('../dist/src/server/secureCaptionCrypto.js');

test('secure caption crypto round-trips JSON with matching token and context', () => {
  const envelope = crypto.encryptSecureCaptionJson('worker-token', 'request', 'job-1', 'item-1', {
    prompt: 'private prompt',
    imageBase64: 'aW1hZ2UtYnl0ZXM=',
  });

  const recovered = crypto.decryptSecureCaptionJson('worker-token', 'request', envelope);

  assert.equal(recovered.prompt, 'private prompt');
  assert.equal(recovered.imageBase64, 'aW1hZ2UtYnl0ZXM=');
});

test('secure caption crypto rejects wrong token and wrong AAD direction', () => {
  const envelope = crypto.encryptSecureCaptionJson('worker-token', 'response', 'job-1', 'item-2', {
    caption: 'private caption',
  });

  assert.throws(() => crypto.decryptSecureCaptionJson('other-token', 'response', envelope));
  assert.throws(() => crypto.decryptSecureCaptionJson('worker-token', 'request', envelope));
});

test('secure caption envelopes do not contain plaintext prompt or caption markers', () => {
  const envelope = crypto.encryptSecureCaptionJson('worker-token', 'request', 'job-1', 'item-3', {
    prompt: 'SECRET_PROMPT_MARKER',
    systemPrompt: 'SECRET_SYSTEM_PROMPT_MARKER',
    caption: 'SECRET_CAPTION_MARKER',
  });
  const serialized = JSON.stringify(envelope);

  assert.equal(serialized.includes('SECRET_PROMPT_MARKER'), false);
  assert.equal(serialized.includes('SECRET_SYSTEM_PROMPT_MARKER'), false);
  assert.equal(serialized.includes('SECRET_CAPTION_MARKER'), false);
});
