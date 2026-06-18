import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const { generateSingleImageRecaption } = require('../dist/src/server/datasetSingleRecaption.js');

function openRouterFetchReturning(content) {
  return async () =>
    new Response(
      JSON.stringify({
        choices: [{ message: { content } }],
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    );
}

test('single-image recaption returns validated OpenRouter text captions', async () => {
  const result = await generateSingleImageRecaption({
    provider: 'openrouter',
    model: 'x-ai/grok-4.3',
    outputFormat: 'text',
    prompt: 'caption it',
    imageDataUrl: 'data:image/png;base64,abc',
    openRouterApiKey: 'test-key',
    fetchImpl: openRouterFetchReturning('A red jacket on a studio chair.'),
  });

  assert.equal(result.caption, 'A red jacket on a studio chair.');
  assert.equal(result.provider, 'openrouter');
});

test('single-image recaption rejects refusal captions', async () => {
  await assert.rejects(
    () =>
      generateSingleImageRecaption({
        provider: 'openrouter',
        model: 'x-ai/grok-4.3',
        outputFormat: 'text',
        prompt: 'caption it',
        imageDataUrl: 'data:image/png;base64,abc',
        openRouterApiKey: 'test-key',
        fetchImpl: openRouterFetchReturning('I cannot fulfill this request.'),
      }),
    /refusal/i,
  );
});
