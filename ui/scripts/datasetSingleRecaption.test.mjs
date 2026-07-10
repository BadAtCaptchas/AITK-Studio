import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const { generateSingleImageRecaption } = require('../dist/src/server/datasetSingleRecaption.js');

function openRouterFetchReturning(content, calls = []) {
  return async (_url, options) => {
    calls.push(JSON.parse(options.body));
    return new Response(
      JSON.stringify({
        choices: [{ message: { content } }],
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      },
    );
  };
}

test('single-image recaption returns validated OpenRouter text captions', async () => {
  const calls = [];
  const result = await generateSingleImageRecaption({
    provider: 'openrouter',
    model: 'x-ai/grok-4.3',
    outputFormat: 'text',
    prompt: 'caption it',
    imageDataUrl: 'data:image/png;base64,abc',
    openRouterApiKey: 'test-key',
    fetchImpl: openRouterFetchReturning('A red jacket on a studio chair.', calls),
  });

  assert.equal(result.caption, 'A red jacket on a studio chair.');
  assert.equal(result.provider, 'openrouter');
  assert.deepEqual(calls[0].provider, { order: ['xai/zdr'] });
});

test('single-image recaption removes triple dash separators from generated captions', async () => {
  const result = await generateSingleImageRecaption({
    provider: 'openrouter',
    model: 'x-ai/grok-4.3',
    outputFormat: 'text',
    prompt: 'caption it',
    imageDataUrl: 'data:image/png;base64,abc',
    openRouterApiKey: 'test-key',
    fetchImpl: openRouterFetchReturning('---\nA red jacket---on a studio chair.\n---'),
  });

  assert.equal(result.caption, 'A red jacket on a studio chair.');
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
