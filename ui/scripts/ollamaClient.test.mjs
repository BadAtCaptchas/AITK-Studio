import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const ollama = require('../dist/src/server/ollama.js');
const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

function response(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

test('ensureOllamaModel skips pull when model is already installed', async () => {
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), method: init?.method || 'GET' });
    assert.match(String(url), /\/api\/tags$/);
    return response({ models: [{ name: 'llava:latest', model: 'llava:latest' }] });
  };

  const result = await ollama.ensureOllamaModel('llava', 'http://ollama.test');

  assert.deepEqual(result, { pulled: false });
  assert.equal(calls.length, 1);
});

test('ensureOllamaModel pulls missing model', async () => {
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), method: init?.method || 'GET', body: init?.body });
    if (String(url).endsWith('/api/tags')) return response({ models: [] });
    if (String(url).endsWith('/api/pull')) return response({ status: 'success' });
    throw new Error(`Unexpected URL: ${url}`);
  };

  const result = await ollama.ensureOllamaModel('llava:13b', 'http://ollama.test');

  assert.deepEqual(result, { pulled: true });
  assert.equal(calls.length, 2);
  assert.equal(calls[1].method, 'POST');
  assert.equal(JSON.parse(calls[1].body).model, 'llava:13b');
});

test('generateOllamaImageCaption returns clean errors from Ollama failures', async () => {
  globalThis.fetch = async url => {
    if (String(url).endsWith('/api/tags')) return response({ models: [{ model: 'llava:latest' }] });
    if (String(url).endsWith('/api/generate')) return response({ error: 'vision model failed' }, 500);
    throw new Error(`Unexpected URL: ${url}`);
  };

  await assert.rejects(
    ollama.generateOllamaImageCaption(
      { model: 'llava', prompt: 'caption', imageBase64: 'aW1n', maxNewTokens: 32 },
      'http://ollama.test',
    ),
    /vision model failed/,
  );
});

test('generateOllamaImageCaption sends system prompt to Ollama generate', async () => {
  const generateBodies = [];
  globalThis.fetch = async (url, init) => {
    if (String(url).endsWith('/api/tags')) return response({ models: [{ model: 'llava:latest' }] });
    if (String(url).endsWith('/api/generate')) {
      generateBodies.push(JSON.parse(init.body));
      return response({ response: 'a caption' });
    }
    throw new Error(`Unexpected URL: ${url}`);
  };

  const caption = await ollama.generateOllamaImageCaption(
    {
      model: 'llava',
      prompt: 'caption',
      systemPrompt: 'Return compact training captions.',
      imageBase64: 'aW1n',
      maxNewTokens: 32,
    },
    'http://ollama.test',
  );

  assert.equal(caption, 'a caption');
  assert.equal(generateBodies.length, 1);
  assert.equal(generateBodies[0].system, 'Return compact training captions.');
  assert.deepEqual(generateBodies[0].options, { num_predict: 32 });
});
