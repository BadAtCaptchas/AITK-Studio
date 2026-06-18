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

async function waitFor(predicate) {
  for (let i = 0; i < 10; i += 1) {
    if (predicate()) return;
    await new Promise(resolve => setTimeout(resolve, 0));
  }
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

test('listOllamaModels reports unreachable Ollama base URL', async () => {
  globalThis.fetch = async () => {
    throw new TypeError('fetch failed');
  };

  await assert.rejects(
    ollama.listOllamaModels('http://ollama.test'),
    /Ollama model list failed at http:\/\/ollama\.test\/api\/tags: fetch failed/,
  );
});

test('listOllamaModels sends optional bearer token', async () => {
  globalThis.fetch = async (url, init = {}) => {
    assert.equal(String(url), 'http://ollama.test/api/tags');
    assert.equal(new Headers(init.headers).get('Authorization'), 'Bearer remote-token');
    return response({ models: [{ model: 'llava:latest' }] });
  };

  const models = await ollama.listOllamaModels({
    baseUrl: 'http://ollama.test',
    authToken: 'remote-token',
  });

  assert.deepEqual(models, [{ model: 'llava:latest' }]);
});

test('startOllamaModelPull warms installed model before reporting ready', async () => {
  const calls = [];
  let resolveWarm;
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), method: init?.method || 'GET' });
    if (String(url).endsWith('/api/tags')) return response({ models: [{ model: 'llava-ready:latest' }] });
    if (String(url).endsWith('/api/generate')) {
      return new Promise(resolve => {
        resolveWarm = () => resolve(response({ done: true, done_reason: 'load' }));
      });
    }
    throw new Error(`Unexpected URL: ${url}`);
  };

  const first = await ollama.startOllamaModelPull('llava-ready', 'http://ollama.test');
  const second = await ollama.startOllamaModelPull('llava-ready', 'http://ollama.test');
  await waitFor(() => calls.some(call => call.url.endsWith('/api/generate')));

  assert.equal(first.status, 'pulling');
  assert.equal(second.status, 'pulling');
  assert.equal(calls.filter(call => call.url.endsWith('/api/generate')).length, 1);

  resolveWarm();
  await new Promise(resolve => setTimeout(resolve, 0));
  const third = await ollama.startOllamaModelPull('llava-ready', 'http://ollama.test');
  assert.equal(third.status, 'ready');
  assert.equal(third.error, null);
});

test('startOllamaModelPull starts missing model pull without awaiting completion', async () => {
  const calls = [];
  let resolvePull;
  let resolveWarm;
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), method: init?.method || 'GET', body: init?.body });
    if (String(url).endsWith('/api/tags')) return response({ models: [] });
    if (String(url).endsWith('/api/pull')) {
      return new Promise(resolve => {
        resolvePull = () => resolve(response({ status: 'success' }));
      });
    }
    if (String(url).endsWith('/api/generate')) {
      return new Promise(resolve => {
        resolveWarm = () => resolve(response({ done: true, done_reason: 'load' }));
      });
    }
    throw new Error(`Unexpected URL: ${url}`);
  };

  const first = await ollama.startOllamaModelPull('llava-pulling:latest', 'http://ollama.test');
  const second = await ollama.startOllamaModelPull('llava-pulling:latest', 'http://ollama.test');
  await waitFor(() => calls.some(call => call.url.endsWith('/api/pull')));

  assert.equal(first.status, 'pulling');
  assert.equal(second.status, 'pulling');
  assert.equal(calls.filter(call => call.url.endsWith('/api/pull')).length, 1);
  assert.equal(JSON.parse(calls.find(call => call.url.endsWith('/api/pull')).body).model, 'llava-pulling:latest');

  resolvePull();
  await waitFor(() => calls.some(call => call.url.endsWith('/api/generate')));
  assert.equal(calls.filter(call => call.url.endsWith('/api/generate')).length, 1);

  resolveWarm();
  await new Promise(resolve => setTimeout(resolve, 0));
  const third = await ollama.startOllamaModelPull('llava-pulling:latest', 'http://ollama.test');
  assert.equal(third.status, 'ready');
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
  assert.equal('think' in generateBodies[0], false);
  assert.deepEqual(generateBodies[0].options, { num_predict: 2048 });
});

test('generateOllamaImageCaption adds thinking budget to larger requested token budgets', async () => {
  const generateBodies = [];
  globalThis.fetch = async (url, init) => {
    if (String(url).endsWith('/api/tags')) return response({ models: [{ model: 'llava:latest' }] });
    if (String(url).endsWith('/api/generate')) {
      generateBodies.push(JSON.parse(init.body));
      return response({ response: 'a longer caption' });
    }
    throw new Error(`Unexpected URL: ${url}`);
  };

  const caption = await ollama.generateOllamaImageCaption(
    { model: 'llava', prompt: 'caption', imageBase64: 'aW1n', maxNewTokens: 768 },
    'http://ollama.test',
  );

  assert.equal(caption, 'a longer caption');
  assert.deepEqual(generateBodies[0].options, { num_predict: 3072 });
});

test('generateOllamaImageCaption falls back to chat when generate is empty', async () => {
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), method: init?.method || 'GET', body: init?.body });
    if (String(url).endsWith('/api/tags')) return response({ models: [{ model: 'llava:latest' }] });
    if (String(url).endsWith('/api/generate')) return response({ response: '', done_reason: 'load' });
    if (String(url).endsWith('/api/chat')) return response({ message: { content: 'chat caption' } });
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

  assert.equal(caption, 'chat caption');
  assert.deepEqual(
    calls.filter(call => call.url.endsWith('/api/generate') || call.url.endsWith('/api/chat')).map(call => call.url.split('/api/')[1]),
    ['generate', 'chat'],
  );
  const chatBody = JSON.parse(calls.find(call => call.url.endsWith('/api/chat')).body);
  assert.deepEqual(chatBody.messages, [
    { role: 'system', content: 'Return compact training captions.' },
    { role: 'user', content: 'caption', images: ['aW1n'] },
  ]);
  assert.equal('think' in chatBody, false);
  assert.deepEqual(chatBody.options, { num_predict: 2048 });
});

test('generateOllamaImageCaption tries chat before generate for Gemma models', async () => {
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), method: init?.method || 'GET', body: init?.body });
    if (String(url).endsWith('/api/tags')) return response({ models: [{ model: 'gemma4:31b' }] });
    if (String(url).endsWith('/api/chat')) return response({ message: { content: '' }, done_reason: 'stop' });
    if (String(url).endsWith('/api/generate')) return response({ response: 'generate caption' });
    throw new Error(`Unexpected URL: ${url}`);
  };

  const caption = await ollama.generateOllamaImageCaption(
    {
      model: 'gemma4:31b',
      prompt: 'caption',
      systemPrompt: 'Return compact training captions.',
      imageBase64: 'aW1n',
      maxNewTokens: 32,
    },
    'http://ollama.test',
  );

  assert.equal(caption, 'generate caption');
  assert.deepEqual(
    calls.filter(call => call.url.endsWith('/api/chat') || call.url.endsWith('/api/generate')).map(call => call.url.split('/api/')[1]),
    ['chat', 'generate'],
  );
  const chatBody = JSON.parse(calls.find(call => call.url.endsWith('/api/chat')).body);
  assert.deepEqual(chatBody.messages, [
    { role: 'system', content: 'Return compact training captions.' },
    { role: 'user', content: 'caption', images: ['aW1n'] },
  ]);
});

test('generateOllamaImageCaption expands retry budget for thinking-only length responses', async () => {
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), method: init?.method || 'GET', body: init?.body });
    if (String(url).endsWith('/api/tags')) return response({ models: [{ model: 'qwen3-vl:latest' }] });
    if (String(url).endsWith('/api/generate')) return response({ response: '', thinking: 'private trace', done_reason: 'length' });
    if (String(url).endsWith('/api/chat')) {
      return response({ message: { content: '', thinking: 'private trace' }, done_reason: 'length' });
    }
    throw new Error(`Unexpected URL: ${url}`);
  };

  await assert.rejects(
    ollama.generateOllamaImageCaption(
      { model: 'qwen3-vl', prompt: 'caption', imageBase64: 'aW1n', maxNewTokens: 180 },
      'http://ollama.test',
    ),
    /with thinking/,
  );

  const generationBodies = calls
    .filter(call => call.url.endsWith('/api/generate') || call.url.endsWith('/api/chat'))
    .map(call => JSON.parse(call.body));
  assert.deepEqual(
    generationBodies.map(body => body.options.num_predict),
    [2048, 2048, 4096, 4096, 4096, 4096],
  );
});

test('generateOllamaImageCaption accepts message content response shape', async () => {
  globalThis.fetch = async (url, init) => {
    if (String(url).endsWith('/api/tags')) return response({ models: [{ model: 'llava:latest' }] });
    if (String(url).endsWith('/api/generate')) return response({ message: { content: 'nested caption' } });
    throw new Error(`Unexpected URL: ${url}`);
  };

  const caption = await ollama.generateOllamaImageCaption(
    { model: 'llava', prompt: 'caption', imageBase64: 'aW1n', maxNewTokens: 32 },
    'http://ollama.test',
  );

  assert.equal(caption, 'nested caption');
});

test('generateOllamaImageCaption rejects empty model responses', async () => {
  globalThis.fetch = async url => {
    if (String(url).endsWith('/api/tags')) return response({ models: [{ model: 'llava:latest' }] });
    if (String(url).endsWith('/api/generate')) return response({ response: '', done_reason: 'stop' });
    if (String(url).endsWith('/api/chat')) return response({ message: { content: '' }, done_reason: 'stop' });
    throw new Error(`Unexpected URL: ${url}`);
  };

  await assert.rejects(
    ollama.generateOllamaImageCaption(
      { model: 'llava', prompt: 'caption', imageBase64: 'aW1n', maxNewTokens: 32 },
      'http://ollama.test',
    ),
    /empty caption/,
  );
});

test('unloadOllamaModel sends keep_alive zero generate request', async () => {
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), method: init?.method || 'GET', body: init?.body });
    if (String(url).endsWith('/api/generate')) return response({ done: true });
    throw new Error(`Unexpected URL: ${url}`);
  };

  const result = await ollama.unloadOllamaModel('llava:latest', 'http://ollama.test');

  assert.deepEqual(result, { unloaded: true });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].method, 'POST');
  assert.deepEqual(JSON.parse(calls[0].body), {
    model: 'llava:latest',
    prompt: '',
    stream: false,
    keep_alive: 0,
  });
});

test('unloadOllamaModel treats missing model as already unloaded', async () => {
  globalThis.fetch = async (url, init) => {
    if (String(url).endsWith('/api/generate')) {
      assert.equal(JSON.parse(init.body).keep_alive, 0);
      return response({ error: "model 'gemma4:31b' not found" }, 404);
    }
    throw new Error(`Unexpected URL: ${url}`);
  };

  const result = await ollama.unloadOllamaModel('gemma4:31b', 'http://ollama.test');

  assert.deepEqual(result, { unloaded: false, reason: 'model_not_found' });
});
