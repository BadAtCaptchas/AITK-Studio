import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const {
  DEFAULT_OLLAMA_VISION_MODEL,
  OLLAMA_VISION_MODELS,
  generateOllamaBoxPatches,
  generateOllamaLayerCaption,
  generateRemoteOllamaVisionCaption,
  normalizeOllamaVisionModel,
} = require('../dist/src/server/ollamaVision.js');

function sampleCaption() {
  return JSON.stringify(
    {
      high_level_description: 'A city street with a taxi.',
      compositional_deconstruction: {
        background: 'Urban street.',
        elements: [
          {
            type: 'obj',
            bbox: [120, 200, 620, 800],
            desc: 'Yellow taxi.',
          },
          {
            type: 'text',
            text: 'TAXI',
            desc: 'Roof sign text.',
          },
        ],
      },
    },
    null,
    2,
  );
}

function emptyElementCaption() {
  return JSON.stringify(
    {
      high_level_description: 'A city street with a taxi.',
      compositional_deconstruction: {
        background: 'Urban street.',
        elements: [],
      },
    },
    null,
    2,
  );
}

test('Ollama model normalization defaults to recommended Qwen vision model', () => {
  assert.equal(DEFAULT_OLLAMA_VISION_MODEL, 'qwen3.5:35b');
  assert.ok(OLLAMA_VISION_MODELS.includes('gemma4:31b'));
  assert.ok(OLLAMA_VISION_MODELS.includes('gemma4:26b'));
  assert.equal(normalizeOllamaVisionModel(''), 'qwen3.5:35b');
  assert.equal(normalizeOllamaVisionModel(null), 'qwen3.5:35b');
  assert.equal(normalizeOllamaVisionModel('custom-vision:latest'), 'custom-vision:latest');
});

test('generateOllamaBoxPatches parses JSON and normalizes pixel boxes', async () => {
  const calls = [];
  const result = await generateOllamaBoxPatches({
    imageDataUrl: 'data:image/jpeg;base64,aW1n',
    caption: sampleCaption(),
    model: '',
    imageSize: { width: 2000, height: 1000 },
    captionRunner: async options => {
      calls.push(options);
      return JSON.stringify({
        boxes: [
          { elementIndex: 0, bbox_px: [100, 380, 640, 1640], color_palette: ['#facc15', 'bad'] },
          { elementIndex: 99, bbox_px: [0, 0, 100, 100], color_palette: ['#000000'] },
        ],
        generatedElements: [],
      });
    },
  });

  assert.equal(result.model, 'qwen3.5:35b');
  assert.deepEqual(result.boxes, [{ elementIndex: 0, bbox: [100, 190, 640, 820], color_palette: ['#FACC15'] }]);
  assert.deepEqual(result.generatedElements, []);
  assert.equal(calls[0].imageBase64, 'aW1n');
  assert.match(calls[0].prompt, /bbox_px/);
  assert.match(calls[0].systemPrompt, /Return only valid JSON/);
});

test('generateOllamaBoxPatches can create elements when caption has no elements', async () => {
  const result = await generateOllamaBoxPatches({
    imageDataUrl: 'data:image/jpeg;base64,aW1n',
    caption: emptyElementCaption(),
    model: 'qwen3.5:27b',
    imageSize: { width: 2000, height: 1000 },
    captionRunner: async () =>
      JSON.stringify({
        boxes: [],
        generatedElements: [
          { type: 'obj', bbox_px: [100, 380, 640, 1640], desc: 'Yellow taxi.', text: '', color_palette: ['#FACC15'] },
          { type: 'text', bbox_px: [60, 720, 120, 1040], desc: 'Taxi roof sign.', text: 'TAXI', color_palette: ['#111111'] },
        ],
      }),
  });

  assert.equal(result.model, 'qwen3.5:27b');
  assert.deepEqual(result.boxes, []);
  assert.deepEqual(result.generatedElements, [
    { type: 'obj', bbox: [100, 190, 640, 820], desc: 'Yellow taxi.', color_palette: ['#FACC15'] },
    { type: 'text', bbox: [60, 360, 120, 520], desc: 'Taxi roof sign.', text: 'TAXI', color_palette: ['#111111'] },
  ]);
});

test('generateOllamaLayerCaption handles no-box targets and color palettes', async () => {
  const result = await generateOllamaLayerCaption({
    imageDataUrl: 'data:image/jpeg;base64,aW1n',
    caption: sampleCaption(),
    elementIndex: 1,
    model: '',
    imageSize: { width: 2000, height: 1000 },
    captionRunner: async () =>
      JSON.stringify({
        desc: 'A yellow taxi roof sign with black letters.',
        text: 'TAXI',
        bbox_px: [80, 900, 150, 1220],
        color_palette: ['#facc15', '#111111', 'invalid'],
      }),
  });

  assert.equal(result.model, 'qwen3.5:35b');
  assert.equal(result.desc, 'A yellow taxi roof sign with black letters.');
  assert.equal(result.text, 'TAXI');
  assert.deepEqual(result.bbox, [80, 450, 150, 610]);
  assert.deepEqual(result.color_palette, ['#FACC15', '#111111']);
});

test('generateRemoteOllamaVisionCaption calls direct Ollama endpoint with optional auth', async () => {
  const originalFetch = globalThis.fetch;
  const token = 'remote-token';
  const worker = { id: 'worker-1', name: 'DGX', base_url: 'https://ollama.test', auth_token: token };
  const calls = [];

  try {
    globalThis.fetch = async (url, init = {}) => {
      calls.push({ url: String(url), init });
      const headers = new Headers(init.headers);
      assert.equal(headers.get('Authorization'), `Bearer ${token}`);
      if (String(url).endsWith('/api/tags')) {
        return new Response(JSON.stringify({ models: [{ model: 'qwen3.5:35b' }] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      if (String(url).endsWith('/api/generate')) {
        const body = JSON.parse(init.body);
        assert.equal(body.model, 'qwen3.5:35b');
        assert.equal(body.prompt, 'Return boxes.');
        assert.equal(body.images[0], 'aW1n');
        assert.match(body.system, /NSFW/);
        return new Response(JSON.stringify({ response: '{"boxes":[],"generatedElements":[]}' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      throw new Error(`Unexpected URL: ${url}`);
    };

    const caption = await generateRemoteOllamaVisionCaption({
      remoteWorkerId: 'worker-1',
      model: '',
      prompt: 'Return boxes.',
      imageBase64: 'aW1n',
      maxNewTokens: 256,
      getWorkerImpl: async workerId => {
        assert.equal(workerId, 'worker-1');
        return worker;
      },
    });

    assert.equal(caption, '{"boxes":[],"generatedElements":[]}');
    assert.deepEqual(
      calls.map(call => call.url),
      ['https://ollama.test/api/tags', 'https://ollama.test/api/generate'],
    );
    assert.equal(calls[1].init.method, 'POST');
  } finally {
    globalThis.fetch = originalFetch;
  }
});
