import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const {
  buildOpenRouterBoxPrompt,
  generateOpenRouterBoxPatches,
} = require('../dist/src/server/openRouterBoxes.js');

function sampleCaption() {
  return JSON.stringify(
    {
      high_level_description: 'A city street with a taxi.',
      style_description: {
        aesthetics: 'realistic',
        lighting: 'daylight',
        photo: 'street photo',
        medium: 'photograph',
        color_palette: ['#111111'],
      },
      compositional_deconstruction: {
        background: 'Urban street.',
        elements: [
          {
            type: 'obj',
            bbox: [120, 200, 620, 800],
            desc: 'Yellow taxi.',
            color_palette: ['#22D3EE'],
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

function mockFetchForResponses(responses, calls = []) {
  return async (_url, options) => {
    calls.push(JSON.parse(options.body));
    const next = responses.shift();
    return new Response(
      JSON.stringify({
        choices: [{ message: { content: JSON.stringify(next.content) } }],
        usage: next.usage,
      }),
      { status: next.status || 200, headers: { 'Content-Type': 'application/json' } },
    );
  };
}

test('buildOpenRouterBoxPrompt includes indexed element context and image dimensions', () => {
  const prompt = buildOpenRouterBoxPrompt(sampleCaption(), { width: 1280, height: 720 });
  assert.match(prompt, /Image pixel size: 1280 x 720/);
  assert.match(prompt, /bbox_px/);
  assert.match(prompt, /currentBbox_px/);
  assert.match(prompt, /"elementIndex": 0/);
  assert.match(prompt, /Yellow taxi/);
  assert.match(prompt, /visibleText/);
  assert.match(prompt, /generatedElements: \[\]/);
});

test('buildOpenRouterBoxPrompt can request new elements when none exist', () => {
  const prompt = buildOpenRouterBoxPrompt(emptyElementCaption(), { width: 1280, height: 720 });
  assert.match(prompt, /currently has no compositional_deconstruction\.elements/);
  assert.match(prompt, /generatedElements/);
  assert.match(prompt, /Return boxes: \[\]/);
});

test('generateOpenRouterBoxPatches defaults to grok 4.3 and filters malformed boxes', async () => {
  const calls = [];
  const fetchImpl = mockFetchForResponses(
    [
      {
        content: {
          boxes: [
            { elementIndex: 0, bbox_px: [100, 380, 640, 1640] },
            { elementIndex: 1, bbox_px: [200, 400, 200, 520] },
            { elementIndex: 99, bbox_px: [0, 0, 100, 100] },
          ],
        },
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      },
    ],
    calls,
  );

  const result = await generateOpenRouterBoxPatches({
    apiKey: 'test-key',
    imageDataUrl: 'data:image/jpeg;base64,abc',
    caption: sampleCaption(),
    imageSize: { width: 2000, height: 1000 },
    fetchImpl,
  });

  assert.equal(result.model, 'x-ai/grok-4.3');
  assert.equal(result.refined, false);
  assert.deepEqual(result.boxes, [{ elementIndex: 0, bbox: [100, 190, 640, 820] }]);
  assert.equal(result.usage.total_tokens, 15);
  assert.equal(calls[0].provider.require_parameters, true);
  assert.equal(calls[0].response_format.json_schema.strict, true);
  assert.deepEqual(calls[0].response_format.json_schema.schema.required, ['boxes', 'generatedElements']);
  assert.deepEqual(calls[0].response_format.json_schema.schema.properties.boxes.items.required, ['elementIndex', 'bbox_px']);
});

test('generateOpenRouterBoxPatches falls back for retired models and can run a refinement pass', async () => {
  const calls = [];
  const fetchImpl = mockFetchForResponses(
    [
      {
        content: { boxes: [{ elementIndex: 0, bbox_px: [100, 380, 640, 1640] }] },
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      },
      {
        content: { boxes: [{ elementIndex: 0, bbox_px: [110, 400, 630, 1620] }] },
        usage: { prompt_tokens: 12, completion_tokens: 6, total_tokens: 18 },
      },
    ],
    calls,
  );

  const result = await generateOpenRouterBoxPatches({
    apiKey: 'test-key',
    imageDataUrl: 'data:image/jpeg;base64,abc',
    caption: sampleCaption(),
    model: 'x-ai/grok-4-fast',
    refine: true,
    imageSize: { width: 2000, height: 1000 },
    fetchImpl,
  });

  assert.equal(calls.length, 2);
  assert.equal(result.model, 'x-ai/grok-4.3');
  assert.equal(result.refined, true);
  assert.deepEqual(result.boxes, [{ elementIndex: 0, bbox: [110, 200, 630, 810] }]);
  assert.equal(result.usage.prompt_tokens, 22);
  assert.equal(result.usage.total_tokens, 33);
  assert.match(calls[1].messages[0].content[0].text, /Current proposed boxes to correct/);
});

test('generateOpenRouterBoxPatches can create elements when caption has no elements', async () => {
  const calls = [];
  const fetchImpl = mockFetchForResponses(
    [
      {
        content: {
          boxes: [],
          generatedElements: [
            { type: 'obj', bbox_px: [100, 380, 640, 1640], desc: 'Yellow taxi.', text: '' },
            { type: 'text', bbox_px: [60, 720, 120, 1040], desc: 'Taxi roof sign.', text: 'TAXI' },
            { type: 'obj', bbox_px: [200, 400, 200, 520], desc: 'flat bad box', text: '' },
          ],
        },
      },
    ],
    calls,
  );

  const result = await generateOpenRouterBoxPatches({
    apiKey: 'test-key',
    imageDataUrl: 'data:image/jpeg;base64,abc',
    caption: emptyElementCaption(),
    imageSize: { width: 2000, height: 1000 },
    fetchImpl,
  });

  assert.deepEqual(result.boxes, []);
  assert.deepEqual(result.generatedElements, [
    { type: 'obj', bbox: [100, 190, 640, 820], desc: 'Yellow taxi.' },
    { type: 'text', bbox: [60, 360, 120, 520], desc: 'Taxi roof sign.', text: 'TAXI' },
  ]);
  assert.match(calls[0].messages[0].content[0].text, /generate new elements/i);
});

test('generateOpenRouterBoxPatches can refine generated elements', async () => {
  const calls = [];
  const fetchImpl = mockFetchForResponses(
    [
      {
        content: {
          boxes: [],
          generatedElements: [{ type: 'obj', bbox_px: [100, 380, 640, 1640], desc: 'Yellow taxi.', text: '' }],
        },
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      },
      {
        content: {
          boxes: [],
          generatedElements: [{ type: 'obj', bbox_px: [110, 400, 630, 1620], desc: 'Yellow taxi.', text: '' }],
        },
        usage: { prompt_tokens: 12, completion_tokens: 6, total_tokens: 18 },
      },
    ],
    calls,
  );

  const result = await generateOpenRouterBoxPatches({
    apiKey: 'test-key',
    imageDataUrl: 'data:image/jpeg;base64,abc',
    caption: emptyElementCaption(),
    refine: true,
    imageSize: { width: 2000, height: 1000 },
    fetchImpl,
  });

  assert.equal(result.refined, true);
  assert.deepEqual(result.generatedElements, [{ type: 'obj', bbox: [110, 200, 630, 810], desc: 'Yellow taxi.' }]);
  assert.equal(result.usage.total_tokens, 33);
  assert.match(calls[1].messages[0].content[0].text, /Current proposed generated elements to correct/);
});

test('generateOpenRouterBoxPatches rejects missing key and unusable model output', async () => {
  await assert.rejects(
    () =>
      generateOpenRouterBoxPatches({
        apiKey: '',
        imageDataUrl: 'data:image/jpeg;base64,abc',
        caption: sampleCaption(),
        fetchImpl: mockFetchForResponses([]),
      }),
    /API key is missing/,
  );

  await assert.rejects(
    () =>
      generateOpenRouterBoxPatches({
        apiKey: 'test-key',
        imageDataUrl: 'data:image/jpeg;base64,abc',
        caption: sampleCaption(),
        imageSize: { width: 2000, height: 1000 },
        fetchImpl: mockFetchForResponses([{ content: { boxes: [] } }]),
      }),
    /usable boxes/,
  );
});
