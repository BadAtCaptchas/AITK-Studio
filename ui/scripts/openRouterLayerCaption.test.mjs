import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const {
  buildOpenRouterLayerCaptionPrompt,
  generateOpenRouterLayerCaption,
} = require('../dist/src/server/openRouterLayerCaption.js');

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
          {
            type: 'obj',
            desc: '',
          },
        ],
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

test('buildOpenRouterLayerCaptionPrompt targets an existing bbox when present', () => {
  const prompt = buildOpenRouterLayerCaptionPrompt(sampleCaption(), 0, { width: 1280, height: 720 });
  assert.match(prompt, /Image pixel size: 1280 x 720/);
  assert.match(prompt, /Target bbox_px: \[86,256,446,1024\]/);
  assert.match(prompt, /bbox_px/);
  assert.match(prompt, /color_palette/);
  assert.match(prompt, /Yellow taxi/);
  assert.doesNotMatch(prompt, /No bbox exists/);
});

test('buildOpenRouterLayerCaptionPrompt uses selected layer clue when no bbox exists', () => {
  const prompt = buildOpenRouterLayerCaptionPrompt(sampleCaption(), 1, { width: 1280, height: 720 });
  assert.match(prompt, /No bbox exists/);
  assert.match(prompt, /return a tight bbox/);
  assert.match(prompt, /TAXI/);
  assert.match(prompt, /Roof sign text/);
});

test('buildOpenRouterLayerCaptionPrompt rejects a no-box layer without a target clue', () => {
  assert.throws(
    () => buildOpenRouterLayerCaptionPrompt(sampleCaption(), 2, { width: 1280, height: 720 }),
    /Add a layer label or draw a box first/,
  );
});

test('generateOpenRouterLayerCaption parses text-layer captions and strict schema calls', async () => {
  const calls = [];
  const fetchImpl = mockFetchForResponses(
    [
      {
        content: {
          desc: 'A yellow taxi roof sign with black letters.',
          text: 'TAXI',
          bbox_px: [80, 900, 150, 1220],
          color_palette: ['#facc15', '#111111', 'invalid', '#FACC15', '#FFFFFF', '#22D3EE', '#EF4444'],
        },
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      },
    ],
    calls,
  );

  const result = await generateOpenRouterLayerCaption({
    apiKey: 'test-key',
    imageDataUrl: 'data:image/jpeg;base64,abc',
    caption: sampleCaption(),
    elementIndex: 1,
    model: 'x-ai/grok-4-fast',
    imageSize: { width: 2000, height: 1000 },
    fetchImpl,
  });

  assert.deepEqual(result, {
    desc: 'A yellow taxi roof sign with black letters.',
    text: 'TAXI',
    bbox: [80, 450, 150, 610],
    color_palette: ['#FACC15', '#111111', '#FFFFFF', '#22D3EE', '#EF4444'],
    model: 'x-ai/grok-4.3',
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  });
  assert.equal(calls[0].provider.require_parameters, true);
  assert.equal(calls[0].response_format.json_schema.strict, true);
  assert.equal(calls[0].response_format.json_schema.name, 'dataset_layer_caption');
  assert.deepEqual(calls[0].response_format.json_schema.schema.required, ['desc', 'text', 'bbox_px', 'color_palette']);
  assert.equal(calls[0].response_format.json_schema.schema.properties.color_palette.maxItems, 5);
});

test('generateOpenRouterLayerCaption ignores returned visible text for object layers', async () => {
  const result = await generateOpenRouterLayerCaption({
    apiKey: 'test-key',
    imageDataUrl: 'data:image/jpeg;base64,abc',
    caption: sampleCaption(),
    elementIndex: 0,
    imageSize: { width: 2000, height: 1000 },
    fetchImpl: mockFetchForResponses([
      {
        content: {
          desc: 'A yellow taxi cab viewed from the side.',
          text: 'TAXI',
          bbox_px: [120, 400, 620, 1600],
          color_palette: ['#FACC15', '#111111'],
        },
      },
    ]),
  });

  assert.equal(result.model, 'x-ai/grok-4.3');
  assert.equal(result.desc, 'A yellow taxi cab viewed from the side.');
  assert.deepEqual(result.bbox, [120, 200, 620, 800]);
  assert.deepEqual(result.color_palette, ['#FACC15', '#111111']);
  assert.equal(Object.prototype.hasOwnProperty.call(result, 'text'), false);
});

test('generateOpenRouterLayerCaption requires a usable bbox for no-box layers', async () => {
  await assert.rejects(
    () =>
      generateOpenRouterLayerCaption({
        apiKey: 'test-key',
        imageDataUrl: 'data:image/jpeg;base64,abc',
        caption: sampleCaption(),
        elementIndex: 1,
        imageSize: { width: 2000, height: 1000 },
        fetchImpl: mockFetchForResponses([
          { content: { desc: 'A yellow taxi roof sign.', text: 'TAXI', bbox_px: [10, 20, 10, 40], color_palette: [] } },
        ]),
      }),
    /usable layer box/,
  );
});
