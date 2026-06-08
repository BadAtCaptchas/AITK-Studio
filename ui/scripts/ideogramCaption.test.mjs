import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import test from 'node:test';

const require = createRequire(import.meta.url);
const {
  addIdeogramElement,
  appendGeneratedIdeogramElements,
  applyGeneratedBoxPatches,
  boxToArray,
  deleteIdeogramElement,
  duplicateIdeogramElement,
  normalizeGeneratedElementBoxes,
  normalizeGeneratedBoxPatches,
  parseIdeogramCaption,
  rectToBox,
  serializeIdeogramCaption,
  updateIdeogramElementBox,
  updateIdeogramElementField,
  updateIdeogramElementPalette,
  updateIdeogramElementType,
} = require('../dist/src/utils/ideogramCaption.js');

function sampleCaption() {
  return {
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
      ],
    },
  };
}

test('parseIdeogramCaption distinguishes plain, invalid JSON, and Ideogram JSON', () => {
  assert.equal(parseIdeogramCaption('plain prompt').kind, 'plain');
  assert.equal(parseIdeogramCaption('{"caption":"plain json"}').kind, 'json');
  const parsed = parseIdeogramCaption(JSON.stringify(sampleCaption()));
  assert.equal(parsed.kind, 'ideogram');
  assert.equal(parsed.boxes.length, 1);
  assert.equal(parsed.boxes[0].label, 'Yellow taxi.');
});

test('parseIdeogramCaption normalizes fractional 0..1 boxes to 0..1000 space', () => {
  const parsed = parseIdeogramCaption(
    JSON.stringify({
      compositional_deconstruction: {
        elements: [
          {
            type: 'obj',
            bbox: [0.1, 0.2, 0.3, 0.4],
            desc: 'Fractional box',
          },
        ],
      },
    }),
  );

  assert.equal(parsed.kind, 'ideogram');
  assert.deepEqual(parsed.boxes[0], {
    y1: 100,
    x1: 200,
    y2: 300,
    x2: 400,
    elementIndex: 0,
    type: 'obj',
    label: 'Fractional box',
    color: '#22D3EE',
  });
});

test('parseIdeogramCaption converts bbox_px and bboxPx using image dimensions when provided', () => {
  const parsed = parseIdeogramCaption(
    JSON.stringify({
      compositional_deconstruction: {
        elements: [
          {
            type: 'obj',
            bbox_px: [50, 100, 250, 400],
            desc: 'Legacy pixels',
          },
          {
            type: 'text',
            bboxPx: [50, 100, 250, 400],
            text: 'Alias pixels',
          },
        ],
      },
    }),
    { width: 500, height: 400 },
  );

  assert.equal(parsed.kind, 'ideogram');
  assert.deepEqual(parsed.boxes.map(box => [box.y1, box.x1, box.y2, box.x2]), [
    [125, 200, 625, 800],
    [125, 200, 625, 800],
  ]);
});

test('parseIdeogramCaption normalizes legacy bbox values that exceed 0..1000 when image dimensions are provided', () => {
  const parsed = parseIdeogramCaption(
    JSON.stringify({
      compositional_deconstruction: {
        elements: [
          {
            type: 'obj',
            bbox: [250, 500, 1250, 1500],
            desc: 'Out of range',
          },
        ],
      },
    }),
    { width: 2000, height: 1000 },
  );

  assert.equal(parsed.kind, 'ideogram');
  assert.deepEqual(parsed.boxes.map(box => [box.y1, box.x1, box.y2, box.x2]), [[250, 250, 1000, 750]]);
});

test('parseIdeogramCaption ignores malformed bbox data instead of throwing', () => {
  const parsed = parseIdeogramCaption(
    JSON.stringify({
      compositional_deconstruction: {
        elements: [
          {
            type: 'obj',
            bbox: ['bad', 2, 3, 4],
            desc: 'Bad box',
          },
          {
            type: 'obj',
            bbox: [10, 20, 30],
            desc: 'Bad length',
          },
          {
            type: 'obj',
            bbox_px: [0.1, 0.2, 0.3, 0.4],
            desc: 'Fractional pixels',
          },
        ],
      },
    }),
    { width: 100, height: 100 },
  );

  assert.equal(parsed.kind, 'ideogram');
  assert.equal(parsed.boxes.length, 1);
  assert.deepEqual(parsed.boxes[0], {
    y1: 100,
    x1: 200,
    y2: 300,
    x2: 400,
    elementIndex: 2,
    type: 'obj',
    label: 'Fractional pixels',
    color: '#22D3EE',
  });
});

test('box helpers clamp and preserve ymin/xmin/ymax/xmax contract', () => {
  assert.deepEqual(boxToArray({ y1: 900.2, x1: -12, y2: 1200, x2: 100.6 }), [900, 0, 1000, 101]);
  assert.deepEqual(rectToBox({ x: 40, y: 60, w: 200, h: 120 }), { y1: 60, x1: 40, y2: 180, x2: 240 });
});

test('caption mutation helpers add, edit, delete, and serialize in schema order', () => {
  const caption = sampleCaption();
  const textIndex = addIdeogramElement(caption, 'text', { y1: 20, x1: 30, y2: 100, x2: 220 });
  updateIdeogramElementField(caption, textIndex, 'text', 'W 34 ST');
  updateIdeogramElementField(caption, textIndex, 'desc', 'Street sign text.');
  updateIdeogramElementPalette(caption, textIndex, ['#00ffcc', 'invalid']);
  updateIdeogramElementType(caption, 0, 'text');
  updateIdeogramElementBox(caption, 0, { y1: 10, x1: 20, y2: 300, x2: 400 });
  deleteIdeogramElement(caption, textIndex);

  const parsed = JSON.parse(serializeIdeogramCaption(caption));
  assert.deepEqual(Object.keys(parsed), [
    'high_level_description',
    'style_description',
    'compositional_deconstruction',
  ]);
  assert.deepEqual(Object.keys(parsed.compositional_deconstruction.elements[0]), [
    'type',
    'bbox',
    'text',
    'desc',
    'color_palette',
  ]);
  assert.deepEqual(parsed.compositional_deconstruction.elements[0].bbox, [10, 20, 300, 400]);
});

test('duplicateIdeogramElement inserts a deep-cloned layer above the source', () => {
  const caption = sampleCaption();
  caption.compositional_deconstruction.elements.push({
    type: 'text',
    bbox: [20, 30, 90, 220],
    text: 'TAXI',
    desc: 'Roof sign.',
    color_palette: ['#F59E0B'],
  });

  const duplicateIndex = duplicateIdeogramElement(caption, 0);
  assert.equal(duplicateIndex, 1);
  assert.equal(caption.compositional_deconstruction.elements.length, 3);
  assert.deepEqual(caption.compositional_deconstruction.elements[1], caption.compositional_deconstruction.elements[0]);
  assert.deepEqual(caption.compositional_deconstruction.elements[2].bbox, [20, 30, 90, 220]);

  caption.compositional_deconstruction.elements[1].color_palette[0] = '#FFFFFF';
  caption.compositional_deconstruction.elements[1].desc = 'Copied taxi.';
  assert.equal(caption.compositional_deconstruction.elements[0].color_palette[0], '#22D3EE');
  assert.equal(caption.compositional_deconstruction.elements[0].desc, 'Yellow taxi.');

  assert.equal(duplicateIdeogramElement(caption, -1), null);
  assert.equal(duplicateIdeogramElement(caption, 99), null);
  assert.equal(caption.compositional_deconstruction.elements.length, 3);

  const parsed = JSON.parse(serializeIdeogramCaption(caption));
  assert.deepEqual(Object.keys(parsed.compositional_deconstruction.elements[1]), [
    'type',
    'bbox',
    'desc',
    'color_palette',
  ]);
  assert.equal(parsed.compositional_deconstruction.elements[1].desc, 'Copied taxi.');
});

test('generated box patches clamp, filter, dedupe, and preserve bbox-only edits', () => {
  const caption = sampleCaption();
  caption.compositional_deconstruction.elements.push({
    type: 'text',
    text: 'TAXI',
    desc: 'Roof text.',
    bbox: [1, 1, 2, 2],
  });

  const patches = normalizeGeneratedBoxPatches(
    {
      boxes: [
        { elementIndex: 0, bbox: [100.4, -20, 650, 1200] },
        { elementIndex: 1, bbox: [200, 200, 200, 260] },
        { elementIndex: 99, bbox: [0, 0, 100, 100] },
        { elementIndex: 0, bbox: [120, 220, 640, 820] },
      ],
    },
    caption.compositional_deconstruction.elements.length,
    2,
  );

  assert.deepEqual(patches, [{ elementIndex: 0, bbox: [120, 220, 640, 820] }]);
  assert.equal(applyGeneratedBoxPatches(caption, patches), 1);

  const parsed = JSON.parse(serializeIdeogramCaption(caption));
  assert.deepEqual(parsed.compositional_deconstruction.elements[0].bbox, [120, 220, 640, 820]);
  assert.equal(parsed.compositional_deconstruction.elements[0].desc, 'Yellow taxi.');
  assert.deepEqual(Object.keys(parsed.compositional_deconstruction.elements[0]), [
    'type',
    'bbox',
    'desc',
    'color_palette',
  ]);
});

test('generated element boxes can append layers for empty captions', () => {
  const caption = sampleCaption();
  caption.compositional_deconstruction.elements = [];

  const generatedElements = normalizeGeneratedElementBoxes(
    {
      generatedElements: [
        { type: 'obj', bbox: [100.4, -20, 650, 1200], desc: 'Yellow taxi.' },
        { type: 'text', bbox: [20, 30, 90, 220], text: 'TAXI', desc: 'Roof sign.' },
        { type: 'obj', bbox: [200, 200, 200, 260], desc: 'flat bad box' },
      ],
    },
    2,
  );

  assert.deepEqual(generatedElements, [
    { type: 'obj', bbox: [100, 0, 650, 1000], desc: 'Yellow taxi.' },
    { type: 'text', bbox: [20, 30, 90, 220], text: 'TAXI', desc: 'Roof sign.' },
  ]);
  const result = appendGeneratedIdeogramElements(caption, generatedElements);
  assert.deepEqual(result, { count: 2, firstElementIndex: 0 });

  const parsed = JSON.parse(serializeIdeogramCaption(caption));
  assert.deepEqual(Object.keys(parsed.compositional_deconstruction.elements[0]), ['type', 'bbox', 'desc']);
  assert.deepEqual(Object.keys(parsed.compositional_deconstruction.elements[1]), ['type', 'bbox', 'text', 'desc']);
  assert.equal(parsed.compositional_deconstruction.elements[0].desc, 'Yellow taxi.');
  assert.equal(parsed.compositional_deconstruction.elements[1].text, 'TAXI');
});

test('Ideogram JSON prompts explicitly allow NSFW captions', () => {
  const uiPromptSource = fs.readFileSync(new URL('../src/helpers/captionOptions.ts', import.meta.url), 'utf8');
  const backendPromptSource = fs.readFileSync(
    new URL('../../extensions_built_in/captioner/BaseCaptioner.py', import.meta.url),
    'utf8',
  );

  assert.match(uiPromptSource, /NSFW content is allowed/);
  assert.match(uiPromptSource, /censoring or omitting/);
  assert.match(backendPromptSource, /NSFW content is allowed/);
  assert.match(backendPromptSource, /censoring or omitting/);
});
