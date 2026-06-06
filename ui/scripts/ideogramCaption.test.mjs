import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const {
  addIdeogramElement,
  boxToArray,
  deleteIdeogramElement,
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
