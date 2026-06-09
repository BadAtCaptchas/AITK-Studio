import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const {
  getRandomPromptCaptionExtCandidates,
  RANDOM_PROMPT_CAPTION_EXTENSIONS,
  normalizeRandomPromptCaptionExt,
  parseRandomPromptCaptionTextAuto,
  parseRandomPromptCaptionText,
} = require('../dist/src/server/randomPromptCaptions.js');

function sampleIdeogramCaption() {
  return {
    high_level_description: 'A realistic product photo on a white table.',
    style_description: {
      aesthetics: 'clean, detailed, commercial',
      lighting: 'soft studio lighting',
      photo: 'studio product photograph',
      medium: 'photograph',
      color_palette: ['#FFFFFF', '#222222'],
    },
    compositional_deconstruction: {
      background: 'A bright white table surface.',
      elements: [
        {
          type: 'obj',
          desc: 'A black camera body centered on the table.',
          bbox: [120, 240, 780, 840],
          color_palette: ['#222222'],
        },
      ],
    },
  };
}

test('random prompt caption extension normalization preserves supported training captions', () => {
  assert.deepEqual(
    Array.from(RANDOM_PROMPT_CAPTION_EXTENSIONS).sort(),
    ['.caption', '.json', '.md', '.sdxl', '.txt'],
  );
  assert.equal(normalizeRandomPromptCaptionExt('json'), '.json');
  assert.equal(normalizeRandomPromptCaptionExt('.JSON'), '.json');
  assert.equal(normalizeRandomPromptCaptionExt('sdxl'), '.sdxl');
  assert.equal(normalizeRandomPromptCaptionExt('md'), '.md');
  assert.equal(normalizeRandomPromptCaptionExt('key'), '.txt');
  assert.equal(normalizeRandomPromptCaptionExt('../key'), '.txt');
  assert.equal(normalizeRandomPromptCaptionExt(''), '.txt');
});

test('random prompt caption candidates prefer JSON before configured text extensions', () => {
  assert.deepEqual(getRandomPromptCaptionExtCandidates('txt'), ['.json', '.txt', '.caption', '.sdxl', '.md']);
  assert.deepEqual(getRandomPromptCaptionExtCandidates('.caption'), ['.json', '.caption', '.txt', '.sdxl', '.md']);
  assert.deepEqual(getRandomPromptCaptionExtCandidates('key'), ['.json', '.txt', '.caption', '.sdxl', '.md']);
});

test('random prompt JSON captions return the caption field only', () => {
  assert.equal(
    parseRandomPromptCaptionText(
      JSON.stringify({
        caption: '  a training prompt  ',
        caption_short: 'short prompt',
        poi: { face: { x: 1, y: 2, width: 3, height: 4 } },
      }),
      'json',
    ),
    'a training prompt',
  );
  assert.equal(parseRandomPromptCaptionText(JSON.stringify({ caption_short: 'short prompt' }), 'json'), '');
  assert.equal(parseRandomPromptCaptionText('{', 'json'), '');
  assert.equal(parseRandomPromptCaptionText('plain prompt', 'txt'), 'plain prompt');
});

test('random prompt JSON captions preserve structured Ideogram captions', () => {
  const ideogramCaption = sampleIdeogramCaption();
  const parsed = parseRandomPromptCaptionText(JSON.stringify(ideogramCaption, null, 2), 'json');

  assert.deepEqual(JSON.parse(parsed), ideogramCaption);
});

test('random prompt JSON captions unwrap nested structured Ideogram strings', () => {
  const ideogramCaption = sampleIdeogramCaption();

  const rawStringCaption = parseRandomPromptCaptionText(JSON.stringify(JSON.stringify(ideogramCaption)), 'json');
  const captionFieldCaption = parseRandomPromptCaptionText(
    JSON.stringify({ caption: JSON.stringify(ideogramCaption) }),
    'json',
  );

  assert.deepEqual(JSON.parse(rawStringCaption), ideogramCaption);
  assert.deepEqual(JSON.parse(captionFieldCaption), ideogramCaption);
  assert.equal(rawStringCaption.includes('\\"high_level_description\\"'), false);
  assert.equal(captionFieldCaption.includes('\\"high_level_description\\"'), false);
});

test('random prompt JSON captions unwrap double-escaped structured Ideogram strings', () => {
  const ideogramCaption = sampleIdeogramCaption();
  const escapedIdeogramCaption = JSON.stringify(ideogramCaption).replace(/"/g, '\\"');

  const rawEscapedCaption = parseRandomPromptCaptionText(escapedIdeogramCaption, 'json');
  const captionFieldCaption = parseRandomPromptCaptionText(
    JSON.stringify({ caption: escapedIdeogramCaption }),
    'json',
  );

  assert.deepEqual(JSON.parse(rawEscapedCaption), ideogramCaption);
  assert.deepEqual(JSON.parse(captionFieldCaption), ideogramCaption);
  assert.equal(rawEscapedCaption.includes('\\"high_level_description\\"'), false);
  assert.equal(captionFieldCaption.includes('\\"high_level_description\\"'), false);
});

test('random prompt auto parsing supports encrypted JSON and plain text captions', () => {
  assert.equal(parseRandomPromptCaptionTextAuto(JSON.stringify({ caption: 'encrypted json prompt' })), 'encrypted json prompt');
  assert.equal(parseRandomPromptCaptionTextAuto('encrypted plain prompt'), 'encrypted plain prompt');
  assert.equal(parseRandomPromptCaptionTextAuto(JSON.stringify({ caption_short: 'short only' })), '');
});

test('random prompt auto parsing unwraps encrypted nested structured Ideogram strings', () => {
  const ideogramCaption = sampleIdeogramCaption();
  const parsed = parseRandomPromptCaptionTextAuto(JSON.stringify({ caption: JSON.stringify(ideogramCaption) }));

  assert.deepEqual(JSON.parse(parsed), ideogramCaption);
  assert.equal(parsed.includes('\\"high_level_description\\"'), false);
});
