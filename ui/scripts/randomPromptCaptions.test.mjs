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
  const ideogramCaption = {
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
  const parsed = parseRandomPromptCaptionText(JSON.stringify(ideogramCaption, null, 2), 'json');

  assert.deepEqual(JSON.parse(parsed), ideogramCaption);
});

test('random prompt auto parsing supports encrypted JSON and plain text captions', () => {
  assert.equal(parseRandomPromptCaptionTextAuto(JSON.stringify({ caption: 'encrypted json prompt' })), 'encrypted json prompt');
  assert.equal(parseRandomPromptCaptionTextAuto('encrypted plain prompt'), 'encrypted plain prompt');
  assert.equal(parseRandomPromptCaptionTextAuto(JSON.stringify({ caption_short: 'short only' })), '');
});
