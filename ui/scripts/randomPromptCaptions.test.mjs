import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const {
  RANDOM_PROMPT_CAPTION_EXTENSIONS,
  normalizeRandomPromptCaptionExt,
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
