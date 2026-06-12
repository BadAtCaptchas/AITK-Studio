import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const {
  captionMatchesKeywords,
  countCaptionKeywordMatches,
  parseCaptionKeywordQuery,
  removeCaptionKeywords,
} = require('../dist/src/utils/captionKeywordSearch.js');

const ideogramCaption = JSON.stringify({
  high_level_description: 'A red cat beside a cathedral.',
  compositional_deconstruction: {
    elements: [
      { type: 'obj', bbox: [10, 20, 200, 300], desc: 'orange cat', color_palette: ['#FF9900'] },
      { type: 'text', bbox: [300, 20, 420, 300], text: 'CATALOG', desc: 'printed catalog text' },
    ],
  },
});

test('parseCaptionKeywordQuery splits whitespace and commas with case-insensitive uniqueness', () => {
  assert.deepEqual(parseCaptionKeywordQuery(' cat, dog  cat DOG '), ['cat', 'dog']);
});

test('whole-word matching does not match partial words by default', () => {
  assert.equal(captionMatchesKeywords('a cat near a catalog', ['cat'], 'whole-word'), true);
  assert.equal(countCaptionKeywordMatches('catalog category', ['cat'], 'whole-word'), 0);
});

test('partial matching can match inside longer words', () => {
  assert.equal(countCaptionKeywordMatches('catalog category', ['cat'], 'partial'), 2);
});

test('multiple keywords use any-keyword semantics', () => {
  assert.equal(captionMatchesKeywords('a small dog', ['cat', 'dog'], 'whole-word'), true);
  assert.equal(captionMatchesKeywords('a small bird', ['cat', 'dog'], 'whole-word'), false);
});

test('removeCaptionKeywords removes whole words from plain captions', () => {
  const result = removeCaptionKeywords('cat, dog, catfish and cat', ['cat'], 'whole-word');
  assert.equal(result.changed, true);
  assert.equal(result.removedCount, 2);
  assert.equal(result.caption, 'dog, catfish and');
});

test('JSON search and removal use text values, not keys or structural values', () => {
  assert.equal(captionMatchesKeywords(ideogramCaption, ['cat'], 'whole-word'), true);
  assert.equal(captionMatchesKeywords(ideogramCaption, ['obj'], 'whole-word'), false);
  assert.equal(captionMatchesKeywords(ideogramCaption, ['FF9900'], 'partial'), false);

  const result = removeCaptionKeywords(ideogramCaption, ['cat'], 'whole-word');
  const parsed = JSON.parse(result.caption);
  assert.equal(result.removedCount, 2);
  assert.equal(parsed.compositional_deconstruction.elements[0].type, 'obj');
  assert.deepEqual(parsed.compositional_deconstruction.elements[0].color_palette, ['#FF9900']);
  assert.equal(parsed.high_level_description, 'A red beside a cathedral.');
  assert.equal(parsed.compositional_deconstruction.elements[0].desc, 'orange');
  assert.equal(parsed.compositional_deconstruction.elements[1].text, 'CATALOG');
});
