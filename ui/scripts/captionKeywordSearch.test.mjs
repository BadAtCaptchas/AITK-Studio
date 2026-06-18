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
const {
  captionFailureReason,
  isFailedCaption,
  isRefusalCaption,
} = require('../dist/src/utils/captionQuality.js');

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

test('refusal captions are treated as failed and not searchable', () => {
  const refusals = [
    'I cannot fulfill this request.',
    "Sorry, but I can't help with that.",
    "I'm unable to help with that.",
    'I can\u2019t provide a caption for this image.',
    "I won't be able to caption this image.",
    "Apologies, but I can't caption this image.",
    'I apologize, but I cannot describe this content.',
    "I'm afraid I can't answer that request.",
    'As an AI language model, I cannot assist with that.',
    'This request violates my content policy.',
    'That request is outside my safety guidelines.',
    'Request denied.',
    'I must refuse this request.',
    'It would be inappropriate to provide that caption.',
    'I cannot access the image.',
    'No image was provided.',
    'The image is not accessible.',
    "I don't feel comfortable helping with this.",
    'Please provide the image or video you would like me to caption.',
    'Please upload the photo you would like me to describe.',
  ];

  for (const refusal of refusals) {
    assert.equal(isRefusalCaption(refusal), true, refusal);
    assert.equal(isFailedCaption(refusal), true, refusal);
    assert.match(captionFailureReason(refusal), /refusal/i);
    assert.equal(captionMatchesKeywords(refusal, ['request'], 'whole-word'), false);
  }
});

test('non-refusal captions with isolated negative wording remain usable', () => {
  const caption = 'A person cannot reach the top shelf in a bright kitchen.';
  assert.equal(isRefusalCaption(caption), false);
  assert.equal(isFailedCaption(caption), false);
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
