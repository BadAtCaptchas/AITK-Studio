import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const {
  analyzeCaptionJsonDraft,
  buildCaptionJsonNavigator,
  collectCaptionJsonSchemaWarnings,
  formatCaptionJsonDraft,
  getCaptionJsonElementLineRanges,
  validateCaptionJsonDraft,
} = require('../dist/src/utils/captionJsonWorkspace.js');

const completeCaption = {
  high_level_description: 'A woman in a structured ivory suit.',
  style_description: { medium: 'photograph', lighting: 'soft window light' },
  compositional_deconstruction: {
    background: 'Warm concrete wall',
    elements: [
      { type: 'obj', bbox: [120, 180, 960, 580], desc: 'Woman in an ivory suit' },
      { type: 'text', bbox: [220, 100, 300, 400], text: 'AITK', desc: 'Visible wordmark' },
    ],
  },
};

test('validates an object draft and formats it without changing its data', () => {
  const compact = JSON.stringify(completeCaption);
  const validation = validateCaptionJsonDraft(compact);
  assert.equal(validation.validity, 'valid');
  if (validation.validity !== 'valid') return;
  assert.deepEqual(validation.data, completeCaption);

  const formatted = formatCaptionJsonDraft(compact);
  assert.equal(formatted.ok, true);
  if (!formatted.ok) return;
  assert.match(formatted.value, /\n  "high_level_description"/);
  assert.deepEqual(JSON.parse(formatted.value), completeCaption);
});

test('reports syntax errors with a usable source location', () => {
  const validation = validateCaptionJsonDraft('{\n  "high_level_description": "broken",\n}');
  assert.equal(validation.validity, 'invalid');
  if (validation.validity !== 'invalid') return;
  assert.equal(validation.errors[0].code, 'invalid-json');
  assert.ok((validation.errors[0].line ?? 0) >= 2);

  const formatted = formatCaptionJsonDraft('{ nope');
  assert.equal(formatted.ok, false);
  if (formatted.ok) return;
  assert.equal(formatted.error.code, 'invalid-json');
  assert.equal(formatted.value, '{ nope');
});

test('rejects syntactically valid non-object JSON captions', () => {
  for (const source of ['[]', 'null', '"caption"', '42']) {
    const validation = validateCaptionJsonDraft(source);
    assert.equal(validation.validity, 'invalid');
    if (validation.validity !== 'invalid') continue;
    assert.equal(validation.errors[0].code, 'non-object-json');
    assert.match(validation.errors[0].message, /must be an object/i);
  }
});

test('aggregates missing schema and bbox warnings', () => {
  const source = JSON.stringify(
    {
      compositional_deconstruction: {
        elements: [{ type: 'obj', desc: 'Unboxed subject' }],
      },
    },
    null,
    2,
  );
  const analysis = analyzeCaptionJsonDraft(source);
  assert.equal(analysis.validity, 'valid');
  const codes = new Set(analysis.warnings.map(warning => warning.code));
  assert.equal(codes.has('missing-high-level-description'), true);
  assert.equal(codes.has('missing-style-description'), true);
  assert.equal(codes.has('missing-background'), true);
  assert.equal(codes.has('missing-element-bbox'), true);
  assert.equal(
    analysis.warnings.find(warning => warning.code === 'missing-element-bbox')?.path,
    'compositional_deconstruction.elements[0].bbox',
  );

  const missingComposition = collectCaptionJsonSchemaWarnings({});
  const missingCodes = new Set(missingComposition.map(warning => warning.code));
  assert.equal(missingCodes.has('missing-compositional-deconstruction'), true);
  assert.equal(missingCodes.has('missing-elements'), true);
});

test('builds useful element labels, types, paths, and fallback labels', () => {
  const data = {
    ...completeCaption,
    compositional_deconstruction: {
      ...completeCaption.compositional_deconstruction,
      elements: [
        { type: 'text', bbox: [0, 0, 1, 1], text: 'Visible text', desc: 'description' },
        { type: 'person', bbox: [0, 0, 1, 1], label: 'Woman' },
        { type: 'obj', bbox: [0, 0, 1, 1], desc: 'Ivory jacket' },
        { bbox: [0, 0, 1, 1] },
      ],
    },
  };
  const source = JSON.stringify(data, null, 2);
  const navigator = buildCaptionJsonNavigator(data, source);

  assert.deepEqual(
    navigator.elements.map(element => element.label),
    ['Visible text', 'Woman', 'Ivory jacket', 'Element 4'],
  );
  assert.deepEqual(
    navigator.elements.map(element => element.type),
    ['text', 'person', 'obj', undefined],
  );
  assert.equal(navigator.elements[1].path, 'compositional_deconstruction.elements[1]');
  assert.equal(navigator.sections.find(section => section.id === 'elements')?.count, 4);
});

test('finds top-level element object ranges without being fooled by escaped strings or nested braces', () => {
  const source = `{
  "high_level_description": "A \\"quoted\\" brace } stays in the string",
  "style_description": {},
  "compositional_deconstruction": {
    "background": "Wall with { decorative } marks",
    "elements": [
      {
        "type": "obj",
        "bbox": [0, 0, 10, 10],
        "desc": "First } subject with \\"escaped { text\\""
      },
      {
        "type": "obj",
        "bbox": [10, 10, 20, 20],
        "desc": "Second subject",
        "attributes": {
          "note": "nested { object } and a closing ] token"
        }
      }
    ]
  }
}`;

  const validation = validateCaptionJsonDraft(source);
  assert.equal(validation.validity, 'valid');
  const ranges = getCaptionJsonElementLineRanges(source);
  assert.equal(ranges.length, 2);
  assert.ok(ranges[0].endLine < ranges[1].startLine);
  assert.equal(source.slice(ranges[0].startOffset, ranges[0].endOffset).trim().startsWith('{'), true);
  assert.match(source.slice(ranges[0].startOffset, ranges[0].endOffset), /First \} subject/);
  assert.match(source.slice(ranges[1].startOffset, ranges[1].endOffset), /nested \{ object \}/);

  const analysis = analyzeCaptionJsonDraft(source);
  assert.equal(analysis.elements[0].startLine, ranges[0].startLine);
  assert.equal(analysis.elements[0].endLine, ranges[0].endLine);
  assert.equal(analysis.elements[1].sourceRange?.endOffset, ranges[1].endOffset);
});

test('supports a flat elements array while preserving canonical schema warnings', () => {
  const source = JSON.stringify(
    {
      caption: 'Editorial portrait',
      elements: [{ type: 'person', label: 'Woman', bbox: [0.1, 0.2, 0.9, 0.8] }],
    },
    null,
    2,
  );
  const analysis = analyzeCaptionJsonDraft(source);
  assert.equal(analysis.validity, 'valid');
  assert.equal(analysis.elements[0].path, 'elements[0]');
  assert.equal(analysis.sections[0].path, 'caption');
  assert.equal(
    analysis.warnings.some(warning => warning.code === 'missing-compositional-deconstruction'),
    true,
  );
});
