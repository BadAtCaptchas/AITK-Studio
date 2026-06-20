import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const {
  filterNavigatorEntries,
  groupNavigatorRows,
  matchesNavigatorSearch,
  navigatorColumnCount,
  navigatorStatusCounts,
  navigatorStatusForCaption,
  parseNavigatorJump,
  sortNavigatorEntries,
} = require('../dist/src/utils/datasetImageNavigator.js');

const boxedCaption = JSON.stringify({
  high_level_description: 'test image',
  compositional_deconstruction: {
    elements: [{ type: 'obj', bbox: [10, 20, 200, 300], desc: 'subject' }],
  },
});

const unboxedCaption = JSON.stringify({
  high_level_description: 'test image',
  compositional_deconstruction: {
    elements: [{ type: 'obj', desc: 'subject' }],
  },
});

test('parseNavigatorJump accepts 1-based direct index input and clamps to dataset length', () => {
  assert.equal(parseNavigatorJump('284', 4812), 283);
  assert.equal(parseNavigatorJump('284 / 4,812', 4812), 283);
  assert.equal(parseNavigatorJump('9999', 4812), 4811);
  assert.equal(parseNavigatorJump('0', 4812), null);
  assert.equal(parseNavigatorJump('nope', 4812), null);
});

test('navigator search matches filename and 1-based index prefixes', () => {
  const entry = { index: 283, name: 'portrait/session_042/image_0284.png' };
  assert.equal(matchesNavigatorSearch(entry, 'portrait'), true);
  assert.equal(matchesNavigatorSearch(entry, '284'), true);
  assert.equal(matchesNavigatorSearch(entry, '28'), true);
  assert.equal(matchesNavigatorSearch(entry, 'kitten'), false);
});

test('navigator status classification mirrors caption semantics', () => {
  assert.equal(navigatorStatusForCaption('', false), 'unknown');
  assert.equal(navigatorStatusForCaption('', true), 'missing');
  assert.equal(navigatorStatusForCaption('I cannot fulfill this request.', true), 'missing');
  assert.equal(navigatorStatusForCaption('plain caption', true), 'plain');
  assert.equal(navigatorStatusForCaption('{"caption":"hello"}', true), 'json');
  assert.equal(navigatorStatusForCaption(unboxedCaption, true), 'json');
  assert.equal(navigatorStatusForCaption(boxedCaption, true), 'has-boxes');
});

test('filterNavigatorEntries combines search and status filters', () => {
  const entries = [
    { index: 0, name: 'a.png', status: 'missing' },
    { index: 1, name: 'boxed.png', status: 'has-boxes' },
    { index: 2, name: 'plain.png', status: 'plain' },
    { index: 3, name: 'pending.png', status: 'unknown' },
  ];
  assert.deepEqual(filterNavigatorEntries(entries, '', 'needs-caption').map(entry => entry.index), [0, 3]);
  assert.deepEqual(filterNavigatorEntries(entries, '', 'has-boxes').map(entry => entry.index), [1]);
  assert.deepEqual(filterNavigatorEntries(entries, 'plain', 'all').map(entry => entry.index), [2]);
});

test('navigator needs-caption counts include pending caption lookups', () => {
  assert.deepEqual(
    navigatorStatusCounts([
      { index: 0, name: 'missing.png', status: 'missing' },
      { index: 1, name: 'pending.png', status: 'unknown' },
      { index: 2, name: 'boxed.png', status: 'has-boxes' },
    ]),
    { total: 3, missing: 2, hasBoxes: 1, unknown: 1 },
  );
});

test('groupNavigatorRows chunks indexes by measured column count', () => {
  assert.deepEqual(groupNavigatorRows([0, 1, 2, 3, 4], 2), [
    [0, 1],
    [2, 3],
    [4],
  ]);
  assert.deepEqual(groupNavigatorRows([0, 1, 2], 0), [[0], [1], [2]]);
});

test('navigatorColumnCount derives stable columns from container and tile widths', () => {
  assert.equal(navigatorColumnCount(500, 120, 8), 3);
  assert.equal(navigatorColumnCount(100, 120, 8), 1);
});

test('sortNavigatorEntries preserves original order for original mode', () => {
  const entries = [
    { index: 2, name: 'c.png', status: 'plain', addedAt: '2026-01-03T00:00:00.000Z' },
    { index: 0, name: 'a.png', status: 'plain', addedAt: '2026-01-01T00:00:00.000Z' },
    { index: 1, name: 'b.png', status: 'plain', addedAt: '2026-01-02T00:00:00.000Z' },
  ];
  assert.deepEqual(sortNavigatorEntries(entries, 'original', 'desc').map(entry => entry.index), [2, 0, 1]);
});

test('sortNavigatorEntries sorts date fields by direction with missing dates last', () => {
  const entries = [
    { index: 0, name: 'missing.png', status: 'plain', addedAt: null, captionedAt: null },
    {
      index: 1,
      name: 'older.png',
      status: 'plain',
      addedAt: '2026-01-01T00:00:00.000Z',
      captionedAt: '2026-01-05T00:00:00.000Z',
    },
    {
      index: 2,
      name: 'newer.png',
      status: 'plain',
      addedAt: '2026-01-03T00:00:00.000Z',
      captionedAt: '2026-01-07T00:00:00.000Z',
    },
  ];
  assert.deepEqual(sortNavigatorEntries(entries, 'added', 'desc').map(entry => entry.index), [2, 1, 0]);
  assert.deepEqual(sortNavigatorEntries(entries, 'added', 'asc').map(entry => entry.index), [1, 2, 0]);
  assert.deepEqual(sortNavigatorEntries(entries, 'captioned', 'desc').map(entry => entry.index), [2, 1, 0]);
});

test('sortNavigatorEntries naturally sorts file names', () => {
  const entries = [
    { index: 0, name: 'image_10.png', status: 'plain' },
    { index: 1, name: 'image_2.png', status: 'plain' },
    { index: 2, name: 'image_1.png', status: 'plain' },
  ];
  assert.deepEqual(sortNavigatorEntries(entries, 'name', 'asc').map(entry => entry.index), [2, 1, 0]);
  assert.deepEqual(sortNavigatorEntries(entries, 'name', 'desc').map(entry => entry.index), [0, 1, 2]);
});

test('sortNavigatorEntries sorts extensions and media types with missing values last', () => {
  const entries = [
    { index: 0, name: 'a', status: 'plain', extension: 'png', mediaType: 'image' },
    { index: 1, name: 'b', status: 'plain', extension: 'txt', mediaType: 'text' },
    { index: 2, name: 'c', status: 'plain', extension: null, mediaType: null },
    { index: 3, name: 'd', status: 'plain', extension: 'mp4', mediaType: 'video' },
  ];
  assert.deepEqual(sortNavigatorEntries(entries, 'extension', 'asc').map(entry => entry.index), [3, 0, 1, 2]);
  assert.deepEqual(sortNavigatorEntries(entries, 'extension', 'desc').map(entry => entry.index), [1, 0, 3, 2]);
  assert.deepEqual(sortNavigatorEntries(entries, 'media-type', 'asc').map(entry => entry.index), [0, 1, 3, 2]);
});

test('sortNavigatorEntries sorts file size and caption length numerically', () => {
  const entries = [
    { index: 0, name: 'small.png', status: 'plain', sizeBytes: 120, captionLength: 14 },
    { index: 1, name: 'missing.png', status: 'plain', sizeBytes: null, captionLength: null },
    { index: 2, name: 'large.png', status: 'plain', sizeBytes: 4000, captionLength: 92 },
  ];
  assert.deepEqual(sortNavigatorEntries(entries, 'size', 'desc').map(entry => entry.index), [2, 0, 1]);
  assert.deepEqual(sortNavigatorEntries(entries, 'size', 'asc').map(entry => entry.index), [0, 2, 1]);
  assert.deepEqual(sortNavigatorEntries(entries, 'caption-length', 'desc').map(entry => entry.index), [2, 0, 1]);
  assert.deepEqual(sortNavigatorEntries(entries, 'caption-length', 'asc').map(entry => entry.index), [0, 2, 1]);
});

test('sortNavigatorEntries sorts caption status with unloaded captions last', () => {
  const entries = [
    { index: 0, name: 'plain.png', status: 'plain' },
    { index: 1, name: 'pending.png', status: 'unknown' },
    { index: 2, name: 'boxed.png', status: 'has-boxes' },
    { index: 3, name: 'missing.png', status: 'missing' },
    { index: 4, name: 'json.png', status: 'json' },
  ];
  assert.deepEqual(sortNavigatorEntries(entries, 'caption-status', 'asc').map(entry => entry.index), [3, 0, 4, 2, 1]);
  assert.deepEqual(sortNavigatorEntries(entries, 'caption-status', 'desc').map(entry => entry.index), [2, 4, 0, 3, 1]);
});

test('sortNavigatorEntries falls back to original index for date ties and invalid dates', () => {
  const entries = [
    { index: 4, name: 'late-tie.png', status: 'plain', captionedAt: '2026-01-01T00:00:00.000Z' },
    { index: 2, name: 'early-tie.png', status: 'plain', captionedAt: '2026-01-01T00:00:00.000Z' },
    { index: 1, name: 'invalid.png', status: 'plain', captionedAt: 'not-a-date' },
    { index: 3, name: 'missing.png', status: 'plain', captionedAt: null },
  ];
  assert.deepEqual(sortNavigatorEntries(entries, 'captioned', 'desc').map(entry => entry.index), [2, 4, 1, 3]);
});
