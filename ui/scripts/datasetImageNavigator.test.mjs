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
