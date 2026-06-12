import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const {
  chooseDragTarget,
  cycleHitSelection,
  detectResizeHandle,
  hitTestBoxes,
  resizeOrMoveBox,
} = require('../dist/src/utils/annotationGeometry.js');

const boxes = [
  { elementIndex: 0, x1: 100, y1: 100, x2: 500, y2: 500 },
  { elementIndex: 1, x1: 200, y1: 200, x2: 600, y2: 600 },
  { elementIndex: 2, x1: 300, y1: 300, x2: 700, y2: 700 },
];

test('hitTestBoxes returns top-first overlap stacks and filters hidden or locked layers', () => {
  assert.deepEqual(
    hitTestBoxes(boxes, { x: 350, y: 350 }).map(box => box.elementIndex),
    [2, 1, 0],
  );
  assert.deepEqual(
    hitTestBoxes(boxes, { x: 350, y: 350 }, { hiddenElementIndexes: new Set([2]) }).map(box => box.elementIndex),
    [1, 0],
  );
  assert.deepEqual(
    hitTestBoxes(boxes, { x: 350, y: 350 }, { lockedElementIndexes: new Set([2]) }).map(box => box.elementIndex),
    [1, 0],
  );
  assert.deepEqual(
    hitTestBoxes(boxes, { x: 350, y: 350 }, { lockedElementIndexes: new Set([2]), includeLocked: true }).map(
      box => box.elementIndex,
    ),
    [2, 1, 0],
  );
});

test('selected box can be prioritized for drag while cycling still walks the hit stack', () => {
  const hits = hitTestBoxes(boxes, { x: 350, y: 350 });
  assert.equal(chooseDragTarget(hits, 0)?.elementIndex, 0);
  assert.equal(chooseDragTarget(hits, 2, new Set([2])), null);
  assert.equal(chooseDragTarget(hits, 99, new Set([2]))?.elementIndex, 1);
  assert.equal(cycleHitSelection(hits, null), 2);
  assert.equal(cycleHitSelection(hits, 2), 1);
  assert.equal(cycleHitSelection(hits, 0), 2);
  assert.equal(cycleHitSelection(hits, 1, -1), 2);
});

test('detectResizeHandle supports corners and edge handles with expanded tolerance', () => {
  const box = { x1: 100, y1: 100, x2: 500, y2: 500 };
  const tolerance = { x: 20, y: 20 };
  assert.equal(detectResizeHandle(box, { x: 104, y: 102 }, tolerance), 'nw');
  assert.equal(detectResizeHandle(box, { x: 250, y: 108 }, tolerance), 'n');
  assert.equal(detectResizeHandle(box, { x: 493, y: 250 }, tolerance), 'e');
  assert.equal(detectResizeHandle(box, { x: 250, y: 488 }, tolerance), 's');
  assert.equal(detectResizeHandle(box, { x: 111, y: 250 }, tolerance), 'w');
  assert.equal(detectResizeHandle(box, { x: 250, y: 250 }, tolerance), null);
});

test('resizeOrMoveBox clamps movement and edge resizing', () => {
  assert.deepEqual(resizeOrMoveBox({ x1: 900, y1: 900, x2: 980, y2: 980 }, 50, 60, 'move'), {
    x1: 920,
    y1: 920,
    x2: 1000,
    y2: 1000,
  });
  assert.deepEqual(resizeOrMoveBox({ x1: 100, y1: 100, x2: 500, y2: 500 }, 0, -300, 'n'), {
    x1: 100,
    y1: 0,
    x2: 500,
    y2: 500,
  });
  assert.deepEqual(resizeOrMoveBox({ x1: 100, y1: 100, x2: 500, y2: 500 }, -300, 0, 'w'), {
    x1: 0,
    y1: 100,
    x2: 500,
    y2: 500,
  });
});
