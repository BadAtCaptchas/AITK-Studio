import { normalizeBox, type NormalizedBox } from './ideogramCaption';

export type AnnotationPoint = {
  x: number;
  y: number;
};

export type AnnotationHitBox = NormalizedBox & {
  elementIndex: number;
};

export type DragHandle = 'move' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w' | 'nw';

export type HitTestOptions = {
  hiddenElementIndexes?: Set<number>;
  lockedElementIndexes?: Set<number>;
  includeLocked?: boolean;
  selectedElementIndex?: number | null;
  selectedFirst?: boolean;
  padding?: number;
};

export type HandleTolerance = {
  x: number;
  y: number;
};

export const RESIZE_HANDLES: Exclude<DragHandle, 'move'>[] = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];

function isHidden(box: AnnotationHitBox, options: HitTestOptions) {
  return options.hiddenElementIndexes?.has(box.elementIndex) ?? false;
}

function isLocked(box: AnnotationHitBox, options: HitTestOptions) {
  return options.lockedElementIndexes?.has(box.elementIndex) ?? false;
}

export function boxContainsPoint(box: NormalizedBox, point: AnnotationPoint, padding = 0) {
  return (
    point.x >= box.x1 - padding &&
    point.x <= box.x2 + padding &&
    point.y >= box.y1 - padding &&
    point.y <= box.y2 + padding
  );
}

export function hitTestBoxes(boxes: AnnotationHitBox[], point: AnnotationPoint, options: HitTestOptions = {}) {
  const hits = boxes
    .filter(box => !isHidden(box, options))
    .filter(box => options.includeLocked || !isLocked(box, options))
    .filter(box => boxContainsPoint(box, point, options.padding ?? 0))
    .reverse();

  if (!options.selectedFirst || options.selectedElementIndex == null) return hits;
  const selectedIndex = hits.findIndex(box => box.elementIndex === options.selectedElementIndex);
  if (selectedIndex <= 0) return hits;

  const selected = hits[selectedIndex];
  return [selected, ...hits.slice(0, selectedIndex), ...hits.slice(selectedIndex + 1)];
}

export function chooseDragTarget(
  hits: AnnotationHitBox[],
  selectedElementIndex: number | null,
  lockedElementIndexes?: Set<number>,
) {
  if (selectedElementIndex != null) {
    const selectedHit = hits.find(box => box.elementIndex === selectedElementIndex);
    if (selectedHit && (lockedElementIndexes?.has(selectedElementIndex) ?? false)) return null;
    if (selectedHit) return selectedHit;
  }
  const unlockedHits = hits.filter(box => !(lockedElementIndexes?.has(box.elementIndex) ?? false));
  return unlockedHits[0] || null;
}

export function cycleHitSelection(hits: AnnotationHitBox[], selectedElementIndex: number | null, direction = 1) {
  if (hits.length === 0) return null;
  if (selectedElementIndex == null) return hits[0].elementIndex;

  const selectedIndex = hits.findIndex(box => box.elementIndex === selectedElementIndex);
  if (selectedIndex < 0) return hits[0].elementIndex;

  const nextIndex = (selectedIndex + (direction >= 0 ? 1 : -1) + hits.length) % hits.length;
  return hits[nextIndex].elementIndex;
}

function near(value: number, target: number, tolerance: number) {
  return Math.abs(value - target) <= tolerance;
}

function within(value: number, min: number, max: number, tolerance: number) {
  return value >= min - tolerance && value <= max + tolerance;
}

export function detectResizeHandle(
  box: NormalizedBox,
  point: AnnotationPoint,
  tolerance: HandleTolerance,
): Exclude<DragHandle, 'move'> | null {
  const west = near(point.x, box.x1, tolerance.x);
  const east = near(point.x, box.x2, tolerance.x);
  const north = near(point.y, box.y1, tolerance.y);
  const south = near(point.y, box.y2, tolerance.y);
  const xInside = within(point.x, box.x1, box.x2, tolerance.x);
  const yInside = within(point.y, box.y1, box.y2, tolerance.y);

  if (west && north) return 'nw';
  if (east && north) return 'ne';
  if (east && south) return 'se';
  if (west && south) return 'sw';
  if (north && xInside) return 'n';
  if (east && yInside) return 'e';
  if (south && xInside) return 's';
  if (west && yInside) return 'w';
  return null;
}

export function resizeOrMoveBox(box: NormalizedBox, dx: number, dy: number, handle: DragHandle, minSpan = 8): NormalizedBox {
  let { x1, y1, x2, y2 } = box;

  if (handle === 'move') {
    const width = x2 - x1;
    const height = y2 - y1;
    x1 = Math.max(0, Math.min(1000 - width, x1 + dx));
    y1 = Math.max(0, Math.min(1000 - height, y1 + dy));
    return { x1, y1, x2: x1 + width, y2: y1 + height };
  }

  if (handle.includes('w')) x1 = Math.max(0, Math.min(x2 - minSpan, x1 + dx));
  if (handle.includes('e')) x2 = Math.min(1000, Math.max(x1 + minSpan, x2 + dx));
  if (handle.includes('n')) y1 = Math.max(0, Math.min(y2 - minSpan, y1 + dy));
  if (handle.includes('s')) y2 = Math.min(1000, Math.max(y1 + minSpan, y2 + dy));
  return normalizeBox({ x1, y1, x2, y2 });
}
