import { useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import type { ReactNode } from 'react';
import classNames from 'classnames';
import { arrayToBox } from '@/utils/ideogramCaption';

export interface OverlayBox {
  y1: number;
  x1: number;
  y2: number;
  x2: number;
  label: string;
  type: 'obj' | 'text';
}

export interface EditableBox extends OverlayBox {
  elementIndex: number;
}

export interface BoxCoords {
  y1: number;
  x1: number;
  y2: number;
  x2: number;
}

export interface EditableCaptionData {
  data: any;
  boxes: EditableBox[];
}

type DragHandle = 'move' | 'nw' | 'ne' | 'sw' | 'se';

const MIN_BOX_SPAN = 8;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

type ImageSize = {
  width: number;
  height: number;
};

function readElements(data: unknown): unknown[] | null {
  if (!isRecord(data)) return null;
  const deconstruction = data.compositional_deconstruction;
  if (!isRecord(deconstruction)) return null;
  return Array.isArray(deconstruction.elements) ? deconstruction.elements : null;
}

function readBox(raw: Record<string, unknown>, imageSize?: ImageSize): BoxCoords | null {
  const direct = arrayToBox(raw.bbox, imageSize);
  if (direct) return direct;
  const pixel = arrayToBox(raw.bbox_px, imageSize, 'bbox_px');
  if (pixel) return pixel;
  return arrayToBox(raw.bboxPx, imageSize, 'bboxPx');
}

function labelForElement(element: Record<string, unknown>, type: 'obj' | 'text') {
  const value = type === 'text' ? element.text : element.desc;
  return value == null ? '' : `${value}`;
}

export function extractEditableBoxes(data: unknown, imageSize?: ImageSize): EditableBox[] {
  const elements = readElements(data);
  if (!elements) return [];

  const boxes: EditableBox[] = [];
  elements.forEach((rawElement, elementIndex) => {
    if (!isRecord(rawElement)) return;
    const coords = readBox(rawElement, imageSize);
    if (!coords) return;
    const type = rawElement.type === 'text' ? 'text' : 'obj';
    boxes.push({
      ...coords,
      label: labelForElement(rawElement, type),
      type,
      elementIndex,
    });
  });
  return boxes;
}

export function parseBoundingBoxes(text: string, imageSize?: ImageSize): OverlayBox[] | null {
  const parsed = parseCaptionForEditing(text, imageSize);
  if (!parsed) return null;
  return parsed.boxes.map(({ elementIndex: _elementIndex, ...box }) => box);
}

export function parseCaptionForEditing(text: string, imageSize?: ImageSize): EditableCaptionData | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith('{')) return null;

  let data: unknown;
  try {
    data = JSON.parse(trimmed);
  } catch {
    return null;
  }

  const boxes = extractEditableBoxes(data, imageSize);
  return boxes.length > 0 ? { data, boxes } : null;
}

export function normalizeBox(box: { y1: number; x1: number; y2: number; x2: number }): BoxCoords {
  const clamp = (value: number) => Math.max(0, Math.min(1000, Math.round(value)));
  return {
    y1: clamp(Math.min(box.y1, box.y2)),
    x1: clamp(Math.min(box.x1, box.x2)),
    y2: clamp(Math.max(box.y1, box.y2)),
    x2: clamp(Math.max(box.x1, box.x2)),
  };
}

export default function BoundingBoxOverlay({ boxes }: { boxes: OverlayBox[] }) {
  return (
    <div className="absolute inset-0 pointer-events-none">
      {boxes.map((box, index) => (
        <BoxFrame key={index} box={box} />
      ))}
    </div>
  );
}

function BoxFrame({
  box,
  selected = false,
  children,
  onPointerDown,
}: {
  box: OverlayBox;
  selected?: boolean;
  children?: ReactNode;
  onPointerDown?: (event: ReactPointerEvent<HTMLDivElement>) => void;
}) {
  const isText = box.type === 'text';
  return (
    <div
      onPointerDown={onPointerDown}
      className={classNames('absolute border-2', {
        'border-cyan-400': !isText && !selected,
        'border-amber-400': isText && !selected,
        'border-white ring-2 ring-blue-400': selected,
      })}
      style={{
        left: `${box.x1 / 10}%`,
        top: `${box.y1 / 10}%`,
        width: `${Math.max(0, box.x2 - box.x1) / 10}%`,
        height: `${Math.max(0, box.y2 - box.y1) / 10}%`,
      }}
    >
      {box.label && (
        <span
          title={box.label}
          className={classNames(
            'absolute left-0 top-0 max-w-full px-1 py-0.5 text-[9px] font-medium leading-tight text-black line-clamp-2 whitespace-pre-line break-words pointer-events-none',
            {
              'bg-cyan-400/90': !isText,
              'bg-amber-400/90': isText,
            },
          )}
        >
          {box.label}
        </span>
      )}
      {children}
    </div>
  );
}

export function BoundingBoxEditor({
  boxes,
  selectedIndex,
  drawing,
  onSelect,
  onChangeBox,
  onCreateBox,
}: {
  boxes: EditableBox[];
  selectedIndex: number | null;
  drawing: boolean;
  onSelect: (elementIndex: number | null) => void;
  onChangeBox: (elementIndex: number, box: BoxCoords) => void;
  onCreateBox: (box: BoxCoords) => void;
}) {
  const layerRef = useRef<HTMLDivElement>(null);
  const [dragPreview, setDragPreview] = useState<{ elementIndex: number; box: OverlayBox } | null>(null);
  const [newBoxPreview, setNewBoxPreview] = useState<BoxCoords | null>(null);

  const pointToNorm = (clientX: number, clientY: number) => {
    const rect = layerRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0 || rect.height === 0) return null;
    return {
      x: Math.max(0, Math.min(1000, ((clientX - rect.left) / rect.width) * 1000)),
      y: Math.max(0, Math.min(1000, ((clientY - rect.top) / rect.height) * 1000)),
    };
  };

  const beginBoxDrag = (event: ReactPointerEvent<HTMLElement>, box: EditableBox, handle: DragHandle) => {
    event.preventDefault();
    event.stopPropagation();
    onSelect(box.elementIndex);

    const startPoint = pointToNorm(event.clientX, event.clientY);
    if (!startPoint) return;
    const startBox = { ...box };
    let latest: OverlayBox = startBox;

    const onMove = (moveEvent: PointerEvent) => {
      const point = pointToNorm(moveEvent.clientX, moveEvent.clientY);
      if (!point) return;
      const dx = point.x - startPoint.x;
      const dy = point.y - startPoint.y;
      latest = resizeOrMoveBox(startBox, dx, dy, handle);
      setDragPreview({ elementIndex: box.elementIndex, box: latest });
    };

    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      setDragPreview(null);
      onChangeBox(box.elementIndex, normalizeBox(latest));
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };

  const beginDraw = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.currentTarget !== event.target) return;
    if (!drawing) {
      onSelect(null);
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    const start = pointToNorm(event.clientX, event.clientY);
    if (!start) return;
    let latest: BoxCoords = { y1: start.y, x1: start.x, y2: start.y, x2: start.x };
    setNewBoxPreview(latest);

    const onMove = (moveEvent: PointerEvent) => {
      const point = pointToNorm(moveEvent.clientX, moveEvent.clientY);
      if (!point) return;
      latest = normalizeBox({ y1: start.y, x1: start.x, y2: point.y, x2: point.x });
      setNewBoxPreview(latest);
    };

    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      setNewBoxPreview(null);
      if (latest.x2 - latest.x1 >= MIN_BOX_SPAN && latest.y2 - latest.y1 >= MIN_BOX_SPAN) {
        onCreateBox(latest);
      }
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };

  return (
    <div
      ref={layerRef}
      onPointerDown={beginDraw}
      className={classNames('absolute inset-0 touch-none', {
        'cursor-crosshair': drawing,
      })}
    >
      {boxes.map(box => {
        const preview = dragPreview?.elementIndex === box.elementIndex ? dragPreview.box : box;
        const selected = selectedIndex === box.elementIndex && !drawing;
        return (
          <BoxFrame
            key={box.elementIndex}
            box={preview}
            selected={selected}
            onPointerDown={drawing ? undefined : event => beginBoxDrag(event, box, 'move')}
          >
            {selected && (
              <>
                {(['nw', 'ne', 'sw', 'se'] as const).map(handle => (
                  <button
                    key={handle}
                    type="button"
                    aria-label={`Resize ${handle}`}
                    onPointerDown={event => beginBoxDrag(event, { ...box, ...preview }, handle)}
                    className={classNames('absolute h-3 w-3 rounded-sm border border-gray-950 bg-white', {
                      'left-0 top-0 -translate-x-1/2 -translate-y-1/2 cursor-nwse-resize': handle === 'nw',
                      'right-0 top-0 translate-x-1/2 -translate-y-1/2 cursor-nesw-resize': handle === 'ne',
                      'bottom-0 left-0 -translate-x-1/2 translate-y-1/2 cursor-nesw-resize': handle === 'sw',
                      'bottom-0 right-0 translate-x-1/2 translate-y-1/2 cursor-nwse-resize': handle === 'se',
                    })}
                  />
                ))}
              </>
            )}
          </BoxFrame>
        );
      })}
      {newBoxPreview && (
        <div
          className="absolute border-2 border-dashed border-white bg-white/10 pointer-events-none"
          style={{
            left: `${newBoxPreview.x1 / 10}%`,
            top: `${newBoxPreview.y1 / 10}%`,
            width: `${(newBoxPreview.x2 - newBoxPreview.x1) / 10}%`,
            height: `${(newBoxPreview.y2 - newBoxPreview.y1) / 10}%`,
          }}
        />
      )}
    </div>
  );
}

function resizeOrMoveBox(box: OverlayBox, dx: number, dy: number, handle: DragHandle): OverlayBox {
  let { x1, y1, x2, y2 } = box;
  if (handle === 'move') {
    const width = x2 - x1;
    const height = y2 - y1;
    x1 = Math.max(0, Math.min(1000 - width, x1 + dx));
    y1 = Math.max(0, Math.min(1000 - height, y1 + dy));
    return { ...box, x1, y1, x2: x1 + width, y2: y1 + height };
  }

  if (handle.includes('w')) x1 = Math.max(0, Math.min(x2 - MIN_BOX_SPAN, x1 + dx));
  if (handle.includes('e')) x2 = Math.min(1000, Math.max(x1 + MIN_BOX_SPAN, x2 + dx));
  if (handle.includes('n')) y1 = Math.max(0, Math.min(y2 - MIN_BOX_SPAN, y1 + dy));
  if (handle.includes('s')) y2 = Math.min(1000, Math.max(y1 + MIN_BOX_SPAN, y2 + dy));
  return { ...box, x1, y1, x2, y2 };
}
