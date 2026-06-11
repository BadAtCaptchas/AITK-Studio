'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import classNames from 'classnames';
import { Lock } from 'lucide-react';
import {
  chooseDragTarget,
  cycleHitSelection,
  detectResizeHandle,
  hitTestBoxes,
  RESIZE_HANDLES,
  resizeOrMoveBox,
  type DragHandle,
} from '@/utils/annotationGeometry';
import { rectToBox, type IdeogramBox, type IdeogramElementType, type NormalizedBox } from '@/utils/ideogramCaption';
import { CLICK_DRAG_TOLERANCE, MIN_BOX_SPAN } from './constants';
import type { ImageSize, ToolMode } from './types';
import { resolveBoxColor } from './utils';

function previewSizeLabel(box: NormalizedBox, imageSize?: ImageSize | null) {
  const w = Math.max(0, box.x2 - box.x1);
  const h = Math.max(0, box.y2 - box.y1);
  if (imageSize?.width && imageSize?.height) {
    return `${Math.round((w / 1000) * imageSize.width)} × ${Math.round((h / 1000) * imageSize.height)} px`;
  }
  return `${(w / 10).toFixed(1)} × ${(h / 10).toFixed(1)} %`;
}

function handleCursor(handle: DragHandle | null) {
  if (!handle) return 'default';
  if (handle === 'move') return 'move';
  if (handle === 'n' || handle === 's') return 'ns-resize';
  if (handle === 'e' || handle === 'w') return 'ew-resize';
  if (handle === 'ne' || handle === 'sw') return 'nesw-resize';
  return 'nwse-resize';
}

function handleClassName(handle: Exclude<DragHandle, 'move'>) {
  return classNames('absolute rounded-sm border border-gray-950 bg-white shadow-[0_0_0_1px_rgba(255,255,255,0.35)]', {
    'left-0 top-0 h-3 w-3 -translate-x-1/2 -translate-y-1/2': handle === 'nw',
    'left-1/2 top-0 h-2 w-5 -translate-x-1/2 -translate-y-1/2': handle === 'n',
    'right-0 top-0 h-3 w-3 translate-x-1/2 -translate-y-1/2': handle === 'ne',
    'right-0 top-1/2 h-5 w-2 -translate-y-1/2 translate-x-1/2': handle === 'e',
    'bottom-0 right-0 h-3 w-3 translate-x-1/2 translate-y-1/2': handle === 'se',
    'bottom-0 left-1/2 h-2 w-5 -translate-x-1/2 translate-y-1/2': handle === 's',
    'bottom-0 left-0 h-3 w-3 -translate-x-1/2 translate-y-1/2': handle === 'sw',
    'left-0 top-1/2 h-5 w-2 -translate-x-1/2 -translate-y-1/2': handle === 'w',
  });
}

export function AnnotationLayer({
  boxes,
  activeTool,
  selectedElementIndex,
  hiddenElementIndexes,
  lockedElementIndexes,
  imageSize,
  onSelect,
  onCreate,
  onChangeBox,
  onOverlapStackChange,
}: {
  boxes: IdeogramBox[];
  activeTool: ToolMode;
  selectedElementIndex: number | null;
  hiddenElementIndexes: Set<number>;
  lockedElementIndexes: Set<number>;
  imageSize?: ImageSize | null;
  onSelect: (elementIndex: number | null) => void;
  onCreate: (type: IdeogramElementType, box: NormalizedBox) => void;
  onChangeBox: (elementIndex: number, box: NormalizedBox) => void;
  onOverlapStackChange: (elementIndexes: number[]) => void;
}) {
  const layerRef = useRef<HTMLDivElement>(null);
  const toastTimerRef = useRef<number | null>(null);
  const [dragPreview, setDragPreview] = useState<{ elementIndex: number; box: NormalizedBox } | null>(null);
  const [newBoxPreview, setNewBoxPreview] = useState<NormalizedBox | null>(null);
  const [cursor, setCursor] = useState('default');
  const [cycleToast, setCycleToast] = useState<{ x: number; y: number; count: number; index: number } | null>(null);
  const drawingType: IdeogramElementType | null = activeTool === 'box' ? 'obj' : activeTool === 'text' ? 'text' : null;

  const visibleBoxes = useMemo(
    () => boxes.filter(box => !hiddenElementIndexes.has(box.elementIndex)),
    [boxes, hiddenElementIndexes],
  );
  const selectedBox = visibleBoxes.find(box => box.elementIndex === selectedElementIndex) || null;

  const orderedBoxes = useMemo(
    () =>
      [...visibleBoxes].sort((left, right) => {
        if (left.elementIndex === selectedElementIndex) return 1;
        if (right.elementIndex === selectedElementIndex) return -1;
        return left.elementIndex - right.elementIndex;
      }),
    [selectedElementIndex, visibleBoxes],
  );

  useEffect(() => {
    onOverlapStackChange([]);
    setCycleToast(null);
  }, [activeTool, boxes, hiddenElementIndexes, onOverlapStackChange]);

  useEffect(
    () => () => {
      if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    },
    [],
  );

  const pointToNorm = useCallback((clientX: number, clientY: number) => {
    const rect = layerRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0 || rect.height === 0) return null;
    return {
      x: Math.max(0, Math.min(1000, ((clientX - rect.left) / rect.width) * 1000)),
      y: Math.max(0, Math.min(1000, ((clientY - rect.top) / rect.height) * 1000)),
    };
  }, []);

  const pointToLayer = useCallback((clientX: number, clientY: number) => {
    const rect = layerRef.current?.getBoundingClientRect();
    if (!rect) return null;
    return {
      x: Math.max(8, Math.min(rect.width - 64, clientX - rect.left + 10)),
      y: Math.max(8, Math.min(rect.height - 32, clientY - rect.top + 10)),
    };
  }, []);

  const handleTolerance = useCallback(() => {
    const rect = layerRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0 || rect.height === 0) return { x: 24, y: 24 };
    return {
      x: Math.max(12, (16 / rect.width) * 1000),
      y: Math.max(12, (16 / rect.height) * 1000),
    };
  }, []);

  const showCycleToast = useCallback(
    (clientX: number, clientY: number, hits: IdeogramBox[], selected: number | null) => {
      if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
      if (hits.length <= 1 || selected == null) {
        setCycleToast(null);
        return;
      }
      const layerPoint = pointToLayer(clientX, clientY);
      if (!layerPoint) return;
      const index = Math.max(0, hits.findIndex(box => box.elementIndex === selected));
      setCycleToast({ ...layerPoint, count: hits.length, index: index + 1 });
      toastTimerRef.current = window.setTimeout(() => setCycleToast(null), 900);
    },
    [pointToLayer],
  );

  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (drawingType) {
        setCursor('crosshair');
        return;
      }
      const point = pointToNorm(event.clientX, event.clientY);
      if (!point) return;
      const handle =
        selectedBox && !lockedElementIndexes.has(selectedBox.elementIndex)
          ? detectResizeHandle(selectedBox, point, handleTolerance())
          : null;
      if (handle) {
        setCursor(handleCursor(handle));
        return;
      }
      const hits = hitTestBoxes(visibleBoxes, point, {
        includeLocked: true,
        hiddenElementIndexes,
      });
      const target = chooseDragTarget(hits, selectedElementIndex, lockedElementIndexes);
      setCursor(target ? 'move' : hits.length > 0 ? 'pointer' : 'default');
    },
    [drawingType, handleTolerance, hiddenElementIndexes, lockedElementIndexes, pointToNorm, selectedBox, selectedElementIndex, visibleBoxes],
  );

  const beginDraw = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, type: IdeogramElementType) => {
      event.preventDefault();
      event.stopPropagation();
      const start = pointToNorm(event.clientX, event.clientY);
      if (!start) return;
      let latest: NormalizedBox = { y1: start.y, x1: start.x, y2: start.y, x2: start.x };
      setNewBoxPreview(latest);

      const onMove = (moveEvent: PointerEvent) => {
        const point = pointToNorm(moveEvent.clientX, moveEvent.clientY);
        if (!point) return;
        latest = rectToBox({
          x: Math.min(start.x, point.x),
          y: Math.min(start.y, point.y),
          w: Math.abs(point.x - start.x),
          h: Math.abs(point.y - start.y),
        });
        setNewBoxPreview(latest);
      };

      const cleanup = () => {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
        window.removeEventListener('keydown', onCancelKey, true);
      };

      const onUp = () => {
        cleanup();
        setNewBoxPreview(null);
        if (latest.x2 - latest.x1 >= MIN_BOX_SPAN && latest.y2 - latest.y1 >= MIN_BOX_SPAN) {
          onCreate(type, latest);
        }
      };

      const onCancelKey = (keyEvent: KeyboardEvent) => {
        if (keyEvent.key !== 'Escape') return;
        keyEvent.preventDefault();
        keyEvent.stopImmediatePropagation();
        cleanup();
        setNewBoxPreview(null);
      };

      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
      window.addEventListener('keydown', onCancelKey, true);
    },
    [onCreate, pointToNorm],
  );

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) return;
      if (drawingType) {
        beginDraw(event, drawingType);
        return;
      }

      event.preventDefault();
      event.stopPropagation();
      const start = pointToNorm(event.clientX, event.clientY);
      if (!start) return;

      const handle =
        selectedBox && !lockedElementIndexes.has(selectedBox.elementIndex)
          ? detectResizeHandle(selectedBox, start, handleTolerance())
          : null;
      const hits = hitTestBoxes(visibleBoxes, start, {
        includeLocked: true,
        hiddenElementIndexes,
      }) as IdeogramBox[];
      onOverlapStackChange(hits.map(box => box.elementIndex));

      if (!handle && hits.length === 0) {
        onSelect(null);
        return;
      }

      if (!handle && hits.length > 0 && !chooseDragTarget(hits, selectedElementIndex, lockedElementIndexes)) {
        const nextSelection = cycleHitSelection(hits, selectedElementIndex, event.shiftKey ? -1 : 1);
        onSelect(nextSelection);
        showCycleToast(event.clientX, event.clientY, hits, nextSelection);
        return;
      }

      const dragBox = handle && selectedBox ? selectedBox : chooseDragTarget(hits, selectedElementIndex, lockedElementIndexes);
      if (!dragBox) return;
      const dragHandle: DragHandle = handle || 'move';
      onSelect(dragBox.elementIndex);

      const startBox = { x1: dragBox.x1, y1: dragBox.y1, x2: dragBox.x2, y2: dragBox.y2 };
      let latest = startBox;
      let moved = false;

      const onMove = (moveEvent: PointerEvent) => {
        const point = pointToNorm(moveEvent.clientX, moveEvent.clientY);
        if (!point) return;
        const dx = point.x - start.x;
        const dy = point.y - start.y;
        moved = moved || Math.abs(dx) > CLICK_DRAG_TOLERANCE || Math.abs(dy) > CLICK_DRAG_TOLERANCE;
        latest = resizeOrMoveBox(startBox, dx, dy, dragHandle, MIN_BOX_SPAN);
        setDragPreview({ elementIndex: dragBox.elementIndex, box: latest });
      };

      const cleanup = () => {
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
        window.removeEventListener('keydown', onCancelKey, true);
      };

      const onUp = (upEvent: PointerEvent) => {
        cleanup();
        setDragPreview(null);
        if (moved) {
          onChangeBox(dragBox.elementIndex, latest);
          return;
        }
        if (!handle) {
          const nextSelection = cycleHitSelection(hits, selectedElementIndex, upEvent.shiftKey ? -1 : 1);
          onSelect(nextSelection);
          showCycleToast(upEvent.clientX, upEvent.clientY, hits, nextSelection);
        }
      };

      const onCancelKey = (keyEvent: KeyboardEvent) => {
        if (keyEvent.key !== 'Escape') return;
        keyEvent.preventDefault();
        keyEvent.stopImmediatePropagation();
        cleanup();
        setDragPreview(null);
      };

      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
      window.addEventListener('keydown', onCancelKey, true);
    },
    [
      beginDraw,
      drawingType,
      handleTolerance,
      hiddenElementIndexes,
      lockedElementIndexes,
      onChangeBox,
      onOverlapStackChange,
      onSelect,
      pointToNorm,
      selectedBox,
      selectedElementIndex,
      showCycleToast,
      visibleBoxes,
    ],
  );

  return (
    <div
      ref={layerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      className="absolute inset-0 touch-none"
      style={{ cursor: drawingType ? 'crosshair' : cursor }}
    >
      {orderedBoxes.map(box => {
        const sourceIndex = boxes.findIndex(candidate => candidate.elementIndex === box.elementIndex);
        const preview = dragPreview?.elementIndex === box.elementIndex ? dragPreview.box : box;
        const selected = selectedElementIndex === box.elementIndex;
        const locked = lockedElementIndexes.has(box.elementIndex);
        const color = resolveBoxColor(box, sourceIndex, selected);
        return (
          <div
            key={box.elementIndex}
            className={classNames('pointer-events-none absolute border-2', {
              'shadow-[0_0_0_1px_rgba(255,255,255,0.75),0_0_0_4px_rgba(34,211,238,0.18)]': selected,
              'border-dashed opacity-75': locked && !selected,
              'opacity-55': selectedElementIndex != null && !selected,
            })}
            style={{
              left: `${preview.x1 / 10}%`,
              top: `${preview.y1 / 10}%`,
              width: `${Math.max(0, preview.x2 - preview.x1) / 10}%`,
              height: `${Math.max(0, preview.y2 - preview.y1) / 10}%`,
              borderColor: color,
              zIndex: selected ? 20 : sourceIndex + 1,
            }}
          >
            <span
              title={box.label}
              className="absolute left-0 top-0 flex max-w-[12rem] items-center gap-1 truncate px-1 py-0.5 text-[10px] font-semibold leading-none text-gray-950"
              style={{ backgroundColor: color }}
            >
              {locked && <Lock className="h-2.5 w-2.5" />}
              <span className="truncate">{box.label || (box.type === 'text' ? 'text' : 'object')}</span>
            </span>
            {selected &&
              !locked &&
              RESIZE_HANDLES.map(handle => <span key={handle} aria-hidden="true" className={handleClassName(handle)} />)}
          </div>
        );
      })}
      {newBoxPreview && (
        <div
          className="pointer-events-none absolute border-2 border-dashed border-white bg-white/10"
          style={{
            left: `${newBoxPreview.x1 / 10}%`,
            top: `${newBoxPreview.y1 / 10}%`,
            width: `${Math.max(0, newBoxPreview.x2 - newBoxPreview.x1) / 10}%`,
            height: `${Math.max(0, newBoxPreview.y2 - newBoxPreview.y1) / 10}%`,
          }}
        />
      )}
      {(() => {
        const activePreview = newBoxPreview ?? dragPreview?.box ?? null;
        if (!activePreview) return null;
        const badgeAbove = activePreview.y1 >= 30;
        return (
          <div
            className="pointer-events-none absolute z-40 whitespace-nowrap rounded border border-gray-700 bg-gray-950/90 px-1.5 py-0.5 font-mono text-[10px] leading-none text-gray-200 shadow"
            style={{
              left: `${activePreview.x1 / 10}%`,
              top: badgeAbove ? `calc(${activePreview.y1 / 10}% - 1.25rem)` : `calc(${activePreview.y1 / 10}% + 2px)`,
            }}
          >
            {previewSizeLabel(activePreview, imageSize)}
          </div>
        );
      })()}
      {cycleToast && (
        <div
          className="pointer-events-none absolute z-40 rounded-md border border-blue-400/40 bg-gray-950/90 px-2 py-1 text-[11px] font-semibold text-blue-100 shadow-xl"
          style={{ left: cycleToast.x, top: cycleToast.y }}
        >
          {cycleToast.index}/{cycleToast.count}
        </div>
      )}
    </div>
  );
}
