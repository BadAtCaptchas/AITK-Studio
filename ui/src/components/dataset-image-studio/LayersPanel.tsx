'use client';

import React, { useEffect, useMemo, useRef } from 'react';
import classNames from 'classnames';
import { Copy, Eye, EyeOff, Layers, Lock, SquareDashed, Trash2, Type, Unlock } from 'lucide-react';
import type { IdeogramBox, IdeogramElementType } from '@/utils/ideogramCaption';
import { BOX_COLORS } from './constants';
import { layerLabelForElement, resolveBoxColor } from './utils';

export function LayersPanel({
  elements,
  boxes,
  selectedElementIndex,
  hiddenElementIndexes,
  lockedElementIndexes,
  onSelect,
  onToggleHidden,
  onToggleLocked,
  onDuplicate,
  onDelete,
}: {
  elements: any[];
  boxes: IdeogramBox[];
  selectedElementIndex: number | null;
  hiddenElementIndexes: Set<number>;
  lockedElementIndexes: Set<number>;
  onSelect: (elementIndex: number) => void;
  onToggleHidden: (elementIndex: number) => void;
  onToggleLocked: (elementIndex: number) => void;
  onDuplicate: (elementIndex: number) => void;
  onDelete: (elementIndex: number) => void;
}) {
  const rowRefs = useRef(new Map<number, HTMLButtonElement | null>());
  const rows = useMemo(
    () =>
      elements
        .map((element, elementIndex) => ({
          element,
          elementIndex,
          box: boxes.find(candidate => candidate.elementIndex === elementIndex) || null,
        }))
        .reverse(),
    [boxes, elements],
  );

  useEffect(() => {
    if (selectedElementIndex == null) return;
    rowRefs.current.get(selectedElementIndex)?.scrollIntoView({ block: 'nearest' });
  }, [selectedElementIndex]);

  return (
    <section className="overflow-hidden rounded-md border border-gray-800 bg-gray-950/80">
      <div className="flex h-12 items-center justify-between border-b border-gray-800 px-4">
        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 text-blue-300" />
          <h3 className="text-sm font-semibold text-gray-100">Layers</h3>
        </div>
        <span className="text-xs text-gray-500">{elements.length}</span>
      </div>
      <div className="max-h-56 overflow-y-auto">
        {rows.length === 0 ? (
          <div className="px-4 py-3 text-sm text-gray-500">No layers</div>
        ) : (
          rows.map(({ element, elementIndex, box }) => {
            const selected = selectedElementIndex === elementIndex;
            const hidden = hiddenElementIndexes.has(elementIndex);
            const locked = lockedElementIndexes.has(elementIndex);
            const type: IdeogramElementType = element?.type === 'text' ? 'text' : 'obj';
            const sourceIndex = boxes.findIndex(candidate => candidate.elementIndex === elementIndex);
            const color = box ? resolveBoxColor(box, sourceIndex, selected) : BOX_COLORS[elementIndex % BOX_COLORS.length];
            const label = layerLabelForElement(element, elementIndex);
            return (
              <div
                key={elementIndex}
                className={classNames('group grid grid-cols-[auto_1fr_auto_auto_auto_auto] items-center gap-2 border-b border-gray-900 px-2 py-1.5 last:border-b-0', {
                  'bg-blue-600/20': selected,
                  'opacity-50': hidden,
                })}
              >
                <button
                  type="button"
                  title={hidden ? 'Show layer' : 'Hide layer'}
                  onClick={() => onToggleHidden(elementIndex)}
                  className="flex h-8 w-8 items-center justify-center rounded-md text-gray-400 hover:bg-gray-800 hover:text-gray-100"
                >
                  {hidden ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
                <button
                  ref={node => {
                    rowRefs.current.set(elementIndex, node);
                  }}
                  type="button"
                  onClick={() => onSelect(elementIndex)}
                  className="grid min-w-0 grid-cols-[auto_auto_1fr] items-center gap-2 rounded-md px-2 py-1 text-left hover:bg-gray-800"
                >
                  {type === 'text' ? <Type className="h-3.5 w-3.5 text-amber-300" /> : <SquareDashed className="h-3.5 w-3.5 text-cyan-300" />}
                  <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
                  <span className="min-w-0">
                    <span className="block truncate text-sm font-medium text-gray-100">{label}</span>
                    <span className="block truncate text-[11px] text-gray-500">
                      {type === 'text' ? 'Text' : 'Object'} - {box ? 'Box' : 'No box'} - #{elementIndex + 1}
                    </span>
                  </span>
                </button>
                <button
                  type="button"
                  title={locked ? 'Unlock layer' : 'Lock layer'}
                  onClick={() => onToggleLocked(elementIndex)}
                  className={classNames('flex h-8 w-8 items-center justify-center rounded-md hover:bg-gray-800 hover:text-gray-100', {
                    'text-amber-300': locked,
                    'text-gray-500': !locked,
                  })}
                >
                  {locked ? <Lock className="h-4 w-4" /> : <Unlock className="h-4 w-4" />}
                </button>
                <span className="rounded border border-gray-800 px-1.5 py-0.5 text-[10px] uppercase text-gray-500">
                  {type === 'text' ? 'TXT' : 'OBJ'}
                </span>
                <button
                  type="button"
                  title={`Duplicate ${label}`}
                  aria-label={`Duplicate ${label}`}
                  onClick={() => onDuplicate(elementIndex)}
                  className="flex h-8 w-8 items-center justify-center rounded-md text-gray-500 hover:bg-gray-800 hover:text-gray-100"
                >
                  <Copy className="h-4 w-4" />
                </button>
                <button
                  type="button"
                  title={`Delete ${label}`}
                  aria-label={`Delete ${label}`}
                  onClick={() => onDelete(elementIndex)}
                  className="flex h-8 w-8 items-center justify-center rounded-md text-gray-500 hover:bg-red-950/40 hover:text-red-300 focus:bg-red-950/40 focus:text-red-300"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            );
          })
        )}
      </div>
    </section>
  );
}
