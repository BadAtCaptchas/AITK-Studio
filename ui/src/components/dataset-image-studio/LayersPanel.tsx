'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import classNames from 'classnames';
import {
  Copy,
  Eye,
  EyeOff,
  Filter,
  GripVertical,
  Layers,
  Lock,
  Plus,
  Search,
  SquareDashed,
  Trash2,
  Type,
  Unlock,
} from 'lucide-react';
import type { IdeogramBox, IdeogramElementType } from '@/utils/ideogramCaption';
import { BOX_COLORS } from './constants';
import { layerLabelForElement, resolveBoxColor } from './utils';

type LayersPanelPresentation = 'default' | 'standalone';
type StandaloneLayerFilter = 'all' | 'visible' | 'hidden' | 'locked';

type LayersPanelProps = {
  elements: unknown[];
  boxes: IdeogramBox[];
  selectedElementIndex: number | null;
  hiddenElementIndexes: Set<number>;
  lockedElementIndexes: Set<number>;
  onSelect: (elementIndex: number) => void;
  onToggleHidden: (elementIndex: number) => void;
  onToggleLocked: (elementIndex: number) => void;
  onDuplicate: (elementIndex: number) => void;
  onDelete: (elementIndex: number) => void;
  presentation?: LayersPanelPresentation;
  onAddLayer?: () => void;
  onReorderLayer?: (sourceElementIndex: number, targetElementIndex: number) => void;
};

function layerTypeForElement(element: unknown): IdeogramElementType {
  if (typeof element !== 'object' || element === null) return 'obj';
  return (element as { type?: unknown }).type === 'text' ? 'text' : 'obj';
}

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
  presentation = 'default',
  onAddLayer,
  onReorderLayer,
}: LayersPanelProps) {
  const rowRefs = useRef(new Map<number, HTMLButtonElement | null>());
  const [searchQuery, setSearchQuery] = useState('');
  const [layerFilter, setLayerFilter] = useState<StandaloneLayerFilter>('all');
  const [filterOpen, setFilterOpen] = useState(false);
  const [dragTargetElementIndex, setDragTargetElementIndex] = useState<number | null>(null);
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
  const standaloneRows = useMemo(() => {
    const query = searchQuery.trim().toLocaleLowerCase();
    return rows.filter(({ element, elementIndex }) => {
      const hidden = hiddenElementIndexes.has(elementIndex);
      const locked = lockedElementIndexes.has(elementIndex);
      if (layerFilter === 'visible' && hidden) return false;
      if (layerFilter === 'hidden' && !hidden) return false;
      if (layerFilter === 'locked' && !locked) return false;

      if (!query) return true;
      const type = layerTypeForElement(element) === 'text' ? 'text' : 'object';
      return `${layerLabelForElement(element, elementIndex)} ${type}`.toLocaleLowerCase().includes(query);
    });
  }, [hiddenElementIndexes, layerFilter, lockedElementIndexes, rows, searchQuery]);

  useEffect(() => {
    if (selectedElementIndex == null) return;
    rowRefs.current.get(selectedElementIndex)?.scrollIntoView({ block: 'nearest' });
  }, [selectedElementIndex]);

  if (presentation === 'default') {
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
              const type = layerTypeForElement(element);
              const sourceIndex = boxes.findIndex(candidate => candidate.elementIndex === elementIndex);
              const color = box
                ? resolveBoxColor(box, sourceIndex, selected)
                : BOX_COLORS[elementIndex % BOX_COLORS.length];
              const label = layerLabelForElement(element, elementIndex);
              return (
                <div
                  key={elementIndex}
                  className={classNames(
                    'group grid grid-cols-[auto_1fr_auto_auto_auto_auto] items-center gap-2 border-b border-gray-900 px-2 py-1.5 last:border-b-0',
                    {
                      'bg-blue-600/20': selected,
                      'opacity-50': hidden,
                    },
                  )}
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
                    {type === 'text' ? (
                      <Type className="h-3.5 w-3.5 text-amber-300" />
                    ) : (
                      <SquareDashed className="h-3.5 w-3.5 text-cyan-300" />
                    )}
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
                    className={classNames(
                      'flex h-8 w-8 items-center justify-center rounded-md hover:bg-gray-800 hover:text-gray-100',
                      {
                        'text-amber-300': locked,
                        'text-gray-500': !locked,
                      },
                    )}
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

  return (
    <section className="flex min-h-0 flex-col bg-[#071019] text-gray-100">
      <div className="flex h-9 flex-none items-center border-b border-[#273039] px-3">
        <h3 className="text-sm font-semibold tracking-tight text-gray-100">Layers</h3>
        <span className="ml-3 text-xs font-medium text-gray-500">{elements.length}</span>
      </div>

      <div className="flex h-10 flex-none items-center gap-1.5 border-b border-[#1d2831] px-3">
        <label className="flex h-8 min-w-0 flex-1 items-center gap-2 rounded-[4px] border border-[#33414d] bg-[#08121a] px-2.5 text-gray-500 focus-within:border-cyan-600 focus-within:text-gray-300">
          <Search className="h-4 w-4 flex-none" />
          <input
            type="search"
            value={searchQuery}
            onChange={event => setSearchQuery(event.target.value)}
            aria-label="Search layers"
            placeholder="Search layers"
            className="min-w-0 flex-1 bg-transparent text-xs text-gray-200 outline-none placeholder:text-gray-500"
          />
        </label>
        <div className="relative flex-none">
          <button
            type="button"
            title="Filter layers"
            aria-label="Filter layers"
            aria-expanded={filterOpen}
            onClick={() => setFilterOpen(open => !open)}
            className={classNames(
              'flex h-8 w-8 items-center justify-center rounded-[4px] border border-[#33414d] bg-[#08121a] transition-colors hover:border-gray-500 hover:text-white',
              layerFilter === 'all' ? 'text-gray-400' : 'border-cyan-800 text-cyan-300',
            )}
          >
            <Filter className="h-4 w-4" />
          </button>
          {filterOpen ? (
            <div className="absolute right-0 top-9 z-20 w-32 overflow-hidden rounded-[4px] border border-[#33414d] bg-[#0b151e] p-1 shadow-xl shadow-black/50">
              {(['all', 'visible', 'hidden', 'locked'] as const).map(filter => (
                <button
                  key={filter}
                  type="button"
                  onClick={() => {
                    setLayerFilter(filter);
                    setFilterOpen(false);
                  }}
                  className={classNames(
                    'flex h-7 w-full items-center rounded-[3px] px-2 text-left text-xs capitalize',
                    layerFilter === filter
                      ? 'bg-cyan-950/60 text-cyan-200'
                      : 'text-gray-400 hover:bg-gray-800 hover:text-gray-100',
                  )}
                >
                  {filter}
                </button>
              ))}
            </div>
          ) : null}
        </div>
        <button
          type="button"
          onClick={onAddLayer}
          disabled={!onAddLayer}
          className="flex h-8 flex-none items-center gap-1.5 rounded-[4px] border border-[#33414d] bg-[#08121a] px-3 text-xs font-medium text-gray-100 transition-colors hover:border-gray-500 hover:bg-[#0d1a24] disabled:cursor-not-allowed disabled:text-gray-600"
        >
          <Plus className="h-4 w-4" />
          Add layer
        </button>
      </div>

      <div className="operator-scrollbar-none min-h-0 flex-1 space-y-[3px] overflow-y-auto px-2.5 py-1">
        {standaloneRows.length === 0 ? (
          <div className="px-2 py-5 text-center text-xs text-gray-500">
            {rows.length === 0 ? 'No layers' : 'No layers match this view'}
          </div>
        ) : (
          standaloneRows.map(({ element, elementIndex, box }) => {
            const selected = selectedElementIndex === elementIndex;
            const hidden = hiddenElementIndexes.has(elementIndex);
            const locked = lockedElementIndexes.has(elementIndex);
            const type = layerTypeForElement(element);
            const sourceIndex = boxes.findIndex(candidate => candidate.elementIndex === elementIndex);
            const color = box
              ? resolveBoxColor(box, sourceIndex, selected)
              : BOX_COLORS[elementIndex % BOX_COLORS.length];
            const label = layerLabelForElement(element, elementIndex);
            return (
              <div
                key={elementIndex}
                onDragOver={event => {
                  if (!onReorderLayer) return;
                  event.preventDefault();
                  event.dataTransfer.dropEffect = 'move';
                  setDragTargetElementIndex(elementIndex);
                }}
                onDragLeave={() => {
                  if (dragTargetElementIndex === elementIndex) setDragTargetElementIndex(null);
                }}
                onDrop={event => {
                  if (!onReorderLayer) return;
                  event.preventDefault();
                  const sourceElementIndex = Number(event.dataTransfer.getData('text/plain'));
                  setDragTargetElementIndex(null);
                  if (!Number.isInteger(sourceElementIndex) || sourceElementIndex === elementIndex) return;
                  onReorderLayer(sourceElementIndex, elementIndex);
                }}
                className={classNames(
                  'grid h-9 grid-cols-[18px_28px_28px_12px_minmax(0,1fr)_28px_28px_28px] items-center gap-1 rounded-[3px] border px-1.5 transition-colors',
                  selected
                    ? 'border-cyan-400 bg-cyan-950/30 shadow-[inset_0_0_0_1px_rgba(34,211,238,0.08)]'
                    : 'border-transparent bg-[#0d1821] hover:bg-[#12202b]',
                  hidden && 'opacity-45',
                  dragTargetElementIndex === elementIndex && 'ring-1 ring-inset ring-cyan-300',
                )}
              >
                <button
                  type="button"
                  draggable={Boolean(onReorderLayer)}
                  title={onReorderLayer ? `Reorder ${label}` : undefined}
                  aria-label={onReorderLayer ? `Reorder ${label}` : undefined}
                  onDragStart={event => {
                    if (!onReorderLayer) {
                      event.preventDefault();
                      return;
                    }
                    event.dataTransfer.effectAllowed = 'move';
                    event.dataTransfer.setData('text/plain', String(elementIndex));
                  }}
                  onDragEnd={() => setDragTargetElementIndex(null)}
                  className={classNames(
                    'flex h-7 w-[18px] items-center justify-center text-gray-500 hover:text-gray-200',
                    onReorderLayer ? 'cursor-grab active:cursor-grabbing' : 'cursor-default',
                  )}
                >
                  <GripVertical className="h-4 w-4" />
                </button>
                <button
                  type="button"
                  title={hidden ? 'Show layer' : 'Hide layer'}
                  aria-label={hidden ? `Show ${label}` : `Hide ${label}`}
                  onClick={() => onToggleHidden(elementIndex)}
                  className="flex h-7 w-7 items-center justify-center rounded-[3px] text-gray-300 hover:bg-white/5 hover:text-white"
                >
                  {hidden ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
                <span className="flex h-7 w-7 items-center justify-center text-gray-300" title={type === 'text' ? 'Text layer' : 'Object layer'}>
                  {type === 'text' ? <Type className="h-4 w-4" /> : <SquareDashed className="h-4 w-4" />}
                </span>
                <span
                  className="h-3 w-3 rounded-full shadow-[0_0_0_1px_rgba(255,255,255,0.08)]"
                  style={{ backgroundColor: color }}
                  aria-hidden="true"
                />
                <button
                  ref={node => {
                    rowRefs.current.set(elementIndex, node);
                  }}
                  type="button"
                  onClick={() => onSelect(elementIndex)}
                  className="min-w-0 py-1 text-left"
                >
                  <span className="block truncate text-xs font-medium leading-4 text-gray-100">{label}</span>
                  <span className="block truncate text-[10px] leading-3.5 text-gray-500">
                    {type === 'text' ? 'Text' : 'Object'} · {box ? 'Box' : 'No box'} · #
                    {String(elementIndex + 1).padStart(2, '0')}
                  </span>
                </button>
                <button
                  type="button"
                  title={locked ? 'Unlock layer' : 'Lock layer'}
                  aria-label={locked ? `Unlock ${label}` : `Lock ${label}`}
                  onClick={() => onToggleLocked(elementIndex)}
                  className={classNames(
                    'flex h-7 w-7 items-center justify-center rounded-[3px] hover:bg-white/5 hover:text-white',
                    locked ? 'text-gray-200' : 'text-gray-500',
                  )}
                >
                  {locked ? <Lock className="h-4 w-4" /> : <Unlock className="h-4 w-4" />}
                </button>
                <button
                  type="button"
                  title={`Duplicate ${label}`}
                  aria-label={`Duplicate ${label}`}
                  onClick={() => onDuplicate(elementIndex)}
                  className="flex h-7 w-7 items-center justify-center rounded-[3px] text-gray-400 hover:bg-white/5 hover:text-white"
                >
                  <Copy className="h-4 w-4" />
                </button>
                <button
                  type="button"
                  title={`Delete ${label}`}
                  aria-label={`Delete ${label}`}
                  onClick={() => onDelete(elementIndex)}
                  className="flex h-7 w-7 items-center justify-center rounded-[3px] text-gray-400 hover:bg-red-950/50 hover:text-red-300 focus:bg-red-950/50 focus:text-red-300"
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
