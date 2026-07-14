'use client';

import classNames from 'classnames';
import { Copy, Eye, EyeOff, Filter, GripVertical, Lock, Plus, Search, Trash2, Unlock } from 'lucide-react';
import { useEffect, useId, useMemo, useRef, useState } from 'react';
import type { IdeogramBox, IdeogramElementType } from '@/utils/ideogramCaption';
import { layerLabelForElement, resolveBoxColor } from './utils';

const BOX_FILTERS = ['all', 'visible', 'hidden', 'locked'] as const;

type BoxFilter = (typeof BOX_FILTERS)[number];

type BoxRow = {
  box: IdeogramBox;
  boxIndex: number;
  element: unknown;
  elementIndex: number;
};

export type StandaloneBoxesPanelProps = {
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
  onAddBox?: () => void;
  onReorderBox?: (sourceElementIndex: number, targetElementIndex: number) => void;
};

function elementType(element: unknown): IdeogramElementType {
  if (typeof element !== 'object' || element === null) return 'obj';
  return (element as { type?: unknown }).type === 'text' ? 'text' : 'obj';
}

function filterLabel(filter: BoxFilter) {
  if (filter === 'all') return 'All boxes';
  if (filter === 'visible') return 'Visible boxes';
  if (filter === 'hidden') return 'Hidden boxes';
  return 'Locked boxes';
}

export function StandaloneBoxesPanel({
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
  onAddBox,
  onReorderBox,
}: StandaloneBoxesPanelProps) {
  const filterMenuId = useId();
  const filterButtonRef = useRef<HTMLButtonElement | null>(null);
  const filterPopoverRef = useRef<HTMLDivElement | null>(null);
  const rowRefs = useRef(new Map<number, HTMLButtonElement | null>());
  const [searchQuery, setSearchQuery] = useState('');
  const [boxFilter, setBoxFilter] = useState<BoxFilter>('all');
  const [filterOpen, setFilterOpen] = useState(false);
  const [dragTargetElementIndex, setDragTargetElementIndex] = useState<number | null>(null);

  const rows = useMemo<BoxRow[]>(() => {
    const rowsByElementIndex = new Map<number, BoxRow>();

    boxes.forEach((box, boxIndex) => {
      const { elementIndex } = box;
      if (!Number.isInteger(elementIndex) || elementIndex < 0 || elementIndex >= elements.length) return;
      if (rowsByElementIndex.has(elementIndex)) return;

      rowsByElementIndex.set(elementIndex, {
        box,
        boxIndex,
        element: elements[elementIndex],
        elementIndex,
      });
    });

    return Array.from(rowsByElementIndex.values()).sort((left, right) => left.elementIndex - right.elementIndex);
  }, [boxes, elements]);

  const visibleRows = useMemo(() => {
    const query = searchQuery.trim().toLocaleLowerCase();

    return rows.filter(({ element, elementIndex }) => {
      const hidden = hiddenElementIndexes.has(elementIndex);
      const locked = lockedElementIndexes.has(elementIndex);
      if (boxFilter === 'visible' && hidden) return false;
      if (boxFilter === 'hidden' && !hidden) return false;
      if (boxFilter === 'locked' && !locked) return false;

      if (!query) return true;
      const type = elementType(element) === 'text' ? 'text' : 'object';
      const label = layerLabelForElement(element, elementIndex);
      return `${label} ${type} box ${elementIndex + 1}`.toLocaleLowerCase().includes(query);
    });
  }, [boxFilter, hiddenElementIndexes, lockedElementIndexes, rows, searchQuery]);

  const boxedElementIndexes = useMemo(() => new Set(rows.map(row => row.elementIndex)), [rows]);

  useEffect(() => {
    if (selectedElementIndex == null) return;
    rowRefs.current.get(selectedElementIndex)?.scrollIntoView({ block: 'nearest' });
  }, [selectedElementIndex]);

  useEffect(() => {
    if (!filterOpen) return;

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (filterButtonRef.current?.contains(target) || filterPopoverRef.current?.contains(target)) return;
      setFilterOpen(false);
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      setFilterOpen(false);
      filterButtonRef.current?.focus();
    };

    document.addEventListener('pointerdown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [filterOpen]);

  return (
    <section className="flex min-h-0 flex-col bg-[#071019] text-gray-100" aria-label="Boxes">
      <div className="flex h-9 flex-none items-center border-b border-[#273039] px-3">
        <h3 className="text-sm font-semibold tracking-tight text-gray-100">Boxes</h3>
        <span className="ml-3 text-xs font-medium tabular-nums text-gray-500" aria-label={`${rows.length} boxes`}>
          {rows.length}
        </span>
      </div>

      <div className="flex h-10 flex-none items-center gap-1.5 border-b border-[#1d2831] px-3">
        <label className="flex h-8 min-w-0 flex-1 items-center gap-2 rounded-[4px] border border-[#33414d] bg-[#08121a] px-2.5 text-gray-500 focus-within:border-cyan-600 focus-within:text-gray-300">
          <Search className="h-4 w-4 flex-none" aria-hidden="true" />
          <input
            type="search"
            value={searchQuery}
            onChange={event => setSearchQuery(event.target.value)}
            aria-label="Search boxes"
            placeholder="Search boxes"
            className="min-w-0 flex-1 bg-transparent text-xs text-gray-200 outline-none placeholder:text-gray-500"
          />
        </label>

        <div className="relative flex-none">
          <button
            ref={filterButtonRef}
            type="button"
            title="Filter boxes"
            aria-label={`Filter boxes: ${filterLabel(boxFilter)}`}
            aria-controls={filterMenuId}
            aria-expanded={filterOpen}
            onClick={() => setFilterOpen(open => !open)}
            className={classNames(
              'flex h-8 w-8 items-center justify-center rounded-[4px] border border-[#33414d] bg-[#08121a] transition-colors hover:border-gray-500 hover:text-white',
              boxFilter === 'all' ? 'text-gray-400' : 'border-cyan-800 text-cyan-300',
            )}
          >
            <Filter className="h-4 w-4" aria-hidden="true" />
          </button>

          {filterOpen ? (
            <div
              ref={filterPopoverRef}
              id={filterMenuId}
              role="group"
              aria-label="Box filters"
              className="absolute right-0 top-9 z-20 w-36 overflow-hidden rounded-[4px] border border-[#33414d] bg-[#0b151e] p-1 shadow-xl shadow-black/50"
            >
              {BOX_FILTERS.map(filter => (
                <button
                  key={filter}
                  type="button"
                  aria-pressed={boxFilter === filter}
                  onClick={() => {
                    setBoxFilter(filter);
                    setFilterOpen(false);
                  }}
                  className={classNames(
                    'flex h-7 w-full items-center rounded-[3px] px-2 text-left text-xs',
                    boxFilter === filter
                      ? 'bg-cyan-950/60 text-cyan-200'
                      : 'text-gray-400 hover:bg-gray-800 hover:text-gray-100',
                  )}
                >
                  {filterLabel(filter)}
                </button>
              ))}
            </div>
          ) : null}
        </div>

        <button
          type="button"
          onClick={onAddBox}
          disabled={!onAddBox}
          aria-label="Add box"
          className="flex h-8 flex-none items-center gap-1.5 rounded-[4px] border border-[#33414d] bg-[#08121a] px-3 text-xs font-medium text-gray-100 transition-colors hover:border-gray-500 hover:bg-[#0d1a24] disabled:cursor-not-allowed disabled:text-gray-600"
        >
          <Plus className="h-4 w-4" aria-hidden="true" />
          Add box
        </button>
      </div>

      <div className="operator-scrollbar-none min-h-0 flex-1 space-y-[3px] overflow-y-auto px-2.5 py-1">
        {visibleRows.length === 0 ? (
          <div className="px-2 py-5 text-center text-xs text-gray-500">
            {rows.length === 0 ? 'No boxes' : 'No boxes match this view'}
          </div>
        ) : (
          visibleRows.map(({ box, boxIndex, element, elementIndex }) => {
            const selected = selectedElementIndex === elementIndex;
            const hidden = hiddenElementIndexes.has(elementIndex);
            const locked = lockedElementIndexes.has(elementIndex);
            const type = elementType(element) === 'text' ? 'Text' : 'Object';
            const color = resolveBoxColor(box, boxIndex, selected);
            const label = layerLabelForElement(element, elementIndex);
            const paddedIndex = String(elementIndex + 1).padStart(2, '0');

            return (
              <div
                key={elementIndex}
                onDragOver={event => {
                  if (!onReorderBox) return;
                  event.preventDefault();
                  event.dataTransfer.dropEffect = 'move';
                  setDragTargetElementIndex(elementIndex);
                }}
                onDragLeave={() => {
                  if (dragTargetElementIndex === elementIndex) setDragTargetElementIndex(null);
                }}
                onDrop={event => {
                  if (!onReorderBox) return;
                  event.preventDefault();
                  const transferValue =
                    event.dataTransfer.getData('application/x-aitk-element-index') ||
                    event.dataTransfer.getData('text/plain');
                  const sourceElementIndex = Number(transferValue);
                  setDragTargetElementIndex(null);
                  if (
                    !Number.isInteger(sourceElementIndex) ||
                    !boxedElementIndexes.has(sourceElementIndex) ||
                    sourceElementIndex === elementIndex
                  ) {
                    return;
                  }
                  onReorderBox(sourceElementIndex, elementIndex);
                }}
                className={classNames(
                  'grid h-[41px] grid-cols-[18px_28px_12px_minmax(0,1fr)_28px_28px_28px] items-center gap-1 rounded-[3px] border px-1.5 transition-colors',
                  selected
                    ? 'border-cyan-400 bg-cyan-950/30 shadow-[inset_0_0_0_1px_rgba(34,211,238,0.08)]'
                    : 'border-transparent bg-[#0d1821] hover:bg-[#12202b]',
                  hidden && 'opacity-45',
                  dragTargetElementIndex === elementIndex && 'ring-1 ring-inset ring-cyan-300',
                )}
              >
                <button
                  type="button"
                  draggable={Boolean(onReorderBox)}
                  disabled={!onReorderBox}
                  title={onReorderBox ? `Reorder ${label}` : 'Box reordering unavailable'}
                  aria-label={onReorderBox ? `Reorder box ${label}` : `Box ${label} cannot be reordered`}
                  onDragStart={event => {
                    if (!onReorderBox) {
                      event.preventDefault();
                      return;
                    }
                    event.dataTransfer.effectAllowed = 'move';
                    event.dataTransfer.setData('application/x-aitk-element-index', String(elementIndex));
                    event.dataTransfer.setData('text/plain', String(elementIndex));
                  }}
                  onDragEnd={() => setDragTargetElementIndex(null)}
                  className={classNames(
                    'flex h-7 w-[18px] items-center justify-center text-gray-500 hover:text-gray-200 disabled:cursor-default disabled:opacity-50',
                    onReorderBox && 'cursor-grab active:cursor-grabbing',
                  )}
                >
                  <GripVertical className="h-4 w-4" aria-hidden="true" />
                </button>

                <button
                  type="button"
                  title={hidden ? 'Show box' : 'Hide box'}
                  aria-label={hidden ? `Show box ${label}` : `Hide box ${label}`}
                  onClick={() => onToggleHidden(elementIndex)}
                  className="flex h-7 w-7 items-center justify-center rounded-[3px] text-gray-300 hover:bg-white/5 hover:text-white"
                >
                  {hidden ? (
                    <EyeOff className="h-4 w-4" aria-hidden="true" />
                  ) : (
                    <Eye className="h-4 w-4" aria-hidden="true" />
                  )}
                </button>

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
                  aria-label={`Select box ${label}, ${type.toLocaleLowerCase()}, number ${elementIndex + 1}`}
                  aria-pressed={selected}
                  className="min-w-0 py-1 text-left"
                >
                  <span className="block truncate text-xs font-medium leading-4 text-gray-100">{label}</span>
                  <span className="block truncate text-[10px] leading-3.5 text-gray-500">
                    {type} · Box · #{paddedIndex}
                  </span>
                </button>

                <button
                  type="button"
                  title={locked ? 'Unlock box' : 'Lock box'}
                  aria-label={locked ? `Unlock box ${label}` : `Lock box ${label}`}
                  onClick={() => onToggleLocked(elementIndex)}
                  className={classNames(
                    'flex h-7 w-7 items-center justify-center rounded-[3px] hover:bg-white/5 hover:text-white',
                    locked ? 'text-gray-200' : 'text-gray-500',
                  )}
                >
                  {locked ? (
                    <Lock className="h-4 w-4" aria-hidden="true" />
                  ) : (
                    <Unlock className="h-4 w-4" aria-hidden="true" />
                  )}
                </button>

                <button
                  type="button"
                  title={`Duplicate ${label}`}
                  aria-label={`Duplicate box ${label}`}
                  onClick={() => onDuplicate(elementIndex)}
                  className="flex h-7 w-7 items-center justify-center rounded-[3px] text-gray-400 hover:bg-white/5 hover:text-white"
                >
                  <Copy className="h-4 w-4" aria-hidden="true" />
                </button>

                <button
                  type="button"
                  title={`Delete ${label}`}
                  aria-label={`Delete box ${label}`}
                  onClick={() => onDelete(elementIndex)}
                  className="flex h-7 w-7 items-center justify-center rounded-[3px] text-gray-400 hover:bg-red-950/50 hover:text-red-300 focus:bg-red-950/50 focus:text-red-300"
                >
                  <Trash2 className="h-4 w-4" aria-hidden="true" />
                </button>
              </div>
            );
          })
        )}
      </div>
    </section>
  );
}
