'use client';

import classNames from 'classnames';
import {
  ArrowRight,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Hand,
  Loader2,
  Maximize2,
  MoreVertical,
  Trash2,
  ZoomIn,
} from 'lucide-react';

export function StudioToolbar({
  selectedIndex,
  itemCount,
  isSaving,
  isDirty,
  zoom,
  isDeletingCurrent,
  canDeleteCurrent,
  onPrevious,
  onNext,
  onCycleZoom,
  onPan,
  onFit,
  onDeleteCurrent,
}: {
  selectedIndex: number;
  itemCount: number;
  isSaving: boolean;
  isDirty: boolean;
  zoom: number;
  isDeletingCurrent?: boolean;
  canDeleteCurrent?: boolean;
  onPrevious: () => void;
  onNext: () => void;
  onCycleZoom: () => void;
  onPan: () => void;
  onFit: () => void;
  onDeleteCurrent: () => void;
}) {
  return (
    <div className="operator-scrollbar-none flex h-14 flex-shrink-0 items-center gap-2 overflow-x-auto border-b border-gray-900 bg-[#070b10] px-2 sm:gap-3 sm:px-3">
      <h2 className="min-w-36 flex-1 text-base font-semibold tracking-normal">Image Studio</h2>
      <div className="hidden items-center overflow-hidden rounded-md border border-gray-800 bg-gray-950 md:flex">
        <button
          type="button"
          className="flex h-9 w-11 items-center justify-center text-gray-300 hover:bg-gray-800"
          onClick={onPrevious}
          title="Previous"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        <div className="h-9 min-w-28 border-x border-gray-800 px-4 text-center text-sm font-medium leading-9">
          {selectedIndex + 1} / {itemCount}
        </div>
        <button
          type="button"
          className="flex h-9 w-11 items-center justify-center text-gray-300 hover:bg-gray-800"
          onClick={onNext}
          title="Next"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
      <div className="hidden h-9 items-center gap-2 rounded-md border border-gray-800 bg-gray-950 px-3 text-sm md:flex">
        {isSaving ? (
          <Loader2 className="h-4 w-4 animate-spin text-blue-400" />
        ) : (
          <span className={classNames('h-2 w-2 rounded-full', isDirty ? 'bg-blue-400' : 'bg-emerald-400')} />
        )}
        <span>{isSaving ? 'Saving' : isDirty ? 'Unsaved' : 'Saved'}</span>
      </div>
      <button
        type="button"
        title="Zoom"
        className="hidden h-9 items-center gap-2 rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 hover:bg-gray-900 md:flex"
        onClick={onCycleZoom}
      >
        <ZoomIn className="h-4 w-4" />
        {Math.round(zoom * 100)}%
        <ChevronDown className="h-3.5 w-3.5 text-gray-500" />
      </button>
      <button
        type="button"
        title="Pan"
        className="hidden h-9 w-10 items-center justify-center rounded-md border border-gray-800 bg-gray-950 text-gray-300 hover:bg-gray-900 md:flex"
        onClick={onPan}
      >
        <Hand className="h-4 w-4" />
      </button>
      <button
        type="button"
        title="Fit"
        className="hidden h-9 w-10 items-center justify-center rounded-md border border-gray-800 bg-gray-950 text-gray-300 hover:bg-gray-900 md:flex"
        onClick={onFit}
      >
        <Maximize2 className="h-4 w-4" />
      </button>
      <button
        type="button"
        title="Delete current image"
        aria-label="Delete current image"
        disabled={!canDeleteCurrent || isDeletingCurrent}
        className="flex h-9 w-10 items-center justify-center rounded-md border border-rose-900/70 bg-rose-950/35 text-rose-100 hover:bg-rose-900/50 disabled:cursor-not-allowed disabled:opacity-45"
        onClick={onDeleteCurrent}
      >
        {isDeletingCurrent ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
      </button>
      <button
        type="button"
        className="inline-flex h-10 items-center gap-2 rounded-md bg-blue-600 px-4 text-sm font-semibold text-white hover:bg-blue-500"
        onClick={onNext}
      >
        Next
        <ArrowRight className="h-4 w-4" />
      </button>
      <button type="button" title="More" className="flex h-9 w-9 items-center justify-center text-gray-400 hover:text-gray-100">
        <MoreVertical className="h-5 w-5" />
      </button>
    </div>
  );
}
