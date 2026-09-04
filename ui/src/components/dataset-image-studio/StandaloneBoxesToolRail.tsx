'use client';

import type { ReactNode } from 'react';
import classNames from 'classnames';
import {
  Hand,
  Maximize2,
  MousePointer2,
  Redo2,
  ScanSearch,
  SquareDashed,
  Type,
  Undo2,
  ZoomIn,
} from 'lucide-react';
import type { ToolMode } from './types';

export type StandaloneBoxesToolRailProps = {
  activeTool: ToolMode;
  canAnnotate: boolean;
  canAutoBoxes?: boolean;
  canUndo: boolean;
  canRedo: boolean;
  onToolChange: (tool: ToolMode) => void;
  onAutoBoxes: () => void;
  onZoomIn: () => void;
  onFit: () => void;
  onUndo: () => void;
  onRedo: () => void;
};

type RailButtonProps = {
  label: string;
  icon: ReactNode;
  onClick: () => void;
  active?: boolean;
  disabled?: boolean;
  action?: boolean;
};

function RailButton({
  label,
  icon,
  onClick,
  active = false,
  disabled = false,
  action = false,
}: RailButtonProps) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      aria-pressed={action ? undefined : active}
      disabled={disabled}
      onClick={onClick}
      className={classNames(
        'group grid min-h-[48px] w-full grid-cols-[38px_minmax(0,1fr)] items-center gap-1 rounded-[4px] px-1 text-left transition-colors',
        'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-brand-300',
        active ? 'text-brand-200' : 'text-gray-400 hover:bg-gray-900 hover:text-gray-100',
        disabled && 'cursor-not-allowed opacity-30 hover:bg-transparent hover:text-gray-400',
      )}
    >
      <span
        className={classNames(
          'flex h-9 w-9 items-center justify-center rounded-[4px] border transition-colors',
          active
            ? 'border-brand-400 bg-brand-400 text-[var(--brand-ink)] shadow-[0_0_16px_rgb(var(--brand-400)/0.12)]'
            : 'border-transparent text-gray-300 group-hover:border-gray-800 group-hover:text-white',
          disabled && 'group-hover:border-transparent group-hover:text-gray-300',
        )}
        aria-hidden="true"
      >
        {icon}
      </span>
      <span className="min-w-0 text-[10px] font-medium leading-[13px]">{label}</span>
    </button>
  );
}

function RailDivider() {
  return <div className="my-1 h-px w-full bg-gray-900" aria-hidden="true" />;
}

export function StandaloneBoxesToolRail({
  activeTool,
  canAnnotate,
  canAutoBoxes = canAnnotate,
  canUndo,
  canRedo,
  onToolChange,
  onAutoBoxes,
  onZoomIn,
  onFit,
  onUndo,
  onRedo,
}: StandaloneBoxesToolRailProps) {
  return (
    <aside
      aria-label="Boxes tools"
      className="operator-scrollbar-none flex w-[100px] flex-none flex-col overflow-y-auto border-r border-gray-800 bg-gray-950 px-2 py-3"
    >
      <RailButton
        label="Select"
        icon={<MousePointer2 className="h-[18px] w-[18px]" />}
        active={activeTool === 'select'}
        disabled={!canAnnotate}
        onClick={() => onToolChange('select')}
      />
      <RailButton
        label="Box"
        icon={<SquareDashed className="h-[18px] w-[18px]" />}
        active={activeTool === 'box'}
        disabled={!canAnnotate}
        onClick={() => onToolChange('box')}
      />
      <RailButton
        label="Text"
        icon={<Type className="h-[18px] w-[18px]" />}
        active={activeTool === 'text'}
        disabled={!canAnnotate}
        onClick={() => onToolChange('text')}
      />
      <RailButton
        label="Auto Boxes"
        icon={<ScanSearch className="h-[18px] w-[18px]" />}
        disabled={!canAutoBoxes}
        action
        onClick={onAutoBoxes}
      />

      <RailDivider />

      <RailButton
        label="Zoom in"
        icon={<ZoomIn className="h-[18px] w-[18px]" />}
        action
        onClick={onZoomIn}
      />
      <RailButton
        label="Pan"
        icon={<Hand className="h-[18px] w-[18px]" />}
        active={activeTool === 'pan'}
        disabled={!canAnnotate}
        onClick={() => onToolChange('pan')}
      />
      <RailButton
        label="Fit"
        icon={<Maximize2 className="h-[18px] w-[18px]" />}
        action
        onClick={onFit}
      />

      <RailDivider />

      <RailButton
        label="Undo"
        icon={<Undo2 className="h-[18px] w-[18px]" />}
        disabled={!canUndo}
        action
        onClick={onUndo}
      />
      <RailButton
        label="Redo"
        icon={<Redo2 className="h-[18px] w-[18px]" />}
        disabled={!canRedo}
        action
        onClick={onRedo}
      />
    </aside>
  );
}
