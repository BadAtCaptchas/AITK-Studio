'use client';

import classNames from 'classnames';
import { Keyboard, MousePointer2, Move, SquareDashed, Tags, Trash2, Type, Undo2, Redo2 } from 'lucide-react';
import type { ReactNode } from 'react';
import type { ToolMode } from './types';
import { ToolButton } from './StudioControls';

type ToolRailProps = {
  activeTool: ToolMode;
  canAnnotate: boolean;
  hasSelection: boolean;
  canUndo: boolean;
  canRedo: boolean;
  canShowJson?: boolean;
  presentation?: 'default' | 'standalone';
  onToolChange: (tool: ToolMode) => void;
  onDelete: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onShowJson: () => void;
};

export function ToolRail({
  activeTool,
  canAnnotate,
  hasSelection,
  canUndo,
  canRedo,
  canShowJson = true,
  presentation = 'default',
  onToolChange,
  onDelete,
  onUndo,
  onRedo,
  onShowJson,
}: ToolRailProps) {
  if (presentation === 'standalone') {
    const toolButton = (
      label: string,
      icon: ReactNode,
      onClick: () => void,
      active = false,
      disabled = false,
    ) => (
      <button
        key={label}
        type="button"
        title={label}
        aria-label={label}
        disabled={disabled}
        onClick={onClick}
        className={classNames(
          'flex h-10 w-10 items-center justify-center rounded-[4px] border transition-colors',
          active
            ? 'border-cyan-500/70 bg-cyan-400 text-[#021014]'
            : 'border-transparent text-gray-400 hover:border-[#33414d] hover:bg-[#101a23] hover:text-gray-100',
          disabled && 'cursor-not-allowed opacity-30 hover:border-transparent hover:bg-transparent',
        )}
      >
        {icon}
      </button>
    );

    return (
      <aside className="operator-scrollbar-none flex w-[65px] flex-none flex-col items-center gap-1 overflow-y-auto border-r border-[#273039] bg-[#071019] py-3">
        {toolButton('Select', <MousePointer2 className="h-[18px] w-[18px]" />, () => onToolChange('select'), activeTool === 'select', !canAnnotate)}
        {toolButton('Box', <SquareDashed className="h-[18px] w-[18px]" />, () => onToolChange('box'), activeTool === 'box', !canAnnotate)}
        {toolButton('Text', <Type className="h-[18px] w-[18px]" />, () => onToolChange('text'), activeTool === 'text', !canAnnotate)}
        {toolButton('Move', <Move className="h-[18px] w-[18px]" />, () => onToolChange('move'), activeTool === 'move', !canAnnotate)}
        {toolButton('Delete', <Trash2 className="h-[18px] w-[18px]" />, onDelete, false, !canAnnotate || !hasSelection)}
        <div className="my-1 h-px w-8 bg-[#273039]" />
        {toolButton('Undo', <Undo2 className="h-[18px] w-[18px]" />, onUndo, false, !canUndo)}
        {toolButton('Redo', <Redo2 className="h-[18px] w-[18px]" />, onRedo, false, !canRedo)}
        <div className="min-h-2 flex-1" />
        {toolButton('JSON labels', <Tags className="h-[18px] w-[18px]" />, onShowJson, false, !canShowJson)}
        {toolButton('Shortcuts', <Keyboard className="h-[18px] w-[18px]" />, () => undefined)}
      </aside>
    );
  }

  return (
    <aside className="operator-scrollbar-none flex h-16 flex-shrink-0 items-center gap-2 overflow-x-auto border-b border-gray-900 bg-[#060a0f] px-2 md:h-auto md:w-20 md:flex-col md:overflow-y-auto md:border-b-0 md:border-r md:px-0 md:py-3">
      <ToolButton
        active={activeTool === 'box'}
        disabled={!canAnnotate}
        label="Box"
        icon={<SquareDashed className="h-5 w-5" />}
        onClick={() => onToolChange('box')}
      />
      <ToolButton
        active={activeTool === 'text'}
        disabled={!canAnnotate}
        label="Text"
        icon={<Type className="h-5 w-5" />}
        onClick={() => onToolChange('text')}
      />
      <div className="hidden h-px w-14 bg-gray-900 md:block" />
      <ToolButton
        active={activeTool === 'select'}
        disabled={!canAnnotate}
        label="Select"
        icon={<MousePointer2 className="h-5 w-5" />}
        onClick={() => onToolChange('select')}
      />
      <ToolButton
        active={activeTool === 'move'}
        disabled={!canAnnotate}
        label="Move"
        icon={<Move className="h-5 w-5" />}
        onClick={() => onToolChange('move')}
      />
      <ToolButton disabled={!canAnnotate || !hasSelection} label="Delete" icon={<Trash2 className="h-5 w-5" />} onClick={onDelete} />
      <div className="hidden h-px w-14 bg-gray-900 md:block" />
      <ToolButton disabled={!canUndo} label="Undo" icon={<Undo2 className="h-5 w-5" />} onClick={onUndo} />
      <ToolButton disabled={!canRedo} label="Redo" icon={<Redo2 className="h-5 w-5" />} onClick={onRedo} />
      <div className="hidden flex-1 md:block" />
      <ToolButton disabled={!canShowJson} label="Labels" icon={<Tags className="h-5 w-5" />} onClick={onShowJson} />
      <ToolButton label="Shortcuts" icon={<Keyboard className="h-5 w-5" />} />
    </aside>
  );
}
