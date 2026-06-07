'use client';

import { Keyboard, MousePointer2, Move, SquareDashed, Tags, Trash2, Type, Undo2, Redo2 } from 'lucide-react';
import type { ToolMode } from './types';
import { ToolButton } from './StudioControls';

export function ToolRail({
  activeTool,
  canAnnotate,
  hasSelection,
  canUndo,
  canRedo,
  onToolChange,
  onDelete,
  onUndo,
  onRedo,
  onShowJson,
}: {
  activeTool: ToolMode;
  canAnnotate: boolean;
  hasSelection: boolean;
  canUndo: boolean;
  canRedo: boolean;
  onToolChange: (tool: ToolMode) => void;
  onDelete: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onShowJson: () => void;
}) {
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
      <ToolButton label="Labels" icon={<Tags className="h-5 w-5" />} onClick={onShowJson} />
      <ToolButton label="Shortcuts" icon={<Keyboard className="h-5 w-5" />} />
    </aside>
  );
}
