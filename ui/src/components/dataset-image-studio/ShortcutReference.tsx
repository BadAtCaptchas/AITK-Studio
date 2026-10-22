'use client';

import { Modal } from '@/components/Modal';

const shortcuts = [
  { keys: 'Ctrl / ⌘ S', action: 'Save caption or JSON draft' },
  { keys: 'Ctrl / ⌘ Enter', action: 'Save and advance to the next image' },
  { keys: '← / →', action: 'Previous / next image when no unlocked box is selected' },
  { keys: 'Arrow keys', action: 'Move the selected unlocked box by 5 units' },
  { keys: 'Shift + arrows', action: 'Move the selected unlocked box by 20 units' },
  { keys: '[ / ]', action: 'Cycle overlapping annotations' },
  { keys: 'Delete / Backspace', action: 'Delete the selected annotation' },
  { keys: 'Ctrl / ⌘ D', action: 'Duplicate the selected annotation' },
  { keys: 'Ctrl / ⌘ Z', action: 'Undo' },
  { keys: 'Ctrl / ⌘ Shift Z', action: 'Redo (also Ctrl / ⌘ Y)' },
  { keys: 'Escape', action: 'Cancel palette sampling or clear the annotation selection' },
] as const;

export function ShortcutReference({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Dataset shortcuts" size="lg">
      <div className="max-h-[calc(100dvh-12rem)] overflow-y-auto">
        <p className="mb-4 text-sm text-gray-400">
          Annotation and image navigation shortcuts apply outside text fields. Save shortcuts also work while editing.
        </p>
        <dl className="divide-y divide-gray-800">
          {shortcuts.map(shortcut => (
            <div
              key={shortcut.keys}
              className="grid grid-cols-[minmax(8rem,0.7fr)_minmax(0,1.3fr)] items-center gap-4 py-2 text-sm"
            >
              <dt>
                <kbd className="rounded border border-gray-700 bg-gray-900 px-1.5 py-0.5 font-mono text-xs text-gray-300">
                  {shortcut.keys}
                </kbd>
              </dt>
              <dd className="text-gray-300">{shortcut.action}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-4 text-xs text-gray-500">
          Apply or discard unfinished JSON edits before modifying annotations. Arrow keys do not navigate images while
          the pan tool is active.
        </p>
        <div className="mt-4 flex justify-end">
          <button type="button" className="operator-button" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </Modal>
  );
}
