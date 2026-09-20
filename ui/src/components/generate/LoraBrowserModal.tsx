'use client';

import { useEffect, useState } from 'react';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { apiClient } from '@/utils/api';
import { uploadLoraFile } from '@/utils/streamedUploads';

import type { LoraPick, CloudLora } from '@/utils/loraTypes';
export type { LoraPick, CloudLora } from '@/utils/loraTypes';
function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
function picks(value: unknown): LoraPick[] {
  if (!record(value) || !Array.isArray(value.loras)) throw new Error('Invalid LoRA library response');
  return value.loras.flatMap((item: unknown) => {
    if (!record(item) || typeof item.path !== 'string' || typeof item.label !== 'string') return [];
    return [
      {
        path: item.path,
        name: item.label,
        triggerWords: Array.isArray(item.triggerWords)
          ? item.triggerWords.filter((v): v is string => typeof v === 'string')
          : [],
        model: record(item.model) ? item.model : undefined,
      },
    ];
  });
}
export default function LoraBrowserModal({
  isOpen,
  onClose,
  onPick,
  cloudLoras = [],
}: {
  isOpen: boolean;
  onClose: () => void;
  onPick: (pick: LoraPick) => void;
  cloudLoras?: CloudLora[];
}) {
  const [items, setItems] = useState<LoraPick[]>([]);
  const [query, setQuery] = useState('');
  const [path, setPath] = useState('');
  const [triggerWords, setTriggerWords] = useState('');
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  const refresh = async () => setItems(picks((await apiClient.get<unknown>('/api/generate/loras')).data));
  useEffect(() => {
    if (isOpen) void refresh().catch(() => setError('Could not load the LoRA library.'));
  }, [isOpen]);
  const choose = (pick: LoraPick) => {
    onPick(pick);
    onClose();
  };
  return (
    <Dialog open={isOpen} onClose={onClose} className="relative z-50">
      <DialogBackdrop className="fixed inset-0 bg-black/70" />
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <DialogPanel className="w-full max-w-2xl max-h-[85vh] overflow-auto rounded-xl border border-gray-700 bg-gray-900 p-5 space-y-4">
          <div className="flex justify-between">
            <DialogTitle className="font-semibold">Add LoRA or LoKr</DialogTitle>
            <button type="button" onClick={onClose}>
              Close
            </button>
          </div>
          {error && (
            <p role="alert" className="text-red-400">
              {error}
            </p>
          )}
          <input
            aria-label="Search LoRAs"
            placeholder="Search library"
            value={query}
            onChange={e => setQuery(e.target.value)}
            className="w-full rounded bg-gray-800 p-2"
          />
          <div className="max-h-72 overflow-auto space-y-2">
            {[...cloudLoras, ...items]
              .filter(item => `${item.name} ${item.path}`.toLowerCase().includes(query.toLowerCase()))
              .map(item => (
                <button
                  type="button"
                  key={item.path}
                  onClick={() => choose(item)}
                  className="block w-full text-left rounded bg-gray-800 hover:bg-gray-700 p-3"
                >
                  <span className="block truncate">{item.name}</span>
                  <span className="block text-xs text-gray-400 truncate">{item.path}</span>
                  {'triggerWords' in item && Array.isArray(item.triggerWords) && (
                    <span className="text-xs text-blue-300">{item.triggerWords.join(', ')}</span>
                  )}
                </button>
              ))}
          </div>
          <label className="block text-sm">
            Local path or Hugging Face repo/file
            <input
              value={path}
              onChange={e => setPath(e.target.value)}
              className="block w-full rounded bg-gray-800 p-2 mt-1"
            />
          </label>
          <button
            type="button"
            disabled={!path.trim()}
            onClick={() => choose({ path: path.trim(), name: path.trim().split(/[\\/]/).pop() || 'LoRA' })}
            className="operator-button"
          >
            Add path
          </button>
          <label className="block text-sm">
            Trigger words for upload
            <input
              value={triggerWords}
              onChange={e => setTriggerWords(e.target.value)}
              className="block w-full rounded bg-gray-800 p-2 mt-1"
            />
          </label>
          <label className="block text-sm">
            {uploading ? 'Uploading…' : 'Upload .safetensors'}
            <input
              className="block mt-2"
              type="file"
              accept=".safetensors"
              disabled={uploading}
              onChange={async e => {
                const file = e.target.files?.[0];
                if (!file) return;
                setUploading(true);
                setError('');
                try {
                  await uploadLoraFile(file, { triggerWords });
                  await refresh();
                } catch {
                  setError('LoRA upload failed.');
                } finally {
                  setUploading(false);
                }
              }}
            />
          </label>
        </DialogPanel>
      </div>
    </Dialog>
  );
}
