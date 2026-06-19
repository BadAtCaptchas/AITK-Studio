'use client';

import { Loader2, RotateCcw } from 'lucide-react';
import { Modal } from '@/components/Modal';
import { AUTO_BOX_PROVIDERS } from './constants';
import type { RecaptionLoadStatus, RecaptionModelOption, RecaptionOutputFormat, RecaptionProvider } from './recaption';

type RecaptionSettingsModalProps = {
  isOpen: boolean;
  provider: RecaptionProvider;
  outputFormat: RecaptionOutputFormat;
  model: string;
  modelOptions: RecaptionModelOption[];
  maxNewTokens: number;
  prompt: string;
  systemPrompt: string;
  rootPrompt: string;
  rootPromptStatus: RecaptionLoadStatus;
  remoteWorkerId: string;
  remoteWorkerOptions: RecaptionModelOption[];
  remoteModelOptions: RecaptionModelOption[];
  remoteModelStatus: RecaptionLoadStatus;
  remoteModelError: string;
  message: string;
  isRecaptioning: boolean;
  canQueueSelectedRecaption: boolean;
  selectedRecaptionIsRunning: boolean;
  selectedRecaptionIsQueued: boolean;
  hasPendingRecaptions: boolean;
  onClose: () => void;
  onSubmit: () => void;
  onProviderChange: (value: string) => void;
  onOutputFormatChange: (value: RecaptionOutputFormat) => void;
  onModelChange: (value: string) => void;
  onMaxNewTokensChange: (value: number) => void;
  onPromptChange: (value: string) => void;
  onSystemPromptChange: (value: string) => void;
  onUseRootPrompt: () => void;
  onRemoteWorkerChange: (value: string) => void;
  onLoadRemoteModels: () => void;
};

export function RecaptionSettingsModal({
  isOpen,
  provider,
  outputFormat,
  model,
  modelOptions,
  maxNewTokens,
  prompt,
  systemPrompt,
  rootPrompt,
  rootPromptStatus,
  remoteWorkerId,
  remoteWorkerOptions,
  remoteModelOptions,
  remoteModelStatus,
  remoteModelError,
  message,
  isRecaptioning,
  canQueueSelectedRecaption,
  selectedRecaptionIsRunning,
  selectedRecaptionIsQueued,
  hasPendingRecaptions,
  onClose,
  onSubmit,
  onProviderChange,
  onOutputFormatChange,
  onModelChange,
  onMaxNewTokensChange,
  onPromptChange,
  onSystemPromptChange,
  onUseRootPrompt,
  onRemoteWorkerChange,
  onLoadRemoteModels,
}: RecaptionSettingsModalProps) {
  const submitLabel = selectedRecaptionIsRunning
    ? 'Recaptioning'
    : selectedRecaptionIsQueued
      ? 'Queued'
      : hasPendingRecaptions
        ? 'Add to Queue'
        : 'Recaption';

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Recaption Image" size="lg" closeOnOverlayClick>
      <form
        className="space-y-4 text-gray-200"
        onSubmit={event => {
          event.preventDefault();
          onSubmit();
        }}
      >
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-gray-400">Provider</span>
            <select
              value={provider}
              onChange={event => onProviderChange(event.target.value)}
              disabled={isRecaptioning}
              className="h-10 w-full rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
            >
              {AUTO_BOX_PROVIDERS.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-gray-400">Output</span>
            <select
              value={outputFormat}
              onChange={event => onOutputFormatChange(event.target.value as RecaptionOutputFormat)}
              disabled={isRecaptioning}
              className="h-10 w-full rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
            >
              <option value="text">Text caption</option>
              <option value="ideogram_json">Ideogram JSON</option>
            </select>
          </label>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-gray-400">Model</span>
            {provider === 'remote_ollama' ? (
              <select
                value={model}
                onChange={event => onModelChange(event.target.value)}
                disabled={isRecaptioning || remoteModelStatus === 'loading' || remoteModelOptions.length === 0}
                className="h-10 w-full rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-600 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {remoteModelStatus === 'loading' && <option value="">Loading models...</option>}
                {remoteModelStatus !== 'loading' && remoteModelOptions.length === 0 && (
                  <option value="">No server models loaded</option>
                )}
                {remoteModelOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            ) : (
              <>
                <input
                  list="recaption-model-options"
                  value={model}
                  onChange={event => onModelChange(event.target.value)}
                  disabled={isRecaptioning}
                  className="h-10 w-full rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
                />
                <datalist id="recaption-model-options">
                  {modelOptions.map(option => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </datalist>
              </>
            )}
          </label>
          <label className="block text-sm">
            <span className="mb-1 block text-xs font-medium text-gray-400">Max tokens</span>
            <input
              type="number"
              min={1}
              value={maxNewTokens}
              onChange={event => onMaxNewTokensChange(Math.max(1, Number(event.target.value) || 1))}
              disabled={isRecaptioning}
              className="h-10 w-full rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
            />
          </label>
        </div>

        {provider === 'remote_ollama' && (
          <div className="space-y-2">
            <div className="grid grid-cols-[1fr_auto] gap-2">
              <label className="block text-sm">
                <span className="mb-1 block text-xs font-medium text-gray-400">Remote Ollama</span>
                <select
                  value={remoteWorkerId}
                  onChange={event => onRemoteWorkerChange(event.target.value)}
                  disabled={isRecaptioning || remoteWorkerOptions.length === 0}
                  className="h-10 w-full rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
                >
                  {remoteWorkerOptions.length === 0 && <option value="">No enabled workers</option>}
                  {remoteWorkerOptions.map(worker => (
                    <option key={worker.value} value={worker.value}>
                      {worker.label}
                    </option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                disabled={isRecaptioning || !remoteWorkerId || remoteModelStatus === 'loading'}
                onClick={onLoadRemoteModels}
                className="mt-5 inline-flex h-10 items-center gap-2 rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-200 hover:bg-gray-900 disabled:cursor-not-allowed disabled:opacity-45"
              >
                {remoteModelStatus === 'loading' ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <RotateCcw className="h-4 w-4" />
                )}
                Models
              </button>
            </div>
            {remoteModelStatus === 'success' && (
              <div className="text-xs text-gray-500">
                {remoteModelOptions.length.toLocaleString()} model
                {remoteModelOptions.length === 1 ? '' : 's'} loaded.
              </div>
            )}
            {remoteModelStatus === 'error' && <div className="text-xs text-red-400">{remoteModelError}</div>}
          </div>
        )}

        <label className="block text-sm">
          <span className="mb-1 block text-xs font-medium text-gray-400">Prompt</span>
          <textarea
            value={prompt}
            onChange={event => onPromptChange(event.target.value)}
            disabled={isRecaptioning}
            rows={6}
            className="w-full resize-none rounded-md border border-gray-800 bg-gray-950 p-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
          />
        </label>

        <label className="block text-sm">
          <span className="mb-1 flex items-center justify-between gap-3 text-xs font-medium text-gray-400">
            <span>System prompt</span>
            {rootPrompt && (
              <button
                type="button"
                disabled={isRecaptioning}
                onClick={onUseRootPrompt}
                className="text-cyan-300 hover:text-cyan-200 disabled:opacity-45"
              >
                Use ROOT_CAPTION.txt
              </button>
            )}
          </span>
          <textarea
            value={systemPrompt}
            onChange={event => onSystemPromptChange(event.target.value)}
            disabled={isRecaptioning}
            rows={3}
            className="w-full resize-none rounded-md border border-gray-800 bg-gray-950 p-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
          />
        </label>
        {rootPromptStatus === 'loading' && <div className="text-xs text-gray-500">Loading ROOT_CAPTION.txt</div>}
        {rootPromptStatus === 'error' && <div className="text-xs text-red-400">Could not load ROOT_CAPTION.txt.</div>}

        {message && <div className="text-sm text-gray-400">{message}</div>}

        <div className="flex justify-end gap-3">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600"
          >
            {isRecaptioning ? 'Close' : 'Cancel'}
          </button>
          <button
            type="submit"
            disabled={!canQueueSelectedRecaption}
            className="rounded-md bg-cyan-600 px-4 py-2 font-medium text-white hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {submitLabel}
          </button>
        </div>
      </form>
    </Modal>
  );
}
