'use client';

import classNames from 'classnames';
import { ChevronRight, ChevronUp, Loader2, Lock, Unlock, WandSparkles } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { rectToBox, type IdeogramElementType, type NormalizedBox } from '@/utils/ideogramCaption';
import { AUTO_BOX_PROVIDERS, OLLAMA_VISION_MODELS, OPENROUTER_BOX_MODELS } from './constants';
import type { BoxRect } from './types';

type ElementRecord = Record<string, unknown>;

function asElementRecord(value: unknown): ElementRecord | null {
  return typeof value === 'object' && value !== null ? (value as ElementRecord) : null;
}

function stringValue(value: unknown) {
  return typeof value === 'string' ? value : '';
}

type StandaloneBoxDetailsPanelProps = {
  selectedElement: unknown | null;
  selectedElementIndex: number | null;
  selectedLayerColor: string;
  selectedRect: BoxRect | null;
  selectedLocked: boolean;
  mutationDisabledReason: string;
  autoBoxesOpen: boolean;
  selectedImageSizeLabel: string;
  canGenerateAutoBoxes: boolean;
  autoBoxDisabledReason: string;
  autoBoxProvider: string;
  autoBoxProviderLabel: string;
  autoBoxModel: string;
  remoteWorkerId: string;
  remoteWorkerOptions: Array<{ value: string; label: string }>;
  autoBoxRefine: boolean;
  isGeneratingBoxes: boolean;
  autoBoxMessage: string;
  onSelectedFieldChange: (field: 'desc' | 'text', value: string) => void;
  onSelectedTypeChange: (type: IdeogramElementType) => void;
  onChangeBox: (elementIndex: number, box: NormalizedBox) => void;
  onToggleLocked: (elementIndex: number) => void;
  onAutoBoxesOpenChange: (open: boolean) => void;
  onGenerateAutoBoxes: () => void;
  onAutoBoxProviderChange: (value: string) => void;
  onAutoBoxModelChange: (value: string) => void;
  onRemoteWorkerChange: (value: string) => void;
  onAutoBoxRefineChange: (value: boolean) => void;
};

export function StandaloneBoxDetailsPanel({
  selectedElement,
  selectedElementIndex,
  selectedLayerColor,
  selectedRect,
  selectedLocked,
  mutationDisabledReason,
  autoBoxesOpen,
  selectedImageSizeLabel,
  canGenerateAutoBoxes,
  autoBoxDisabledReason,
  autoBoxProvider,
  autoBoxProviderLabel,
  autoBoxModel,
  remoteWorkerId,
  remoteWorkerOptions,
  autoBoxRefine,
  isGeneratingBoxes,
  autoBoxMessage,
  onSelectedFieldChange,
  onSelectedTypeChange,
  onChangeBox,
  onToggleLocked,
  onAutoBoxesOpenChange,
  onGenerateAutoBoxes,
  onAutoBoxProviderChange,
  onAutoBoxModelChange,
  onRemoteWorkerChange,
  onAutoBoxRefineChange,
}: StandaloneBoxDetailsPanelProps) {
  const element = asElementRecord(selectedElement);
  const isText = element?.type === 'text';
  const description = stringValue(element?.desc);
  const label = isText
    ? stringValue(element?.text) || stringValue(element?.label) || description
    : stringValue(element?.label) || description;
  const [labelDraft, setLabelDraft] = useState(label);
  const [descriptionDraft, setDescriptionDraft] = useState(description);
  const [rectDraft, setRectDraft] = useState<BoxRect | null>(selectedRect);
  const mutationDisabled = Boolean(mutationDisabledReason);
  const editDisabled = mutationDisabled || selectedLocked;
  const autoBoxModels = autoBoxProvider === 'openrouter' ? OPENROUTER_BOX_MODELS : OLLAMA_VISION_MODELS;
  const modelListId = `standalone-auto-box-models-${autoBoxProvider}`;

  useEffect(() => {
    setLabelDraft(label);
    setDescriptionDraft(description);
  }, [description, label, selectedElementIndex]);

  useEffect(() => {
    setRectDraft(selectedRect);
  }, [selectedElementIndex, selectedRect]);

  const selectedId = useMemo(
    () => (selectedElementIndex == null ? '' : String(selectedElementIndex + 1).padStart(2, '0')),
    [selectedElementIndex],
  );

  const commitLabel = () => {
    if (!element || !isText || editDisabled || labelDraft === label) return;
    onSelectedFieldChange('text', labelDraft);
  };

  const commitDescription = () => {
    if (!element || editDisabled || descriptionDraft === description) return;
    onSelectedFieldChange('desc', descriptionDraft);
  };

  const commitRect = () => {
    if (selectedElementIndex == null || !rectDraft || editDisabled) return;
    if (
      selectedRect &&
      rectDraft.x === selectedRect.x &&
      rectDraft.y === selectedRect.y &&
      rectDraft.w === selectedRect.w &&
      rectDraft.h === selectedRect.h
    ) {
      return;
    }
    onChangeBox(selectedElementIndex, rectToBox(rectDraft));
  };

  return (
    <section className="flex min-h-0 flex-1 flex-col border-t border-gray-800 bg-gray-950">
      <div className="flex h-9 flex-none items-center px-3 text-xs font-medium text-gray-400">Box properties</div>

      <div className="operator-scrollbar-none flex min-h-0 flex-1 flex-col overflow-y-auto">
        {!element || selectedElementIndex == null ? (
          <div className="mx-3 rounded-[4px] border border-dashed border-gray-800 px-3 py-4 text-xs leading-5 text-gray-500">
            Select a box to edit its label, type, geometry, and description.
          </div>
        ) : (
          <div className="space-y-1 px-3 pb-2">
            <div className="grid grid-cols-[minmax(0,1fr)_144px] items-end gap-3">
              <label className="min-w-0">
                <span className="mb-1 block text-[11px] text-gray-500">Label</span>
                <span className="flex h-8 items-center gap-2 rounded-[4px] border border-gray-800 bg-gray-950 px-2.5 focus-within:border-brand-600">
                  <span className="h-2.5 w-2.5 flex-none rounded-full" style={{ backgroundColor: selectedLayerColor }} />
                  <input
                    value={labelDraft}
                    disabled={editDisabled}
                    readOnly={!isText}
                    aria-readonly={!isText}
                    title={!isText ? 'Object names are derived from their description.' : undefined}
                    onChange={event => {
                      if (isText) setLabelDraft(event.target.value);
                    }}
                    onBlur={commitLabel}
                    onKeyDown={event => {
                      if (event.key === 'Enter') event.currentTarget.blur();
                      if (event.key === 'Escape') setLabelDraft(label);
                    }}
                    aria-label="Box label"
                    className="min-w-0 flex-1 bg-transparent text-xs text-gray-100 outline-none disabled:opacity-50"
                  />
                </span>
              </label>
              <div className="grid h-8 grid-cols-2 overflow-hidden rounded-[4px] border border-gray-800 bg-gray-950">
                {([['obj', 'Object'], ['text', 'Text']] as const).map(([value, text]) => {
                  const active = value === (isText ? 'text' : 'obj');
                  return (
                    <button
                      key={value}
                      type="button"
                      disabled={editDisabled}
                      aria-pressed={active}
                      onClick={() => onSelectedTypeChange(value)}
                      className={classNames('text-xs transition-colors disabled:opacity-45', {
                        'bg-brand-900/70 text-brand-50': active,
                        'text-gray-400 hover:bg-white/5 hover:text-gray-100': !active,
                      })}
                    >
                      {text}
                    </button>
                  );
                })}
              </div>
            </div>

            {rectDraft ? (
              <div>
                <div className="mb-1 flex items-center justify-between text-[11px] text-gray-500">
                  <span>Bounding box</span>
                  <span className="tabular-nums">ID #{selectedId}</span>
                </div>
                <div className="grid grid-cols-[repeat(4,minmax(0,1fr))_36px] gap-2">
                  {(['x', 'y', 'w', 'h'] as const).map(field => (
                    <label
                      key={field}
                      className="flex h-8 min-w-0 items-center gap-1 rounded-[4px] border border-gray-800 bg-gray-950 px-2"
                    >
                      <span className="text-[10px] uppercase text-gray-600">{field}</span>
                      <input
                        type="number"
                        min={0}
                        max={1000}
                        value={rectDraft[field]}
                        disabled={editDisabled}
                        onChange={event => setRectDraft(current => current ? { ...current, [field]: Number(event.target.value) } : current)}
                        onBlur={commitRect}
                        onKeyDown={event => {
                          if (event.key === 'Enter') event.currentTarget.blur();
                          if (event.key === 'Escape') setRectDraft(selectedRect);
                        }}
                        aria-label={`Box ${field}`}
                        className="min-w-0 flex-1 bg-transparent text-right text-xs tabular-nums text-gray-100 outline-none disabled:opacity-50"
                      />
                    </label>
                  ))}
                  <button
                    type="button"
                    onClick={() => onToggleLocked(selectedElementIndex)}
                    disabled={mutationDisabled}
                    title={selectedLocked ? 'Unlock selected box' : 'Lock selected box'}
                    aria-label={selectedLocked ? 'Unlock selected box' : 'Lock selected box'}
                    className={classNames(
                      'flex h-8 w-9 items-center justify-center rounded-[4px] border border-gray-800 bg-gray-950 hover:text-white disabled:opacity-45',
                      selectedLocked ? 'text-amber-300' : 'text-gray-500',
                    )}
                  >
                    {selectedLocked ? <Lock className="h-4 w-4" /> : <Unlock className="h-4 w-4" />}
                  </button>
                </div>
              </div>
            ) : (
              <div className="rounded-[4px] border border-dashed border-gray-800 px-3 py-2 text-[11px] text-gray-500">
                No box yet. Choose Box or Text, then draw on the image.
              </div>
            )}

            <label className="block">
              <span className="mb-1 block text-[11px] text-gray-500">Description</span>
              <span className="relative block h-14">
                <textarea
                  value={descriptionDraft}
                  disabled={editDisabled}
                  rows={2}
                  maxLength={500}
                  onChange={event => setDescriptionDraft(event.target.value)}
                  onBlur={commitDescription}
                  onKeyDown={event => {
                    if (event.key === 'Escape') setDescriptionDraft(description);
                  }}
                  aria-label="Box description"
                  className="operator-scrollbar-none h-14 w-full resize-none rounded-[4px] border border-gray-800 bg-gray-950 px-2.5 pb-4 pt-1.5 text-xs leading-4 text-gray-100 outline-none focus:border-brand-600 disabled:opacity-50"
                />
                <span className="pointer-events-none absolute bottom-1.5 right-2.5 text-[10px] tabular-nums text-gray-600">
                  {descriptionDraft.length} / 500
                </span>
              </span>
            </label>

            {mutationDisabledReason ? (
              <div className="rounded-[4px] border border-amber-900/50 bg-amber-950/20 px-2 py-1.5 text-[10px] leading-4 text-amber-200/80">
                {mutationDisabledReason}
              </div>
            ) : null}
          </div>
        )}

        {autoBoxesOpen ? (
          <div className="order-first mx-3 mb-2 space-y-2 rounded-[4px] border border-brand-900/60 bg-brand-950/10 p-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-[11px] font-medium text-gray-300">Generate boxes</div>
                <div className="mt-0.5 text-[10px] text-gray-600">{selectedImageSizeLabel}</div>
              </div>
              <button
                type="button"
                disabled={!canGenerateAutoBoxes}
                onClick={onGenerateAutoBoxes}
                title={autoBoxDisabledReason || `Generate boxes with ${autoBoxProviderLabel}`}
                className="inline-flex h-8 items-center gap-1.5 rounded-[4px] border border-brand-600/50 bg-brand-500/15 px-3 text-[11px] font-medium text-brand-100 hover:bg-brand-500/25 disabled:cursor-not-allowed disabled:border-gray-700 disabled:bg-gray-900 disabled:text-gray-500"
              >
                {isGeneratingBoxes ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <WandSparkles className="h-3.5 w-3.5" />}
                Generate
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <label>
                <span className="mb-1 block text-[10px] text-gray-500">Provider</span>
                <select
                  value={autoBoxProvider}
                  onChange={event => onAutoBoxProviderChange(event.target.value)}
                  className="h-8 w-full rounded-[4px] border border-gray-800 bg-gray-950 px-2 text-[11px] text-gray-100 outline-none focus:border-brand-600"
                >
                  {AUTO_BOX_PROVIDERS.map(provider => (
                    <option key={provider.value} value={provider.value}>{provider.label}</option>
                  ))}
                </select>
              </label>
              <label>
                <span className="mb-1 block text-[10px] text-gray-500">Model</span>
                <input
                  value={autoBoxModel}
                  list={modelListId}
                  onChange={event => onAutoBoxModelChange(event.target.value)}
                  className="h-8 w-full rounded-[4px] border border-gray-800 bg-gray-950 px-2 text-[11px] text-gray-100 outline-none focus:border-brand-600"
                />
                <datalist id={modelListId}>
                  {autoBoxModels.map(model => <option key={model.value} value={model.value}>{model.label}</option>)}
                </datalist>
              </label>
              {autoBoxProvider === 'remote_ollama' ? (
                <label>
                  <span className="mb-1 block text-[10px] text-gray-500">Remote Ollama</span>
                  <select
                    value={remoteWorkerId}
                    onChange={event => onRemoteWorkerChange(event.target.value)}
                    className="h-8 w-full rounded-[4px] border border-gray-800 bg-gray-950 px-2 text-[11px] text-gray-100 outline-none focus:border-brand-600"
                  >
                    {remoteWorkerOptions.map(worker => <option key={worker.value} value={worker.value}>{worker.label}</option>)}
                  </select>
                </label>
              ) : null}
              <label className="flex h-8 items-center gap-2 self-end rounded-[4px] border border-gray-800 bg-gray-950 px-2 text-[11px] text-gray-300">
                <input
                  type="checkbox"
                  checked={autoBoxRefine}
                  onChange={event => onAutoBoxRefineChange(event.target.checked)}
                  className="h-3.5 w-3.5 accent-brand-400"
                />
                Refine pass
              </label>
            </div>
            {autoBoxProvider !== 'openrouter' ? (
              <div className="text-[10px] leading-4 text-amber-200/80">Use a vision model for reliable box JSON.</div>
            ) : null}
            {autoBoxMessage || autoBoxDisabledReason ? (
              <div className="text-[10px] leading-4 text-gray-500">{autoBoxMessage || autoBoxDisabledReason}</div>
            ) : null}
          </div>
        ) : null}
      </div>

      <button
        type="button"
        onClick={() => onAutoBoxesOpenChange(!autoBoxesOpen)}
        aria-expanded={autoBoxesOpen}
        className="mx-3 mb-2 flex h-12 flex-none items-center gap-2 rounded-[4px] border border-gray-800 bg-gray-950 px-3 text-left text-xs text-gray-200 hover:border-brand-800 hover:bg-brand-950/20"
      >
        <WandSparkles className="h-4 w-4 flex-none text-gray-400" />
        <span className="min-w-0 flex-1">
          <span className="block font-medium">Auto Boxes</span>
          <span className="block truncate text-[10px] text-gray-600">Generate suggested boxes for this image</span>
        </span>
        {autoBoxesOpen ? <ChevronUp className="h-4 w-4 text-gray-500" /> : <ChevronRight className="h-4 w-4 text-gray-500" />}
      </button>
    </section>
  );
}
