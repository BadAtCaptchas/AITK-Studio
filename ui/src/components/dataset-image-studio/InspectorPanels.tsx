'use client';

import classNames from 'classnames';
import { Button } from '@headlessui/react';
import { CheckCircle2, ChevronDown, FileJson2, Loader2, Pipette, Save, SlidersHorizontal, Trash2, WandSparkles } from 'lucide-react';
import { rectToBox, type IdeogramElementType, type NormalizedBox } from '@/utils/ideogramCaption';
import { AUTO_BOX_PROVIDERS, BOX_COLORS, OLLAMA_VISION_MODELS, OPENROUTER_BOX_MODELS } from './constants';
import { SegmentedButton } from './StudioControls';
import type { BoxRect, CaptionStatus, CaptionTab, ImageSize } from './types';
import { normalizeHexColor } from './utils';

export function ObjectDetailsPanel({
  canAnnotate,
  isCaptionLoaded,
  canConvertDataset,
  isPlainTextItem,
  selectedImageSize,
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
  selectedElement,
  selectedElementIndex,
  selectedLayerColor,
  selectedRect,
  selectedPalette,
  activePaletteSamplerIndex,
  layerCaptionStatus,
  selectedLayerIsCaptioning,
  canCaptionSelectedLayer,
  layerCaptionDisabledReason,
  onConvertDatasetToJson,
  onGenerateAutoBoxes,
  onAutoBoxProviderChange,
  onAutoBoxModelChange,
  onRemoteWorkerChange,
  onAutoBoxRefineChange,
  onSelectedFieldChange,
  onSelectedTypeChange,
  onChangeBox,
  onSelectedPaletteChange,
  onStartPaletteSample,
  onCancelPaletteSample,
  onCaptionSelectedLayer,
}: {
  canAnnotate: boolean;
  isCaptionLoaded: boolean;
  canConvertDataset: boolean;
  isPlainTextItem: boolean;
  selectedImageSize: ImageSize | null;
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
  selectedElement: any | null;
  selectedElementIndex: number | null;
  selectedLayerColor: string;
  selectedRect: BoxRect | null;
  selectedPalette: string[];
  activePaletteSamplerIndex: number | null;
  layerCaptionStatus: string;
  selectedLayerIsCaptioning: boolean;
  canCaptionSelectedLayer: boolean;
  layerCaptionDisabledReason: string;
  onConvertDatasetToJson?: () => void;
  onGenerateAutoBoxes: () => void;
  onAutoBoxProviderChange: (value: string) => void;
  onAutoBoxModelChange: (value: string) => void;
  onRemoteWorkerChange: (value: string) => void;
  onAutoBoxRefineChange: (value: boolean) => void;
  onSelectedFieldChange: (field: 'desc' | 'text', value: string) => void;
  onSelectedTypeChange: (type: IdeogramElementType) => void;
  onChangeBox: (elementIndex: number, box: NormalizedBox) => void;
  onSelectedPaletteChange: (colors: string[]) => void;
  onStartPaletteSample: (index: number) => void;
  onCancelPaletteSample: () => void;
  onCaptionSelectedLayer: () => void;
}) {
  const autoBoxModelOptions = autoBoxProvider === 'openrouter' ? OPENROUTER_BOX_MODELS : OLLAMA_VISION_MODELS;
  const autoBoxModelListId = `auto-box-models-${autoBoxProvider}`;

  return (
    <section className={classNames('overflow-hidden rounded-md border border-gray-800 bg-gray-950/80', canAnnotate ? 'mt-3' : '')}>
      <div className="flex h-12 items-center justify-between border-b border-gray-800 px-4">
        <h3 className="text-sm font-semibold text-gray-100">Object Details</h3>
        <ChevronDown className="h-4 w-4 text-gray-500" />
      </div>
      <div className="space-y-4 p-4">
        {canAnnotate && (
          <div className="space-y-3 rounded-md border border-cyan-500/25 bg-cyan-950/10 p-3">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2 text-sm font-semibold text-cyan-100">
                  <WandSparkles className="h-4 w-4 text-cyan-300" />
                  Auto Boxes
                </div>
                <div className="mt-1 truncate text-xs text-gray-500">
                  {selectedImageSize ? `${selectedImageSize.width} x ${selectedImageSize.height}` : 'Image size pending'}
                </div>
              </div>
              <button
                type="button"
                disabled={!canGenerateAutoBoxes}
                onClick={onGenerateAutoBoxes}
                title={autoBoxDisabledReason || `Generate boxes with ${autoBoxProviderLabel}`}
                className="inline-flex h-9 flex-shrink-0 items-center gap-2 rounded-md border border-cyan-500/40 bg-cyan-500/15 px-3 text-sm font-medium text-cyan-100 hover:bg-cyan-500/25 disabled:cursor-not-allowed disabled:border-gray-700 disabled:bg-gray-900 disabled:text-gray-500"
              >
                {isGeneratingBoxes ? <Loader2 className="h-4 w-4 animate-spin" /> : <WandSparkles className="h-4 w-4" />}
                Generate
              </button>
            </div>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              <label>
                <span className="mb-1 block text-xs text-gray-500">Provider</span>
                <select
                  value={autoBoxProvider}
                  onChange={event => onAutoBoxProviderChange(event.target.value)}
                  className="h-9 w-full rounded-md border border-gray-800 bg-gray-900 px-2 text-sm text-gray-100 outline-none focus:border-cyan-500"
                >
                  {AUTO_BOX_PROVIDERS.map(provider => (
                    <option key={provider.value} value={provider.value}>
                      {provider.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span className="mb-1 block text-xs text-gray-500">Model</span>
                <input
                  value={autoBoxModel}
                  list={autoBoxModelListId}
                  onChange={event => onAutoBoxModelChange(event.target.value)}
                  className="h-9 w-full rounded-md border border-gray-800 bg-gray-900 px-2 text-sm text-gray-100 outline-none focus:border-cyan-500"
                />
                <datalist id={autoBoxModelListId}>
                  {autoBoxModelOptions.map(model => (
                    <option key={model.value} value={model.value}>
                      {model.label}
                    </option>
                  ))}
                </datalist>
              </label>
              {autoBoxProvider === 'remote_ollama' && (
                <label>
                  <span className="mb-1 block text-xs text-gray-500">Remote Ollama</span>
                  <select
                    value={remoteWorkerId}
                    onChange={event => onRemoteWorkerChange(event.target.value)}
                    className="h-9 w-full rounded-md border border-gray-800 bg-gray-900 px-2 text-sm text-gray-100 outline-none focus:border-cyan-500"
                  >
                    {remoteWorkerOptions.map(worker => (
                      <option key={worker.value} value={worker.value}>
                        {worker.label}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <label className="flex h-9 items-center gap-2 self-end rounded-md border border-gray-800 bg-gray-900 px-3 text-sm text-gray-300">
                <input
                  type="checkbox"
                  checked={autoBoxRefine}
                  onChange={event => onAutoBoxRefineChange(event.target.checked)}
                  className="h-4 w-4"
                />
                Refine pass
              </label>
            </div>
            {autoBoxProvider !== 'openrouter' && (
              <div className="text-xs text-amber-200">
                Use vision models only. Smaller models may produce invalid JSON, miss NSFW details, or create weak boxes.
              </div>
            )}
            {(autoBoxMessage || autoBoxDisabledReason) && (
              <div className="text-xs text-gray-400">{autoBoxMessage || autoBoxDisabledReason}</div>
            )}
          </div>
        )}
        {!isCaptionLoaded ? (
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading caption
          </div>
        ) : isPlainTextItem ? (
          <div className="space-y-3 rounded-md border border-gray-800 bg-gray-900/60 p-3 text-sm text-gray-300">
            <div className="flex items-center gap-2 text-gray-100">
              <FileJson2 className="h-4 w-4 text-blue-300" />
              Text file editing
            </div>
            <p className="text-gray-400">
              This is a regular text caption file. Caption editing and basic bulk actions are available; JSON boxes,
              colors, and layer tools remain disabled.
            </p>
          </div>
        ) : !canAnnotate ? (
          <div className="space-y-3 rounded-md border border-gray-800 bg-gray-900/60 p-3 text-sm text-gray-300">
            <div className="flex items-center gap-2 text-gray-100">
              <FileJson2 className="h-4 w-4 text-amber-300" />
              JSON boxes unavailable
            </div>
            <p className="text-gray-400">
              Box and text-region tools are enabled for Ideogram JSON captions. This item can still be captioned normally.
            </p>
            <button
              type="button"
              disabled={!canConvertDataset}
              onClick={onConvertDatasetToJson}
              className="inline-flex h-9 w-full items-center justify-center gap-2 rounded-md border border-blue-500/50 bg-blue-600/20 text-sm font-medium text-blue-100 hover:bg-blue-600/30 disabled:cursor-not-allowed disabled:border-gray-700 disabled:bg-gray-900 disabled:text-gray-500"
            >
              <WandSparkles className="h-4 w-4" />
              Convert dataset to JSON
            </button>
          </div>
        ) : selectedElement ? (
          <>
            <div className="grid grid-cols-[1fr_auto] gap-3">
              <label className="min-w-0">
                <span className="mb-1 block text-xs text-gray-400">Label</span>
                <div className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 flex-shrink-0 rounded-full" style={{ backgroundColor: selectedLayerColor }} />
                  <input
                    value={selectedElement.type === 'text' ? selectedElement.text || '' : selectedElement.desc || ''}
                    onChange={event => onSelectedFieldChange(selectedElement.type === 'text' ? 'text' : 'desc', event.target.value)}
                    className="h-10 min-w-0 flex-1 rounded-md border border-gray-800 bg-gray-900 px-3 text-sm text-gray-100 outline-none focus:border-blue-500"
                  />
                </div>
              </label>
              <div className="pt-6 text-xs text-gray-500">ID: {String(selectedElementIndex ?? 0).padStart(3, '0')}</div>
            </div>
            <div>
              <span className="mb-2 block text-xs text-gray-400">Type</span>
              <div className="inline-flex overflow-hidden rounded-md border border-gray-800">
                <SegmentedButton active={selectedElement.type !== 'text'} onClick={() => onSelectedTypeChange('obj')}>
                  Object
                </SegmentedButton>
                <SegmentedButton active={selectedElement.type === 'text'} onClick={() => onSelectedTypeChange('text')}>
                  Text
                </SegmentedButton>
              </div>
            </div>
            <div className="flex min-h-9 items-center justify-between gap-3">
              <div className="min-w-0 truncate text-xs text-gray-500">{layerCaptionStatus}</div>
              <button
                type="button"
                disabled={!canCaptionSelectedLayer}
                onClick={onCaptionSelectedLayer}
                title={selectedLayerIsCaptioning ? 'Captioning layer' : layerCaptionDisabledReason || `Caption selected layer with ${autoBoxProviderLabel}`}
                className="inline-flex h-9 flex-shrink-0 items-center gap-2 overflow-hidden whitespace-nowrap rounded-md border border-cyan-500/40 bg-cyan-500/15 px-3 text-sm font-medium text-cyan-100 hover:bg-cyan-500/25 disabled:cursor-not-allowed disabled:border-gray-700 disabled:bg-gray-900 disabled:text-gray-500"
              >
                {selectedLayerIsCaptioning ? <Loader2 className="h-4 w-4 animate-spin" /> : <WandSparkles className="h-4 w-4" />}
                Caption Layer
              </button>
            </div>
            {selectedRect && (
              <div>
                <span className="mb-2 block text-xs text-gray-400">Bounding Box (x, y, w, h)</span>
                <div className="grid grid-cols-4 gap-2">
                  {(['x', 'y', 'w', 'h'] as const).map(field => (
                    <label key={field} className="flex h-9 items-center gap-1 rounded-md border border-gray-800 bg-gray-900 px-2">
                      <span className="text-xs text-gray-500">{field}</span>
                      <input
                        type="number"
                        min={0}
                        max={1000}
                        value={selectedRect[field]}
                        onChange={event => {
                          const nextRect = { ...selectedRect, [field]: Number(event.target.value) };
                          if (selectedElementIndex != null) onChangeBox(selectedElementIndex, rectToBox(nextRect));
                        }}
                        className="min-w-0 flex-1 bg-transparent text-right text-sm text-gray-100 outline-none"
                      />
                    </label>
                  ))}
                </div>
              </div>
            )}
            {selectedElement.type === 'text' && (
              <label>
                <span className="mb-1 block text-xs text-gray-400">Visible Text</span>
                <textarea
                  value={selectedElement.text || ''}
                  rows={2}
                  onChange={event => onSelectedFieldChange('text', event.target.value)}
                  className="h-16 w-full resize-none rounded-md border border-gray-800 bg-gray-900 p-3 text-sm text-gray-100 outline-none focus:border-blue-500"
                />
              </label>
            )}
            <label>
              <span className="mb-1 block text-xs text-gray-400">Object Description</span>
              <textarea
                value={selectedElement.desc || ''}
                rows={4}
                onChange={event => onSelectedFieldChange('desc', event.target.value)}
                className="h-28 w-full resize-none rounded-md border border-gray-800 bg-gray-900 p-3 text-sm text-gray-100 outline-none focus:border-blue-500"
              />
            </label>
            <div>
              <span className="mb-2 block text-xs text-gray-400">Color Palette</span>
              <div className="flex flex-wrap gap-2">
                {selectedPalette.map((rawColor, index) => {
                  const color = normalizeHexColor(rawColor) || BOX_COLORS[index % BOX_COLORS.length];
                  const sampling = activePaletteSamplerIndex === index;
                  return (
                    <div
                      key={`${rawColor}-${index}`}
                      className={classNames('inline-flex h-9 items-center gap-1 rounded-md border bg-gray-900 px-1.5', {
                        'border-cyan-400 shadow-[0_0_0_1px_rgba(34,211,238,0.25)]': sampling,
                        'border-gray-800': !sampling,
                      })}
                    >
                      <span className="h-6 w-6 flex-shrink-0 rounded border border-gray-700" style={{ backgroundColor: color }} title={color} />
                      <span className="w-[4.5rem] font-mono text-[11px] text-gray-400">{color}</span>
                      <button
                        type="button"
                        title={sampling ? 'Cancel color sample' : `Sample ${color} from image`}
                        onClick={() => (sampling ? onCancelPaletteSample() : onStartPaletteSample(index))}
                        className={classNames('flex h-7 w-7 items-center justify-center rounded hover:bg-gray-800', {
                          'bg-cyan-500/20 text-cyan-100': sampling,
                          'text-gray-400 hover:text-gray-100': !sampling,
                        })}
                      >
                        <Pipette className="h-3.5 w-3.5" />
                      </button>
                      <button
                        type="button"
                        title={`Remove ${color}`}
                        onClick={() => onSelectedPaletteChange(selectedPalette.filter((_, i) => i !== index))}
                        className="flex h-7 w-7 items-center justify-center rounded text-gray-500 hover:bg-gray-800 hover:text-rose-300"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  );
                })}
                <button
                  type="button"
                  onClick={() => onSelectedPaletteChange([...selectedPalette, BOX_COLORS[selectedPalette.length % BOX_COLORS.length]])}
                  className="h-8 rounded-md border border-gray-700 px-3 text-xs text-gray-300 hover:bg-gray-800"
                >
                  Add
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="rounded-md border border-gray-800 bg-gray-900/60 p-3 text-sm text-gray-400">
            Select a layer, or use Box/Text to draw a new region.
          </div>
        )}
      </div>
    </section>
  );
}

export function CaptionEditorPanel({
  captionTab,
  captionStatus,
  captionText,
  highLevelDescription,
  isIdeogram,
  isPlainTextItem,
  isAutoCaptioning,
  isCaptionLoaded,
  isDirty,
  isSaving,
  isRecaptioning,
  canRecaption,
  isSelectedRecaptionQueued,
  hasActiveRecaptions,
  hasQueuedRecaptions,
  recaptionFeedback,
  onCaptionTabChange,
  onCaptionDescriptionChange,
  onCaptionTextChange,
  onRecaption,
  onRecaptionSettings,
  onSave,
}: {
  captionTab: CaptionTab;
  captionStatus: CaptionStatus;
  captionText: string;
  highLevelDescription: string;
  isIdeogram: boolean;
  isPlainTextItem: boolean;
  isAutoCaptioning: boolean;
  isCaptionLoaded: boolean;
  isDirty: boolean;
  isSaving: boolean;
  isRecaptioning?: boolean;
  canRecaption?: boolean;
  isSelectedRecaptionQueued?: boolean;
  hasActiveRecaptions?: boolean;
  hasQueuedRecaptions?: boolean;
  recaptionFeedback?: string;
  onCaptionTabChange: (tab: CaptionTab) => void;
  onCaptionDescriptionChange: (value: string) => void;
  onCaptionTextChange: (value: string) => void;
  onRecaption?: () => void;
  onRecaptionSettings?: () => void;
  onSave: () => void;
}) {
  const recaptionButtonLabel = isRecaptioning
    ? 'Recaptioning'
    : isSelectedRecaptionQueued
      ? 'Queued'
      : hasActiveRecaptions || hasQueuedRecaptions
        ? 'Add to Queue'
        : 'Recaption';

  return (
    <section className="mt-3 overflow-hidden rounded-md border border-gray-800 bg-gray-950/80">
      <div className="flex h-12 items-center border-b border-gray-800 px-4">
        <button
          type="button"
          onClick={() => onCaptionTabChange('caption')}
          className={classNames('mr-5 h-12 border-b-2 text-sm font-semibold', {
            'border-blue-500 text-gray-100': captionTab === 'caption',
            'border-transparent text-gray-400 hover:text-gray-200': captionTab !== 'caption',
          })}
        >
          Caption
        </button>
        {!isPlainTextItem && (
          <button
            type="button"
            onClick={() => onCaptionTabChange('json')}
            className={classNames('h-12 border-b-2 text-sm font-semibold', {
              'border-blue-500 text-gray-100': captionTab === 'json',
              'border-transparent text-gray-400 hover:text-gray-200': captionTab !== 'json',
            })}
          >
            JSON
          </button>
        )}
        <div className="flex-1" />
        <span className={classNames('h-2 w-2 rounded-full', captionStatus.dot)} title={captionStatus.title} />
      </div>
      <div className="p-4">
        {captionTab === 'caption' ? (
          <label>
            <textarea
              value={highLevelDescription}
              rows={6}
              readOnly={isAutoCaptioning || !isCaptionLoaded}
              onChange={event => onCaptionDescriptionChange(event.target.value)}
              className="h-36 w-full resize-none rounded-md border border-gray-800 bg-gray-900 p-3 text-sm text-gray-100 outline-none focus:border-blue-500 disabled:opacity-50"
            />
            <span className="mt-1 flex items-center justify-between text-xs text-gray-500">
              <span>{highLevelDescription.trim() ? highLevelDescription.trim().split(/\s+/).length : 0} words</span>
              <span>
                {highLevelDescription.length} / {isIdeogram ? 2000 : 4000}
              </span>
            </span>
          </label>
        ) : (
          <label>
            <textarea
              value={captionText}
              rows={10}
              readOnly={isAutoCaptioning || !isCaptionLoaded}
              onChange={event => onCaptionTextChange(event.target.value)}
              className="h-64 w-full resize-none rounded-md border border-gray-800 bg-gray-900 p-3 font-mono text-xs leading-relaxed text-gray-100 outline-none focus:border-blue-500 disabled:opacity-50"
            />
          </label>
        )}
        <div className="mt-3 flex min-h-9 items-center justify-between gap-3 overflow-hidden">
          <div className="flex min-w-0 items-center gap-2 overflow-hidden whitespace-nowrap text-xs leading-none text-gray-500">
            {isDirty ? (
              <>
                <span className="h-2 w-2 rounded-full bg-blue-400" />
                Unsaved changes (Ctrl+S to save)
              </>
            ) : (
              <>
                <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                Saved
              </>
            )}
          </div>
          <div className="flex flex-shrink-0 items-center gap-2">
            <Button
              className="inline-flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-md border border-gray-700 bg-gray-900 text-gray-300 hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-40"
              disabled={!canRecaption || isAutoCaptioning || isRecaptioning || isSaving}
              onClick={onRecaptionSettings}
              title="Recaption settings"
              aria-label="Recaption settings"
            >
              <SlidersHorizontal className="h-4 w-4" />
            </Button>
            <Button
              className="inline-flex h-9 flex-shrink-0 items-center gap-2 overflow-hidden whitespace-nowrap rounded-md border border-cyan-500/40 bg-cyan-600/20 px-3 text-sm font-medium leading-none text-cyan-100 hover:bg-cyan-600/30 disabled:cursor-not-allowed disabled:opacity-40"
              disabled={!canRecaption || isAutoCaptioning || isSaving || isRecaptioning || isSelectedRecaptionQueued}
              onClick={onRecaption}
            >
              {isRecaptioning ? <Loader2 className="h-4 w-4 animate-spin" /> : <WandSparkles className="h-4 w-4" />}
              {recaptionButtonLabel}
            </Button>
            <Button
              className="inline-flex h-9 flex-shrink-0 items-center gap-2 overflow-hidden whitespace-nowrap rounded-md border border-emerald-500/40 bg-emerald-600/20 px-3 text-sm font-medium leading-none text-emerald-100 hover:bg-emerald-600/30 disabled:cursor-not-allowed disabled:opacity-40"
              disabled={!isDirty || !isCaptionLoaded || isSaving}
              onClick={onSave}
            >
              {isSaving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
              Save
            </Button>
          </div>
        </div>
        {recaptionFeedback && (
          <div
            className="mt-2 max-h-20 overflow-y-auto break-words rounded-md border border-cyan-500/10 bg-cyan-950/20 px-2 py-1.5 text-xs leading-relaxed text-cyan-200/80"
            title={recaptionFeedback}
          >
            {recaptionFeedback}
          </div>
        )}
      </div>
    </section>
  );
}
