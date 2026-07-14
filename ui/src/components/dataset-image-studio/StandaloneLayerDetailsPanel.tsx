'use client';

import classNames from 'classnames';
import { ChevronRight, Loader2, MessageSquareText } from 'lucide-react';
import { rectToBox, type IdeogramElementType, type NormalizedBox } from '@/utils/ideogramCaption';
import type { BoxRect } from './types';

type LayerRecord = Record<string, unknown>;

function layerRecord(value: unknown): LayerRecord | null {
  return typeof value === 'object' && value !== null ? (value as LayerRecord) : null;
}

function layerString(value: unknown) {
  return typeof value === 'string' ? value : '';
}

export function StandaloneLayerDetailsPanel({
  selectedElement,
  selectedElementIndex,
  selectedLayerColor,
  selectedRect,
  layerCaptionStatus,
  selectedLayerIsCaptioning,
  canCaptionSelectedLayer,
  layerCaptionDisabledReason,
  onSelectedFieldChange,
  onSelectedTypeChange,
  onChangeBox,
  onCaptionSelectedLayer,
}: {
  selectedElement: unknown | null;
  selectedElementIndex: number | null;
  selectedLayerColor: string;
  selectedRect: BoxRect | null;
  layerCaptionStatus: string;
  selectedLayerIsCaptioning: boolean;
  canCaptionSelectedLayer: boolean;
  layerCaptionDisabledReason: string;
  onSelectedFieldChange: (field: 'desc' | 'text', value: string) => void;
  onSelectedTypeChange: (type: IdeogramElementType) => void;
  onChangeBox: (elementIndex: number, box: NormalizedBox) => void;
  onCaptionSelectedLayer: () => void;
}) {
  const element = layerRecord(selectedElement);
  const isText = element?.type === 'text';
  const label = isText ? layerString(element?.text) : layerString(element?.desc);
  const description = layerString(element?.desc);

  return (
    <section className="flex min-h-0 flex-1 flex-col border-t border-[#273039] bg-[#071019]">
      <div className="flex h-9 flex-none items-center px-3 text-xs font-medium text-gray-400">Layer properties</div>
      {!element || selectedElementIndex == null ? (
        <div className="mx-3 rounded-[4px] border border-dashed border-[#33414d] px-3 py-4 text-xs leading-5 text-gray-500">
          Select a layer to edit its label, type, geometry, and description.
        </div>
      ) : (
        <div className="operator-scrollbar-none min-h-0 flex-1 space-y-1.5 overflow-y-auto px-3 pb-2">
          <div className="grid grid-cols-[minmax(0,1fr)_144px] items-end gap-3">
            <label className="min-w-0">
              <span className="mb-1 block text-[11px] text-gray-500">Label</span>
              <span className="flex h-9 items-center gap-2 rounded-[4px] border border-[#33414d] bg-[#08121a] px-2.5 focus-within:border-cyan-600">
                <span className="h-2.5 w-2.5 flex-none rounded-full" style={{ backgroundColor: selectedLayerColor }} />
                <input
                  value={label}
                  onChange={event => onSelectedFieldChange(isText ? 'text' : 'desc', event.target.value)}
                  className="min-w-0 flex-1 bg-transparent text-xs text-gray-100 outline-none"
                />
              </span>
            </label>
            <div className="grid h-9 grid-cols-2 overflow-hidden rounded-[4px] border border-[#33414d] bg-[#08121a]">
              {([
                ['obj', 'Object'],
                ['text', 'Text'],
              ] as const).map(([value, text]) => {
                const active = value === (isText ? 'text' : 'obj');
                return (
                  <button
                    key={value}
                    type="button"
                    onClick={() => onSelectedTypeChange(value)}
                    className={classNames('text-xs transition-colors', {
                      'bg-cyan-900/70 text-cyan-50': active,
                      'text-gray-400 hover:bg-white/5 hover:text-gray-100': !active,
                    })}
                  >
                    {text}
                  </button>
                );
              })}
            </div>
          </div>

          {selectedRect ? (
            <div>
              <div className="mb-1 flex items-center justify-between text-[11px] text-gray-500">
                <span>Bounding box</span>
                <span className="tabular-nums">ID #{String(selectedElementIndex + 1).padStart(2, '0')}</span>
              </div>
              <div className="grid grid-cols-4 gap-2">
                {(['x', 'y', 'w', 'h'] as const).map(field => (
                  <label
                    key={field}
                    className="flex h-9 min-w-0 items-center gap-1 rounded-[4px] border border-[#33414d] bg-[#08121a] px-2"
                  >
                    <span className="text-[10px] uppercase text-gray-600">{field}</span>
                    <input
                      type="number"
                      min={0}
                      max={1000}
                      value={selectedRect[field]}
                      onChange={event => {
                        const nextRect = { ...selectedRect, [field]: Number(event.target.value) };
                        onChangeBox(selectedElementIndex, rectToBox(nextRect));
                      }}
                      className="min-w-0 flex-1 bg-transparent text-right text-xs tabular-nums text-gray-100 outline-none"
                    />
                  </label>
                ))}
              </div>
            </div>
          ) : (
            <div className="rounded-[4px] border border-dashed border-[#33414d] px-3 py-2 text-[11px] text-gray-500">
              No box yet. Choose Box or Text, then draw on the image.
            </div>
          )}

          <label className="block">
            <span className="mb-1 block text-[11px] text-gray-500">Description</span>
            <textarea
              value={description}
              rows={2}
              onChange={event => onSelectedFieldChange('desc', event.target.value)}
              className="h-11 w-full resize-none rounded-[4px] border border-[#33414d] bg-[#08121a] px-2.5 py-2 text-xs leading-4 text-gray-100 outline-none focus:border-cyan-600"
            />
            <span className="mt-1 block text-right text-[10px] tabular-nums text-gray-600">{description.length} / 500</span>
          </label>

          <button
            type="button"
            disabled={!canCaptionSelectedLayer}
            onClick={onCaptionSelectedLayer}
            title={layerCaptionDisabledReason || 'Caption selected layer'}
            className="flex min-h-10 w-full items-center gap-2 rounded-[4px] border border-[#33414d] bg-[#08121a] px-3 text-left text-xs text-gray-200 hover:border-cyan-800 hover:bg-cyan-950/20 disabled:cursor-not-allowed disabled:opacity-45"
          >
            {selectedLayerIsCaptioning ? (
              <Loader2 className="h-4 w-4 flex-none animate-spin text-cyan-300" />
            ) : (
              <MessageSquareText className="h-4 w-4 flex-none text-gray-400" />
            )}
            <span className="min-w-0 flex-1">
              <span className="block font-medium">Caption layer</span>
              <span className="block truncate text-[10px] text-gray-600">
                {layerCaptionStatus || 'Generate or refine this layer description'}
              </span>
            </span>
            <ChevronRight className="h-4 w-4 flex-none text-gray-500" />
          </button>
        </div>
      )}
    </section>
  );
}
