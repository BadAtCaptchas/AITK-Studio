'use client';

import classNames from 'classnames';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { extractIdeogramBoxes, parseIdeogramCaption } from '@/utils/ideogramCaption';
import type { CaptionCacheEntry, DatasetStudioItem } from './types';
import { EncryptedThumb, PlainThumb } from './StudioMedia';
import { itemKey, itemName, statusForCaption } from './utils';

export function ThumbnailStrip({
  items,
  thumbRange,
  selectedIndex,
  datasetName,
  workerID,
  encryptedKey,
  captionCache,
  onSelectIndex,
}: {
  items: DatasetStudioItem[];
  thumbRange: { start: number; end: number };
  selectedIndex: number;
  datasetName: string;
  workerID: string;
  encryptedKey?: CryptoKey | null;
  captionCache: Map<string, CaptionCacheEntry>;
  onSelectIndex: (index: number) => void;
}) {
  const visibleThumbs = items.slice(thumbRange.start, thumbRange.end);

  return (
    <div className="flex h-20 flex-shrink-0 items-center gap-2 border-t border-gray-900 bg-[#080d12] px-2 sm:h-24 xl:h-28 xl:gap-3 xl:px-3">
      <button
        type="button"
        className="flex h-[70px] w-10 flex-shrink-0 items-center justify-center rounded-md border border-gray-900 bg-gray-950 text-gray-300 hover:bg-gray-900 sm:h-[86px] sm:w-12 xl:h-[104px] xl:w-16"
        onClick={() => onSelectIndex(selectedIndex - 1)}
        title="Previous"
      >
        <ArrowLeft className="h-7 w-7" />
      </button>
      <div className="operator-scrollbar-none flex min-w-0 flex-1 gap-2 overflow-x-auto xl:gap-3">
        {visibleThumbs.map((item, offset) => {
          const index = thumbRange.start + offset;
          const key = itemKey(item);
          const cached = captionCache.get(key);
          const status = statusForCaption(cached?.caption || '', Boolean(cached?.loaded));
          const selected = index === selectedIndex;
          const parsedCaption = cached?.loaded ? parseIdeogramCaption(cached.caption) : null;
          const previewBoxes = parsedCaption?.kind === 'ideogram' ? extractIdeogramBoxes(parsedCaption.data).slice(0, 3) : [];
          return (
            <button
              key={key}
              type="button"
              title={`${itemName(item)} - ${status.title}`}
              onClick={() => onSelectIndex(index)}
              className={classNames(
                'h-[70px] w-28 flex-shrink-0 overflow-hidden rounded-md border bg-gray-950 text-left transition-colors sm:h-[86px] sm:w-32 xl:h-[104px] xl:w-40',
                {
                  'border-blue-500 shadow-[0_0_0_1px_rgba(59,130,246,0.9)]': selected,
                  'border-gray-800 hover:border-gray-700': !selected,
                },
              )}
            >
              <div className="relative h-10 overflow-hidden bg-gray-900 sm:h-12 xl:h-[70px]">
                {item.kind === 'plain' ? (
                  <PlainThumb path={item.path} alt={itemName(item)} />
                ) : (
                  <EncryptedThumb datasetName={datasetName} workerID={workerID} cryptoKey={encryptedKey} item={item.item} />
                )}
                {previewBoxes.length > 0 && (
                  <div className="pointer-events-none absolute inset-0">
                    {previewBoxes.map(box => (
                      <span
                        key={box.elementIndex}
                        className="absolute border border-white/80"
                        style={{
                          left: `${box.x1 / 10}%`,
                          top: `${box.y1 / 10}%`,
                          width: `${(box.x2 - box.x1) / 10}%`,
                          height: `${(box.y2 - box.y1) / 10}%`,
                        }}
                      />
                    ))}
                  </div>
                )}
              </div>
              <div className="flex h-6 items-center gap-2 overflow-hidden px-2 text-[11px] leading-none text-gray-300 sm:h-8 sm:text-xs">
                <span className="font-medium text-gray-100">{index + 1}</span>
                <span className={classNames('h-2 w-2 flex-shrink-0 rounded-full', status.dot)} />
                <span className="truncate">{status.label}</span>
              </div>
            </button>
          );
        })}
      </div>
      <button
        type="button"
        className="flex h-[70px] w-10 flex-shrink-0 items-center justify-center rounded-md border border-gray-900 bg-gray-950 text-gray-300 hover:bg-gray-900 sm:h-[86px] sm:w-12 xl:h-[104px] xl:w-16"
        onClick={() => onSelectIndex(selectedIndex + 1)}
        title="Next"
      >
        <ArrowRight className="h-7 w-7" />
      </button>
    </div>
  );
}
