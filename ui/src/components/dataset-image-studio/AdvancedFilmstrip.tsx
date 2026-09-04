'use client';

import classNames from 'classnames';
import { Check, ChevronLeft, ChevronRight } from 'lucide-react';
import { useEffect, useRef } from 'react';
import type { DatasetStudioItem } from './types';
import { EncryptedThumb, PlainThumb } from './StudioMedia';
import { itemKey, itemName } from './utils';

export function AdvancedFilmstrip({
  items,
  selectedIndex,
  datasetName,
  workerID,
  projectID,
  encryptedKey,
  onSelectIndex,
}: {
  items: DatasetStudioItem[];
  selectedIndex: number;
  datasetName: string;
  workerID: string;
  projectID?: string | null;
  encryptedKey: CryptoKey | null | undefined;
  onSelectIndex: (index: number) => void;
}) {
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const selectedRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    const scroller = scrollerRef.current;
    const selected = selectedRef.current;
    if (!scroller || !selected) return;
    const nextLeft = selected.offsetLeft - Math.max(0, (scroller.clientWidth - selected.clientWidth) / 2);
    scroller.scrollTo({ left: Math.max(0, nextLeft), behavior: 'smooth' });
  }, [selectedIndex]);

  return (
    <nav className="flex h-[105px] flex-none border-b border-gray-800 bg-gray-950" aria-label="Dataset filmstrip">
      <button
        type="button"
        onClick={() => onSelectIndex(selectedIndex - 1)}
        disabled={selectedIndex === 0}
        className="flex w-9 flex-none items-center justify-center border-r border-gray-800 text-gray-400 hover:bg-gray-900 hover:text-white disabled:opacity-25"
        title="Previous asset"
      >
        <ChevronLeft className="h-5 w-5" />
      </button>
      <div ref={scrollerRef} className="operator-scrollbar-none flex min-w-0 flex-1 gap-1 overflow-x-auto px-1 py-1.5">
        {items.map((item, index) => {
          const selected = index === selectedIndex;
          const name = itemName(item);
          return (
            <button
              ref={selected ? selectedRef : undefined}
              key={itemKey(item)}
              type="button"
              onClick={() => onSelectIndex(index)}
              className={classNames(
                'relative h-[91px] w-[142px] flex-none overflow-hidden rounded-[2px] border-2 bg-gray-950 text-left',
                selected ? 'border-brand-400' : 'border-transparent hover:border-gray-600',
              )}
              title={name}
              aria-label={`Select ${name}`}
              aria-current={selected ? 'true' : undefined}
            >
              {item.kind === 'plain' ? (
                <PlainThumb path={item.path} mediaUrl={item.mediaUrl} alt={name} />
              ) : (
                <EncryptedThumb
                  datasetName={datasetName}
                  workerID={workerID}
                  projectID={projectID}
                  cryptoKey={encryptedKey}
                  item={item.item}
                />
              )}
              {selected ? (
                <span className="absolute right-1.5 top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-brand-400 text-[var(--brand-ink)] shadow-md shadow-black/50">
                  <Check className="h-3.5 w-3.5" />
                </span>
              ) : null}
            </button>
          );
        })}
      </div>
      <button
        type="button"
        onClick={() => onSelectIndex(selectedIndex + 1)}
        disabled={selectedIndex >= items.length - 1}
        className="flex w-8 flex-none items-center justify-center border-l border-gray-800 text-gray-400 hover:bg-gray-900 hover:text-white disabled:opacity-25"
        title="Next asset"
      >
        <ChevronRight className="h-5 w-5" />
      </button>
    </nav>
  );
}
