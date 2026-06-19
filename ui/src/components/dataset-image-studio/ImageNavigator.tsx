'use client';

import classNames from 'classnames';
import {
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Eraser,
  FolderInput,
  Grid2X2,
  Loader2,
  Maximize2,
  Minimize2,
  Search,
  Square,
  Trash2,
  X,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Virtuoso, type VirtuosoHandle } from 'react-virtuoso';
import { apiClient } from '@/utils/api';
import {
  filterNavigatorEntries,
  groupNavigatorRows,
  navigatorColumnCount,
  navigatorStatusCounts,
  navigatorStatusForCaption,
  parseNavigatorJump,
  type DatasetNavigatorFilter,
} from '@/utils/datasetImageNavigator';
import {
  captionMatchesKeywords,
  parseCaptionKeywordQuery,
  type CaptionKeywordMatchMode,
} from '@/utils/captionKeywordSearch';
import { decryptEncryptedObjectBlob } from '@/utils/encryptedDatasets';
import { buildEncryptedObjectRequestBody } from '@/utils/encryptedObjectMediaCache';
import { extractIdeogramBoxes, parseIdeogramCaption } from '@/utils/ideogramCaption';
import type {
  BulkCaptionAction,
  BulkCaptionActionResult,
  BulkCaptionMatch,
  CaptionCacheEntry,
  DatasetStudioItem,
  DeleteImagesResult,
} from './types';
import { EncryptedThumb, PlainThumb } from './StudioMedia';
import { captionResponseToText, clampIndex, isPlainTextCaptionItem, itemKey, itemName, statusForCaption } from './utils';

type ThumbSize = 'sm' | 'md' | 'lg';
type ScanState = {
  status: 'idle' | 'scanning' | 'done' | 'error';
  scanned: number;
  total: number;
  error?: string;
};

const SCAN_CHUNK_SIZE = 160;
const ENCRYPTED_SCAN_CONCURRENCY = 6;
const THUMB_SIZE_CONFIG: Record<ThumbSize, { label: string; tileWidth: number; imageHeight: number; tileHeight: number }> = {
  sm: { label: 'S', tileWidth: 92, imageHeight: 58, tileHeight: 84 },
  md: { label: 'M', tileWidth: 120, imageHeight: 76, tileHeight: 104 },
  lg: { label: 'L', tileWidth: 152, imageHeight: 96, tileHeight: 128 },
};

function useElementSize<T extends HTMLElement>() {
  const [element, setElement] = useState<T | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (!element) return;

    const update = () => {
      const rect = element.getBoundingClientRect();
      setSize({ width: Math.max(0, rect.width), height: Math.max(0, rect.height) });
    };

    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    window.addEventListener('resize', update);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', update);
    };
  }, [element]);

  return { ref: setElement, size };
}

function previewBoxesForCaption(cached?: CaptionCacheEntry) {
  if (!cached?.loaded) return [];
  const parsed = parseIdeogramCaption(cached.caption);
  return parsed.kind === 'ideogram' ? extractIdeogramBoxes(parsed.data).slice(0, 3) : [];
}

function StatusDot({ status }: { status: ReturnType<typeof statusForCaption> }) {
  return <span className={classNames('h-2 w-2 flex-shrink-0 rounded-full', status.dot)} />;
}

function ThumbnailTile({
  item,
  index,
  selected,
  bulkSelected,
  datasetName,
  workerID,
  projectID,
  encryptedKey,
  captionCache,
  onSelect,
  onToggleBulkSelect,
  mode,
  tileSize,
}: {
  item: DatasetStudioItem;
  index: number;
  selected: boolean;
  bulkSelected?: boolean;
  datasetName: string;
  workerID: string;
  projectID?: string | null;
  encryptedKey?: CryptoKey | null;
  captionCache: Map<string, CaptionCacheEntry>;
  onSelect: (index: number) => void;
  onToggleBulkSelect?: (index: number) => void;
  mode: 'compact' | 'drawer';
  tileSize?: { imageHeight: number; tileHeight: number };
}) {
  const key = itemKey(item);
  const name = itemName(item);
  const cached = captionCache.get(key);
  const status = statusForCaption(cached?.caption || '', Boolean(cached?.loaded));
  const previewBoxes = previewBoxesForCaption(cached);
  const compact = mode === 'compact';

  return (
    <div
      className={classNames(
        'relative flex-shrink-0 overflow-hidden rounded-md border bg-gray-950 text-left transition-colors',
        compact ? 'h-[70px] w-24 sm:h-[82px] sm:w-28 xl:h-[92px] xl:w-36' : 'w-full',
        {
          'border-blue-500 shadow-[0_0_0_1px_rgba(59,130,246,0.9)]': selected,
          'border-cyan-500 shadow-[0_0_0_1px_rgba(6,182,212,0.8)]': !selected && bulkSelected,
          'border-gray-800 hover:border-gray-700': !selected && !bulkSelected,
        },
      )}
      style={!compact && tileSize ? { height: `${tileSize.tileHeight}px` } : undefined}
    >
      <button
        type="button"
        title={`${name} - ${status.title}`}
        onClick={() => onSelect(index)}
        className="block h-full w-full text-left"
      >
        <div
          className={classNames('relative overflow-hidden bg-gray-900', compact ? 'h-10 sm:h-12 xl:h-[58px]' : '')}
          style={!compact && tileSize ? { height: `${tileSize.imageHeight}px` } : undefined}
        >
          {item.kind === 'plain' ? (
            <PlainThumb path={item.path} alt={name} />
          ) : (
            <EncryptedThumb
              datasetName={datasetName}
              workerID={workerID}
              projectID={projectID}
              cryptoKey={encryptedKey}
              item={item.item}
            />
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
        <div
          className={classNames(
            'flex items-center gap-2 overflow-hidden px-2 leading-none text-gray-300',
            compact ? 'h-6 text-[11px] sm:h-7 sm:text-xs' : 'h-7 text-[11px]',
          )}
        >
          <span className="font-medium text-gray-100">{index + 1}</span>
          <StatusDot status={status} />
          <span className="truncate">{status.label}</span>
        </div>
      </button>
      {!compact && onToggleBulkSelect && (
        <button
          type="button"
          title={bulkSelected ? 'Deselect image' : 'Select image'}
          aria-label={bulkSelected ? 'Deselect image' : 'Select image'}
          aria-pressed={Boolean(bulkSelected)}
          onClick={event => {
            event.stopPropagation();
            onToggleBulkSelect(index);
          }}
          className={classNames(
            'absolute right-1.5 top-1.5 flex h-7 w-7 items-center justify-center rounded-md border backdrop-blur',
            bulkSelected
              ? 'border-cyan-400 bg-cyan-500 text-gray-950'
              : 'border-gray-700 bg-gray-950/80 text-gray-300 hover:border-gray-500',
          )}
        >
          {bulkSelected ? <Check className="h-4 w-4" /> : <Square className="h-4 w-4" />}
        </button>
      )}
    </div>
  );
}

function IconButton({
  title,
  onClick,
  disabled,
  children,
  className,
}: {
  title: string;
  onClick: () => void;
  disabled?: boolean;
  children: ReactNode;
  className?: string;
}) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      disabled={disabled}
      onClick={onClick}
      className={classNames(
        'flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-md border border-gray-800 bg-gray-950 text-gray-300 hover:bg-gray-900 disabled:cursor-not-allowed disabled:opacity-45',
        className,
      )}
    >
      {children}
    </button>
  );
}

export function ImageNavigator({
  items,
  selectedIndex,
  datasetName,
  workerID,
  projectID,
  encryptedKey,
  isAutoCaptioning,
  liveCaptionRefresh,
  captionCache,
  captionCacheVersion,
  onCaptionCacheChange,
  onSelectIndex,
  onBulkCaptionAction,
  onDeleteImages,
}: {
  items: DatasetStudioItem[];
  selectedIndex: number;
  datasetName: string;
  workerID: string;
  projectID?: string | null;
  encryptedKey?: CryptoKey | null;
  isAutoCaptioning?: boolean;
  liveCaptionRefresh?: boolean;
  captionCache: Map<string, CaptionCacheEntry>;
  captionCacheVersion: number;
  onCaptionCacheChange: () => void;
  onSelectIndex: (index: number) => void;
  onBulkCaptionAction?: (request: {
    action: BulkCaptionAction;
    query: string;
    matchMode: CaptionKeywordMatchMode;
    destinationName?: string;
    matches: BulkCaptionMatch[];
  }) => Promise<BulkCaptionActionResult>;
  onDeleteImages?: (items: DatasetStudioItem[], label?: string) => Promise<DeleteImagesResult>;
}) {
  const projectPayload = useMemo(() => (projectID ? { project_id: projectID } : {}), [projectID]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [captionKeywordQuery, setCaptionKeywordQuery] = useState('');
  const [captionKeywordMode, setCaptionKeywordMode] = useState<CaptionKeywordMatchMode>('whole-word');
  const [bulkDestinationName, setBulkDestinationName] = useState(`${datasetName}_matches`);
  const [bulkBusyAction, setBulkBusyAction] = useState<BulkCaptionAction | null>(null);
  const [bulkMessage, setBulkMessage] = useState('');
  const [selectedBulkKeys, setSelectedBulkKeys] = useState<Set<string>>(() => new Set());
  const [isBulkDeletingImages, setIsBulkDeletingImages] = useState(false);
  const [deleteSelectionMessage, setDeleteSelectionMessage] = useState('');
  const [filter, setFilter] = useState<DatasetNavigatorFilter>('all');
  const [thumbSize, setThumbSize] = useState<ThumbSize>('md');
  const [jumpText, setJumpText] = useState('');
  const [scrubValue, setScrubValue] = useState(selectedIndex + 1);
  const [localCacheVersion, setLocalCacheVersion] = useState(0);
  const [scanState, setScanState] = useState<ScanState>({ status: 'idle', scanned: 0, total: items.length });
  const scanStartedRef = useRef(false);
  const scanControllerRef = useRef<AbortController | null>(null);
  const [gridScroller, setGridScroller] = useState<HTMLDivElement | null>(null);
  const virtuosoRef = useRef<VirtuosoHandle>(null);
  const { ref: gridMeasureRef, size: gridSize } = useElementSize<HTMLDivElement>();

  useEffect(() => {
    setJumpText(`${selectedIndex + 1} / ${items.length.toLocaleString()}`);
    setScrubValue(selectedIndex + 1);
  }, [items.length, selectedIndex]);

  useEffect(() => {
    scanControllerRef.current?.abort();
    scanControllerRef.current = null;
    scanStartedRef.current = false;
    setScanState({ status: 'idle', scanned: 0, total: items.length });
    setSearchQuery('');
    setCaptionKeywordQuery('');
    setCaptionKeywordMode('whole-word');
    setBulkDestinationName(`${datasetName}_matches`);
    setBulkBusyAction(null);
    setBulkMessage('');
    setSelectedBulkKeys(new Set());
    setIsBulkDeletingImages(false);
    setDeleteSelectionMessage('');
    setFilter('all');
  }, [datasetName, items, workerID]);

  const notifyCaptionCacheChange = useCallback(() => {
    setLocalCacheVersion(version => version + 1);
    onCaptionCacheChange();
  }, [onCaptionCacheChange]);

  useEffect(() => {
    const availableKeys = new Set(items.map(item => itemKey(item)));
    setSelectedBulkKeys(previous => {
      const next = new Set([...previous].filter(key => availableKeys.has(key)));
      return next.size === previous.size ? previous : next;
    });
  }, [items]);

  const entries = useMemo(
    () =>
      items.map((item, index) => {
        const cached = captionCache.get(itemKey(item));
        return {
          index,
          name: itemName(item),
          status: navigatorStatusForCaption(cached?.caption || '', Boolean(cached?.loaded)),
        };
      }),
    [captionCache, captionCacheVersion, items, localCacheVersion],
  );

  const filteredEntries = useMemo(
    () => filterNavigatorEntries(entries, searchQuery, filter),
    [entries, filter, searchQuery],
  );
  const captionKeywordTerms = useMemo(() => parseCaptionKeywordQuery(captionKeywordQuery), [captionKeywordQuery]);
  const hasCaptionKeywordFilter = captionKeywordTerms.length > 0;
  const captionKeywordMatches = useMemo<BulkCaptionMatch[]>(() => {
    if (!hasCaptionKeywordFilter) return [];
    return entries.flatMap(entry => {
      const item = items[entry.index];
      if (!item) return [];
      const key = itemKey(item);
      const cached = captionCache.get(key);
      if (!cached?.loaded) return [];
      if (!captionMatchesKeywords(cached.caption, captionKeywordTerms, captionKeywordMode)) return [];
      return [{ key, index: entry.index, item, caption: cached.caption }];
    });
  }, [
    captionCache,
    captionCacheVersion,
    captionKeywordMode,
    captionKeywordTerms,
    entries,
    hasCaptionKeywordFilter,
    items,
    localCacheVersion,
  ]);
  const captionKeywordMatchIndexSet = useMemo(
    () => new Set(captionKeywordMatches.map(match => match.index)),
    [captionKeywordMatches],
  );
  const shownEntries = useMemo(
    () =>
      hasCaptionKeywordFilter
        ? filteredEntries.filter(entry => captionKeywordMatchIndexSet.has(entry.index))
        : filteredEntries,
    [captionKeywordMatchIndexSet, filteredEntries, hasCaptionKeywordFilter],
  );
  const filteredIndexes = useMemo(() => shownEntries.map(entry => entry.index), [shownEntries]);
  const shownCaptionKeywordMatches = useMemo(() => {
    if (!hasCaptionKeywordFilter) return [];
    const shownIndexes = new Set(filteredIndexes);
    return captionKeywordMatches.filter(match => shownIndexes.has(match.index));
  }, [captionKeywordMatches, filteredIndexes, hasCaptionKeywordFilter]);
  const selectedBulkItems = useMemo(
    () => items.filter(item => selectedBulkKeys.has(itemKey(item))),
    [items, selectedBulkKeys],
  );
  const selectedBulkCount = selectedBulkItems.length;
  const allShownSelected =
    filteredIndexes.length > 0 &&
    filteredIndexes.every(index => {
      const item = items[index];
      return Boolean(item && selectedBulkKeys.has(itemKey(item)));
    });
  const statusCounts = useMemo(() => navigatorStatusCounts(entries), [entries]);
  const captionPendingCount = useMemo(() => entries.filter(entry => entry.status === 'unknown').length, [entries]);
  const thumbConfig = THUMB_SIZE_CONFIG[thumbSize];
  const gridColumns = useMemo(
    () => navigatorColumnCount(gridSize.width, thumbConfig.tileWidth, 8),
    [gridSize.width, thumbConfig.tileWidth],
  );
  const gridRows = useMemo(() => groupNavigatorRows(filteredIndexes, gridColumns), [filteredIndexes, gridColumns]);
  const visibleStart = Math.max(0, Math.min(items.length, selectedIndex - 5));
  const visibleEnd = Math.min(items.length, Math.max(visibleStart + 11, selectedIndex + 6));
  const thumbRange = {
    start: Math.max(0, visibleEnd - 11),
    end: visibleEnd,
  };
  const visibleThumbs = items.slice(thumbRange.start, thumbRange.end);
  const canGoPrevious = selectedIndex > 0;
  const canGoNext = selectedIndex < items.length - 1;
  const scanProgress = scanState.total > 0 ? Math.round((scanState.scanned / scanState.total) * 100) : 100;
  const canRunBulkAction =
    Boolean(onBulkCaptionAction) &&
    captionKeywordTerms.length > 0 &&
    shownCaptionKeywordMatches.length > 0 &&
    captionPendingCount === 0 &&
    !bulkBusyAction;

  const commitIndex = useCallback(
    (index: number) => {
      const next = clampIndex(index, items.length);
      if (next !== selectedIndex) onSelectIndex(next);
    },
    [items.length, onSelectIndex, selectedIndex],
  );

  const commitJump = useCallback(() => {
    const parsed = parseNavigatorJump(jumpText, items.length);
    if (parsed != null) commitIndex(parsed);
    setJumpText(`${(parsed ?? selectedIndex) + 1} / ${items.length.toLocaleString()}`);
  }, [commitIndex, items.length, jumpText, selectedIndex]);

  const commitScrub = useCallback(() => {
    commitIndex(scrubValue - 1);
  }, [commitIndex, scrubValue]);

  const toggleBulkSelectedIndex = useCallback(
    (index: number) => {
      const item = items[index];
      if (!item) return;
      const key = itemKey(item);
      setDeleteSelectionMessage('');
      setSelectedBulkKeys(previous => {
        const next = new Set(previous);
        if (next.has(key)) {
          next.delete(key);
        } else {
          next.add(key);
        }
        return next;
      });
    },
    [items],
  );

  const selectShownBulkItems = useCallback(() => {
    setDeleteSelectionMessage('');
    setSelectedBulkKeys(previous => {
      const next = new Set(previous);
      filteredIndexes.forEach(index => {
        const item = items[index];
        if (item) next.add(itemKey(item));
      });
      return next;
    });
  }, [filteredIndexes, items]);

  const clearBulkSelection = useCallback(() => {
    setDeleteSelectionMessage('');
    setSelectedBulkKeys(new Set());
  }, []);

  const runSelectedImageDelete = useCallback(async () => {
    if (!onDeleteImages || selectedBulkItems.length === 0 || isBulkDeletingImages) return;
    setIsBulkDeletingImages(true);
    setDeleteSelectionMessage('');
    try {
      const result = await onDeleteImages(selectedBulkItems, 'selected images');
      const removedKeys = new Set(
        result.removedKeys ||
          (result.deleted > 0 && (result.failed || 0) === 0 ? selectedBulkItems.map(item => itemKey(item)) : []),
      );
      setSelectedBulkKeys(previous => {
        const next = new Set(previous);
        removedKeys.forEach(key => next.delete(key));
        return next;
      });
      const failed = result.failed || 0;
      const skipped = result.skipped || 0;
      setDeleteSelectionMessage(
        result.deleted === 0 && failed === 0 && skipped === 0
          ? 'No images deleted.'
          : failed > 0
          ? `${result.deleted.toLocaleString()} of ${result.requested.toLocaleString()} deleted, ${failed.toLocaleString()} failed.`
          : skipped > 0
            ? `${result.deleted.toLocaleString()} deleted, ${skipped.toLocaleString()} already missing.`
            : `${result.deleted.toLocaleString()} deleted.`,
      );
    } catch (error: any) {
      setDeleteSelectionMessage(error?.response?.data?.error || error?.message || 'Delete failed.');
    } finally {
      setIsBulkDeletingImages(false);
    }
  }, [isBulkDeletingImages, onDeleteImages, selectedBulkItems]);

  const bulkResultMessage = useCallback((result: BulkCaptionActionResult) => {
    if (result.action === 'move') {
      return `${result.found.toLocaleString()} found, ${result.affected.toLocaleString()} moved${
        result.destinationName ? ` to ${result.destinationName}` : ''
      }.`;
    }
    if (result.action === 'delete') {
      return `${result.found.toLocaleString()} found, ${result.affected.toLocaleString()} deleted.`;
    }
    return `${result.found.toLocaleString()} found, ${result.affected.toLocaleString()} captions updated${
      result.removedWords ? `, ${result.removedWords.toLocaleString()} words removed` : ''
    }.`;
  }, []);

  const runBulkAction = useCallback(
    async (action: BulkCaptionAction) => {
      if (!onBulkCaptionAction || !canRunBulkAction) return;
      if (action === 'move' && !bulkDestinationName.trim()) {
        setBulkMessage('Enter a destination dataset name.');
        return;
      }
      if (action === 'delete') {
        const confirmed = window.confirm(`Delete ${shownCaptionKeywordMatches.length.toLocaleString()} matching item(s)?`);
        if (!confirmed) return;
      }
      if (action === 'move') {
        const confirmed = window.confirm(
          `Move ${shownCaptionKeywordMatches.length.toLocaleString()} matching item(s) to "${bulkDestinationName.trim()}"?`,
        );
        if (!confirmed) return;
      }

      setBulkBusyAction(action);
      setBulkMessage('');
      try {
        const result = await onBulkCaptionAction({
          action,
          query: captionKeywordQuery,
          matchMode: captionKeywordMode,
          destinationName: action === 'move' ? bulkDestinationName.trim() : undefined,
          matches: shownCaptionKeywordMatches,
        });
        setBulkMessage(bulkResultMessage(result));
      } catch (error: any) {
        setBulkMessage(error?.response?.data?.error || error?.message || 'Bulk action failed.');
      } finally {
        setBulkBusyAction(null);
      }
    },
    [
      bulkDestinationName,
      bulkResultMessage,
      canRunBulkAction,
      captionKeywordMode,
      captionKeywordQuery,
      onBulkCaptionAction,
      shownCaptionKeywordMatches,
    ],
  );

  const setCacheEntry = useCallback(
    (item: DatasetStudioItem, caption: string) => {
      captionCache.set(itemKey(item), { caption, saved: caption, loaded: true });
    },
    [captionCache],
  );

  const scanPlainChunk = useCallback(
    async (chunk: DatasetStudioItem[], signal: AbortSignal) => {
      const plainItems = chunk.filter((item): item is Extract<DatasetStudioItem, { kind: 'plain' }> => item.kind === 'plain');
      if (plainItems.length === 0) return 0;
      const directItems = plainItems.filter(isPlainTextCaptionItem);
      const sidecarItems = plainItems.filter(item => !isPlainTextCaptionItem(item));
      await Promise.all(
        directItems.map(async item => {
          try {
            const response = await apiClient.post('/api/caption/get', { imgPath: item.path, direct: true, ...projectPayload }, { signal });
            setCacheEntry(item, captionResponseToText(response.data));
          } catch {
            setCacheEntry(item, '');
          }
        }),
      );
      if (sidecarItems.length > 0) {
        const response = await apiClient.post(
          '/api/caption/getBatch',
          { imgPaths: sidecarItems.map(item => item.path), ...projectPayload },
          { signal },
        );
        const captions: Record<string, unknown> = response.data?.captions || {};
        sidecarItems.forEach(item => {
          setCacheEntry(item, captionResponseToText(captions[item.path]));
        });
      }
      return plainItems.length;
    },
    [setCacheEntry],
  );

  const scanEncryptedChunk = useCallback(
    async (chunk: DatasetStudioItem[], signal: AbortSignal) => {
      const encryptedItems = chunk.filter(
        (item): item is Extract<DatasetStudioItem, { kind: 'encrypted' }> => item.kind === 'encrypted',
      );
      if (encryptedItems.length === 0) return 0;
      if (!encryptedKey) {
        encryptedItems.forEach(item => setCacheEntry(item, ''));
        return encryptedItems.length;
      }

      let cursor = 0;
      let completed = 0;
      const workerCount = Math.min(ENCRYPTED_SCAN_CONCURRENCY, encryptedItems.length);
      await Promise.all(
        Array.from({ length: workerCount }).map(async () => {
          while (cursor < encryptedItems.length) {
            if (signal.aborted) return;
            const item = encryptedItems[cursor++];
            let caption = '';
            const captionPath = item.item.captionObjectPath;
            if (captionPath) {
              try {
                const response = await apiClient.post(
                  '/api/datasets/encrypted/object',
                  buildEncryptedObjectRequestBody({ datasetName, workerID, projectID, objectPath: captionPath }),
                  { responseType: 'blob', signal },
                );
                const decrypted = await decryptEncryptedObjectBlob(encryptedKey, captionPath, response.data as Blob);
                caption = new TextDecoder().decode(decrypted);
              } catch (error) {
                if (signal.aborted) return;
                console.error('Encrypted caption scan failed:', error);
              }
            }
            setCacheEntry(item, caption);
            completed += 1;
          }
        }),
      );
      return completed;
    },
    [datasetName, encryptedKey, projectID, projectPayload, setCacheEntry, workerID],
  );

  useEffect(() => {
    if (!drawerOpen || scanStartedRef.current) return;
    scanStartedRef.current = true;

    const controller = new AbortController();
    scanControllerRef.current = controller;
    const runScan = async () => {
      const missingItems = items.filter(item => !captionCache.get(itemKey(item))?.loaded);
      const cachedCount = items.length - missingItems.length;
      setScanState({ status: missingItems.length > 0 ? 'scanning' : 'done', scanned: cachedCount, total: items.length });
      if (missingItems.length === 0) {
        scanControllerRef.current = null;
        return;
      }

      let scanned = cachedCount;
      try {
        for (let index = 0; index < missingItems.length; index += SCAN_CHUNK_SIZE) {
          if (controller.signal.aborted) return;
          const chunk = missingItems.slice(index, index + SCAN_CHUNK_SIZE);
          const plainCount = await scanPlainChunk(chunk, controller.signal);
          const encryptedCount = await scanEncryptedChunk(chunk, controller.signal);
          scanned += plainCount + encryptedCount;
          notifyCaptionCacheChange();
          setScanState({ status: 'scanning', scanned, total: items.length });
        }
        if (!controller.signal.aborted) {
          setScanState({ status: 'done', scanned: items.length, total: items.length });
          scanControllerRef.current = null;
          notifyCaptionCacheChange();
        }
      } catch (error) {
        if (controller.signal.aborted) return;
        scanControllerRef.current = null;
        setScanState({
          status: 'error',
          scanned,
          total: items.length,
          error: error instanceof Error ? error.message : 'Caption scan failed.',
        });
      }
    };

    void runScan();
  }, [captionCache, drawerOpen, items, notifyCaptionCacheChange, scanEncryptedChunk, scanPlainChunk]);

  useEffect(() => {
    const shouldLiveRefreshCaptions = isAutoCaptioning || liveCaptionRefresh;
    if (!shouldLiveRefreshCaptions || items.length === 0) return;
    const controller = new AbortController();
    let busy = false;

    const refreshMissingCaptions = async () => {
      if (busy || controller.signal.aborted) return;
      const targets = items
        .filter(item => {
          const cached = captionCache.get(itemKey(item));
          if (!cached?.loaded) return true;
          return navigatorStatusForCaption(cached.caption, true) === 'missing';
        })
        .slice(0, SCAN_CHUNK_SIZE);
      if (targets.length === 0) return;

      busy = true;
      try {
        await scanPlainChunk(targets, controller.signal);
        await scanEncryptedChunk(targets, controller.signal);
        if (!controller.signal.aborted) notifyCaptionCacheChange();
      } catch (error: any) {
        if (error?.name !== 'AbortError') {
          console.error('Auto-caption navigator refresh failed:', error);
        }
      } finally {
        busy = false;
      }
    };

    void refreshMissingCaptions();
    const interval = window.setInterval(refreshMissingCaptions, 5000);
    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  }, [
    captionCache,
    isAutoCaptioning,
    items,
    liveCaptionRefresh,
    notifyCaptionCacheChange,
    scanEncryptedChunk,
    scanPlainChunk,
  ]);

  useEffect(() => {
    return () => scanControllerRef.current?.abort();
  }, []);

  useEffect(() => {
    if (!drawerOpen || gridRows.length === 0) return;
    const rowIndex = gridRows.findIndex(row => row.includes(selectedIndex));
    if (rowIndex >= 0) {
      window.setTimeout(() => virtuosoRef.current?.scrollToIndex({ index: rowIndex, align: 'center' }), 0);
    }
  }, [drawerOpen, gridRows, selectedIndex]);

  useEffect(() => {
    if (!drawerOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setDrawerOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [drawerOpen]);

  const scrubberMarkers = useMemo(() => {
    const marked = entries.filter(entry => entry.status === 'missing' || entry.status === 'has-boxes');
    const step = Math.max(1, Math.ceil(marked.length / 140));
    return marked.filter((_, index) => index % step === 0);
  }, [entries]);

  return (
    <div className="flex min-w-0 max-w-full flex-shrink-0 flex-col overflow-hidden border-t border-gray-900 bg-[#080d12]">
      {drawerOpen && (
        <div className="flex h-[42dvh] min-h-[230px] flex-col border-b border-gray-900 bg-[#080d12] xl:max-h-[440px]">
          <div className="flex flex-shrink-0 flex-wrap items-center gap-2 border-b border-gray-900 px-2 py-2 xl:px-3">
            <div className="relative min-w-[180px] flex-1 sm:max-w-sm">
              <Search className="pointer-events-none absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
              <input
                value={searchQuery}
                onChange={event => setSearchQuery(event.target.value)}
                placeholder="Search"
                className="h-9 w-full rounded-md border border-gray-800 bg-gray-950 pl-8 pr-8 text-sm text-gray-100 outline-none focus:border-blue-600"
              />
              {searchQuery && (
                <button
                  type="button"
                  title="Clear search"
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-200"
                  onClick={() => setSearchQuery('')}
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>

            <div className="flex h-9 overflow-hidden rounded-md border border-gray-800 bg-gray-950 text-xs">
              {[
                { value: 'all', label: `All ${items.length}` },
                { value: 'needs-caption', label: `Needs Caption ${statusCounts.missing}` },
                { value: 'has-boxes', label: `Has Boxes ${statusCounts.hasBoxes}` },
              ].map(option => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => setFilter(option.value as DatasetNavigatorFilter)}
                  className={classNames('border-r border-gray-800 px-3 last:border-r-0 hover:bg-gray-900', {
                    'bg-blue-600 text-white hover:bg-blue-600': filter === option.value,
                    'text-gray-300': filter !== option.value,
                  })}
                >
                  {option.label}
                </button>
              ))}
            </div>

            <div className="flex h-9 overflow-hidden rounded-md border border-gray-800 bg-gray-950 text-xs">
              {(['sm', 'md', 'lg'] as ThumbSize[]).map(size => (
                <button
                  key={size}
                  type="button"
                  title={`${THUMB_SIZE_CONFIG[size].label} thumbnails`}
                  onClick={() => setThumbSize(size)}
                  className={classNames('w-9 border-r border-gray-800 last:border-r-0 hover:bg-gray-900', {
                    'bg-gray-800 text-white': thumbSize === size,
                    'text-gray-400': thumbSize !== size,
                  })}
                >
                  {THUMB_SIZE_CONFIG[size].label}
                </button>
              ))}
            </div>

            <div className="flex h-9 items-center gap-2 rounded-md border border-gray-800 bg-gray-950 px-2 text-xs text-gray-300">
              {scanState.status === 'scanning' && <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400" />}
              {scanState.status === 'done' && <span className="h-2 w-2 rounded-full bg-emerald-400" />}
              {scanState.status === 'error' && <span className="h-2 w-2 rounded-full bg-rose-400" />}
              <span>
                {scanState.status === 'scanning'
                  ? `Scanning ${scanProgress}%`
                  : scanState.status === 'error'
                    ? scanState.error || 'Scan failed'
                    : `${shownEntries.length.toLocaleString()} shown`}
              </span>
              {statusCounts.unknown > 0 && scanState.status !== 'done' && (
                <span className="text-gray-500">{statusCounts.unknown.toLocaleString()} pending</span>
              )}
            </div>

            <IconButton title="Collapse grid" onClick={() => setDrawerOpen(false)}>
              <Minimize2 className="h-4 w-4" />
            </IconButton>
          </div>

          <div className="flex flex-shrink-0 flex-wrap items-center gap-2 border-b border-gray-900 px-2 py-2 xl:px-3">
            <div className="flex h-9 items-center gap-2 rounded-md border border-gray-800 bg-gray-950 px-2 text-xs text-gray-300">
              <span className="h-2 w-2 rounded-full bg-cyan-400" />
              <span>{selectedBulkCount.toLocaleString()} selected</span>
            </div>
            <button
              type="button"
              disabled={filteredIndexes.length === 0 || allShownSelected}
              onClick={selectShownBulkItems}
              className="inline-flex h-9 items-center rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-200 hover:bg-gray-900 disabled:cursor-not-allowed disabled:opacity-45"
            >
              Select Shown
            </button>
            <button
              type="button"
              disabled={selectedBulkCount === 0}
              onClick={clearBulkSelection}
              className="inline-flex h-9 items-center rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-200 hover:bg-gray-900 disabled:cursor-not-allowed disabled:opacity-45"
            >
              Clear
            </button>
            <button
              type="button"
              title="Delete selected images"
              aria-label="Delete selected images"
              disabled={!onDeleteImages || selectedBulkCount === 0 || isBulkDeletingImages}
              onClick={() => void runSelectedImageDelete()}
              className="inline-flex h-9 items-center gap-2 rounded-md border border-rose-900/70 bg-rose-950/40 px-3 text-sm text-rose-100 hover:bg-rose-900/50 disabled:cursor-not-allowed disabled:opacity-45"
            >
              {isBulkDeletingImages ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
              Delete Selected
            </button>
            {deleteSelectionMessage && <div className="min-w-[180px] flex-1 text-xs text-gray-400">{deleteSelectionMessage}</div>}
          </div>

          <div className="flex flex-shrink-0 flex-wrap items-center gap-2 border-b border-gray-900 px-2 py-2 xl:px-3">
            <div className="relative min-w-[210px] flex-1 sm:max-w-md">
              <Search className="pointer-events-none absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
              <input
                value={captionKeywordQuery}
                onChange={event => {
                  setCaptionKeywordQuery(event.target.value);
                  setBulkMessage('');
                }}
                placeholder="Caption keywords"
                className="h-9 w-full rounded-md border border-gray-800 bg-gray-950 pl-8 pr-8 text-sm text-gray-100 outline-none focus:border-cyan-600"
              />
              {captionKeywordQuery && (
                <button
                  type="button"
                  title="Clear caption keywords"
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-200"
                  onClick={() => {
                    setCaptionKeywordQuery('');
                    setBulkMessage('');
                  }}
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>

            <label className="flex h-9 items-center gap-2 rounded-md border border-gray-800 bg-gray-950 px-2 text-xs text-gray-300">
              <input
                type="checkbox"
                checked={captionKeywordMode === 'partial'}
                onChange={event => setCaptionKeywordMode(event.target.checked ? 'partial' : 'whole-word')}
                className="h-4 w-4"
              />
              Partial
            </label>

            <input
              value={bulkDestinationName}
              onChange={event => setBulkDestinationName(event.target.value)}
              className="h-9 min-w-[180px] rounded-md border border-gray-800 bg-gray-950 px-2 text-sm text-gray-100 outline-none focus:border-cyan-600"
              aria-label="Destination dataset"
              title="Destination dataset"
            />

            <div className="flex h-9 items-center gap-2 rounded-md border border-gray-800 bg-gray-950 px-2 text-xs text-gray-300">
              {scanState.status === 'scanning' && <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400" />}
              <span>{shownCaptionKeywordMatches.length.toLocaleString()} found</span>
              {captionPendingCount > 0 && <span className="text-gray-500">{captionPendingCount.toLocaleString()} pending</span>}
            </div>

            <button
              type="button"
              title="Remove words from matching captions"
              aria-label="Remove words from matching captions"
              disabled={!canRunBulkAction || bulkBusyAction === 'remove_words'}
              onClick={() => void runBulkAction('remove_words')}
              className="inline-flex h-9 items-center gap-2 rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-200 hover:bg-gray-900 disabled:cursor-not-allowed disabled:opacity-45"
            >
              {bulkBusyAction === 'remove_words' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Eraser className="h-4 w-4" />}
              Remove Words
            </button>
            <button
              type="button"
              title="Move matching items"
              aria-label="Move matching items"
              disabled={!canRunBulkAction || bulkBusyAction === 'move'}
              onClick={() => void runBulkAction('move')}
              className="inline-flex h-9 items-center gap-2 rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-200 hover:bg-gray-900 disabled:cursor-not-allowed disabled:opacity-45"
            >
              {bulkBusyAction === 'move' ? <Loader2 className="h-4 w-4 animate-spin" /> : <FolderInput className="h-4 w-4" />}
              Move
            </button>
            <button
              type="button"
              title="Delete matching items"
              aria-label="Delete matching items"
              disabled={!canRunBulkAction || bulkBusyAction === 'delete'}
              onClick={() => void runBulkAction('delete')}
              className="inline-flex h-9 items-center gap-2 rounded-md border border-rose-900/70 bg-rose-950/40 px-3 text-sm text-rose-100 hover:bg-rose-900/50 disabled:cursor-not-allowed disabled:opacity-45"
            >
              {bulkBusyAction === 'delete' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
              Delete
            </button>

            {bulkMessage && <div className="min-w-[180px] flex-1 text-xs text-gray-400">{bulkMessage}</div>}
          </div>

          <div ref={gridMeasureRef} className="relative min-h-0 flex-1">
            <div ref={setGridScroller} className="operator-scrollbar-none absolute inset-0 overflow-y-auto px-2 py-2 xl:px-3">
              {gridScroller && gridRows.length > 0 ? (
                <Virtuoso
                  ref={virtuosoRef}
                  customScrollParent={gridScroller}
                  totalCount={gridRows.length}
                  increaseViewportBy={600}
                  computeItemKey={index => gridRows[index]?.[0] ?? index}
                  itemContent={rowIndex => {
                    const row = gridRows[rowIndex] || [];
                    return (
                      <div className="grid gap-2 pb-2" style={{ gridTemplateColumns: `repeat(${gridColumns}, minmax(0, 1fr))` }}>
                        {row.map(index => {
                          const item = items[index];
                          if (!item) return null;
                          return (
                            <ThumbnailTile
                              key={itemKey(item)}
                              item={item}
                              index={index}
                              selected={index === selectedIndex}
                              bulkSelected={selectedBulkKeys.has(itemKey(item))}
                              datasetName={datasetName}
                              workerID={workerID}
                              projectID={projectID}
                              encryptedKey={encryptedKey}
                              captionCache={captionCache}
                              onSelect={commitIndex}
                              onToggleBulkSelect={toggleBulkSelectedIndex}
                              mode="drawer"
                              tileSize={thumbConfig}
                            />
                          );
                        })}
                      </div>
                    );
                  }}
                />
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-gray-500">
                  {shownEntries.length === 0 ? 'No matching images' : 'Measuring grid'}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="flex min-w-0 max-w-full flex-col gap-2 overflow-hidden px-2 py-2 xl:px-3">
        <div className="operator-scrollbar-none flex min-w-0 max-w-full items-center gap-2 overflow-x-auto">
          <IconButton title="First image" disabled={!canGoPrevious} onClick={() => commitIndex(0)}>
            <ChevronsLeft className="h-4 w-4" />
          </IconButton>
          <IconButton title="Back 100" disabled={!canGoPrevious} onClick={() => commitIndex(selectedIndex - 100)}>
            <span className="text-[11px] font-semibold">-100</span>
          </IconButton>
          <IconButton title="Previous" disabled={!canGoPrevious} onClick={() => commitIndex(selectedIndex - 1)}>
            <ChevronLeft className="h-4 w-4" />
          </IconButton>

          <div className="relative min-w-0 flex-1">
            <div className="pointer-events-none absolute inset-x-1 top-1/2 z-0 h-3 -translate-y-1/2">
              <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-gray-700" />
              {scrubberMarkers.map(marker => (
                <span
                  key={`${marker.index}-${marker.status}`}
                  className={classNames('absolute top-1/2 h-2 w-px -translate-y-1/2', {
                    'bg-rose-400/80': marker.status === 'missing',
                    'bg-emerald-400/80': marker.status === 'has-boxes',
                  })}
                  style={{ left: `${items.length > 1 ? (marker.index / (items.length - 1)) * 100 : 0}%` }}
                />
              ))}
            </div>
            <input
              type="range"
              min={1}
              max={Math.max(1, items.length)}
              value={scrubValue}
              onChange={event => setScrubValue(Number(event.target.value))}
              onPointerUp={commitScrub}
              onKeyUp={event => {
                if (event.key === 'Enter') commitScrub();
              }}
              onBlur={commitScrub}
              className="relative z-10 h-9 w-full cursor-pointer accent-blue-500"
              title="Dataset scrubber"
            />
          </div>

          <input
            value={jumpText}
            onChange={event => setJumpText(event.target.value)}
            onFocus={event => event.currentTarget.select()}
            onBlur={commitJump}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                event.preventDefault();
                commitJump();
                event.currentTarget.blur();
              }
              if (event.key === 'Escape') {
                setJumpText(`${selectedIndex + 1} / ${items.length.toLocaleString()}`);
                event.currentTarget.blur();
              }
            }}
            className="h-9 w-28 flex-shrink-0 rounded-md border border-gray-800 bg-gray-950 px-2 text-center text-sm font-medium text-gray-100 outline-none focus:border-blue-600 sm:w-36"
            aria-label="Jump"
            title="Jump"
          />

          <IconButton title="Next" disabled={!canGoNext} onClick={() => commitIndex(selectedIndex + 1)}>
            <ChevronRight className="h-4 w-4" />
          </IconButton>
          <IconButton title="Forward 100" disabled={!canGoNext} onClick={() => commitIndex(selectedIndex + 100)}>
            <span className="text-[11px] font-semibold">+100</span>
          </IconButton>
          <IconButton title="Last image" disabled={!canGoNext} onClick={() => commitIndex(items.length - 1)}>
            <ChevronsRight className="h-4 w-4" />
          </IconButton>
          <IconButton title={drawerOpen ? 'Hide grid' : 'Show grid'} onClick={() => setDrawerOpen(open => !open)} className="hidden sm:flex">
            {drawerOpen ? <Minimize2 className="h-4 w-4" /> : <Grid2X2 className="h-4 w-4" />}
          </IconButton>
        </div>

        <div className="flex h-[74px] min-w-0 max-w-full items-center gap-2 overflow-hidden sm:h-[86px] xl:h-[96px]">
          <button
            type="button"
            className="flex h-full w-10 flex-shrink-0 items-center justify-center rounded-md border border-gray-900 bg-gray-950 text-gray-300 hover:bg-gray-900 disabled:cursor-not-allowed disabled:opacity-45 sm:w-12"
            onClick={() => commitIndex(selectedIndex - 1)}
            disabled={!canGoPrevious}
            title="Previous"
          >
            <ArrowLeft className="h-6 w-6" />
          </button>
          <div className="operator-scrollbar-none flex min-w-0 flex-1 gap-2 overflow-x-auto">
            {visibleThumbs.map((item, offset) => {
              const index = thumbRange.start + offset;
              return (
                <ThumbnailTile
                  key={itemKey(item)}
                  item={item}
                  index={index}
                  selected={index === selectedIndex}
                  datasetName={datasetName}
                  workerID={workerID}
                  projectID={projectID}
                  encryptedKey={encryptedKey}
                  captionCache={captionCache}
                  onSelect={commitIndex}
                  mode="compact"
                />
              );
            })}
          </div>
          <button
            type="button"
            className="flex h-full w-10 flex-shrink-0 items-center justify-center rounded-md border border-gray-900 bg-gray-950 text-gray-300 hover:bg-gray-900 disabled:cursor-not-allowed disabled:opacity-45 sm:w-12"
            onClick={() => commitIndex(selectedIndex + 1)}
            disabled={!canGoNext}
            title="Next"
          >
            <ArrowRight className="h-6 w-6" />
          </button>
          <IconButton title={drawerOpen ? 'Hide grid' : 'Show grid'} onClick={() => setDrawerOpen(open => !open)} className="sm:hidden">
            {drawerOpen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </IconButton>
        </div>
      </div>
    </div>
  );
}
