'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import classNames from 'classnames';
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  FileJson2,
  Hand,
  Keyboard,
  Loader2,
  Maximize2,
  MoreVertical,
  MousePointer2,
  Move,
  Redo2,
  Save,
  SquareDashed,
  Tags,
  Trash2,
  Type,
  Undo2,
  WandSparkles,
  ZoomIn,
} from 'lucide-react';
import { Button } from '@headlessui/react';
import AudioPlayer from '@/components/AudioPlayer';
import type { EncryptedDatasetItem } from '@/types';
import { apiClient } from '@/utils/api';
import { isAudio, isVideo } from '@/utils/basic';
import {
  captionObjectPath,
  decryptEncryptedObjectBlob,
  encryptCaptionObject,
  randomId,
} from '@/utils/encryptedDatasets';
import { getDisplayPath, getMediaUrl } from '@/utils/media';
import {
  addIdeogramElement,
  boxToRect,
  cloneIdeogramData,
  deleteIdeogramElement,
  extractIdeogramBoxes,
  parseIdeogramCaption,
  rectToBox,
  serializeIdeogramCaption,
  type IdeogramBox,
  type IdeogramElementType,
  type NormalizedBox,
  updateIdeogramElementBox,
  updateIdeogramElementField,
  updateIdeogramElementPalette,
  updateIdeogramElementType,
  updateIdeogramHighLevelDescription,
} from '@/utils/ideogramCaption';

export type DatasetStudioItem =
  | {
      kind: 'plain';
      path: string;
    }
  | {
      kind: 'encrypted';
      item: EncryptedDatasetItem;
    };

type ToolMode = 'box' | 'text' | 'select' | 'move' | 'pan';
type DragHandle = 'move' | 'nw' | 'ne' | 'sw' | 'se';
type CaptionTab = 'caption' | 'json';

type DatasetImageStudioProps = {
  datasetName: string;
  workerID: string;
  datasetPath?: string | null;
  items: DatasetStudioItem[];
  isAutoCaptioning: boolean;
  encryptedKey?: CryptoKey | null;
  encryptedRawKeyB64?: string | null;
  onRefresh?: () => void;
  onAddImages: () => void;
  onConvertDatasetToJson?: () => void;
  onSaveEncryptedCaption?: (
    item: EncryptedDatasetItem,
    captionObjectPath: string,
    encryptedCaptionJson: string,
  ) => Promise<void>;
};

const MIN_BOX_SPAN = 8;
const MAX_HISTORY = 50;
const THUMB_WINDOW = 11;
const BOX_COLORS = ['#22D3EE', '#F59E0B', '#A3E635', '#FB7185', '#818CF8', '#34D399'];

function itemKey(item: DatasetStudioItem) {
  return item.kind === 'plain' ? item.path : item.item.id;
}

function itemName(item: DatasetStudioItem) {
  if (item.kind === 'encrypted') return item.item.name;
  const displayPath = getDisplayPath(item.path);
  if (displayPath !== item.path) return displayPath;
  return item.path.split(/[\\/]/).pop() || item.path;
}

function itemKind(item: DatasetStudioItem) {
  if (item.kind === 'encrypted') return item.item.mediaKind;
  if (isAudio(item.path)) return 'audio';
  if (isVideo(item.path)) return 'video';
  return 'image';
}

function clampIndex(value: number, length: number) {
  if (length <= 0) return 0;
  return Math.max(0, Math.min(length - 1, value));
}

function statusForCaption(caption: string, loaded: boolean) {
  if (!loaded) return { dot: 'bg-gray-500', label: '...', title: 'Caption not loaded' };
  if (!caption.trim()) return { dot: 'bg-rose-400', label: '0%', title: 'Missing caption' };
  const parsed = parseIdeogramCaption(caption);
  if (parsed.kind === 'ideogram') {
    return {
      dot: parsed.boxes.length > 0 ? 'bg-emerald-400' : 'bg-blue-400',
      label: parsed.boxes.length > 0 ? '100%' : 'JSON',
      title: parsed.boxes.length > 0 ? `${parsed.boxes.length} box${parsed.boxes.length === 1 ? '' : 'es'}` : 'JSON caption',
    };
  }
  if (parsed.kind === 'json') return { dot: 'bg-amber-400', label: 'JSON', title: parsed.error };
  return { dot: 'bg-amber-400', label: 'TXT', title: 'Plain text caption' };
}

function captionResponseToText(value: unknown) {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  return JSON.stringify(value, null, 2);
}

function resolveBoxColor(box: IdeogramBox, index: number, selected: boolean) {
  if (selected) return '#22D3EE';
  if (box.type === 'text') return '#F59E0B';
  return box.color || BOX_COLORS[index % BOX_COLORS.length];
}

function resizeOrMoveBox(box: NormalizedBox, dx: number, dy: number, handle: DragHandle): NormalizedBox {
  let { x1, y1, x2, y2 } = box;
  if (handle === 'move') {
    const width = x2 - x1;
    const height = y2 - y1;
    x1 = Math.max(0, Math.min(1000 - width, x1 + dx));
    y1 = Math.max(0, Math.min(1000 - height, y1 + dy));
    return { x1, y1, x2: x1 + width, y2: y1 + height };
  }

  if (handle.includes('w')) x1 = Math.max(0, Math.min(x2 - MIN_BOX_SPAN, x1 + dx));
  if (handle.includes('e')) x2 = Math.min(1000, Math.max(x1 + MIN_BOX_SPAN, x2 + dx));
  if (handle.includes('n')) y1 = Math.max(0, Math.min(y2 - MIN_BOX_SPAN, y1 + dy));
  if (handle.includes('s')) y2 = Math.min(1000, Math.max(y1 + MIN_BOX_SPAN, y2 + dy));
  return { x1, y1, x2, y2 };
}

function ToolButton({
  active,
  disabled,
  label,
  icon,
  onClick,
}: {
  active?: boolean;
  disabled?: boolean;
  label: string;
  icon: React.ReactNode;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      title={label}
      onClick={onClick}
      className={classNames(
        'group inline-flex h-14 w-14 flex-shrink-0 flex-col items-center justify-center gap-1 rounded-md border text-[10px] transition-colors md:h-16 md:w-16 md:text-[11px]',
        {
          'border-blue-500 bg-blue-600/20 text-blue-100 shadow-[0_0_0_1px_rgba(59,130,246,0.25)]': active,
          'border-transparent text-gray-300 hover:border-gray-700 hover:bg-gray-800 hover:text-gray-100': !active,
          'cursor-not-allowed opacity-35 hover:border-transparent hover:bg-transparent': disabled,
        },
      )}
    >
      <span className="flex h-6 w-6 items-center justify-center">{icon}</span>
      <span>{label}</span>
    </button>
  );
}

function SegmentedButton({
  active,
  children,
  onClick,
}: {
  active: boolean;
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={classNames('h-8 min-w-24 border-r border-gray-800 px-3 text-sm last:border-r-0', {
        'bg-blue-600/30 text-blue-100': active,
        'text-gray-300 hover:bg-gray-800': !active,
      })}
    >
      {children}
    </button>
  );
}

function useEncryptedObjectUrl(
  datasetName: string,
  workerID: string,
  cryptoKey: CryptoKey | null | undefined,
  item: EncryptedDatasetItem | null,
  enabled = true,
) {
  const [url, setUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!cryptoKey || !item || !enabled) {
      setUrl(null);
      return;
    }

    let objectUrl: string | null = null;
    let cancelled = false;
    setLoading(true);
    setUrl(null);

    apiClient
      .post('/api/datasets/encrypted/object', { datasetName, worker_id: workerID, objectPath: item.objectPath }, { responseType: 'blob' })
      .then(async response => {
        const decrypted = await decryptEncryptedObjectBlob(cryptoKey, item.objectPath, response.data as Blob);
        if (cancelled) return;
        objectUrl = URL.createObjectURL(new Blob([decrypted], { type: item.mimeType || 'application/octet-stream' }));
        setUrl(objectUrl);
      })
      .catch(error => {
        if (!cancelled) console.error('Encrypted media load failed:', error);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [cryptoKey, datasetName, enabled, item, workerID]);

  return { url, loading };
}

function PlainThumb({ path, alt }: { path: string; alt: string }) {
  if (isAudio(path)) {
    return <div className="flex h-full w-full items-center justify-center bg-gray-900 text-[10px] text-gray-400">Audio</div>;
  }
  if (isVideo(path)) {
    return <video src={getMediaUrl(path)} className="h-full w-full object-cover" muted preload="metadata" />;
  }
  return <img src={getMediaUrl(path)} alt={alt} loading="lazy" className="h-full w-full object-cover" />;
}

function EncryptedThumb({
  datasetName,
  workerID,
  cryptoKey,
  item,
}: {
  datasetName: string;
  workerID: string;
  cryptoKey: CryptoKey | null | undefined;
  item: EncryptedDatasetItem;
}) {
  const { url, loading } = useEncryptedObjectUrl(datasetName, workerID, cryptoKey, item);

  if (loading || !url) {
    return <div className="flex h-full w-full items-center justify-center bg-gray-900 text-[10px] text-gray-500">Decrypting</div>;
  }
  if (item.mediaKind === 'audio') {
    return <div className="flex h-full w-full items-center justify-center bg-gray-900 text-[10px] text-gray-400">Audio</div>;
  }
  if (item.mediaKind === 'video') {
    return <video src={url} className="h-full w-full object-cover" muted preload="metadata" />;
  }
  return <img src={url} alt={item.name} loading="lazy" className="h-full w-full object-cover" />;
}

function useElementSize<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const updateSize = () => {
      const rect = element.getBoundingClientRect();
      setSize({
        width: Math.max(0, rect.width),
        height: Math.max(0, rect.height),
      });
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(element);
    window.addEventListener('resize', updateSize);

    return () => {
      observer.disconnect();
      window.removeEventListener('resize', updateSize);
    };
  }, []);

  return { ref, size };
}

function StudioMedia({
  item,
  datasetName,
  workerID,
  cryptoKey,
  children,
  zoom,
}: {
  item: DatasetStudioItem;
  datasetName: string;
  workerID: string;
  cryptoKey?: CryptoKey | null;
  children: React.ReactNode;
  zoom: number;
}) {
  const encryptedItem = item.kind === 'encrypted' ? item.item : null;
  const { url, loading } = useEncryptedObjectUrl(datasetName, workerID, cryptoKey, encryptedItem);
  const kind = itemKind(item);
  const src = item.kind === 'plain' ? getMediaUrl(item.path) : url;
  const name = itemName(item);
  const { ref: frameRef, size: frameSize } = useElementSize<HTMLDivElement>();
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    setNaturalSize(null);
  }, [src]);

  const fittedSize = useMemo(() => {
    if (!naturalSize || frameSize.width <= 0 || frameSize.height <= 0) return null;
    const fitScale = Math.min(frameSize.width / naturalSize.width, frameSize.height / naturalSize.height);
    const scaledWidth = Math.max(1, naturalSize.width * fitScale * zoom);
    const scaledHeight = Math.max(1, naturalSize.height * fitScale * zoom);
    return {
      width: Math.round(scaledWidth),
      height: Math.round(scaledHeight),
    };
  }, [frameSize.height, frameSize.width, naturalSize, zoom]);

  if (item.kind === 'encrypted' && (loading || !src)) {
    return (
      <div className="flex h-full w-full items-center justify-center text-sm text-gray-400">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Decrypting media
      </div>
    );
  }

  if (kind === 'audio' && src) {
    return (
      <div className="flex h-full w-full items-center justify-center p-4">
        <AudioPlayer src={src} title={name} />
      </div>
    );
  }

  if (kind === 'video' && src) {
    return (
      <div className="flex h-full w-full items-center justify-center overflow-hidden">
        <video src={src} className="h-full w-full object-contain" controls loop />
      </div>
    );
  }

  return (
    <div ref={frameRef} className="relative flex h-full w-full min-h-0 items-center justify-center overflow-auto">
      <div
        className="relative shrink-0 leading-[0]"
        style={
          fittedSize
            ? {
                width: `${fittedSize.width}px`,
                height: `${fittedSize.height}px`,
              }
            : undefined
        }
      >
        {src ? (
          <img
            src={src}
            alt={name}
            draggable={false}
            onLoad={event => {
              const { naturalWidth, naturalHeight } = event.currentTarget;
              if (naturalWidth > 0 && naturalHeight > 0) {
                setNaturalSize({ width: naturalWidth, height: naturalHeight });
              }
            }}
            className={classNames('block select-none object-contain', {
              'h-full w-full': fittedSize,
              'max-h-full max-w-full': !fittedSize,
            })}
          />
        ) : null}
        {fittedSize ? children : null}
      </div>
    </div>
  );
}

function AnnotationLayer({
  boxes,
  activeTool,
  selectedElementIndex,
  onSelect,
  onCreate,
  onChangeBox,
}: {
  boxes: IdeogramBox[];
  activeTool: ToolMode;
  selectedElementIndex: number | null;
  onSelect: (elementIndex: number | null) => void;
  onCreate: (type: IdeogramElementType, box: NormalizedBox) => void;
  onChangeBox: (elementIndex: number, box: NormalizedBox) => void;
}) {
  const layerRef = useRef<HTMLDivElement>(null);
  const [dragPreview, setDragPreview] = useState<{ elementIndex: number; box: NormalizedBox } | null>(null);
  const [newBoxPreview, setNewBoxPreview] = useState<NormalizedBox | null>(null);
  const drawingType: IdeogramElementType | null = activeTool === 'box' ? 'obj' : activeTool === 'text' ? 'text' : null;

  const pointToNorm = useCallback((clientX: number, clientY: number) => {
    const rect = layerRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0 || rect.height === 0) return null;
    return {
      x: Math.max(0, Math.min(1000, ((clientX - rect.left) / rect.width) * 1000)),
      y: Math.max(0, Math.min(1000, ((clientY - rect.top) / rect.height) * 1000)),
    };
  }, []);

  const beginBoxDrag = (event: ReactPointerEvent<HTMLElement>, box: IdeogramBox, handle: DragHandle) => {
    if (drawingType) return;
    event.preventDefault();
    event.stopPropagation();
    onSelect(box.elementIndex);

    const start = pointToNorm(event.clientX, event.clientY);
    if (!start) return;
    const startBox = { x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 };
    let latest = startBox;

    const onMove = (moveEvent: PointerEvent) => {
      const point = pointToNorm(moveEvent.clientX, moveEvent.clientY);
      if (!point) return;
      latest = resizeOrMoveBox(startBox, point.x - start.x, point.y - start.y, handle);
      setDragPreview({ elementIndex: box.elementIndex, box: latest });
    };

    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      setDragPreview(null);
      onChangeBox(box.elementIndex, latest);
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };

  const beginDraw = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.currentTarget !== event.target) return;
    if (!drawingType) {
      onSelect(null);
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    const start = pointToNorm(event.clientX, event.clientY);
    if (!start) return;
    let latest: NormalizedBox = { y1: start.y, x1: start.x, y2: start.y, x2: start.x };
    setNewBoxPreview(latest);

    const onMove = (moveEvent: PointerEvent) => {
      const point = pointToNorm(moveEvent.clientX, moveEvent.clientY);
      if (!point) return;
      latest = rectToBox({
        x: Math.min(start.x, point.x),
        y: Math.min(start.y, point.y),
        w: Math.abs(point.x - start.x),
        h: Math.abs(point.y - start.y),
      });
      setNewBoxPreview(latest);
    };

    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      setNewBoxPreview(null);
      if (latest.x2 - latest.x1 >= MIN_BOX_SPAN && latest.y2 - latest.y1 >= MIN_BOX_SPAN) {
        onCreate(drawingType, latest);
      }
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };

  return (
    <div
      ref={layerRef}
      onPointerDown={beginDraw}
      className={classNames('absolute inset-0 touch-none', {
        'cursor-crosshair': drawingType,
        'cursor-default': !drawingType,
      })}
    >
      {boxes.map((box, index) => {
        const preview = dragPreview?.elementIndex === box.elementIndex ? dragPreview.box : box;
        const selected = selectedElementIndex === box.elementIndex;
        const color = resolveBoxColor(box, index, selected);
        return (
          <div
            key={box.elementIndex}
            onPointerDown={event => beginBoxDrag(event, box, 'move')}
            className={classNames('absolute border-2', {
              'shadow-[0_0_0_1px_rgba(255,255,255,0.75)]': selected,
            })}
            style={{
              left: `${preview.x1 / 10}%`,
              top: `${preview.y1 / 10}%`,
              width: `${Math.max(0, preview.x2 - preview.x1) / 10}%`,
              height: `${Math.max(0, preview.y2 - preview.y1) / 10}%`,
              borderColor: color,
            }}
          >
            <span
              title={box.label}
              className="absolute left-0 top-0 max-w-[12rem] truncate px-1 py-0.5 text-[10px] font-semibold leading-none text-gray-950"
              style={{ backgroundColor: color }}
            >
              {box.label || (box.type === 'text' ? 'text' : 'object')}
            </span>
            {selected &&
              (['nw', 'ne', 'sw', 'se'] as const).map(handle => (
                <button
                  key={handle}
                  type="button"
                  aria-label={`Resize ${handle}`}
                  onPointerDown={event => beginBoxDrag(event, { ...box, ...preview }, handle)}
                  className={classNames('absolute h-3 w-3 rounded-sm border border-gray-950 bg-white', {
                    'left-0 top-0 -translate-x-1/2 -translate-y-1/2 cursor-nwse-resize': handle === 'nw',
                    'right-0 top-0 translate-x-1/2 -translate-y-1/2 cursor-nesw-resize': handle === 'ne',
                    'bottom-0 left-0 -translate-x-1/2 translate-y-1/2 cursor-nesw-resize': handle === 'sw',
                    'bottom-0 right-0 translate-x-1/2 translate-y-1/2 cursor-nwse-resize': handle === 'se',
                  })}
                />
              ))}
          </div>
        );
      })}
      {newBoxPreview && (
        <div
          className="absolute border-2 border-dashed border-white bg-white/10"
          style={{
            left: `${newBoxPreview.x1 / 10}%`,
            top: `${newBoxPreview.y1 / 10}%`,
            width: `${Math.max(0, newBoxPreview.x2 - newBoxPreview.x1) / 10}%`,
            height: `${Math.max(0, newBoxPreview.y2 - newBoxPreview.y1) / 10}%`,
          }}
        />
      )}
    </div>
  );
}

export default function DatasetImageStudio({
  datasetName,
  workerID,
  datasetPath,
  items,
  isAutoCaptioning,
  encryptedKey,
  encryptedRawKeyB64,
  onRefresh,
  onAddImages,
  onConvertDatasetToJson,
  onSaveEncryptedCaption,
}: DatasetImageStudioProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [captionText, setCaptionText] = useState('');
  const [savedCaption, setSavedCaption] = useState('');
  const [isCaptionLoaded, setIsCaptionLoaded] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [activeTool, setActiveTool] = useState<ToolMode>('select');
  const [selectedElementIndex, setSelectedElementIndex] = useState<number | null>(null);
  const [captionTab, setCaptionTab] = useState<CaptionTab>('caption');
  const [zoom, setZoom] = useState(1);
  const [undoStack, setUndoStack] = useState<string[]>([]);
  const [redoStack, setRedoStack] = useState<string[]>([]);
  const [encryptedCaptionPaths, setEncryptedCaptionPaths] = useState<Record<string, string>>({});
  const captionCacheRef = useRef(new Map<string, { caption: string; saved: string; loaded: boolean }>());
  const saveCaptionRef = useRef<() => Promise<void>>(async () => undefined);
  const autoSelectKeyRef = useRef('');

  useEffect(() => {
    setSelectedIndex(index => clampIndex(index, items.length));
  }, [items.length]);

  const selectedItem = items[selectedIndex] || null;
  const selectedKey = selectedItem ? itemKey(selectedItem) : '';
  const selectedName = selectedItem ? itemName(selectedItem) : '';
  const selectedKind = selectedItem ? itemKind(selectedItem) : 'image';
  const captionParse = useMemo(() => parseIdeogramCaption(captionText), [captionText]);
  const isIdeogram = captionParse.kind === 'ideogram';
  const boxes = isIdeogram ? captionParse.boxes : [];
  const selectedElement =
    isIdeogram && selectedElementIndex != null ? captionParse.elements[selectedElementIndex] ?? null : null;
  const selectedBox = boxes.find(box => box.elementIndex === selectedElementIndex) || null;
  const isDirty = captionText.trim() !== savedCaption.trim();
  const captionStatus = statusForCaption(captionText, isCaptionLoaded);
  const canAnnotate = isIdeogram && selectedKind === 'image' && isCaptionLoaded;
  const canConvertDataset = Boolean(datasetPath && onConvertDatasetToJson);

  useEffect(() => {
    if (!selectedKey) {
      setCaptionText('');
      setSavedCaption('');
      setIsCaptionLoaded(false);
      return;
    }

    const cached = captionCacheRef.current.get(selectedKey);
    if (cached?.loaded) {
      setCaptionText(cached.caption);
      setSavedCaption(cached.saved);
      setIsCaptionLoaded(true);
      return;
    }

    let cancelled = false;
    setCaptionText('');
    setSavedCaption('');
    setIsCaptionLoaded(false);

    async function loadCaption() {
      try {
        let text = '';
        if (!selectedItem) return;
        if (selectedItem.kind === 'plain') {
          const response = await apiClient.post('/api/caption/get', { imgPath: selectedItem.path });
          text = captionResponseToText(response.data);
        } else if (encryptedKey) {
          const captionPath = selectedItem.item.captionObjectPath;
          if (captionPath) {
            const response = await apiClient.post(
              '/api/datasets/encrypted/object',
              { datasetName, worker_id: workerID, objectPath: captionPath },
              { responseType: 'blob' },
            );
            const decrypted = await decryptEncryptedObjectBlob(encryptedKey, captionPath, response.data as Blob);
            text = new TextDecoder().decode(decrypted);
          }
        }
        if (cancelled) return;
        setCaptionText(text);
        setSavedCaption(text);
        setIsCaptionLoaded(true);
        captionCacheRef.current.set(selectedKey, { caption: text, saved: text, loaded: true });
      } catch (error) {
        if (!cancelled) {
          console.error('Caption load failed:', error);
          setIsCaptionLoaded(true);
          captionCacheRef.current.set(selectedKey, { caption: '', saved: '', loaded: true });
        }
      }
    }

    void loadCaption();
    return () => {
      cancelled = true;
    };
  }, [datasetName, encryptedKey, selectedItem, selectedKey, workerID]);

  useEffect(() => {
    if (!selectedKey) return;
    captionCacheRef.current.set(selectedKey, { caption: captionText, saved: savedCaption, loaded: isCaptionLoaded });
  }, [captionText, isCaptionLoaded, savedCaption, selectedKey]);

  useEffect(() => {
    if (!isIdeogram || selectedElementIndex == null) return;
    if (!captionParse.elements[selectedElementIndex]) setSelectedElementIndex(null);
  }, [captionParse, isIdeogram, selectedElementIndex]);

  useEffect(() => {
    if (!selectedKey || autoSelectKeyRef.current === selectedKey) return;
    if (!isIdeogram || boxes.length === 0) return;
    autoSelectKeyRef.current = selectedKey;
    setSelectedElementIndex(boxes[0].elementIndex);
  }, [boxes, isIdeogram, selectedKey]);

  const saveCaption = useCallback(async () => {
    if (!selectedItem || !isCaptionLoaded || isSaving || !isDirty) return;
    const value = captionText.trim();
    setIsSaving(true);
    try {
      if (selectedItem.kind === 'plain') {
        await apiClient.post('/api/img/caption', { imgPath: selectedItem.path, caption: value });
      } else if (encryptedKey && onSaveEncryptedCaption) {
        const key = itemKey(selectedItem);
        const targetCaptionPath =
          encryptedCaptionPaths[key] || selectedItem.item.captionObjectPath || captionObjectPath(randomId());
        const encryptedCaption = await encryptCaptionObject(encryptedKey, targetCaptionPath, value);
        await onSaveEncryptedCaption(selectedItem.item, targetCaptionPath, JSON.stringify(encryptedCaption));
        setEncryptedCaptionPaths(previous => ({ ...previous, [key]: targetCaptionPath }));
      }
      setSavedCaption(value);
      captionCacheRef.current.set(selectedKey, { caption: value, saved: value, loaded: true });
      onRefresh?.();
    } catch (error) {
      console.error('Caption save failed:', error);
      alert('Failed to save caption. Please try again.');
    } finally {
      setIsSaving(false);
    }
  }, [
    captionText,
    encryptedCaptionPaths,
    encryptedKey,
    isCaptionLoaded,
    isDirty,
    isSaving,
    onRefresh,
    onSaveEncryptedCaption,
    selectedItem,
    selectedKey,
  ]);

  useEffect(() => {
    saveCaptionRef.current = saveCaption;
  }, [saveCaption]);

  useEffect(() => {
    return () => {
      void saveCaptionRef.current();
    };
  }, []);

  const selectIndex = useCallback(
    (nextIndex: number) => {
      void saveCaption();
      setSelectedIndex(clampIndex(nextIndex, items.length));
      setSelectedElementIndex(null);
      setActiveTool('select');
      setUndoStack([]);
      setRedoStack([]);
    },
    [items.length, saveCaption],
  );

  const mutateCaption = useCallback(
    (mutator: (data: Record<string, any>) => void, nextSelectedElementIndex?: number | null) => {
      const parsed = parseIdeogramCaption(captionText);
      if (parsed.kind !== 'ideogram') return;
      const data = cloneIdeogramData(parsed.data);
      mutator(data);
      const next = serializeIdeogramCaption(data);
      if (next === captionText) return;
      setUndoStack(previous => [...previous.slice(Math.max(0, previous.length - MAX_HISTORY + 1)), captionText]);
      setRedoStack([]);
      setCaptionText(next);
      if (nextSelectedElementIndex !== undefined) setSelectedElementIndex(nextSelectedElementIndex);
    },
    [captionText],
  );

  const undo = useCallback(() => {
    setUndoStack(previous => {
      const nextCaption = previous[previous.length - 1];
      if (!nextCaption) return previous;
      setRedoStack(redo => [captionText, ...redo].slice(0, MAX_HISTORY));
      setCaptionText(nextCaption);
      return previous.slice(0, -1);
    });
  }, [captionText]);

  const redo = useCallback(() => {
    setRedoStack(previous => {
      const nextCaption = previous[0];
      if (!nextCaption) return previous;
      setUndoStack(undoStackValue => [...undoStackValue.slice(Math.max(0, undoStackValue.length - MAX_HISTORY + 1)), captionText]);
      setCaptionText(nextCaption);
      return previous.slice(1);
    });
  }, [captionText]);

  const handleCreateBox = useCallback(
    (type: IdeogramElementType, box: NormalizedBox) => {
      let createdIndex: number | null = null;
      mutateCaption(data => {
        createdIndex = addIdeogramElement(data, type, box);
      }, createdIndex);
      if (createdIndex != null) {
        setSelectedElementIndex(createdIndex);
        setActiveTool('select');
      }
    },
    [mutateCaption],
  );

  const handleChangeBox = useCallback(
    (elementIndex: number, box: NormalizedBox) => {
      mutateCaption(data => updateIdeogramElementBox(data, elementIndex, box));
    },
    [mutateCaption],
  );

  const handleDeleteSelectedElement = useCallback(() => {
    if (selectedElementIndex == null) return;
    mutateCaption(data => deleteIdeogramElement(data, selectedElementIndex), null);
  }, [mutateCaption, selectedElementIndex]);

  const handleSelectedFieldChange = useCallback(
    (field: 'desc' | 'text', value: string) => {
      if (selectedElementIndex == null) return;
      mutateCaption(data => updateIdeogramElementField(data, selectedElementIndex, field, value));
    },
    [mutateCaption, selectedElementIndex],
  );

  const handleSelectedTypeChange = useCallback(
    (type: IdeogramElementType) => {
      if (selectedElementIndex == null) return;
      mutateCaption(data => updateIdeogramElementType(data, selectedElementIndex, type));
    },
    [mutateCaption, selectedElementIndex],
  );

  const handleSelectedPaletteChange = useCallback(
    (colors: string[]) => {
      if (selectedElementIndex == null) return;
      mutateCaption(data => updateIdeogramElementPalette(data, selectedElementIndex, colors));
    },
    [mutateCaption, selectedElementIndex],
  );

  const handleCaptionDescriptionChange = useCallback(
    (value: string) => {
      if (!isIdeogram) {
        setCaptionText(value);
        return;
      }
      mutateCaption(data => updateIdeogramHighLevelDescription(data, value));
    },
    [isIdeogram, mutateCaption],
  );

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      const isTyping = tag === 'TEXTAREA' || tag === 'INPUT' || (target?.isContentEditable ?? false);
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
        event.preventDefault();
        void saveCaption();
        return;
      }
      if (isTyping) return;
      if (event.key === 'ArrowLeft') selectIndex(selectedIndex - 1);
      if (event.key === 'ArrowRight') selectIndex(selectedIndex + 1);
      if (event.key === 'Delete' || event.key === 'Backspace') handleDeleteSelectedElement();
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        undo();
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'y') {
        event.preventDefault();
        redo();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [handleDeleteSelectedElement, redo, saveCaption, selectIndex, selectedIndex, undo]);

  const thumbRange = useMemo(() => {
    const half = Math.floor(THUMB_WINDOW / 2);
    let start = Math.max(0, selectedIndex - half);
    let end = Math.min(items.length, start + THUMB_WINDOW);
    start = Math.max(0, end - THUMB_WINDOW);
    return { start, end };
  }, [items.length, selectedIndex]);

  const visibleThumbs = items.slice(thumbRange.start, thumbRange.end);
  const highLevelDescription =
    isIdeogram && typeof captionParse.data.high_level_description === 'string'
      ? captionParse.data.high_level_description
      : captionText;
  const selectedRect = selectedBox ? boxToRect(selectedBox) : null;
  const selectedPalette = Array.isArray(selectedElement?.color_palette) ? selectedElement.color_palette : [];

  if (items.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-gray-400">
        <div className="border border-dashed border-gray-700 bg-gray-900/60 px-6 py-5 text-sm">No media found.</div>
      </div>
    );
  }

  if (!selectedItem) return null;

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#02060a] text-gray-100">
      <div className="operator-scrollbar-none flex h-14 flex-shrink-0 items-center gap-2 overflow-x-auto border-b border-gray-900 bg-[#070b10] px-2 sm:gap-3 sm:px-3">
        <h2 className="min-w-36 flex-1 text-base font-semibold tracking-normal">Image Studio</h2>
        <div className="hidden items-center overflow-hidden rounded-md border border-gray-800 bg-gray-950 md:flex">
          <button
            type="button"
            className="flex h-9 w-11 items-center justify-center text-gray-300 hover:bg-gray-800"
            onClick={() => selectIndex(selectedIndex - 1)}
            title="Previous"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <div className="h-9 min-w-28 border-x border-gray-800 px-4 text-center text-sm font-medium leading-9">
            {selectedIndex + 1} / {items.length}
          </div>
          <button
            type="button"
            className="flex h-9 w-11 items-center justify-center text-gray-300 hover:bg-gray-800"
            onClick={() => selectIndex(selectedIndex + 1)}
            title="Next"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
        <div className="hidden h-9 items-center gap-2 rounded-md border border-gray-800 bg-gray-950 px-3 text-sm md:flex">
          {isSaving ? <Loader2 className="h-4 w-4 animate-spin text-blue-400" /> : <span className={classNames('h-2 w-2 rounded-full', isDirty ? 'bg-blue-400' : 'bg-emerald-400')} />}
          <span>{isSaving ? 'Saving' : isDirty ? 'Unsaved' : 'Saved'}</span>
        </div>
        <button
          type="button"
          title="Zoom"
          className="hidden h-9 items-center gap-2 rounded-md border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 hover:bg-gray-900 md:flex"
          onClick={() => setZoom(value => (value >= 1.5 ? 1 : Number((value + 0.25).toFixed(2))))}
        >
          <ZoomIn className="h-4 w-4" />
          {Math.round(zoom * 100)}%
          <ChevronDown className="h-3.5 w-3.5 text-gray-500" />
        </button>
        <button
          type="button"
          title="Pan"
          className="hidden h-9 w-10 items-center justify-center rounded-md border border-gray-800 bg-gray-950 text-gray-300 hover:bg-gray-900 md:flex"
          onClick={() => setActiveTool('pan')}
        >
          <Hand className="h-4 w-4" />
        </button>
        <button
          type="button"
          title="Fit"
          className="hidden h-9 w-10 items-center justify-center rounded-md border border-gray-800 bg-gray-950 text-gray-300 hover:bg-gray-900 md:flex"
          onClick={() => setZoom(1)}
        >
          <Maximize2 className="h-4 w-4" />
        </button>
        <button
          type="button"
          className="inline-flex h-10 items-center gap-2 rounded-md bg-blue-600 px-4 text-sm font-semibold text-white hover:bg-blue-500"
          onClick={() => selectIndex(selectedIndex + 1)}
        >
          Next
          <ArrowRight className="h-4 w-4" />
        </button>
        <button type="button" title="More" className="flex h-9 w-9 items-center justify-center text-gray-400 hover:text-gray-100">
          <MoreVertical className="h-5 w-5" />
        </button>
      </div>

      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <aside className="operator-scrollbar-none flex h-16 flex-shrink-0 items-center gap-2 overflow-x-auto border-b border-gray-900 bg-[#060a0f] px-2 md:h-auto md:w-20 md:flex-col md:overflow-y-auto md:border-b-0 md:border-r md:px-0 md:py-3">
          <ToolButton active={activeTool === 'box'} disabled={!canAnnotate} label="Box" icon={<SquareDashed className="h-5 w-5" />} onClick={() => setActiveTool('box')} />
          <ToolButton active={activeTool === 'text'} disabled={!canAnnotate} label="Text" icon={<Type className="h-5 w-5" />} onClick={() => setActiveTool('text')} />
          <div className="hidden h-px w-14 bg-gray-900 md:block" />
          <ToolButton active={activeTool === 'select'} disabled={!canAnnotate} label="Select" icon={<MousePointer2 className="h-5 w-5" />} onClick={() => setActiveTool('select')} />
          <ToolButton active={activeTool === 'move'} disabled={!canAnnotate} label="Move" icon={<Move className="h-5 w-5" />} onClick={() => setActiveTool('move')} />
          <ToolButton disabled={!canAnnotate || selectedElementIndex == null} label="Delete" icon={<Trash2 className="h-5 w-5" />} onClick={handleDeleteSelectedElement} />
          <div className="hidden h-px w-14 bg-gray-900 md:block" />
          <ToolButton disabled={undoStack.length === 0} label="Undo" icon={<Undo2 className="h-5 w-5" />} onClick={undo} />
          <ToolButton disabled={redoStack.length === 0} label="Redo" icon={<Redo2 className="h-5 w-5" />} onClick={redo} />
          <div className="hidden flex-1 md:block" />
          <ToolButton label="Labels" icon={<Tags className="h-5 w-5" />} onClick={() => setCaptionTab('json')} />
          <ToolButton label="Shortcuts" icon={<Keyboard className="h-5 w-5" />} />
        </aside>

        <div className="flex min-h-0 flex-1 flex-col xl:flex-row">
        <main className="relative flex min-h-0 flex-1 flex-col bg-[#03070b]">
          <div className="relative flex min-h-[260px] flex-1 items-stretch justify-stretch overflow-hidden">
            <div className="absolute left-3 top-3 z-10 max-w-[calc(100%-1.5rem)] truncate rounded-md border border-gray-800 bg-gray-950/80 px-2 py-1 text-xs text-gray-300 backdrop-blur">
              {selectedName}
            </div>
            <StudioMedia item={selectedItem} datasetName={datasetName} workerID={workerID} cryptoKey={encryptedKey} zoom={zoom}>
              {canAnnotate && (
                <AnnotationLayer
                  boxes={boxes}
                  activeTool={activeTool}
                  selectedElementIndex={selectedElementIndex}
                  onSelect={setSelectedElementIndex}
                  onCreate={handleCreateBox}
                  onChangeBox={handleChangeBox}
                />
              )}
            </StudioMedia>
          </div>
          <div className="flex h-20 flex-shrink-0 items-center gap-2 border-t border-gray-900 bg-[#080d12] px-2 sm:h-24 xl:h-28 xl:gap-3 xl:px-3">
            <button
              type="button"
              className="flex h-[70px] w-10 flex-shrink-0 items-center justify-center rounded-md border border-gray-900 bg-gray-950 text-gray-300 hover:bg-gray-900 sm:h-[86px] sm:w-12 xl:h-[104px] xl:w-16"
              onClick={() => selectIndex(selectedIndex - 1)}
              title="Previous"
            >
              <ArrowLeft className="h-7 w-7" />
            </button>
            <div className="operator-scrollbar-none flex min-w-0 flex-1 gap-2 overflow-x-auto xl:gap-3">
              {visibleThumbs.map((item, offset) => {
                const index = thumbRange.start + offset;
                const key = itemKey(item);
                const cached = captionCacheRef.current.get(key);
                const status = statusForCaption(cached?.caption || '', Boolean(cached?.loaded));
                const selected = index === selectedIndex;
                return (
                  <button
                    key={key}
                    type="button"
                    title={`${itemName(item)} - ${status.title}`}
                    onClick={() => selectIndex(index)}
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
                      {cached?.loaded && parseIdeogramCaption(cached.caption).kind === 'ideogram' && (
                        <div className="absolute inset-0 pointer-events-none">
                          {extractIdeogramBoxes((parseIdeogramCaption(cached.caption) as any).data)
                            .slice(0, 3)
                            .map(box => (
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
              onClick={() => selectIndex(selectedIndex + 1)}
              title="Next"
            >
              <ArrowRight className="h-7 w-7" />
            </button>
          </div>
        </main>

        <aside className="flex max-h-[34dvh] min-h-[190px] flex-shrink-0 flex-col overflow-hidden border-t border-gray-900 bg-[#080d12] xl:max-h-none xl:min-h-0 xl:w-[410px] xl:border-l xl:border-t-0">
          <div className="operator-scrollbar-none min-h-0 flex-1 overflow-y-auto p-2 md:p-3">
            <section className="overflow-hidden rounded-md border border-gray-800 bg-gray-950/80">
              <div className="flex h-12 items-center justify-between border-b border-gray-800 px-4">
                <h3 className="text-sm font-semibold text-gray-100">Object Details</h3>
                <ChevronDown className="h-4 w-4 text-gray-500" />
              </div>
              <div className="space-y-4 p-4">
                {!isCaptionLoaded ? (
                  <div className="flex items-center gap-2 text-sm text-gray-400">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading caption
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
                ) : selectedElement && selectedBox ? (
                  <>
                    <div className="grid grid-cols-[1fr_auto] gap-3">
                      <label className="min-w-0">
                        <span className="mb-1 block text-xs text-gray-400">Label</span>
                        <div className="flex items-center gap-2">
                          <span className="h-2.5 w-2.5 flex-shrink-0 rounded-full" style={{ backgroundColor: selectedBox.color }} />
                          <input
                            value={selectedElement.type === 'text' ? selectedElement.text || '' : selectedElement.desc || ''}
                            onChange={event =>
                              handleSelectedFieldChange(selectedElement.type === 'text' ? 'text' : 'desc', event.target.value)
                            }
                            className="h-10 min-w-0 flex-1 rounded-md border border-gray-800 bg-gray-900 px-3 text-sm text-gray-100 outline-none focus:border-blue-500"
                          />
                        </div>
                      </label>
                      <div className="pt-6 text-xs text-gray-500">ID: {String(selectedElementIndex).padStart(3, '0')}</div>
                    </div>
                    <div>
                      <span className="mb-2 block text-xs text-gray-400">Type</span>
                      <div className="inline-flex overflow-hidden rounded-md border border-gray-800">
                        <SegmentedButton active={selectedElement.type !== 'text'} onClick={() => handleSelectedTypeChange('obj')}>
                          Object
                        </SegmentedButton>
                        <SegmentedButton active={selectedElement.type === 'text'} onClick={() => handleSelectedTypeChange('text')}>
                          Text
                        </SegmentedButton>
                      </div>
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
                                  handleChangeBox(selectedBox.elementIndex, rectToBox(nextRect));
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
                          onChange={event => handleSelectedFieldChange('text', event.target.value)}
                          className="h-16 w-full resize-none rounded-md border border-gray-800 bg-gray-900 p-3 text-sm text-gray-100 outline-none focus:border-blue-500"
                        />
                      </label>
                    )}
                    <label>
                      <span className="mb-1 block text-xs text-gray-400">Object Description</span>
                      <textarea
                        value={selectedElement.desc || ''}
                        rows={4}
                        onChange={event => handleSelectedFieldChange('desc', event.target.value)}
                        className="h-28 w-full resize-none rounded-md border border-gray-800 bg-gray-900 p-3 text-sm text-gray-100 outline-none focus:border-blue-500"
                      />
                    </label>
                    <div>
                      <span className="mb-2 block text-xs text-gray-400">Color Palette</span>
                      <div className="flex flex-wrap gap-2">
                        {selectedPalette.map((color: string, index: number) => (
                          <button
                            key={`${color}-${index}`}
                            type="button"
                            title={`Remove ${color}`}
                            onClick={() => handleSelectedPaletteChange(selectedPalette.filter((_: string, i: number) => i !== index))}
                            className="h-8 w-8 rounded-md border border-gray-700"
                            style={{ backgroundColor: color }}
                          />
                        ))}
                        <button
                          type="button"
                          onClick={() => handleSelectedPaletteChange([...selectedPalette, BOX_COLORS[selectedPalette.length % BOX_COLORS.length]])}
                          className="h-8 rounded-md border border-gray-700 px-3 text-xs text-gray-300 hover:bg-gray-800"
                        >
                          Add
                        </button>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="rounded-md border border-gray-800 bg-gray-900/60 p-3 text-sm text-gray-400">
                    Select a box, or use Box/Text to draw a new region.
                  </div>
                )}
              </div>
            </section>

            <section className="mt-3 overflow-hidden rounded-md border border-gray-800 bg-gray-950/80">
              <div className="flex h-12 items-center border-b border-gray-800 px-4">
                <button
                  type="button"
                  onClick={() => setCaptionTab('caption')}
                  className={classNames('mr-5 h-12 border-b-2 text-sm font-semibold', {
                    'border-blue-500 text-gray-100': captionTab === 'caption',
                    'border-transparent text-gray-400 hover:text-gray-200': captionTab !== 'caption',
                  })}
                >
                  Caption
                </button>
                <button
                  type="button"
                  onClick={() => setCaptionTab('json')}
                  className={classNames('h-12 border-b-2 text-sm font-semibold', {
                    'border-blue-500 text-gray-100': captionTab === 'json',
                    'border-transparent text-gray-400 hover:text-gray-200': captionTab !== 'json',
                  })}
                >
                  JSON
                </button>
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
                      onChange={event => handleCaptionDescriptionChange(event.target.value)}
                      className="h-36 w-full resize-none rounded-md border border-gray-800 bg-gray-900 p-3 text-sm text-gray-100 outline-none focus:border-blue-500 disabled:opacity-50"
                    />
                    <span className="mt-1 block text-right text-xs text-gray-500">
                      {highLevelDescription.length} / {isIdeogram ? 2000 : 4000}
                    </span>
                  </label>
                ) : (
                  <label>
                    <textarea
                      value={captionText}
                      rows={10}
                      readOnly={isAutoCaptioning || !isCaptionLoaded}
                      onChange={event => setCaptionText(event.target.value)}
                      className="h-64 w-full resize-none rounded-md border border-gray-800 bg-gray-900 p-3 font-mono text-xs leading-relaxed text-gray-100 outline-none focus:border-blue-500 disabled:opacity-50"
                    />
                  </label>
                )}
                <div className="mt-3 flex min-h-9 items-center justify-between gap-3 overflow-hidden">
                  <div className="flex min-w-0 items-center gap-2 overflow-hidden whitespace-nowrap text-xs leading-none text-gray-500">
                    {isDirty ? (
                      <>
                        <span className="h-2 w-2 rounded-full bg-blue-400" />
                        Unsaved changes
                      </>
                    ) : (
                      <>
                        <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                        Saved
                      </>
                    )}
                  </div>
                  <Button
                    className="inline-flex h-9 flex-shrink-0 items-center gap-2 overflow-hidden whitespace-nowrap rounded-md border border-emerald-500/40 bg-emerald-600/20 px-3 text-sm font-medium leading-none text-emerald-100 hover:bg-emerald-600/30 disabled:cursor-not-allowed disabled:opacity-40"
                    disabled={!isDirty || !isCaptionLoaded || isSaving}
                    onClick={() => void saveCaption()}
                  >
                    {isSaving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                    Save
                  </Button>
                </div>
              </div>
            </section>
          </div>
        </aside>
        </div>
      </div>
    </div>
  );
}
