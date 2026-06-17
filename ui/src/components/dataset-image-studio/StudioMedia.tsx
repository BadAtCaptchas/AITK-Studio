'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { FileText, Loader2, Pipette } from 'lucide-react';
import classNames from 'classnames';
import AudioPlayer from '@/components/AudioPlayer';
import type { EncryptedDatasetItem } from '@/types';
import { apiClient } from '@/utils/api';
import { isAudio, isTextCaption, isVideo } from '@/utils/basic';
import { decryptEncryptedObjectBlob } from '@/utils/encryptedDatasets';
import { getMediaUrl } from '@/utils/media';
import { itemKind, itemName, sampleImageColorAt } from './utils';
import type { DatasetStudioItem, ImageSize } from './types';

export function useEncryptedObjectUrl(
  datasetName: string,
  workerID: string,
  projectID: string | null | undefined,
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
      .post(
        '/api/datasets/encrypted/object',
        { datasetName, worker_id: workerID, objectPath: item.objectPath, ...(projectID ? { project_id: projectID } : {}) },
        { responseType: 'blob' },
      )
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
  }, [cryptoKey, datasetName, enabled, item, projectID, workerID]);

  return { url, loading };
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

export function PlainThumb({ path, alt }: { path: string; alt: string }) {
  if (isTextCaption(path)) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-1 bg-gray-900 text-[10px] text-gray-400">
        <FileText className="h-4 w-4 text-blue-300" />
        <span className="max-w-full truncate px-1">Text</span>
      </div>
    );
  }
  if (isAudio(path)) {
    return <div className="flex h-full w-full items-center justify-center bg-gray-900 text-[10px] text-gray-400">Audio</div>;
  }
  if (isVideo(path)) {
    return <video src={getMediaUrl(path)} className="h-full w-full object-cover" muted preload="metadata" />;
  }
  return <img src={getMediaUrl(path)} alt={alt} loading="lazy" className="h-full w-full object-cover" />;
}

export function EncryptedThumb({
  datasetName,
  workerID,
  projectID,
  cryptoKey,
  item,
}: {
  datasetName: string;
  workerID: string;
  projectID?: string | null;
  cryptoKey: CryptoKey | null | undefined;
  item: EncryptedDatasetItem;
}) {
  const { url, loading } = useEncryptedObjectUrl(datasetName, workerID, projectID, cryptoKey, item);

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

export function StudioMedia({
  item,
  datasetName,
  workerID,
  projectID,
  cryptoKey,
  children,
  zoom,
  onNaturalSizeChange,
  isSamplingColor,
  onSampleColor,
  onCancelColorSample,
}: {
  item: DatasetStudioItem;
  datasetName: string;
  workerID: string;
  projectID?: string | null;
  cryptoKey?: CryptoKey | null;
  children: React.ReactNode;
  zoom: number;
  onNaturalSizeChange?: (size: ImageSize | null) => void;
  isSamplingColor?: boolean;
  onSampleColor?: (color: string) => void;
  onCancelColorSample?: () => void;
}) {
  const encryptedItem = item.kind === 'encrypted' ? item.item : null;
  const { url, loading } = useEncryptedObjectUrl(datasetName, workerID, projectID, cryptoKey, encryptedItem);
  const kind = itemKind(item);
  const src = item.kind === 'plain' ? getMediaUrl(item.path) : url;
  const name = itemName(item);
  const { ref: frameRef, size: frameSize } = useElementSize<HTMLDivElement>();
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    setNaturalSize(null);
    onNaturalSizeChange?.(null);
  }, [onNaturalSizeChange, src]);

  const fittedSize = useMemo(() => {
    if (!naturalSize || frameSize.width <= 0 || frameSize.height <= 0) return null;
    const availableWidth = Math.max(1, frameSize.width - 24);
    const availableHeight = Math.max(1, frameSize.height - 24);
    const fitScale = Math.min(availableWidth / naturalSize.width, availableHeight / naturalSize.height);
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

  if (kind === 'text') {
    return (
      <div className="flex h-full w-full items-center justify-center p-6 text-center text-gray-300">
        <div className="max-w-md rounded-md border border-gray-800 bg-gray-950/80 px-6 py-5">
          <FileText className="mx-auto mb-3 h-10 w-10 text-blue-300" />
          <div className="text-sm font-semibold text-gray-100">Text Caption File</div>
          <div className="mt-2 break-all text-xs text-gray-500">{name}</div>
          <p className="mt-3 text-xs text-gray-400">Edit the file contents in the caption panel. JSON-only box and layer tools are disabled.</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={frameRef} className="relative flex h-full w-full min-w-0 min-h-0 items-center justify-center overflow-auto p-3">
      <div
        className={classNames('relative shrink-0 leading-[0]', {
          'max-h-full max-w-full': !fittedSize || zoom <= 1,
        })}
        style={
          fittedSize
            ? {
                width: `${fittedSize.width}px`,
                height: `${fittedSize.height}px`,
                maxWidth: zoom <= 1 ? '100%' : undefined,
                maxHeight: zoom <= 1 ? '100%' : undefined,
              }
            : undefined
        }
      >
        {src ? (
          <img
            ref={imageRef}
            src={src}
            alt={name}
            draggable={false}
            onLoad={event => {
              const { naturalWidth, naturalHeight } = event.currentTarget;
              if (naturalWidth > 0 && naturalHeight > 0) {
                const nextSize = { width: naturalWidth, height: naturalHeight };
                setNaturalSize(nextSize);
                onNaturalSizeChange?.(nextSize);
              }
            }}
            className={classNames('block select-none object-contain', {
              'h-full w-full': fittedSize,
              'max-h-full max-w-full': !fittedSize,
            })}
          />
        ) : null}
        {fittedSize ? children : null}
        {fittedSize && isSamplingColor && (
          <div
            className="absolute inset-0 z-50 cursor-crosshair bg-cyan-400/5"
            onPointerDown={event => {
              event.preventDefault();
              event.stopPropagation();
              const image = imageRef.current;
              if (!image) return;
              try {
                const color = sampleImageColorAt(image, event.clientX, event.clientY);
                if (color) onSampleColor?.(color);
              } catch (error) {
                console.error('Image color sample failed:', error);
              }
            }}
            onDoubleClick={event => {
              event.preventDefault();
              event.stopPropagation();
              onCancelColorSample?.();
            }}
          >
            <div className="pointer-events-none absolute left-3 top-3 inline-flex items-center gap-2 rounded-md border border-cyan-400/40 bg-gray-950/90 px-2 py-1 text-xs font-medium text-cyan-100 shadow-xl">
              <Pipette className="h-3.5 w-3.5" />
              Pick color
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
