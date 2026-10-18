'use client';
import { useEffect } from 'react';
import useScopedState from './useScopedState';
import { apiClient } from '@/utils/api';

// Module-level batcher: many cards mount at once when the virtualized grid scrolls;
// instead of N HTTP requests, queue paths and flush them as a single batch.
type Resolver = { resolve: (caption: string) => void; reject: (err: unknown) => void };
const pending = new Map<string, Resolver[]>();
const cache = new Map<string, string>();
let flushTimer: ReturnType<typeof setTimeout> | null = null;
const FLUSH_DELAY_MS = 30;
const MAX_BATCH = 200;

function scheduleFlush() {
  if (flushTimer) return;
  flushTimer = setTimeout(flush, FLUSH_DELAY_MS);
}

async function flush() {
  flushTimer = null;
  if (pending.size === 0) return;

  // Drain up to MAX_BATCH paths; if more arrived, reschedule.
  const paths: string[] = [];
  for (const path of pending.keys()) {
    paths.push(path);
    if (paths.length >= MAX_BATCH) break;
  }
  const batchResolvers = paths.map(p => ({ path: p, resolvers: pending.get(p)! }));
  for (const p of paths) pending.delete(p);

  try {
    const res = await apiClient.post('/api/caption/getBatch', { imgPaths: paths });
    const captions: Record<string, string> = res.data?.captions ?? {};
    for (const { path, resolvers } of batchResolvers) {
      const value = captions[path] ?? '';
      cache.set(path, value);
      for (const r of resolvers) r.resolve(value);
    }
  } catch (err) {
    for (const { resolvers } of batchResolvers) {
      for (const r of resolvers) r.reject(err);
    }
  }

  if (pending.size > 0) scheduleFlush();
}

function requestCaption(path: string, signal?: AbortSignal): Promise<string> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const resolver: Resolver = { resolve, reject };
    const list = pending.get(path);
    if (list) {
      list.push(resolver);
    } else {
      pending.set(path, [resolver]);
    }
    if (signal) {
      const onAbort = () => {
        // Remove this resolver from the pending batch. If no other card is
        // still waiting on the same path, drop the path entirely so the next
        // batch doesn't include it.
        const arr = pending.get(path);
        if (arr) {
          const idx = arr.indexOf(resolver);
          if (idx >= 0) arr.splice(idx, 1);
          if (arr.length === 0) pending.delete(path);
        }
        reject(new DOMException('Aborted', 'AbortError'));
      };
      signal.addEventListener('abort', onAbort, { once: true });
    }
    scheduleFlush();
  });
}

export function invalidateCaption(path: string) {
  cache.delete(path);
}

export function setCachedCaption(path: string, caption: string) {
  cache.set(path, caption);
}

// Fetches caption for a path, using the module-level batcher + cache.
// `refreshKey` busts the cache (e.g. after external edits or auto-captioning poll).
export default function useCaptionBatch(imgPath: string | null, refreshKey: number = 0) {
  const scope = JSON.stringify([imgPath, refreshKey]);
  const [caption, setCaption] = useScopedState<string>(scope, () => (imgPath ? (cache.get(imgPath) ?? '') : ''));
  const [isLoaded, setIsLoaded] = useScopedState(scope, () =>
    Boolean(imgPath && refreshKey === 0 && cache.has(imgPath)),
  );

  useEffect(() => {
    if (!imgPath) return;
    if (refreshKey > 0) invalidateCaption(imgPath);
    if (cache.has(imgPath)) return;

    let cancelled = false;
    const controller = new AbortController();
    requestCaption(imgPath, controller.signal)
      .then(value => {
        if (cancelled) return;
        setCaption(value);
        setIsLoaded(true);
      })
      .catch(err => {
        if (err?.name === 'AbortError' || cancelled) return;
        console.error('Error fetching caption:', err);
        setIsLoaded(true);
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [imgPath, refreshKey, setCaption, setIsLoaded]);

  return { caption, isLoaded, setCaption };
}
