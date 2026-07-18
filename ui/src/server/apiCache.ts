type CacheEntry = {
  promise: Promise<unknown>;
  timestamp: number;
};

const cache = new Map<string, CacheEntry>();
const DEFAULT_STALE_TIME_MS = 5000;

export async function cached<T>(
  key: string,
  fetcher: () => Promise<T>,
  staleTimeMs: number = DEFAULT_STALE_TIME_MS,
  params: unknown = null,
): Promise<T> {
  const cacheKey = params === null ? key : `${key}:${JSON.stringify(params)}`;
  const now = Date.now();
  const entry = cache.get(cacheKey);
  if (entry && now - entry.timestamp < staleTimeMs) {
    return entry.promise as Promise<T>;
  }

  const promise = fetcher();
  cache.set(cacheKey, { promise, timestamp: now });
  promise.catch(() => {
    if (cache.get(cacheKey)?.promise === promise) cache.delete(cacheKey);
  });
  return promise;
}

export function invalidateCache(key: string, params: unknown = null): void {
  if (params !== null) {
    cache.delete(`${key}:${JSON.stringify(params)}`);
    return;
  }
  cache.delete(key);
  for (const cacheKey of cache.keys()) {
    if (cacheKey.startsWith(`${key}:`)) cache.delete(cacheKey);
  }
}
