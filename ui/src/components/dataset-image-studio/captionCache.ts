import type { CaptionCacheEntry } from './types';

/** The editor and background scans share a cache. Publish immutable snapshots
 * so a concurrent render cannot observe a map changing underneath it. */
export function createCaptionCache() {
  let snapshot = { entries: new Map<string, CaptionCacheEntry>(), version: 0 };
  const listeners = new Set<() => void>();
  const publish = (entries: Map<string, CaptionCacheEntry>) => {
    snapshot = { entries, version: snapshot.version + 1 };
    listeners.forEach(listener => listener());
  };
  return {
    getSnapshot: () => snapshot,
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => { listeners.delete(listener); };
    },
    get: (key: string) => snapshot.entries.get(key),
    keys: () => snapshot.entries.keys(),
    set: (key: string, entry: CaptionCacheEntry) => {
      const previous = snapshot.entries.get(key);
      if (previous?.caption === entry.caption && previous.saved === entry.saved && previous.loaded === entry.loaded) return;
      publish(new Map(snapshot.entries).set(key, entry));
    },
    delete: (key: string) => {
      if (!snapshot.entries.has(key)) return false;
      const entries = new Map(snapshot.entries);
      entries.delete(key);
      publish(entries);
      return true;
    },
  };
}
