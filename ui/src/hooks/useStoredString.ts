'use client';
import { useCallback, useSyncExternalStore } from 'react';

const changed = 'aitk-storage-changed';
// Keep preferences usable for this tab when storage is denied or full.
const temporaryValues = new Map<string, string>();
function subscribe(listener: () => void) {
  const onStorage = (event: StorageEvent) => {
    if (event.key === null) temporaryValues.clear();
    else temporaryValues.delete(event.key);
    listener();
  };
  window.addEventListener('storage', onStorage);
  window.addEventListener(changed, listener);
  return () => {
    window.removeEventListener('storage', onStorage);
    window.removeEventListener(changed, listener);
  };
}

export default function useStoredString(key: string, fallback: string) {
  const value = useSyncExternalStore(subscribe, () => {
    if (temporaryValues.has(key)) return temporaryValues.get(key)!;
    try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
  }, () => fallback);
  const setValue = useCallback((next: string) => {
    try {
      localStorage.setItem(key, next);
      temporaryValues.delete(key);
    } catch {
      temporaryValues.set(key, next);
    }
    window.dispatchEvent(new Event(changed));
  }, [key]);
  return [value, setValue] as const;
}
