'use client';
import { useCallback, useSyncExternalStore } from 'react';

const changed = 'aitk-storage-changed';
function subscribe(listener: () => void) {
  window.addEventListener('storage', listener);
  window.addEventListener(changed, listener);
  return () => {
    window.removeEventListener('storage', listener);
    window.removeEventListener(changed, listener);
  };
}

export default function useStoredString(key: string, fallback: string) {
  const value = useSyncExternalStore(subscribe, () => {
    try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
  }, () => fallback);
  const setValue = useCallback((next: string) => {
    localStorage.setItem(key, next);
    window.dispatchEvent(new Event(changed));
  }, [key]);
  return [value, setValue] as const;
}
