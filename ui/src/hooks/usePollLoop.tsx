'use client';

import { useEffect, useRef } from 'react';
import type { DependencyList } from 'react';
import { startSettledPollLoop } from '@/utils/basic';

type PollCallback = (signal: AbortSignal) => void | Promise<unknown>;

/**
 * Run a poll immediately, then schedule the next run only after the current
 * promise settles. Changing dependencies aborts the active request and starts
 * a fresh loop. A null/zero interval preserves the one-shot loading behavior
 * used by the data hooks.
 */
export default function usePollLoop(
  callback: PollCallback,
  intervalMs: number | null | undefined,
  dependencies: DependencyList = [],
) {
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    return startSettledPollLoop(signal => callbackRef.current(signal), intervalMs);
    // The caller controls restart semantics explicitly through dependencies.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, ...dependencies]);
}
