'use client';

import { useCallback, useEffect, useMemo, useRef } from 'react';
import type { ComfyInstallProgress, HFDownloadProgress } from '@/types';
import { apiClient } from '@/utils/api';
import useCommittedRef from './useCommittedRef';
import useScopedState from './useScopedState';

type TimedProgress = { updatedAt: string; status: string };
type ProgressState<T> = {
  progress: T | null;
  initial: T | null;
  interval: number | null;
  updatedAt: string | null;
  revision: object;
  emptyPolls: number;
  hasSeenProgress: boolean;
  isPolling: boolean;
  status: 'idle' | 'loading' | 'success' | 'error';
};

const MAX_EMPTY_POLLS = 10;
const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const isNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const isNullableNumber = (value: unknown) => value === null || isNumber(value);
const isNullableString = (value: unknown) => value === null || typeof value === 'string';
const isTimestamp = (value: unknown): value is string =>
  typeof value === 'string' && Number.isFinite(Date.parse(value));

function hasProgressFields(value: unknown): value is Record<string, unknown> {
  return isRecord(value) && isNumber(value.version) && typeof value.status === 'string' &&
    typeof value.message === 'string' && isNullableNumber(value.percent) &&
    isNullableString(value.error) && isTimestamp(value.updatedAt);
}

export function isDownloadProgress(value: unknown): value is HFDownloadProgress {
  return hasProgressFields(value) && ['idle', 'downloading', 'completed', 'failed'].includes(String(value.status)) &&
    isNullableString(value.fileName) && isNumber(value.activeCount) && isNumber(value.bytesDownloaded) &&
    isNullableNumber(value.bytesTotal) && Array.isArray(value.downloads) && value.downloads.every(item =>
      isRecord(item) && isNumber(item.id) && typeof item.fileName === 'string' &&
      (item.source === undefined || typeof item.source === 'string') && isNumber(item.bytesDownloaded) &&
      isNullableNumber(item.bytesTotal) && isTimestamp(item.startedAt) && isTimestamp(item.updatedAt));
}

export function isComfyInstallProgress(value: unknown): value is ComfyInstallProgress {
  return hasProgressFields(value) &&
    ['idle', 'checking', 'installing', 'launching', 'ready', 'completed', 'failed'].includes(String(value.status)) &&
    typeof value.step === 'string' && isNullableString(value.root) && isTimestamp(value.startedAt);
}

function compareTimestamp(next: string, previous: string | null): number {
  return previous === null ? 1 : Date.parse(next) - Date.parse(previous);
}

function isTerminal(progress: TimedProgress | null): boolean {
  return progress?.status === 'completed' || progress?.status === 'failed';
}

/** Shared ownership for job-provided snapshots and their independently polled endpoint. */
export default function useJobProgress<T extends TimedProgress>(
  jobID: string,
  initialProgress: T | null,
  reloadInterval: number | null,
  endpoint: 'hf-download-progress' | 'comfy-install-progress',
  isProgress: (value: unknown) => value is T,
  stopAtTerminal: boolean,
) {
  const [state, setState] = useScopedState<ProgressState<T>>(jobID, () => ({
    progress: initialProgress, initial: initialProgress, interval: reloadInterval,
    updatedAt: initialProgress?.updatedAt ?? null, revision: {}, emptyPolls: 0,
    hasSeenProgress: Boolean(initialProgress),
    isPolling: !stopAtTerminal || !isTerminal(initialProgress), status: 'idle',
  }));
  let current = state;
  if (state.initial !== initialProgress || state.interval !== reloadInterval) {
    current = { ...state, initial: initialProgress, interval: reloadInterval };
    if (state.initial !== initialProgress) {
      if (initialProgress) {
        const age = compareTimestamp(initialProgress.updatedAt, state.updatedAt);
        // Equal timestamps are common in HF's second-resolution writer. A parent
        // snapshot may advance its own value, but cannot replace a newer polled value.
        if (age > 0 || (age === 0 && state.progress !== null && state.progress === state.initial)) {
          current = {
            ...current, progress: initialProgress, updatedAt: initialProgress.updatedAt, revision: {},
            emptyPolls: 0, hasSeenProgress: true,
            isPolling: !stopAtTerminal || !isTerminal(initialProgress), status: 'success',
          };
        }
      } else if (state.initial !== null) {
        // Keep the timestamp watermark after expiry so a cached response cannot
        // resurrect the old band. A genuinely newer operation is still accepted.
        current = { ...current, progress: null, revision: {}, isPolling: false, status: 'success' };
      }
    }
    if (state.interval !== reloadInterval) {
      current = { ...current, emptyPolls: 0, isPolling: !stopAtTerminal || !isTerminal(current.progress) };
    }
    setState(current);
  }

  const visit = useMemo(() => ({ jobID }), [jobID]);
  const activeVisit = useCommittedRef(visit);
  const committedState = useCommittedRef(current);
  const mounted = useRef(false);
  const requestGeneration = useRef(0);
  const refreshProgress = useCallback((signal?: AbortSignal) => {
    if (signal?.aborted || !mounted.current || activeVisit.current !== visit) return;
    const generation = ++requestGeneration.current;
    const requestedRevision = committedState.current.revision;
    const isCurrent = () => !signal?.aborted && mounted.current && activeVisit.current === visit &&
      generation === requestGeneration.current;
    setState(previous => previous.status === 'idle' ? { ...previous, status: 'loading' } : previous);
    return apiClient.get<unknown>(`/api/jobs/${jobID}/${endpoint}`, { signal }).then(({ data }) => {
      if (!isCurrent()) return;
      if (!isRecord(data) || !Object.hasOwn(data, 'progress') || (data.progress !== null && !isProgress(data.progress))) {
        throw new Error('Invalid job progress response');
      }
      const nextProgress = data.progress;
      setState(previous => {
        if (!isCurrent()) return previous;
        if (nextProgress !== null) {
          const age = compareTimestamp(nextProgress.updatedAt, previous.updatedAt);
          if (age < 0 || (age === 0 && (previous.progress === null || previous.revision !== requestedRevision))) {
            return { ...previous, status: 'success' };
          }
          return {
            ...previous, progress: nextProgress, updatedAt: nextProgress.updatedAt, revision: {},
            emptyPolls: 0, hasSeenProgress: true, status: 'success',
            isPolling: !stopAtTerminal || !isTerminal(nextProgress),
          };
        }
        // An empty result has no timestamp. Only the observation that issued the
        // request can be expired by it; a newer parent snapshot takes precedence.
        if (previous.revision !== requestedRevision) return previous;
        const emptyPolls = previous.emptyPolls + 1;
        return {
          ...previous, progress: null, revision: {}, emptyPolls, status: 'success',
          isPolling: !previous.hasSeenProgress && emptyPolls < MAX_EMPTY_POLLS,
        };
      });
    }).catch(() => {
      if (!isCurrent()) return;
      setState(previous => isCurrent() && previous.revision === requestedRevision ? { ...previous, status: 'error' } : previous);
    });
  }, [jobID, endpoint, isProgress, stopAtTerminal, visit, activeVisit, committedState, setState]);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      requestGeneration.current += 1;
    };
  }, [visit]);

  return { progress: current.progress, status: current.status, isPolling: current.isPolling, refreshProgress };
}
