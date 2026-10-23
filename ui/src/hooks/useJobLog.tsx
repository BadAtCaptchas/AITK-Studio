'use client';
import useScopedState from './useScopedState';
import useCommittedRef from './useCommittedRef';

import { useCallback, useEffect, useRef } from 'react';
import { apiClient } from '@/utils/api';
import { TerminalEmulator } from '@/utils/terminalEmulator';
import usePollLoop from './usePollLoop';

type JobLogResponse = {
  log: string;
  offset: number | null;
  reset: boolean;
};

function parseJobLogResponse(data: unknown): JobLogResponse {
  if (!data || typeof data !== 'object') {
    return { log: '', offset: null, reset: true };
  }

  const record = data as Record<string, unknown>;
  const offset = typeof record.offset === 'number' && Number.isFinite(record.offset) ? record.offset : null;
  return {
    log: typeof record.log === 'string' ? record.log : '',
    offset,
    reset: typeof record.reset === 'boolean' ? record.reset : offset === null,
  };
}

export default function useJobLog(
  jobID: string,
  reloadInterval: null | number = null,
  attemptId: string | null = null,
) {
  const requestScope = JSON.stringify([jobID, attemptId]);
  const [log, setLog] = useScopedState<string>(requestScope, '');
  const terminalRef = useRef<TerminalEmulator | null>(null);
  if (terminalRef.current === null) {
    terminalRef.current = new TerminalEmulator();
  }
  const didInitialLoadRef = useRef(false);
  const offsetRef = useRef<number | null>(null);
  const inFlightRef = useRef<AbortController | null>(null);
  const activeScopeRef = useCommittedRef(requestScope);
  const [status, setStatus] = useScopedState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>(
    requestScope,
    'idle',
  );

  const refresh = useCallback(
    (signal?: AbortSignal) => {
      if (!jobID || signal?.aborted || inFlightRef.current || activeScopeRef.current !== requestScope) return;
      const controller = new AbortController();
      inFlightRef.current = controller;
      const abort = () => {
        // A changed poll interval can immediately need a final one-shot request.
        if (inFlightRef.current === controller) inFlightRef.current = null;
        controller.abort();
      };
      signal?.addEventListener('abort', abort, { once: true });
      const isCurrent = () =>
        !controller.signal.aborted && inFlightRef.current === controller && activeScopeRef.current === requestScope;
      let loadStatus: 'loading' | 'refreshing' = 'loading';
      if (didInitialLoadRef.current) {
        loadStatus = 'refreshing';
      }
      setStatus(loadStatus);
      const offset = offsetRef.current;
      return apiClient
        .get(`/api/jobs/${jobID}/log`, {
          ...(offset !== null ? { params: { offset } } : {}),
          signal: controller.signal,
        })
        .then(res => res.data)
        .then(data => {
          if (!isCurrent()) return;
          const payload = parseJobLogResponse(data);
          offsetRef.current = payload.offset;
          const terminal = terminalRef.current;
          if (!terminal) return;
          if (payload.reset) {
            terminal.reset();
          }
          terminal.write(payload.log);
          setLog(terminal.toString());
          setStatus('success');
          didInitialLoadRef.current = true;
        })
        .catch(error => {
          if (!isCurrent()) return;
          console.error('Error fetching log:', error);
          setStatus('error');
        })
        .finally(() => {
          signal?.removeEventListener('abort', abort);
          if (inFlightRef.current === controller) inFlightRef.current = null;
        });
    },
    [jobID, requestScope, activeScopeRef, setStatus, setLog],
  );

  useEffect(() => {
    offsetRef.current = null;
    didInitialLoadRef.current = false;
    terminalRef.current?.reset();
    return () => {
      const controller = inFlightRef.current;
      inFlightRef.current = null;
      controller?.abort();
    };
  }, [requestScope]);

  usePollLoop(signal => refresh(signal), reloadInterval, [requestScope]);

  return { log, status, refresh: () => refresh() };
}
