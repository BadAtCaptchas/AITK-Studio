'use client';

import { useCallback, useEffect, useState, useRef } from 'react';
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

export default function useJobLog(jobID: string, reloadInterval: null | number = null) {
  const [log, setLog] = useState<string>('');
  const terminalRef = useRef<TerminalEmulator | null>(null);
  if (terminalRef.current === null) {
    terminalRef.current = new TerminalEmulator();
  }
  const didInitialLoadRef = useRef(false);
  const offsetRef = useRef<number | null>(null);
  const isRefreshingRef = useRef(false);
  const activeJobIDRef = useRef(jobID);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const refresh = useCallback((signal?: AbortSignal) => {
    if (isRefreshingRef.current) return;
    isRefreshingRef.current = true;
    const requestJobID = jobID;
    let loadStatus: 'loading' | 'refreshing' = 'loading';
    if (didInitialLoadRef.current) {
      loadStatus = 'refreshing';
    }
    setStatus(loadStatus);
    const offset = offsetRef.current;
    return apiClient
      .get(`/api/jobs/${jobID}/log`, {
        ...(offset !== null ? { params: { offset } } : {}),
        signal,
      })
      .then(res => res.data)
      .then(data => {
        if (activeJobIDRef.current !== requestJobID) return;
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
        if (signal?.aborted) return;
        if (activeJobIDRef.current !== requestJobID) return;
        console.error('Error fetching log:', error);
        setStatus('error');
      })
      .finally(() => {
        if (activeJobIDRef.current === requestJobID) {
          isRefreshingRef.current = false;
        }
      });
  }, [jobID]);

  useEffect(() => {
    activeJobIDRef.current = jobID;
    isRefreshingRef.current = false;
    offsetRef.current = null;
    didInitialLoadRef.current = false;
    terminalRef.current?.reset();
    setLog('');
  }, [jobID]);

  usePollLoop(signal => refresh(signal), reloadInterval, [jobID]);

  return { log, status, refresh: () => refresh() };
}
