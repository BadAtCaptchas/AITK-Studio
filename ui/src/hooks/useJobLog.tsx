'use client';

import { useCallback, useEffect, useState, useRef } from 'react';
import { apiClient } from '@/utils/api';

const clean = (text: string): string => {
  // remove \x1B[A\x1B[A
  text = text.replace(/\x1B\[A/g, '');
  return text;
};

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
  const didInitialLoadRef = useRef(false);
  const offsetRef = useRef<number | null>(null);
  const isRefreshingRef = useRef(false);
  const activeJobIDRef = useRef(jobID);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const refresh = useCallback(() => {
    if (isRefreshingRef.current) return;
    isRefreshingRef.current = true;
    const requestJobID = jobID;
    let loadStatus: 'loading' | 'refreshing' = 'loading';
    if (didInitialLoadRef.current) {
      loadStatus = 'refreshing';
    }
    setStatus(loadStatus);
    const offset = offsetRef.current;
    apiClient
      .get(`/api/jobs/${jobID}/log`, offset !== null ? { params: { offset } } : undefined)
      .then(res => res.data)
      .then(data => {
        if (activeJobIDRef.current !== requestJobID) return;
        const payload = parseJobLogResponse(data);
        offsetRef.current = payload.offset;
        const cleanLog = clean(payload.log);
        if (payload.reset) {
          setLog(cleanLog);
        } else if (cleanLog) {
          setLog(previous => previous + cleanLog);
        }
        setStatus('success');
        didInitialLoadRef.current = true;
      })
      .catch(error => {
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
    setLog('');
    refresh();

    if (reloadInterval) {
      const interval = setInterval(refresh, reloadInterval);

      return () => {
        clearInterval(interval);
      };
    }
  }, [jobID, reloadInterval, refresh]);

  return { log, setLog, status, refresh };
}
