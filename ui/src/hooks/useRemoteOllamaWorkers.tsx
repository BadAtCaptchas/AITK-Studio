'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import useScopedState from './useScopedState';
import useCommittedRef from './useCommittedRef';
import type { RemoteOllamaWorker } from '@/types';
import { apiClient } from '@/utils/api';

type UseRemoteOllamaWorkersOptions = {
  enabled?: boolean;
};

export default function useRemoteOllamaWorkers(options: UseRemoteOllamaWorkersOptions = {}) {
  const enabled = options.enabled !== false;
  const [workers, setWorkerData] = useState<RemoteOllamaWorker[]>([]);
  const [status, setStatus] = useScopedState<'idle' | 'loading' | 'success' | 'error'>(
    enabled,
    enabled ? 'loading' : 'idle',
  );
  const scope = useMemo(() => ({ enabled }), [enabled]);
  const activeScope = useCommittedRef(scope);
  const mounted = useRef(false);
  const requestGeneration = useRef(0);
  const activeRequest = useRef<AbortController | null>(null);
  const setWorkers = useCallback<Dispatch<SetStateAction<RemoteOllamaWorker[]>>>(
    update => {
      if (scope.enabled && mounted.current && activeScope.current === scope) setWorkerData(update);
    },
    [scope, activeScope],
  );

  const loadWorkers = useCallback(() => {
    if (!scope.enabled || !mounted.current || activeScope.current !== scope) return;
    const generation = ++requestGeneration.current;
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    const isCurrent = () =>
      mounted.current &&
      !controller.signal.aborted &&
      activeScope.current === scope &&
      generation === requestGeneration.current;
    setStatus('loading');
    return apiClient
      .get('/api/ollama-workers', { signal: controller.signal })
      .then(res => res.data)
      .then(data => {
        if (!isCurrent()) return;
        setWorkerData(data.workers || []);
        setStatus('success');
      })
      .catch(error => {
        if (!isCurrent()) return;
        console.error('Error fetching Remote Ollama workers:', error);
        setStatus('error');
      })
      .finally(() => {
        if (activeRequest.current === controller) activeRequest.current = null;
      });
  }, [scope, activeScope, setStatus]);

  useEffect(() => {
    mounted.current = true;
    void loadWorkers();
    return () => {
      mounted.current = false;
      requestGeneration.current += 1;
      activeRequest.current?.abort();
      activeRequest.current = null;
    };
  }, [loadWorkers]);

  return { workers, setWorkers, status, refreshWorkers: loadWorkers };
}
