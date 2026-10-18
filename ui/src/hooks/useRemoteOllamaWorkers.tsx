'use client';

import { useCallback, useEffect, useState } from 'react';
import useScopedState from './useScopedState';
import type { RemoteOllamaWorker } from '@/types';
import { apiClient } from '@/utils/api';

type UseRemoteOllamaWorkersOptions = {
  enabled?: boolean;
};

export default function useRemoteOllamaWorkers(options: UseRemoteOllamaWorkersOptions = {}) {
  const enabled = options.enabled !== false;
  const [workers, setWorkers] = useState<RemoteOllamaWorker[]>([]);
  const [status, setStatus] = useScopedState<'idle' | 'loading' | 'success' | 'error'>(enabled, enabled ? 'loading' : 'idle');

  const loadWorkers = useCallback(() => {
    if (!enabled) return;
    apiClient
      .get('/api/ollama-workers')
      .then(res => res.data)
      .then(data => {
        setWorkers(data.workers || []);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching Remote Ollama workers:', error);
        setStatus('error');
      });
  }, [enabled, setStatus]);

  useEffect(() => {
    loadWorkers();
  }, [loadWorkers]);

  const refreshWorkers = () => {
    if (!enabled) return;
    setStatus('loading');
    loadWorkers();
  };

  return { workers, setWorkers, status, refreshWorkers };
}
