'use client';

import { useCallback, useEffect, useState } from 'react';
import type { RemoteOllamaWorker } from '@/types';
import { apiClient } from '@/utils/api';

type UseRemoteOllamaWorkersOptions = {
  enabled?: boolean;
};

export default function useRemoteOllamaWorkers(options: UseRemoteOllamaWorkersOptions = {}) {
  const enabled = options.enabled !== false;
  const [workers, setWorkers] = useState<RemoteOllamaWorker[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshWorkers = useCallback(() => {
    if (!enabled) return;
    setStatus('loading');
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
  }, [enabled]);

  useEffect(() => {
    refreshWorkers();
  }, [refreshWorkers]);

  return { workers, setWorkers, status, refreshWorkers };
}
