'use client';

import { useEffect, useState } from 'react';
import type { RemoteOllamaWorker } from '@/types';
import { apiClient } from '@/utils/api';

export default function useRemoteOllamaWorkers() {
  const [workers, setWorkers] = useState<RemoteOllamaWorker[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshWorkers = () => {
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
  };

  useEffect(() => {
    refreshWorkers();
  }, []);

  return { workers, setWorkers, status, refreshWorkers };
}
