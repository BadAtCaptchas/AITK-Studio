'use client';

import { useEffect, useState } from 'react';
import type { WorkerNode } from '@/types';
import { apiClient } from '@/utils/api';

export default function useWorkers() {
  const [workers, setWorkers] = useState<WorkerNode[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('loading');

  const loadWorkers = () => {
    apiClient
      .get('/api/workers')
      .then(res => res.data)
      .then(data => {
        setWorkers(data.workers || []);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching workers:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    loadWorkers();
  }, []);

  const refreshWorkers = () => {
    setStatus('loading');
    loadWorkers();
  };

  return { workers, setWorkers, status, refreshWorkers };
}
