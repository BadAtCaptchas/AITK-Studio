'use client';

import { useCallback, useState } from 'react';
import type { Queue } from '@/types';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';

export default function useQueueList(reloadInterval: number | null = null) {
  const [queues, setQueues] = useState<Queue[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const fetchQueues = useCallback((signal?: AbortSignal) => {
    setStatus('loading');
    return apiClient
      .get('/api/queue', { signal })
      .then(res => res.data)
      .then(data => {
        if (signal?.aborted) return;
        if (data.error) {
          setStatus('error');
        } else {
          setQueues(data.queues);
          setStatus('success');
        }
      })
      .catch(() => {
        if (signal?.aborted) return;
        setStatus('error');
      });
  }, []);
  usePollLoop(signal => fetchQueues(signal), reloadInterval);

  return { queues, setQueues, status, refreshQueues: () => fetchQueues() };
}
