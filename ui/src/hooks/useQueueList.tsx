'use client';

import { useCallback, useState } from 'react';
import type { Queue } from '@/types';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';
import { parseQueueSnapshot, type QueueFreshness } from '@/utils/queueSnapshot';

export default function useQueueList(reloadInterval: number | null = null) {
  const [queues, setQueues] = useState<Queue[]>([]);
  const [freshness, setFreshness] = useState<QueueFreshness[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const fetchQueues = useCallback((signal?: AbortSignal) => {
    setStatus('loading');
    return apiClient
      .get<unknown>('/api/queue', { signal })
      .then(res => parseQueueSnapshot(res.data))
      .then(data => {
        if (signal?.aborted) return;
        setQueues(data.queues);
        setFreshness(data.freshness);
        setStatus('success');
      })
      .catch(() => {
        if (signal?.aborted) return;
        setStatus('error');
      });
  }, []);
  usePollLoop(signal => fetchQueues(signal), reloadInterval);

  return { queues, setQueues, status, freshness, refreshQueues: () => fetchQueues() };
}
