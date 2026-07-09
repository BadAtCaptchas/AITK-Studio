'use client';

import { useEffect, useState } from 'react';
import type { Queue } from '@/types';
import { apiClient } from '@/utils/api';

export default function useQueueList() {
  const [queues, setQueues] = useState<Queue[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshQueues = () => {
    setStatus('loading');
    apiClient
      .get('/api/queue')
      .then(res => res.data)
      .then(data => {
        if (data.error) {
          setStatus('error');
        } else {
          setQueues(data.queues);
          setStatus('success');
        }
      })
      .catch(() => {
        setStatus('error');
      });
  };
  useEffect(() => {
    refreshQueues();
  }, []);

  return { queues, setQueues, status, refreshQueues };
}
