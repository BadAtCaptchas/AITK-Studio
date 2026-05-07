'use client';

import { useCallback, useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export type TensorBoardStatus = {
  enabled: boolean;
  running: boolean;
  port: number;
  url: string | null;
  logDir: string | null;
  pid: number | null;
  source: 'managed' | 'external' | null;
  error?: string;
};

export default function useTensorBoardStatus(refreshMs = 30000) {
  const [tensorBoardStatus, setTensorBoardStatus] = useState<TensorBoardStatus | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refresh = useCallback(async () => {
    setStatus(current => (current === 'idle' ? 'loading' : current));
    try {
      const response = await apiClient.get<TensorBoardStatus>('/api/tensorboard');
      setTensorBoardStatus(response.data);
      setStatus('success');
    } catch (error) {
      setStatus('error');
    }
  }, []);

  useEffect(() => {
    let isMounted = true;

    const load = async () => {
      if (!isMounted) {
        return;
      }
      await refresh();
    };

    void load();
    if (refreshMs <= 0) {
      return () => {
        isMounted = false;
      };
    }

    const interval = window.setInterval(load, refreshMs);
    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, [refresh, refreshMs]);

  return { tensorBoardStatus, status, refresh };
}
