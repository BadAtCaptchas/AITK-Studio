'use client';

import { useCallback, useState } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';

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

  const refresh = useCallback(async (signal?: AbortSignal) => {
    setStatus(current => (current === 'idle' ? 'loading' : current));
    try {
      const response = await apiClient.get<TensorBoardStatus>('/api/tensorboard', { signal });
      if (signal?.aborted) return;
      setTensorBoardStatus(response.data);
      setStatus('success');
    } catch {
      if (signal?.aborted) return;
      setStatus('error');
    }
  }, []);

  usePollLoop(signal => refresh(signal), refreshMs > 0 ? refreshMs : null);

  return { tensorBoardStatus, status, refresh: () => refresh() };
}
