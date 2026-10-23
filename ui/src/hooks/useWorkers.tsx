'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { WorkerNode } from '@/types';
import { apiClient } from '@/utils/api';

export default function useWorkers() {
  const [workers, setWorkerData] = useState<WorkerNode[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('loading');
  const mounted = useRef(false);
  const requestGeneration = useRef(0);
  const activeRequest = useRef<AbortController | null>(null);
  const setWorkers = useCallback<Dispatch<SetStateAction<WorkerNode[]>>>(update => {
    if (mounted.current) setWorkerData(update);
  }, []);

  const loadWorkers = useCallback(() => {
    if (!mounted.current) return;
    const generation = ++requestGeneration.current;
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    const isCurrent = () => mounted.current && !controller.signal.aborted && generation === requestGeneration.current;
    setStatus('loading');
    return apiClient
      .get('/api/workers', { signal: controller.signal })
      .then(res => res.data)
      .then(data => {
        if (!isCurrent()) return;
        setWorkerData(data.workers || []);
        setStatus('success');
      })
      .catch(error => {
        if (!isCurrent()) return;
        console.error('Error fetching workers:', error);
        setStatus('error');
      })
      .finally(() => {
        if (activeRequest.current === controller) activeRequest.current = null;
      });
  }, []);

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
