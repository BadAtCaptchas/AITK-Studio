'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';
import type { EvaluationRun } from '@/types';

export default function useEvaluations(reloadInterval: number | null = 5000) {
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const refresh = () => {
    setStatus(prev => (prev === 'success' ? 'refreshing' : 'loading'));
    apiClient
      .get('/api/evaluations')
      .then(res => res.data)
      .then(data => {
        setRuns(data.runs ?? []);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching evaluations:', error);
        setStatus('error');
      });
  };

  const createRun = async (payload: { name?: string; jobIds: string[]; referencePath?: string }) => {
    const run = await apiClient.post('/api/evaluations', payload).then(res => res.data);
    await refresh();
    return run;
  };

  useEffect(() => {
    refresh();
    if (!reloadInterval) return;
    const id = window.setInterval(refresh, reloadInterval);
    return () => window.clearInterval(id);
  }, [reloadInterval]);

  return { runs, status, refresh, createRun };
}
