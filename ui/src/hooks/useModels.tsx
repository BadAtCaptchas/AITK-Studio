'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';
import type { ModelArtifact } from '@/types';

export default function useModels(reloadInterval: number | null = null) {
  const [artifacts, setArtifacts] = useState<ModelArtifact[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const refresh = () => {
    setStatus(prev => (prev === 'success' ? 'refreshing' : 'loading'));
    apiClient
      .get('/api/models')
      .then(res => res.data)
      .then(data => {
        setArtifacts(data.artifacts ?? []);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching models:', error);
        setStatus('error');
      });
  };

  const reindex = async () => {
    setStatus('refreshing');
    const data = await apiClient.post('/api/models/reindex').then(res => res.data);
    setArtifacts(data.artifacts ?? []);
    setStatus('success');
  };

  useEffect(() => {
    refresh();
    if (!reloadInterval) return;
    const id = window.setInterval(refresh, reloadInterval);
    return () => window.clearInterval(id);
  }, [reloadInterval]);

  return { artifacts, status, refresh, reindex };
}
