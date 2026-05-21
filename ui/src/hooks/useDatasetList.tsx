'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';
import type { DatasetSummary } from '@/types';

export default function useDatasetList() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshDatasets = () => {
    setStatus('loading');
    apiClient
      .get('/api/datasets/list')
      .then(res => res.data)
      .then(data => {
        console.log('Datasets:', data);
        const normalized: DatasetSummary[] = (Array.isArray(data) ? data : []).map(item =>
          typeof item === 'string' ? { name: item, encrypted: false } : item,
        );
        normalized.sort((a, b) => a.name.localeCompare(b.name));
        setDatasets(normalized);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  };
  useEffect(() => {
    refreshDatasets();
  }, []);

  return { datasets, setDatasets, status, refreshDatasets };
}
