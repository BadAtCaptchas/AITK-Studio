'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';
import type { DatasetSummary } from '@/types';

type DatasetListError = {
  worker_id: string;
  worker_name: string;
  error: string;
};

type UseDatasetListOptions = {
  includeRemote?: boolean;
  workerID?: string;
  projectID?: string | null;
};

export default function useDatasetList(options: UseDatasetListOptions = {}) {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [errors, setErrors] = useState<DatasetListError[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const includeRemote = options.includeRemote === true;
  const workerID = options.workerID || 'local';
  const projectID = options.projectID || null;

  const refreshDatasets = () => {
    setStatus('loading');
    const params = new URLSearchParams();
    if (includeRemote) params.set('include_remote', '1');
    if (workerID !== 'local') params.set('worker_id', workerID);
    if (projectID) params.set('project_id', projectID);
    const query = params.toString();
    apiClient
      .get(`/api/datasets/list${query ? `?${query}` : ''}`)
      .then(res => res.data)
      .then(data => {
        console.log('Datasets:', data);
        const rawDatasets = Array.isArray(data) ? data : Array.isArray(data?.datasets) ? data.datasets : [];
        const normalized: DatasetSummary[] = rawDatasets.map((item: DatasetSummary | string) =>
          typeof item === 'string' ? { name: item, encrypted: false } : item,
        );
        normalized.sort((a, b) => {
          if ((a.source || 'local') !== (b.source || 'local')) {
            return (a.source || 'local') === 'local' ? -1 : 1;
          }
          const workerCompare = (a.worker_name || '').localeCompare(b.worker_name || '');
          return workerCompare || a.name.localeCompare(b.name);
        });
        setErrors(Array.isArray(data?.errors) ? data.errors : []);
        setDatasets(normalized);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching datasets:', error);
        setErrors([]);
        setStatus('error');
      });
  };
  useEffect(() => {
    refreshDatasets();
  }, [includeRemote, workerID, projectID]);

  return { datasets, setDatasets, errors, status, refreshDatasets };
}
