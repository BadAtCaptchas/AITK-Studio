'use client';
import { useEffect } from 'react';
import { apiClient } from '@/utils/api';
import useScopedState from './useScopedState';
import type { DatasetSummary } from '@/types';
type DatasetListError = {
  worker_id: string;
  worker_name: string;
  error: string;
};
type UseDatasetListOptions = {
  includeRemote?: boolean;
  workerID?: string;
};
export default function useDatasetList(options: UseDatasetListOptions = {}) {
  const includeRemote = options.includeRemote === true;
  const workerID = options.workerID || 'local';
  const scope = JSON.stringify([includeRemote, workerID]);
  const [datasets, setDatasets] = useScopedState<DatasetSummary[]>(scope, []);
  const [errors, setErrors] = useScopedState<DatasetListError[]>(scope, []);
  const [status, setStatus] = useScopedState<'idle' | 'loading' | 'success' | 'error'>(scope, 'loading');
  const loadDatasets = (
    refreshOptions: {
      background?: boolean;
      signal?: AbortSignal;
    } = {},
  ) => {
    const params = new URLSearchParams();
    if (includeRemote) params.set('include_remote', '1');
    if (workerID !== 'local') params.set('worker_id', workerID);
    const query = params.toString();
    return apiClient
      .get(`/api/datasets/list${query ? `?${query}` : ''}`, { signal: refreshOptions.signal })
      .then(res => res.data)
      .then(data => {
        if (refreshOptions.signal?.aborted) return;
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
        if (refreshOptions.signal?.aborted) return;
        console.error('Error fetching datasets:', error);
        setErrors([]);
        if (!refreshOptions.background) setStatus('error');
      });
  };
  useEffect(() => {
    const controller = new AbortController();
    void loadDatasets({ signal: controller.signal });
    return () => controller.abort();
  }, [includeRemote, workerID]);
  const refreshDatasets = (options: { background?: boolean } = {}) => {
    if (!options.background) setStatus('loading');
    return loadDatasets(options);
  };
  return { datasets, setDatasets, errors, status, refreshDatasets };
}
