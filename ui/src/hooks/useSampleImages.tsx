'use client';
import useCommittedRef from './useCommittedRef';
import useScopedState from './useScopedState';

import { useCallback } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';

export default function useSampleImages(jobID: string, reloadInterval: null | number = null) {
  const [sampleImages, setSampleImages] = useScopedState<string[]>(jobID, []);
  const activeJobIDRef = useCommittedRef(jobID);
  const [status, setStatus] = useScopedState<'idle' | 'loading' | 'success' | 'error'>(jobID, 'idle');

  const refreshSampleImages = useCallback(
    (signal?: AbortSignal) => {
      const requestJobID = jobID;
      if (activeJobIDRef.current !== requestJobID) return;
      setStatus('loading');
      return apiClient
        .get<unknown>(`/api/jobs/${encodeURIComponent(jobID)}/samples`, { signal })
        .then(res => res.data)
        .then(data => {
          if (signal?.aborted || activeJobIDRef.current !== requestJobID) return;
          if (
            !data ||
            typeof data !== 'object' ||
            !('samples' in data) ||
            !Array.isArray(data.samples) ||
            !data.samples.every((sample: unknown): sample is string => typeof sample === 'string')
          )
            throw new Error('Invalid sample response');
          setSampleImages(data.samples);
          setStatus('success');
        })
        .catch(error => {
          if (signal?.aborted || activeJobIDRef.current !== requestJobID) return;
          console.error('Error fetching samples:', error);
          setStatus('error');
        });
    },
    [jobID, setStatus, setSampleImages],
  );

  usePollLoop(signal => refreshSampleImages(signal), reloadInterval, [jobID]);

  return { sampleImages, setSampleImages, status, refreshSampleImages: () => refreshSampleImages() };
}
