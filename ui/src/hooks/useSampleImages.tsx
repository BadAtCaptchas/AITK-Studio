'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';

export default function useSampleImages(jobID: string, reloadInterval: null | number = null) {
  const [sampleImages, setSampleImages] = useState<string[]>([]);
  const activeJobIDRef = useRef(jobID);
  activeJobIDRef.current = jobID;
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

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
    [jobID],
  );

  useEffect(() => {
    setSampleImages([]);
    setStatus('idle');
  }, [jobID]);

  usePollLoop(signal => refreshSampleImages(signal), reloadInterval, [jobID]);

  return { sampleImages, setSampleImages, status, refreshSampleImages: () => refreshSampleImages() };
}
