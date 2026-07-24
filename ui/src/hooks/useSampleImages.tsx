'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';

export default function useSampleImages(jobID: string, reloadInterval: null | number = null) {
  const [sampleImages, setSampleImages] = useState<string[]>([]);
  const activeJobIDRef = useRef(jobID);
  activeJobIDRef.current = jobID;
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshSampleImages = useCallback((signal?: AbortSignal) => {
    const requestJobID = jobID;
    if (activeJobIDRef.current !== requestJobID) return;
    setStatus('loading');
    return apiClient
      .get(`/api/jobs/${jobID}/samples`, { signal })
      .then(res => res.data)
      .then(data => {
        if (signal?.aborted || activeJobIDRef.current !== requestJobID) return;
        console.log('Fetched sample images:', data);
        if (data.samples) {
          setSampleImages(data.samples);
        }
        setStatus('success');
      })
      .catch(error => {
        if (signal?.aborted || activeJobIDRef.current !== requestJobID) return;
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  }, [jobID]);

  useEffect(() => {
    setSampleImages([]);
    setStatus('idle');
  }, [jobID]);

  usePollLoop(signal => refreshSampleImages(signal), reloadInterval, [jobID]);

  return { sampleImages, setSampleImages, status, refreshSampleImages: () => refreshSampleImages() };
}
