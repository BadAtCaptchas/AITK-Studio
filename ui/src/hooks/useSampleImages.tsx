'use client';
import useCommittedRef from './useCommittedRef';
import useScopedState from './useScopedState';

import { useCallback, useEffect, useRef } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';
import { parseSampleMetadataMap, type SampleMetadata } from '@/utils/sampleMetadata';

export default function useSampleImages(jobID: string, reloadInterval: null | number = null, includeMetadata = false) {
  const [sampleImages, setSampleImages] = useScopedState<string[]>(jobID, []);
  const [sampleMetadata, setSampleMetadata] = useScopedState<Record<string, SampleMetadata>>(jobID, {});
  const requestScope = JSON.stringify([jobID, includeMetadata]);
  const activeScopeRef = useCommittedRef(requestScope);
  const requestGeneration = useRef(0);
  const [status, setStatus] = useScopedState<'idle' | 'loading' | 'success' | 'error'>(jobID, 'idle');

  const refreshSampleImages = useCallback(
    (signal?: AbortSignal) => {
      const currentRequestScope = requestScope;
      if (signal?.aborted || activeScopeRef.current !== currentRequestScope) return;
      const generation = ++requestGeneration.current;
      setStatus('loading');
      return apiClient
        .get<unknown>(`/api/jobs/${encodeURIComponent(jobID)}/samples${includeMetadata ? '?metadata=1' : ''}`, { signal })
        .then(res => res.data)
        .then(data => {
          if (
            signal?.aborted ||
            activeScopeRef.current !== currentRequestScope ||
            generation !== requestGeneration.current
          )
            return;
          if (
            !data ||
            typeof data !== 'object' ||
            !('samples' in data) ||
            !Array.isArray(data.samples) ||
            !data.samples.every((sample: unknown): sample is string => typeof sample === 'string')
          )
            throw new Error('Invalid sample response');
          setSampleImages(data.samples);
          setSampleMetadata(parseSampleMetadataMap('metadata' in data ? data.metadata : undefined, data.samples));
          setStatus('success');
        })
        .catch(error => {
          if (
            signal?.aborted ||
            activeScopeRef.current !== currentRequestScope ||
            generation !== requestGeneration.current
          )
            return;
          console.error('Error fetching samples:', error);
          setStatus('error');
        });
    },
    [jobID, includeMetadata, requestScope, setStatus, setSampleImages, setSampleMetadata],
  );

  useEffect(() => {
    return () => {
      requestGeneration.current += 1;
    };
  }, [requestScope]);

  usePollLoop(signal => refreshSampleImages(signal), reloadInterval, [jobID, includeMetadata]);

  return { sampleImages, sampleMetadata, setSampleImages, status, refreshSampleImages: () => refreshSampleImages() };
}
