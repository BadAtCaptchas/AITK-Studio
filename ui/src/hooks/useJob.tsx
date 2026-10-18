'use client';
import useCommittedRef from './useCommittedRef';
import useScopedState from './useScopedState';
import { useCallback, useEffect, useRef } from 'react';
import type { Job } from '@/types';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';
export default function useJob(jobID: string, reloadInterval: null | number = null) {
  const [job, setJob] = useScopedState<Job | null>(jobID, null);
  const [status, setStatus] = useScopedState<'idle' | 'loading' | 'success' | 'error'>(jobID, 'idle');
  const requestGeneration = useRef(0);
  const requestScope = jobID;
  const activeScopeRef = useCommittedRef(requestScope);
  const refreshJob = useCallback(
    (signal?: AbortSignal) => {
      if (!jobID) return;
      const currentRequestScope = requestScope;
      if (activeScopeRef.current !== currentRequestScope) return;
      const generation = ++requestGeneration.current;
      setStatus('loading');
      return apiClient
        .get('/api/jobs', {
          params: {
            id: jobID,
          },
          signal,
        })
        .then(res => res.data)
        .then(data => {
          if (
            signal?.aborted ||
            activeScopeRef.current !== currentRequestScope ||
            generation !== requestGeneration.current
          )
            return;
          setJob(data);
          setStatus('success');
        })
        .catch(error => {
          if (
            signal?.aborted ||
            activeScopeRef.current !== currentRequestScope ||
            generation !== requestGeneration.current
          )
            return;
          console.error('Error fetching training run:', error);
          setStatus('error');
        });
    },
    [jobID, requestScope, setStatus, setJob],
  );
  useEffect(() => {
    return () => {
      requestGeneration.current += 1;
    };
  }, [jobID]);
  usePollLoop(signal => refreshJob(signal), reloadInterval, [jobID]);
  return { job, setJob, status, refreshJob: () => refreshJob() };
}
