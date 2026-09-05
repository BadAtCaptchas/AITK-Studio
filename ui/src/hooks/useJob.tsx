'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { Job } from '@/types';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';
export default function useJob(jobID: string, reloadInterval: null | number = null) {
  const [job, setJob] = useState<Job | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const requestScope = jobID;
  const activeScopeRef = useRef(requestScope);
  activeScopeRef.current = requestScope;
  const refreshJob = useCallback(
    (signal?: AbortSignal) => {
      if (!jobID) return;
      const currentRequestScope = requestScope;
      if (activeScopeRef.current !== currentRequestScope) return;
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
          if (signal?.aborted || activeScopeRef.current !== currentRequestScope) return;
          setJob(data);
          setStatus('success');
        })
        .catch(error => {
          if (signal?.aborted || activeScopeRef.current !== currentRequestScope) return;
          console.error('Error fetching training run:', error);
          setStatus('error');
        });
    },
    [jobID, requestScope],
  );
  useEffect(() => {
    setJob(null);
    setStatus('idle');
  }, [jobID]);
  usePollLoop(signal => refreshJob(signal), reloadInterval, [jobID]);
  return { job, setJob, status, refreshJob: () => refreshJob() };
}
