'use client';
import useCommittedRef from './useCommittedRef';
import useScopedState from './useScopedState';
import { useCallback } from 'react';
import type { Job } from '@/types';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';
export default function useJobByRef(
  jobRef: string | null,
  reloadInterval: null | number = null,
  jobType: string | null = null,
) {
  const requestScope = JSON.stringify([jobRef, jobType]);
  const [job, setJob] = useScopedState<Job | null>(requestScope, null);
  const [status, setStatus] = useScopedState<'idle' | 'loading' | 'success' | 'error'>(requestScope, 'idle');
  const activeScopeRef = useCommittedRef(requestScope);
  const refreshJob = useCallback(
    (signal?: AbortSignal) => {
      if (!jobRef) {
        setJob(null);
        setStatus('idle');
        return;
      }
      const currentRequestScope = requestScope;
      setStatus('loading');
      const params = new URLSearchParams({ job_ref: jobRef });
      if (jobType) params.set('job_type', jobType);
      return apiClient
        .get(`/api/jobs?${params.toString()}`, { signal })
        .then(res => res.data)
        .then(data => {
          if (signal?.aborted || activeScopeRef.current !== currentRequestScope) return;
          setJob(data);
          setStatus('success');
        })
        .catch(error => {
          if (signal?.aborted || activeScopeRef.current !== currentRequestScope) return;
          console.error('Error fetching job:', error);
          setJob(null);
          setStatus('error');
        });
    },
    [jobRef, jobType, requestScope, setJob, setStatus],
  );
  usePollLoop(signal => refreshJob(signal), reloadInterval, [jobRef, jobType]);
  return { job, setJob, status, refreshJob: () => refreshJob() };
}
