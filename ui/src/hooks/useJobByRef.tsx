'use client';
import useCommittedRef from './useCommittedRef';
import useScopedState from './useScopedState';
import { useCallback, useEffect, useMemo, useRef } from 'react';
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
  const visit = useMemo(() => ({ requestScope }), [requestScope]);
  const activeVisit = useCommittedRef(visit);
  const mounted = useRef(false);
  const requestGeneration = useRef(0);
  const refreshJob = useCallback(
    (signal?: AbortSignal) => {
      if (signal?.aborted || !mounted.current || activeVisit.current !== visit) return;
      const generation = ++requestGeneration.current;
      if (!jobRef) {
        setJob(null);
        setStatus('idle');
        return;
      }
      setStatus('loading');
      const params = new URLSearchParams({ job_ref: jobRef });
      if (jobType) params.set('job_type', jobType);
      return apiClient
        .get(`/api/jobs?${params.toString()}`, { signal })
        .then(res => res.data)
        .then(data => {
          if (signal?.aborted || !mounted.current || activeVisit.current !== visit ||
              generation !== requestGeneration.current) return;
          setJob(data);
          setStatus('success');
        })
        .catch(error => {
          if (signal?.aborted || !mounted.current || activeVisit.current !== visit ||
              generation !== requestGeneration.current) return;
          console.error('Error fetching job:', error);
          // A failed lookup does not establish that a known caption job disappeared.
          setStatus('error');
        });
    },
    [jobRef, jobType, visit, activeVisit, setJob, setStatus],
  );
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      requestGeneration.current += 1;
    };
  }, [visit]);
  usePollLoop(signal => refreshJob(signal), reloadInterval, [jobRef, jobType]);
  return { job, setJob, status, refreshJob: () => refreshJob() };
}
