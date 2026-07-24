'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { Job } from '@/types';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';

export default function useJobByRef(
  jobRef: string | null,
  reloadInterval: null | number = null,
  jobType: string | null = null,
  projectID: string | null = null,
) {
  const [job, setJob] = useState<Job | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const requestScope = JSON.stringify([jobRef, jobType, projectID]);
  const activeScopeRef = useRef(requestScope);
  activeScopeRef.current = requestScope;

  const refreshJob = useCallback((signal?: AbortSignal) => {
    if (!jobRef) {
      setJob(null);
      setStatus('idle');
      return;
    }
    const currentRequestScope = requestScope;
    setStatus('loading');
    const params = new URLSearchParams({ job_ref: jobRef });
    if (jobType) params.set('job_type', jobType);
    if (projectID) params.set('project_id', projectID);
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
  }, [jobRef, jobType, projectID, requestScope]);

  useEffect(() => {
    setJob(null);
    setStatus('idle');
  }, [jobRef, jobType, projectID]);

  usePollLoop(signal => refreshJob(signal), reloadInterval, [jobRef, jobType, projectID]);

  return { job, setJob, status, refreshJob: () => refreshJob() };
}
