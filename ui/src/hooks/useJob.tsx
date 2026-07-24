'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { Job } from '@/types';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';

export default function useJob(
  jobID: string,
  reloadInterval: null | number = null,
  projectID?: string | null,
) {
  const [job, setJob] = useState<Job | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const requestScope = `${jobID}\0${projectID ?? ''}`;
  const activeScopeRef = useRef(requestScope);
  activeScopeRef.current = requestScope;

  const refreshJob = useCallback((signal?: AbortSignal) => {
    if (projectID === null) return;
    const currentRequestScope = requestScope;
    if (activeScopeRef.current !== currentRequestScope) return;
    setStatus('loading');
    return apiClient
      .get('/api/jobs', {
        params: {
          id: jobID,
          ...(projectID ? { scope: 'project', project_id: projectID } : {}),
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
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  }, [jobID, projectID, requestScope]);

  useEffect(() => {
    setJob(null);
    setStatus('idle');
  }, [jobID, projectID]);

  usePollLoop(signal => refreshJob(signal), reloadInterval, [jobID, projectID]);

  return { job, setJob, status, refreshJob: () => refreshJob() };
}
