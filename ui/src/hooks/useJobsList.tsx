'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { Job } from '@/types';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';
type UseJobsListProps = {
  onlyActive?: boolean;
  reloadInterval?: number | null;
  job_type?: string | null;
};
export default function useJobsList({
  onlyActive = false,
  reloadInterval = null,
  job_type = null,
}: UseJobsListProps = {}) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const requestScope = JSON.stringify([job_type, onlyActive]);
  const activeScopeRef = useRef(requestScope);
  activeScopeRef.current = requestScope;
  const refreshJobs = useCallback(
    (signal?: AbortSignal) => {
      const currentRequestScope = requestScope;
      if (activeScopeRef.current !== currentRequestScope) return;
      setStatus('loading');
      return apiClient
        .get('/api/jobs', {
          params: {
            ...(job_type ? { job_type } : {}),
          },
          signal,
        })
        .then(res => res.data)
        .then(data => {
          if (signal?.aborted || activeScopeRef.current !== currentRequestScope) return;
          if (data.error) {
            setStatus('error');
          } else {
            if (onlyActive) {
              data.jobs = data.jobs.filter((job: Job) => ['running', 'queued', 'stopping'].includes(job.status));
            }
            setJobs(data.jobs);
            setStatus('success');
          }
        })
        .catch(() => {
          if (signal?.aborted || activeScopeRef.current !== currentRequestScope) return;
          setStatus('error');
        });
    },
    [job_type, onlyActive, requestScope],
  );
  useEffect(() => {
    setJobs([]);
    setStatus('idle');
  }, [job_type, onlyActive]);
  usePollLoop(signal => refreshJobs(signal), reloadInterval, [job_type, onlyActive]);
  return { jobs, setJobs, status, refreshJobs: () => refreshJobs() };
}
