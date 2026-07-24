'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { Job } from '@/types';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';

type UseJobsListProps = {
  onlyActive?: boolean;
  reloadInterval?: number | null;
  job_type?: string | null;
  projectID?: string | null;
  scope?: 'global' | 'all' | 'project';
  includeProjectActive?: boolean;
};

export default function useJobsList({
  onlyActive = false,
  reloadInterval = null,
  job_type = null,
  projectID = null,
  scope,
  includeProjectActive = false,
}: UseJobsListProps = {}) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const requestScope = JSON.stringify([job_type, projectID, scope, onlyActive, includeProjectActive]);
  const activeScopeRef = useRef(requestScope);
  activeScopeRef.current = requestScope;

  const refreshJobs = useCallback((signal?: AbortSignal) => {
    const currentRequestScope = requestScope;
    if (activeScopeRef.current !== currentRequestScope) return;
    setStatus('loading');
    return apiClient
      .get('/api/jobs', {
        params: {
          ...(job_type ? { job_type } : {}),
          ...(projectID ? { project_id: projectID } : {}),
          ...(scope ? { scope } : {}),
          ...(includeProjectActive && !projectID ? { include_project_active: '1' } : {}),
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
  }, [includeProjectActive, job_type, onlyActive, projectID, requestScope, scope]);

  useEffect(() => {
    setJobs([]);
    setStatus('idle');
  }, [includeProjectActive, job_type, onlyActive, projectID, scope]);

  usePollLoop(signal => refreshJobs(signal), reloadInterval, [
    job_type,
    projectID,
    scope,
    onlyActive,
    includeProjectActive,
  ]);

  return { jobs, setJobs, status, refreshJobs: () => refreshJobs() };
}
