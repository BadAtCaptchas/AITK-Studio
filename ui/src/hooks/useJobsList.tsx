'use client';

import { useEffect, useState } from 'react';
import type { Job } from '@/types';
import { apiClient } from '@/utils/api';

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

  const refreshJobs = () => {
    setStatus('loading');
    apiClient
      .get('/api/jobs', {
        params: {
          ...(job_type ? { job_type } : {}),
          ...(projectID ? { project_id: projectID } : {}),
          ...(scope ? { scope } : {}),
          ...(includeProjectActive && !projectID ? { include_project_active: '1' } : {}),
        },
      })
      .then(res => res.data)
      .then(data => {
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
        setStatus('error');
      });
  };
  useEffect(() => {
    refreshJobs();

    if (reloadInterval) {
      const interval = setInterval(() => {
        refreshJobs();
      }, reloadInterval);
      return () => clearInterval(interval);
    }
  }, [job_type, projectID, scope, onlyActive, reloadInterval, includeProjectActive]);

  return { jobs, setJobs, status, refreshJobs };
}
