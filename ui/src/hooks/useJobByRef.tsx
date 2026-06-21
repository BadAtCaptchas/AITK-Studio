'use client';

import { useCallback, useEffect, useState } from 'react';
import type { Job } from '@/types';
import { apiClient } from '@/utils/api';

export default function useJobByRef(
  jobRef: string | null,
  reloadInterval: null | number = null,
  jobType: string | null = null,
  projectID: string | null = null,
) {
  const [job, setJob] = useState<Job | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshJob = useCallback(() => {
    if (!jobRef) {
      setJob(null);
      setStatus('idle');
      return;
    }
    setStatus('loading');
    const params = new URLSearchParams({ job_ref: jobRef });
    if (jobType) params.set('job_type', jobType);
    if (projectID) params.set('project_id', projectID);
    apiClient
      .get(`/api/jobs?${params.toString()}`)
      .then(res => res.data)
      .then(data => {
        setJob(data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching job:', error);
        setJob(null);
        setStatus('error');
      });
  }, [jobRef, jobType, projectID]);

  useEffect(() => {
    refreshJob();

    if (reloadInterval) {
      const interval = setInterval(refreshJob, reloadInterval);

      return () => {
        clearInterval(interval);
      };
    }
  }, [refreshJob, reloadInterval]);

  return { job, setJob, status, refreshJob };
}
