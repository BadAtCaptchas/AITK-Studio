'use client';

import { useCallback, useEffect, useState } from 'react';
import type { Job } from '@/types';
import { apiClient } from '@/utils/api';

export default function useJob(
  jobID: string,
  reloadInterval: null | number = null,
  projectID?: string | null,
) {
  const [job, setJob] = useState<Job | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshJob = useCallback(() => {
    if (projectID === null) return;
    setStatus('loading');
    apiClient
      .get('/api/jobs', {
        params: {
          id: jobID,
          ...(projectID ? { scope: 'project', project_id: projectID } : {}),
        },
      })
      .then(res => res.data)
      .then(data => {
        setJob(data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  }, [jobID, projectID]);

  useEffect(() => {
    if (projectID === null) {
      setJob(null);
      setStatus('idle');
      return;
    }
    refreshJob();

    if (reloadInterval) {
      const interval = setInterval(refreshJob, reloadInterval);

      return () => {
        clearInterval(interval);
      };
    }
  }, [projectID, refreshJob, reloadInterval]);

  return { job, setJob, status, refreshJob };
}
