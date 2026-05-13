'use client';

import { useCallback, useEffect, useState } from 'react';
import type { Job } from '@/types';
import { apiClient } from '@/utils/api';

export default function useJob(jobID: string, reloadInterval: null | number = null) {
  const [job, setJob] = useState<Job | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshJob = useCallback(() => {
    setStatus('loading');
    apiClient
      .get(`/api/jobs?id=${jobID}`)
      .then(res => res.data)
      .then(data => {
        setJob(data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  }, [jobID]);

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
