'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export default function useJobDashboard(jobID: string, reloadInterval: number | null = 5000) {
  const [dashboard, setDashboard] = useState<any | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const refresh = () => {
    setStatus(prev => (prev === 'success' ? 'refreshing' : 'loading'));
    apiClient
      .get(`/api/jobs/${jobID}/dashboard`)
      .then(res => res.data)
      .then(data => {
        setDashboard(data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching job dashboard:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    if (!jobID) return;
    refresh();
    if (!reloadInterval) return;
    const id = window.setInterval(refresh, reloadInterval);
    return () => window.clearInterval(id);
  }, [jobID, reloadInterval]);

  return { dashboard, status, refresh };
}
