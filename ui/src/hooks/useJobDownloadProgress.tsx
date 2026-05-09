'use client';

import { useCallback, useEffect, useState } from 'react';
import type { HFDownloadProgress } from '@/types';
import { apiClient } from '@/utils/api';

export default function useJobDownloadProgress(
  jobID: string,
  initialProgress: HFDownloadProgress | null = null,
  reloadInterval: number | null = null,
) {
  const [progress, setProgress] = useState<HFDownloadProgress | null>(initialProgress);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshProgress = useCallback(() => {
    setStatus(current => (current === 'idle' ? 'loading' : current));
    apiClient
      .get(`/api/jobs/${jobID}/hf-download-progress`)
      .then(res => res.data)
      .then(data => {
        setProgress(data.progress || null);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching Hugging Face download progress:', error);
        setStatus('error');
      });
  }, [jobID]);

  useEffect(() => {
    setProgress(initialProgress);
  }, [initialProgress, jobID]);

  useEffect(() => {
    refreshProgress();

    if (!reloadInterval) return;
    const interval = setInterval(refreshProgress, reloadInterval);
    return () => clearInterval(interval);
  }, [refreshProgress, reloadInterval]);

  return { progress, status, refreshProgress };
}
