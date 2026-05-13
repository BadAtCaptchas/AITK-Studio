'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { HFDownloadProgress } from '@/types';
import { apiClient } from '@/utils/api';

const MAX_EMPTY_POLLS = 10;

function isTerminalProgress(progress: HFDownloadProgress | null) {
  return progress?.status === 'completed' || progress?.status === 'failed';
}

export default function useJobDownloadProgress(
  jobID: string,
  initialProgress: HFDownloadProgress | null = null,
  reloadInterval: number | null = null,
) {
  const [progress, setProgress] = useState<HFDownloadProgress | null>(initialProgress);
  const [isPolling, setIsPolling] = useState(Boolean(reloadInterval));
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const emptyPollsRef = useRef(0);
  const hasSeenProgressRef = useRef(Boolean(initialProgress));

  const refreshProgress = useCallback(() => {
    setStatus(current => (current === 'idle' ? 'loading' : current));
    apiClient
      .get(`/api/jobs/${jobID}/hf-download-progress`)
      .then(res => res.data)
      .then(data => {
        const nextProgress = data.progress || null;
        setProgress(nextProgress);
        if (nextProgress) {
          hasSeenProgressRef.current = true;
          emptyPollsRef.current = 0;
          if (isTerminalProgress(nextProgress)) {
            setIsPolling(false);
          }
        } else {
          emptyPollsRef.current += 1;
          if (hasSeenProgressRef.current || emptyPollsRef.current >= MAX_EMPTY_POLLS) {
            setIsPolling(false);
          }
        }
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching Hugging Face download progress:', error);
        setStatus('error');
      });
  }, [jobID]);

  useEffect(() => {
    emptyPollsRef.current = 0;
    hasSeenProgressRef.current = Boolean(initialProgress);
    setProgress(initialProgress);
    setIsPolling(Boolean(reloadInterval) && !isTerminalProgress(initialProgress));
  }, [jobID]);

  useEffect(() => {
    if (!reloadInterval) {
      setIsPolling(false);
      return;
    }
    if (isTerminalProgress(initialProgress)) {
      setIsPolling(false);
      return;
    }

    emptyPollsRef.current = 0;
    setIsPolling(true);
  }, [jobID, reloadInterval, initialProgress?.status]);

  useEffect(() => {
    refreshProgress();
  }, [refreshProgress]);

  useEffect(() => {
    if (!reloadInterval || !isPolling) return;
    const interval = setInterval(refreshProgress, reloadInterval);
    return () => clearInterval(interval);
  }, [isPolling, refreshProgress, reloadInterval]);

  return { progress, status, refreshProgress };
}
