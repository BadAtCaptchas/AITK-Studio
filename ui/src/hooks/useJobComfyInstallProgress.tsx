'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { ComfyInstallProgress } from '@/types';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';

const MAX_EMPTY_POLLS = 10;

function isTerminalProgress(progress: ComfyInstallProgress | null) {
  return progress?.status === 'completed' || progress?.status === 'failed';
}

export default function useJobComfyInstallProgress(
  jobID: string,
  initialProgress: ComfyInstallProgress | null = null,
  reloadInterval: number | null = null,
) {
  const [progress, setProgress] = useState<ComfyInstallProgress | null>(initialProgress);
  const [isPolling, setIsPolling] = useState(Boolean(reloadInterval));
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const emptyPollsRef = useRef(0);
  const hasSeenProgressRef = useRef(Boolean(initialProgress));
  const activeJobIDRef = useRef(jobID);
  activeJobIDRef.current = jobID;

  const refreshProgress = useCallback((signal?: AbortSignal) => {
    const requestJobID = jobID;
    setStatus(current => (current === 'idle' ? 'loading' : current));
    return apiClient
      .get(`/api/jobs/${jobID}/comfy-install-progress`, { signal })
      .then(res => res.data)
      .then(data => {
        if (signal?.aborted || activeJobIDRef.current !== requestJobID) return;
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
        if (signal?.aborted || activeJobIDRef.current !== requestJobID) return;
        console.error('Error fetching ComfyUI install progress:', error);
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

  usePollLoop(
    signal => refreshProgress(signal),
    reloadInterval && isPolling ? reloadInterval : null,
    [jobID, isPolling],
  );

  return { progress, status, refreshProgress: () => refreshProgress() };
}
