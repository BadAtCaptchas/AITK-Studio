'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { HFDownloadProgress } from '@/types';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';

const MAX_EMPTY_POLLS = 10;

function getUpdatedMs(progress: HFDownloadProgress) {
  const updatedMs = new Date(progress.updatedAt).getTime();
  return Number.isFinite(updatedMs) ? updatedMs : null;
}

function isNewerProgress(nextProgress: HFDownloadProgress, previousUpdatedAt: string | null) {
  if (!previousUpdatedAt) return true;

  const nextMs = getUpdatedMs(nextProgress);
  const previousMs = new Date(previousUpdatedAt).getTime();
  if (nextMs !== null && Number.isFinite(previousMs)) {
    return nextMs > previousMs;
  }

  return nextProgress.updatedAt !== previousUpdatedAt;
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
  const latestProgressUpdatedAtRef = useRef(initialProgress?.updatedAt || null);
  const activeJobIDRef = useRef(jobID);
  activeJobIDRef.current = jobID;

  const refreshProgress = useCallback((signal?: AbortSignal) => {
    const requestJobID = jobID;
    setStatus(current => (current === 'idle' ? 'loading' : current));
    return apiClient
      .get(`/api/jobs/${jobID}/hf-download-progress`, { signal })
      .then(res => res.data)
      .then(data => {
        if (signal?.aborted || activeJobIDRef.current !== requestJobID) return;
        const nextProgress = data.progress || null;
        setProgress(nextProgress);
        if (nextProgress) {
          hasSeenProgressRef.current = true;
          emptyPollsRef.current = 0;
          latestProgressUpdatedAtRef.current = nextProgress.updatedAt || null;
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
        console.error('Error fetching Hugging Face download progress:', error);
        setStatus('error');
      });
  }, [jobID]);

  useEffect(() => {
    emptyPollsRef.current = 0;
    hasSeenProgressRef.current = Boolean(initialProgress);
    latestProgressUpdatedAtRef.current = initialProgress?.updatedAt || null;
    setProgress(initialProgress);
    setIsPolling(Boolean(reloadInterval));
  }, [jobID]);

  useEffect(() => {
    if (!reloadInterval) {
      setIsPolling(false);
      return;
    }

    emptyPollsRef.current = 0;
    setIsPolling(true);
  }, [jobID, reloadInterval]);

  useEffect(() => {
    if (!initialProgress || !isNewerProgress(initialProgress, latestProgressUpdatedAtRef.current)) return;

    latestProgressUpdatedAtRef.current = initialProgress.updatedAt || null;
    hasSeenProgressRef.current = true;
    emptyPollsRef.current = 0;
    setProgress(initialProgress);
    if (reloadInterval) {
      setIsPolling(true);
    }
  }, [initialProgress, reloadInterval]);

  usePollLoop(
    signal => refreshProgress(signal),
    reloadInterval && isPolling ? reloadInterval : null,
    [jobID, isPolling],
  );

  return { progress, status, refreshProgress: () => refreshProgress() };
}
