'use client';

import { useCallback, useEffect, useRef } from 'react';
import type { ComfyInstallProgress } from '@/types';
import { apiClient } from '@/utils/api';
import useScopedState from './useScopedState';
import useCommittedRef from './useCommittedRef';
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
  const [progress, setProgress] = useScopedState<ComfyInstallProgress | null>(jobID, initialProgress);
  const pollingScope = JSON.stringify([jobID, reloadInterval, initialProgress?.status]);
  const [isPolling, setIsPolling] = useScopedState(
    pollingScope,
    Boolean(reloadInterval) && !isTerminalProgress(initialProgress),
  );
  const [status, setStatus] = useScopedState<'idle' | 'loading' | 'success' | 'error'>(jobID, 'idle');
  const emptyPollsRef = useRef(0);
  const hasSeenProgressRef = useRef(Boolean(initialProgress));
  const activeJobIDRef = useCommittedRef(jobID);

  const refreshProgress = useCallback(
    (signal?: AbortSignal) => {
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
    },
    [jobID, setProgress, setStatus, setIsPolling],
  );

  useEffect(() => {
    emptyPollsRef.current = 0;
    hasSeenProgressRef.current = Boolean(initialProgress);
  }, [jobID]);

  useEffect(() => {
    emptyPollsRef.current = 0;
  }, [pollingScope]);

  usePollLoop(signal => refreshProgress(signal), reloadInterval && isPolling ? reloadInterval : null, [
    jobID,
    isPolling,
  ]);

  return { progress, status, refreshProgress: () => refreshProgress() };
}
