'use client';

import type { HFDownloadProgress } from '@/types';
import useJobProgress, { isDownloadProgress } from './useJobProgress';
import usePollLoop from './usePollLoop';

export default function useJobDownloadProgress(
  jobID: string,
  initialProgress: HFDownloadProgress | null = null,
  reloadInterval: number | null = null,
) {
  const { progress, status, isPolling, refreshProgress } = useJobProgress(
    jobID, initialProgress, reloadInterval, 'hf-download-progress', isDownloadProgress, false,
  );
  usePollLoop(signal => refreshProgress(signal), reloadInterval && isPolling ? reloadInterval : null, [jobID, isPolling]);
  return { progress, status, refreshProgress: () => refreshProgress() };
}
