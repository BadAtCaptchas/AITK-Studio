'use client';

import type { ComfyInstallProgress } from '@/types';
import useJobProgress, { isComfyInstallProgress } from './useJobProgress';
import usePollLoop from './usePollLoop';

export default function useJobComfyInstallProgress(
  jobID: string,
  initialProgress: ComfyInstallProgress | null = null,
  reloadInterval: number | null = null,
) {
  const { progress, status, isPolling, refreshProgress } = useJobProgress(
    jobID, initialProgress, reloadInterval, 'comfy-install-progress', isComfyInstallProgress, true,
  );
  usePollLoop(signal => refreshProgress(signal), reloadInterval && isPolling ? reloadInterval : null, [jobID, isPolling]);
  return { progress, status, refreshProgress: () => refreshProgress() };
}
