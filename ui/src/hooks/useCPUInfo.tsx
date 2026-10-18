'use client';

import { CpuInfo } from '@/types';
import { useCallback } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';
import useScopedState from './useScopedState';
import useCommittedRef from './useCommittedRef';
import useMonitorStream from './useMonitorStream';

export default function useCPUInfo(reloadInterval: null | number = null, workerID = 'local') {
  const monitor = useMonitorStream();
  const useLocalMonitor = workerID === 'local';
  const useLiveMonitor = useLocalMonitor && monitor.connected && !!monitor.cpu;
  const scope = JSON.stringify([workerID, useLiveMonitor]);
  const [cpuInfo, setCpuInfo] = useScopedState<CpuInfo | null>(scope, null);
  const [isCPUInfoLoaded, setIsLoaded] = useScopedState(scope, false);
  const [status, setStatus] = useScopedState<'idle' | 'loading' | 'success' | 'error'>(scope, 'idle');
  const activeWorkerRef = useCommittedRef(workerID);

  const fetchCpuInfo = useCallback(
    async (signal?: AbortSignal) => {
      const requestWorkerID = workerID;
      setStatus('loading');
      try {
        const data: CpuInfo = await apiClient
          .get('/api/cpu', { params: { worker_id: workerID }, signal })
          .then(res => res.data);
        if (signal?.aborted || activeWorkerRef.current !== requestWorkerID) return;
        setCpuInfo(data);
        setStatus('success');
      } catch (err) {
        if (signal?.aborted || activeWorkerRef.current !== requestWorkerID) return;
        console.error(`Failed to fetch CPU data: ${err instanceof Error ? err.message : String(err)}`);
        setStatus('error');
      } finally {
        if (!signal?.aborted && activeWorkerRef.current === requestWorkerID) setIsLoaded(true);
      }
    },
    [workerID, setStatus, setCpuInfo, setIsLoaded],
  );

  usePollLoop(
    signal => {
      if (useLiveMonitor) return;
      return fetchCpuInfo(signal);
    },
    useLiveMonitor ? null : useLocalMonitor ? (reloadInterval ?? 5000) : reloadInterval,
    [workerID, useLiveMonitor],
  );

  return {
    cpuInfo: useLiveMonitor ? monitor.cpu : cpuInfo,
    isCPUInfoLoaded: useLiveMonitor ? true : isCPUInfoLoaded,
    status: useLiveMonitor ? ('success' as const) : status,
    refreshCpuInfo: () => fetchCpuInfo(),
  };
}
