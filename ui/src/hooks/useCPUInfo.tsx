'use client';

import { CpuInfo } from '@/types';
import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';
import useMonitorStream from './useMonitorStream';

export default function useCPUInfo(reloadInterval: null | number = null, workerID = 'local') {
  const [cpuInfo, setCpuInfo] = useState<CpuInfo | null>(null);
  const [isCPUInfoLoaded, setIsLoaded] = useState(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const activeWorkerRef = useRef(workerID);
  const monitor = useMonitorStream();
  const useLocalMonitor = workerID === 'local';
  activeWorkerRef.current = workerID;

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
    [workerID],
  );

  useEffect(() => {
    setCpuInfo(null);
    setIsLoaded(false);
    setStatus('idle');
  }, [workerID]);

  useEffect(() => {
    if (!useLocalMonitor || !monitor.cpu) return;
    setCpuInfo(monitor.cpu);
    setIsLoaded(true);
    setStatus('success');
  }, [monitor.cpu, useLocalMonitor]);

  usePollLoop(signal => fetchCpuInfo(signal), useLocalMonitor ? null : reloadInterval, [workerID, useLocalMonitor]);

  return {
    cpuInfo,
    isCPUInfoLoaded,
    status,
    refreshCpuInfo: () => (useLocalMonitor ? Promise.resolve() : fetchCpuInfo()),
  };
}
