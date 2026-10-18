'use client';

import { GPUApiResponse, GpuInfo } from '@/types';
import { useCallback, useMemo } from 'react';
import { apiClient } from '@/utils/api';
import { SharedAbortableRequestPool } from '@/utils/sharedAbortableRequest';
import usePollLoop from './usePollLoop';
import useScopedState from './useScopedState';
import useCommittedRef from './useCommittedRef';
import useMonitorStream from './useMonitorStream';

const DEFAULT_GPU_CACHE_TTL_MS = 5000;

type UseGPUInfoOptions = {
  enabled?: boolean;
  cacheTtlMs?: number;
};

type FetchGpuInfoOptions = {
  force?: boolean;
  signal?: AbortSignal;
};

const gpuCache = new Map<string, { data: GPUApiResponse; fetchedAt: number }>();
const gpuRequestPool = new SharedAbortableRequestPool<string, GPUApiResponse>(async (workerID, signal) => {
  const data = await apiClient
    .get('/api/gpu', { params: { worker_id: workerID }, signal, timeout: 10_000 })
    .then(res => res.data as GPUApiResponse);
  if (data.stale || data.error) throw new Error(data.error || 'Waiting for fresh GPU readings.');
  gpuCache.set(workerID, { data, fetchedAt: Date.now() });
  return data;
});

async function loadGpuInfo(workerID: string, cacheTtlMs: number, signal?: AbortSignal) {
  const cacheKey = workerID || 'local';
  const cached = gpuCache.get(cacheKey);
  const now = Date.now();
  if (cached && now - cached.fetchedAt < cacheTtlMs) {
    return cached.data;
  }

  return gpuRequestPool.subscribe(cacheKey, signal);
}

export default function useGPUInfo(
  gpuIds: null | number[] = null,
  reloadInterval: null | number = null,
  workerID = 'local',
  options: UseGPUInfoOptions = {},
) {
  const enabled = options.enabled ?? true;
  const cacheTtlMs = options.cacheTtlMs ?? (reloadInterval ? 0 : DEFAULT_GPU_CACHE_TTL_MS);
  const gpuIDsKey = gpuIds?.join(',') ?? '';
  const monitor = useMonitorStream();
  const useLocalMonitor = workerID === 'local';
  const useLiveMonitor =
    enabled && useLocalMonitor && monitor.connected && !!monitor.gpu && !monitor.gpu.stale && !monitor.gpu.error;
  // A stream disconnect starts a new polling scope; old readings cannot leak into it.
  const scope = JSON.stringify([enabled, workerID, gpuIDsKey, useLiveMonitor]);
  const activeScopeRef = useCommittedRef(scope);
  const [polledData, setGpuData] = useScopedState<GPUApiResponse | null>(scope, null);
  const [isGPUInfoLoaded, setIsLoaded] = useScopedState(scope, false);
  const [status, setStatus] = useScopedState<'idle' | 'loading' | 'success' | 'error'>(scope, 'idle');
  const [polledAt, setPolledAt] = useScopedState<Date | null>(scope, null);
  const gpuData = useLiveMonitor ? monitor.gpu : polledData;
  const gpuList = useMemo(
    () => (gpuData?.gpus ?? []).filter(gpu => !gpuIds || gpuIds.includes(gpu.index)).sort((a, b) => a.index - b.index),
    [gpuData, gpuIds],
  );

  const fetchGpuInfo = useCallback(
    async (fetchOptions?: FetchGpuInfoOptions) => {
      if (!enabled) {
        setStatus('idle');
        return;
      }
      const requestScope = scope;
      setStatus('loading');
      try {
        const data = await loadGpuInfo(workerID, fetchOptions?.force ? 0 : cacheTtlMs, fetchOptions?.signal);
        if (fetchOptions?.signal?.aborted) return;
        if (activeScopeRef.current !== requestScope) return;
        setGpuData(data);
        setPolledAt(new Date());
        setStatus('success');
      } catch (err) {
        if (fetchOptions?.signal?.aborted) return;
        if (activeScopeRef.current !== requestScope) return;
        console.error(`Failed to fetch GPU data: ${err instanceof Error ? err.message : String(err)}`);
        setGpuData(null);
        setStatus('error');
      } finally {
        if (!fetchOptions?.signal?.aborted && activeScopeRef.current === requestScope) {
          setIsLoaded(true);
        }
      }
    },
    [cacheTtlMs, enabled, scope, workerID, activeScopeRef, setGpuData, setStatus, setIsLoaded, setPolledAt],
  );

  usePollLoop(
    signal => {
      if (!enabled || useLiveMonitor) return;
      return fetchGpuInfo({ signal, force: useLocalMonitor });
    },
    enabled && !useLiveMonitor ? (useLocalMonitor ? (reloadInterval ?? 5000) : reloadInterval) : null,
    [enabled, workerID, cacheTtlMs, gpuIDsKey, useLiveMonitor],
  );

  return {
    gpuData,
    lastUpdated: useLiveMonitor ? monitor.lastUpdated : polledAt,
    gpuList,
    isGPUInfoLoaded: useLiveMonitor || isGPUInfoLoaded,
    status: useLiveMonitor ? ('success' as const) : status,
    refreshGpuInfo: () => fetchGpuInfo({ force: true }),
  };
}
