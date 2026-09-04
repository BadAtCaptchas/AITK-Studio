'use client';

import { GPUApiResponse, GpuInfo } from '@/types';
import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';
import { SharedAbortableRequestPool } from '@/utils/sharedAbortableRequest';
import usePollLoop from './usePollLoop';
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
const gpuRequestPool = new SharedAbortableRequestPool<string, GPUApiResponse>(
  async (workerID, signal) => {
    const data = await apiClient
      .get('/api/gpu', { params: { worker_id: workerID }, signal })
      .then(res => res.data as GPUApiResponse);
    gpuCache.set(workerID, { data, fetchedAt: Date.now() });
    return data;
  },
);

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
  const [gpuData, setGpuData] = useState<GPUApiResponse | null>(null);
  const [gpuList, setGpuList] = useState<GpuInfo[]>([]);
  const [isGPUInfoLoaded, setIsLoaded] = useState(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const gpuIDsKey = gpuIds?.join(',') ?? '';
  const activeScopeRef = useRef('');
  const monitor = useMonitorStream();
  const useLocalMonitor = workerID === 'local';
  activeScopeRef.current = `${enabled}:${workerID}:${gpuIDsKey}`;

  const fetchGpuInfo = useCallback(async (fetchOptions?: FetchGpuInfoOptions) => {
    if (!enabled) {
      setStatus('idle');
      return;
    }
    const requestScope = `${enabled}:${workerID}:${gpuIDsKey}`;
    setStatus('loading');
    try {
      const data = await loadGpuInfo(
        workerID,
        fetchOptions?.force ? 0 : cacheTtlMs,
        fetchOptions?.signal,
      );
      if (fetchOptions?.signal?.aborted) return;
      if (activeScopeRef.current !== requestScope) return;
      setGpuData(data);
      let gpus = [...data.gpus].sort((a, b) => a.index - b.index);
      if (gpuIds) {
        gpus = gpus.filter(gpu => gpuIds.includes(gpu.index));
      }
      setGpuList(gpus);
      setStatus('success');
    } catch (err) {
      if (fetchOptions?.signal?.aborted) return;
      if (activeScopeRef.current !== requestScope) return;
      console.error(`Failed to fetch GPU data: ${err instanceof Error ? err.message : String(err)}`);
      setStatus('error');
    } finally {
      if (!fetchOptions?.signal?.aborted && activeScopeRef.current === requestScope) {
        setIsLoaded(true);
      }
    }
  }, [cacheTtlMs, enabled, gpuIDsKey, gpuIds, workerID]);

  useEffect(() => {
    setGpuList([]);
    setGpuData(null);
    setIsLoaded(false);
    setStatus('idle');
  }, [enabled, gpuIDsKey, workerID]);

  useEffect(() => {
    if (!enabled || !useLocalMonitor || !monitor.gpu) return;
    const data = monitor.gpu;
    let gpus = [...data.gpus].sort((a, b) => a.index - b.index);
    if (gpuIds) gpus = gpus.filter(gpu => gpuIds.includes(gpu.index));
    setGpuData(data);
    setGpuList(gpus);
    setIsLoaded(true);
    setStatus('success');
  }, [enabled, gpuIDsKey, gpuIds, monitor.gpu, useLocalMonitor]);

  usePollLoop(
    signal => fetchGpuInfo({ signal }),
    enabled && !useLocalMonitor ? reloadInterval : null,
    [enabled, workerID, cacheTtlMs, gpuIDsKey],
  );

  return {
    gpuData,
    gpuList,
    setGpuList,
    isGPUInfoLoaded,
    status,
    refreshGpuInfo: () => useLocalMonitor ? Promise.resolve() : fetchGpuInfo({ force: true }),
  };
}
