'use client';

import { GPUApiResponse, GpuInfo } from '@/types';
import { useCallback, useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

const DEFAULT_GPU_CACHE_TTL_MS = 5000;

type UseGPUInfoOptions = {
  enabled?: boolean;
  cacheTtlMs?: number;
};

type FetchGpuInfoOptions = {
  force?: boolean;
};

const gpuCache = new Map<string, { data: GPUApiResponse; fetchedAt: number }>();
const gpuRequests = new Map<string, Promise<GPUApiResponse>>();

async function loadGpuInfo(workerID: string, cacheTtlMs: number) {
  const cacheKey = workerID || 'local';
  const cached = gpuCache.get(cacheKey);
  const now = Date.now();
  if (cached && now - cached.fetchedAt < cacheTtlMs) {
    return cached.data;
  }

  const existingRequest = gpuRequests.get(cacheKey);
  if (existingRequest) {
    return existingRequest;
  }

  const request = apiClient
    .get('/api/gpu', { params: { worker_id: workerID } })
    .then(res => res.data as GPUApiResponse)
    .then(data => {
      gpuCache.set(cacheKey, { data, fetchedAt: Date.now() });
      return data;
    })
    .finally(() => {
      gpuRequests.delete(cacheKey);
    });

  gpuRequests.set(cacheKey, request);
  return request;
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

  const fetchGpuInfo = useCallback(async (fetchOptions?: FetchGpuInfoOptions) => {
    if (!enabled) {
      setStatus('idle');
      return;
    }
    setStatus('loading');
    try {
      const data = await loadGpuInfo(workerID, fetchOptions?.force ? 0 : cacheTtlMs);
      setGpuData(data);
      let gpus = [...data.gpus].sort((a, b) => a.index - b.index);
      if (gpuIds) {
        gpus = gpus.filter(gpu => gpuIds.includes(gpu.index));
      }
      setGpuList(gpus);
      setStatus('success');
    } catch (err) {
      console.error(`Failed to fetch GPU data: ${err instanceof Error ? err.message : String(err)}`);
      setStatus('error');
    } finally {
      setIsLoaded(true);
    }
  }, [cacheTtlMs, enabled, gpuIds, workerID]);

  useEffect(() => {
    if (!enabled) {
      setGpuList([]);
      setGpuData(null);
      setIsLoaded(false);
      setStatus('idle');
      return;
    }

    // Fetch immediately on component mount
    fetchGpuInfo();

    // Set up interval if specified
    if (reloadInterval) {
      const interval = setInterval(() => {
        fetchGpuInfo();
      }, reloadInterval);

      // Cleanup interval on unmount
      return () => {
        clearInterval(interval);
      };
    }
  }, [enabled, fetchGpuInfo, reloadInterval]);

  return {
    gpuData,
    gpuList,
    setGpuList,
    isGPUInfoLoaded,
    status,
    refreshGpuInfo: () => fetchGpuInfo({ force: true }),
  };
}
