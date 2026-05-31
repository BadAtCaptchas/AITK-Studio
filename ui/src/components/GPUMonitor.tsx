import React, { useState, useEffect, useMemo } from 'react';
import Loading from '@/components/Loading';
import GPUWidget from '@/components/GPUWidget';
import useGPUInfo from '@/hooks/useGPUInfo';
import { RotateCcw } from 'lucide-react';
import { PageNotice } from '@/components/OperatorPrimitives';

const GpuMonitor: React.FC = () => {
  const { gpuData, gpuList, isGPUInfoLoaded, status, refreshGpuInfo } = useGPUInfo();
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  useEffect(() => {
    if (status === 'success') {
      setLastUpdated(new Date());
    }
  }, [gpuList, status]);

  const getGridClasses = (gpuCount: number): string => {
    switch (gpuCount) {
      case 1:
        return 'grid-cols-1';
      case 2:
        return 'grid-cols-1 xl:grid-cols-2';
      case 3:
        return 'grid-cols-1 lg:grid-cols-2 2xl:grid-cols-3';
      case 4:
        return 'grid-cols-1 lg:grid-cols-2 2xl:grid-cols-4';
      case 5:
      case 6:
        return 'grid-cols-1 md:grid-cols-2 2xl:grid-cols-3';
      case 7:
      case 8:
        return 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4';
      case 9:
      case 10:
        return 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-5';
      default:
        return 'grid-cols-1 md:grid-cols-2 2xl:grid-cols-3';
    }
  };

  const content = useMemo(() => {
    if (!isGPUInfoLoaded && !gpuData) {
      return <Loading />;
    }

    if (status === 'error') {
      return (
        <PageNotice tone="danger" title="GPU telemetry unavailable">
          Failed to fetch GPU data. Refresh or check the worker connection.
        </PageNotice>
      );
    }

    if (!gpuData) {
      return (
        <PageNotice tone="warning" title="No GPU telemetry">
          The GPU endpoint returned no data for this worker.
        </PageNotice>
      );
    }

    if (!gpuData.hasNvidiaSmi && !gpuData.isMac) {
      return (
        <PageNotice tone="warning" title="No NVIDIA GPUs detected">
          nvidia-smi is not available on this system. {gpuData.error || ''}
        </PageNotice>
      );
    }

    if (gpuList.length === 0) {
      return (
        <PageNotice tone="warning" title="No GPUs found">
          nvidia-smi is available, but it did not return any GPU devices.
        </PageNotice>
      );
    }

    const gridClass = getGridClasses(gpuList.length || 1);

    return (
      <div className={`grid ${gridClass} gap-3`}>
        {gpuList.map(gpu => (
          <GPUWidget key={gpu.index} gpu={gpu} />
        ))}
      </div>
    );
  }, [gpuData, gpuList, isGPUInfoLoaded, status]);

  return (
    <div className="w-full">
      <div className="mb-2 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <h1 className="text-sm font-semibold uppercase tracking-wide text-gray-400">GPU Monitor</h1>
        <div className="flex w-full items-center justify-between gap-3 sm:w-auto sm:justify-end">
          <div className="text-xs text-gray-500">Last updated: {lastUpdated?.toLocaleTimeString() || '--'}</div>
          <button
            type="button"
            onClick={refreshGpuInfo}
            disabled={status === 'loading'}
            className="operator-button py-1 text-xs"
          >
            <RotateCcw className={`h-3.5 w-3.5 ${status === 'loading' ? 'animate-spin' : ''}`} />
            <span className="hidden sm:inline">Refresh</span>
          </button>
        </div>
      </div>
      {content}
    </div>
  );
};

export default GpuMonitor;
