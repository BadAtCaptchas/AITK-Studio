import React, { useState, useEffect, useMemo } from 'react';
import Loading from '@/components/Loading';
import GPUWidget from '@/components/GPUWidget';
import useGPUInfo from '@/hooks/useGPUInfo';
import { RotateCcw } from 'lucide-react';

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
        return 'grid-cols-2';
      case 3:
        return 'grid-cols-3';
      case 4:
        return 'grid-cols-4';
      case 5:
      case 6:
        return 'grid-cols-3';
      case 7:
      case 8:
        return 'grid-cols-4';
      case 9:
      case 10:
        return 'grid-cols-5';
      default:
        return 'grid-cols-3';
    }
  };

  const content = useMemo(() => {
    if (!isGPUInfoLoaded && !gpuData) {
      return <Loading />;
    }

    if (status === 'error') {
      return (
        <div className="bg-red-900 border border-red-600 text-red-200 px-4 py-3 rounded relative" role="alert">
          <strong className="font-bold">Error!</strong>
          <span className="block sm:inline"> Failed to fetch GPU data.</span>
        </div>
      );
    }

    if (!gpuData) {
      return (
        <div className="bg-yellow-900 border border-yellow-700 text-yellow-300 px-4 py-3 rounded relative" role="alert">
          <span className="block sm:inline">No GPU data available.</span>
        </div>
      );
    }

    if (!gpuData.hasNvidiaSmi && !gpuData.isMac) {
      return (
        <div className="bg-yellow-900 border border-yellow-700 text-yellow-300 px-4 py-3 rounded relative" role="alert">
          <strong className="font-bold">No NVIDIA GPUs detected!</strong>
          <span className="block sm:inline"> nvidia-smi is not available on this system.</span>
          {gpuData.error && <p className="mt-2 text-sm">{gpuData.error}</p>}
        </div>
      );
    }

    if (gpuList.length === 0) {
      return (
        <div className="bg-yellow-900 border border-yellow-700 text-yellow-300 px-4 py-3 rounded relative" role="alert">
          <span className="block sm:inline">No GPUs found, but nvidia-smi is available.</span>
        </div>
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
      <div className="flex justify-between items-center mb-2">
        <h1 className="text-md">GPU Monitor</h1>
        <div className="flex items-center gap-3">
          <div className="text-xs text-gray-500">Last updated: {lastUpdated?.toLocaleTimeString() || '--'}</div>
          <button
            type="button"
            onClick={refreshGpuInfo}
            disabled={status === 'loading'}
            className="inline-flex items-center gap-1.5 rounded-md border border-gray-700 bg-gray-800 px-2.5 py-1 text-xs text-gray-200 hover:bg-gray-700 disabled:opacity-60"
          >
            <RotateCcw className={`h-3.5 w-3.5 ${status === 'loading' ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>
      {content}
    </div>
  );
};

export default GpuMonitor;
