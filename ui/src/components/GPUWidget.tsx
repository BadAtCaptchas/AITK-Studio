import { GpuInfo } from '@/types';
import { Thermometer, Zap, Clock, HardDrive, Fan, Cpu } from 'lucide-react';

interface GPUWidgetProps {
  gpu: GpuInfo;
}

export default function GPUWidget({ gpu }: GPUWidgetProps) {
  const formatMemory = (mb: number): string => {
    return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
  };

  const getUtilizationColor = (value: number): string => {
    return value < 30 ? 'bg-emerald-500' : value < 70 ? 'bg-amber-500' : 'bg-rose-500';
  };

  const getTemperatureColor = (temp: number): string => {
    return temp < 50 ? 'text-emerald-500' : temp < 80 ? 'text-amber-500' : 'text-rose-500';
  };

  const memoryPercent = gpu.memory.total > 0 ? (gpu.memory.used / gpu.memory.total) * 100 : 0;

  return (
    <div className="operator-panel overflow-hidden">
      <div className="operator-panel-header">
        <div className="flex min-w-0 items-center gap-2">
          <h2 className="truncate font-semibold text-gray-100">{gpu.name}</h2>
          <span className="rounded-sm border border-gray-700 bg-gray-950 px-2 py-0.5 text-xs text-gray-300">
            GPU {gpu.index}
          </span>
        </div>
      </div>

      <div className="space-y-3 p-3">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <Thermometer className={`h-4 w-4 ${getTemperatureColor(gpu.temperature)}`} />
              <div>
                <p className="text-xs text-gray-400">Temperature</p>
                <p className={`text-sm font-medium ${getTemperatureColor(gpu.temperature)}`}>{gpu.temperature}&deg;C</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Fan className="h-4 w-4 text-cyan-400" />
              <div>
                <p className="text-xs text-gray-400">Fan Speed</p>
                <p className="text-sm font-medium text-cyan-300">{gpu.fan.speed}%</p>
              </div>
            </div>
          </div>
          <div>
            <div className="mb-1 flex items-center space-x-2">
              <Cpu className="h-4 w-4 text-gray-400" />
              <p className="text-xs text-gray-400">GPU Load</p>
              <span className="ml-auto text-xs text-gray-300">{gpu.utilization.gpu}%</span>
            </div>
            <div className="h-1 w-full rounded-sm bg-gray-800">
              <div
                className={`h-1 rounded-sm transition-all ${getUtilizationColor(gpu.utilization.gpu)}`}
                style={{ width: `${gpu.utilization.gpu}%` }}
              />
            </div>
            <div className="mb-1 mt-3 flex items-center space-x-2">
              <HardDrive className="h-4 w-4 text-cyan-400" />
              <p className="text-xs text-gray-400">Memory</p>
              <span className="ml-auto text-xs text-gray-300">{memoryPercent.toFixed(1)}%</span>
            </div>
            <div className="h-1 w-full rounded-sm bg-gray-800">
              <div className="h-1 rounded-sm bg-cyan-500 transition-all" style={{ width: `${memoryPercent}%` }} />
            </div>
            <p className="mt-0.5 text-xs text-gray-400">
              {formatMemory(gpu.memory.used)} / {formatMemory(gpu.memory.total)}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 border-t border-gray-800 pt-3 sm:grid-cols-2">
          <div className="flex items-start space-x-2">
            <Clock className="h-4 w-4 text-gray-400" />
            <div>
              <p className="text-xs text-gray-400">Clock Speed</p>
              <p className="text-sm text-gray-200">{gpu.clocks.graphics} MHz</p>
            </div>
          </div>
          <div className="flex items-start space-x-2">
            <Zap className="h-4 w-4 text-amber-400" />
            <div>
              <p className="text-xs text-gray-400">Power Draw</p>
              <p className="text-sm text-gray-200">
                {gpu.power.draw?.toFixed(1)}W
                <span className="text-xs text-gray-400"> / {gpu.power.limit?.toFixed(1) || ' ? '}W</span>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
