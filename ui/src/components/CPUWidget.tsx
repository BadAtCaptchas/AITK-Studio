import { CpuInfo } from '@/types';
import { HardDrive, Cpu } from 'lucide-react';

interface CPUWidgetProps {
  cpu: CpuInfo | null;
}

export default function CPUWidget({ cpu }: CPUWidgetProps) {
  const formatMemory = (mb: number): string => {
    return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
  };

  const getUtilizationColor = (value: number): string => {
    return value < 30 ? 'bg-emerald-500' : value < 70 ? 'bg-amber-500' : 'bg-rose-500';
  };

  if (!cpu) {
    return (
      <div className="operator-panel overflow-hidden">
        <div className="operator-panel-header">
          <h2 className="font-semibold text-gray-100">CPU Info</h2>
        </div>
        <div className="p-3">
          <p className="text-sm text-gray-400">No CPU data available.</p>
        </div>
      </div>
    );
  }

  const usedMemory = cpu.totalMemory - cpu.availableMemory;
  const memoryPercent = cpu.totalMemory > 0 ? (usedMemory / cpu.totalMemory) * 100 : 0;

  return (
    <div className="operator-panel overflow-hidden">
      <div className="operator-panel-header">
        <h2 className="truncate font-semibold text-gray-100">{cpu.name}</h2>
      </div>

      <div className="space-y-3 p-3">
        <div>
          <div className="mb-1 flex items-center space-x-2">
            <Cpu className="h-4 w-4 text-gray-400" />
            <p className="text-xs text-gray-400">CPU Load</p>
            <span className="ml-auto text-xs text-gray-300">{cpu.currentLoad.toFixed(1)}%</span>
          </div>
          <div className="h-1 w-full rounded-sm bg-gray-800">
            <div
              className={`h-1 rounded-sm transition-all ${getUtilizationColor(cpu.currentLoad)}`}
              style={{ width: `${cpu.currentLoad}%` }}
            />
          </div>
        </div>
        <div>
          <div className="mb-1 flex items-center space-x-2">
            <HardDrive className="h-4 w-4 text-cyan-400" />
            <p className="text-xs text-gray-400">Memory</p>
            <span className="ml-auto text-xs text-gray-300">{memoryPercent.toFixed(1)}%</span>
          </div>
          <div className="h-1 w-full rounded-sm bg-gray-800">
            <div className="h-1 rounded-sm bg-cyan-500 transition-all" style={{ width: `${memoryPercent}%` }} />
          </div>
          <p className="mt-0.5 text-xs text-gray-400">
            {formatMemory(usedMemory)} / {formatMemory(cpu.totalMemory)}
          </p>
        </div>
      </div>
    </div>
  );
}
