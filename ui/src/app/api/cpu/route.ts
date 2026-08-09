import { NextResponse } from 'next/server';
import si from 'systeminformation';
import os from 'os';
import { CpuInfo } from '@/types';
import { fetchWorkerCpu, getRemoteWorker, isLocalWorker, runRemoteBackgroundPoll } from '@/server/remoteClient';
import { cached } from '@/server/apiCache';
import { loadMacstats, numberFromRecord } from '@/server/macstats';

const isMac = os.platform() === 'darwin';

export async function GET(request: Request) {
  try {
    const workerID = new URL(request.url).searchParams.get('worker_id') || 'local';
    if (!isLocalWorker(workerID)) {
      const worker = await getRemoteWorker(workerID);
      const poll = await runRemoteBackgroundPoll(worker, 'CPU telemetry', () => fetchWorkerCpu(worker));
      if ('reason' in poll) {
        return NextResponse.json({ error: `Remote CPU telemetry unavailable: ${poll.reason}` }, { status: 503 });
      }
      return NextResponse.json(poll.value);
    }

    const localCpuInfo = await cached('cpu-info:local', async () => {
      const cpuInfoRaw = await si.cpu();
      let cpuInfo: CpuInfo;

      if (isMac) {
        try {
          const ms = loadMacstats();
          if (!ms) throw new Error('macstats unavailable');
          const ramData = ms.getRAMUsageSync();
          const cpuData = ms.getCpuDataSync();
          const totalMemoryBytes = numberFromRecord(ramData, 'total', Number.NaN);
          const freeMemoryBytes = numberFromRecord(ramData, 'free', Number.NaN);
          if (!Number.isFinite(totalMemoryBytes) || !Number.isFinite(freeMemoryBytes)) {
            throw new Error('macstats returned invalid memory data');
          }

          cpuInfo = {
            name: `${cpuInfoRaw.manufacturer} ${cpuInfoRaw.brand}`,
            cores: cpuInfoRaw.cores,
            temperature: numberFromRecord(cpuData, 'temperature'),
            totalMemory: totalMemoryBytes / (1024 * 1024),
            availableMemory: freeMemoryBytes / (1024 * 1024),
            freeMemory: freeMemoryBytes / (1024 * 1024),
            currentLoad: (await si.currentLoad()).currentLoad || 0,
          };
        } catch {
          // Fallback to systeminformation if macstats fails
          const memoryData = await si.mem();
          cpuInfo = {
            name: `${cpuInfoRaw.manufacturer} ${cpuInfoRaw.brand}`,
            cores: cpuInfoRaw.cores,
            temperature: (await si.cpuTemperature()).main || 0,
            totalMemory: memoryData.total / (1024 * 1024),
            availableMemory: memoryData.available / (1024 * 1024),
            freeMemory: memoryData.free / (1024 * 1024),
            currentLoad: (await si.currentLoad()).currentLoad || 0,
          };
        }
      } else {
        const memoryData = await si.mem();
        cpuInfo = {
          name: `${cpuInfoRaw.manufacturer} ${cpuInfoRaw.brand}`,
          cores: cpuInfoRaw.cores,
          temperature: (await si.cpuTemperature()).main || 0,
          totalMemory: memoryData.total / (1024 * 1024),
          availableMemory: memoryData.available / (1024 * 1024),
          freeMemory: memoryData.free / (1024 * 1024),
          currentLoad: (await si.currentLoad()).currentLoad || 0,
        };
      }

      return cpuInfo;
    });

    return NextResponse.json(localCpuInfo);
  } catch (error) {
    console.error('Error fetching CPU stats:', error);
    return NextResponse.json(
      {
        error: `Failed to fetch CPU stats: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 500 },
    );
  }
}
