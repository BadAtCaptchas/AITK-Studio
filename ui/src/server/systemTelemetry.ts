import { exec } from 'child_process';
import os from 'os';
import { promisify } from 'util';
import si from 'systeminformation';
import { db, type SystemMetricSampleCreateInput } from './db';

const execAsync = promisify(exec);

export type TelemetryRange = '1h' | '6h' | '24h' | 'all';

const DEFAULT_INTERVAL_SEC = 5;
const DEFAULT_RETENTION_HOURS = 24;

function asNumber(value: unknown, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

export async function getTelemetrySettings() {
  const [intervalRow, retentionRow] = await Promise.all([
    db.settings.get('SYSTEM_TELEMETRY_INTERVAL_SEC'),
    db.settings.get('SYSTEM_TELEMETRY_RETENTION_HOURS'),
  ]);
  const intervalSec = Math.max(1, asNumber(intervalRow?.value, DEFAULT_INTERVAL_SEC));
  const retentionHours = Math.max(1, asNumber(retentionRow?.value, DEFAULT_RETENTION_HOURS));
  return { intervalSec, retentionHours };
}

export function rangeToSince(range: TelemetryRange) {
  if (range === 'all') return null;
  const hours = range === '1h' ? 1 : range === '6h' ? 6 : 24;
  return new Date(Date.now() - hours * 60 * 60 * 1000);
}

async function readGpuSamples(createdAt: Date): Promise<SystemMetricSampleCreateInput[]> {
  if (os.platform() === 'darwin') return [];
  try {
    const { stdout } = await execAsync(
      'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,power.limit --format=csv,noheader,nounits',
      { env: { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' }, timeout: 4000 },
    );
    return stdout
      .trim()
      .split('\n')
      .filter(Boolean)
      .flatMap(line => {
        const [index, name, temp, gpuUtil, memUtil, memTotal, memUsed, powerDraw, powerLimit] = line
          .split(', ')
          .map(part => part.trim());
        const device_id = index || '0';
        const metadata = JSON.stringify({ name });
        return [
          { created_at: createdAt, scope: 'gpu', device_id, metric: 'gpu_utilization', value: asNumber(gpuUtil), unit: '%', metadata },
          { created_at: createdAt, scope: 'gpu', device_id, metric: 'vram_utilization', value: asNumber(memUtil), unit: '%', metadata },
          { created_at: createdAt, scope: 'gpu', device_id, metric: 'vram_used_mb', value: asNumber(memUsed), unit: 'MB', metadata: JSON.stringify({ name, total_mb: asNumber(memTotal) }) },
          { created_at: createdAt, scope: 'gpu', device_id, metric: 'temperature', value: asNumber(temp), unit: 'C', metadata },
          { created_at: createdAt, scope: 'gpu', device_id, metric: 'power_draw', value: asNumber(powerDraw), unit: 'W', metadata: JSON.stringify({ name, limit_w: asNumber(powerLimit) }) },
        ];
      });
  } catch {
    return [];
  }
}

export async function readCurrentSystemMetricSamples() {
  const createdAt = new Date();
  const [cpu, load, mem, gpuSamples] = await Promise.all([
    si.cpu().catch(() => null),
    si.currentLoad().catch(() => null),
    si.mem().catch(() => null),
    readGpuSamples(createdAt),
  ]);

  const samples: SystemMetricSampleCreateInput[] = [];
  if (load) {
    samples.push({
      created_at: createdAt,
      scope: 'cpu',
      device_id: 'host',
      metric: 'cpu_utilization',
      value: asNumber(load.currentLoad),
      unit: '%',
      metadata: JSON.stringify({ name: cpu ? `${cpu.manufacturer} ${cpu.brand}` : 'CPU' }),
    });
  }
  if (mem) {
    const used = mem.total - mem.available;
    samples.push(
      {
        created_at: createdAt,
        scope: 'ram',
        device_id: 'host',
        metric: 'ram_utilization',
        value: mem.total > 0 ? (used / mem.total) * 100 : 0,
        unit: '%',
        metadata: JSON.stringify({ total_mb: mem.total / (1024 * 1024) }),
      },
      {
        created_at: createdAt,
        scope: 'ram',
        device_id: 'host',
        metric: 'ram_used_mb',
        value: used / (1024 * 1024),
        unit: 'MB',
        metadata: JSON.stringify({ total_mb: mem.total / (1024 * 1024) }),
      },
    );
  }
  samples.push(...gpuSamples);
  return samples;
}

export async function collectSystemTelemetry() {
  const samples = await readCurrentSystemMetricSamples();
  await db.systemMetrics.createMany(samples);
  return samples.length;
}

export async function pruneSystemTelemetry() {
  const { retentionHours } = await getTelemetrySettings();
  await db.systemMetrics.prune(new Date(Date.now() - retentionHours * 60 * 60 * 1000));
}

export async function listTelemetry(range: TelemetryRange) {
  const since = rangeToSince(range);
  return db.systemMetrics.list({ since, limit: range === 'all' ? 50000 : 15000 });
}
