'use client';

import { ChevronDown, RefreshCw } from 'lucide-react';
import type useGPUInfo from '@/hooks/useGPUInfo';
import type { GpuInfo } from '@/types';

interface DashboardHardwareProps {
  telemetry: ReturnType<typeof useGPUInfo>;
  connected: boolean;
  lastUpdated: Date | null;
}

function memoryLabel(mb: number): string {
  if (!Number.isFinite(mb) || mb < 0) return '—';
  return mb >= 1024 ? `${(mb / 1024).toFixed(mb >= 10240 ? 0 : 1)} GB` : `${Math.round(mb)} MB`;
}

function measurement(value: number | null | undefined, unit: string): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${Math.round(value * 10) / 10}${unit}` : '—';
}

function HardwareDevice({ gpu }: { gpu: GpuInfo }) {
  const freePercent = gpu.memory.total > 0 ? Math.min(100, Math.max(0, (gpu.memory.free / gpu.memory.total) * 100)) : 0;
  return (
    <article className="studio-device" aria-label={gpu.name}>
      <div className="studio-device-label">GPU {gpu.index}</div>
      <h3>{gpu.name}</h3>
      <div className="studio-memory-value">{memoryLabel(gpu.memory.free)}</div>
      <p className="studio-muted">free VRAM of {memoryLabel(gpu.memory.total)}</p>
      <div
        className="studio-memory-track"
        role="meter"
        aria-label={`GPU ${gpu.index} free VRAM`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Number.isFinite(freePercent) ? freePercent : 0}
        aria-valuetext={`${memoryLabel(gpu.memory.free)} free of ${memoryLabel(gpu.memory.total)}`}
      >
        <div style={{ width: `${Number.isFinite(freePercent) ? freePercent : 0}%` }} />
      </div>
      <div className="studio-memory-scale">
        <span>0 GB</span>
        <span>{memoryLabel(gpu.memory.total)}</span>
      </div>
      <dl className="studio-device-metrics">
        <div>
          <dd>{measurement(gpu.utilization.gpu, '%')}</dd>
          <dt>GPU load</dt>
        </div>
        <div>
          <dd className={gpu.temperature >= 80 ? 'text-amber-500' : ''}>{measurement(gpu.temperature, '°C')}</dd>
          <dt>Temperature</dt>
        </div>
      </dl>
      <details className="studio-device-details">
        <summary>
          Device details <ChevronDown size={15} aria-hidden="true" />
        </summary>
        <dl>
          <div>
            <dt>Temperature</dt>
            <dd>{measurement(gpu.temperature, '°C')}</dd>
          </div>
          <div>
            <dt>Fan speed</dt>
            <dd>{measurement(gpu.fan.speed, '%')}</dd>
          </div>
          <div>
            <dt>Clock speed</dt>
            <dd>{measurement(gpu.clocks.graphics, ' MHz')}</dd>
          </div>
          <div>
            <dt>Power draw</dt>
            <dd>
              {measurement(gpu.power.draw, ' W')} / {measurement(gpu.power.limit, ' W')}
            </dd>
          </div>
        </dl>
      </details>
    </article>
  );
}

export default function DashboardHardware({ telemetry, connected, lastUpdated }: DashboardHardwareProps) {
  const { gpuData, gpuList, isGPUInfoLoaded, status, refreshGpuInfo } = telemetry;
  const ready = isGPUInfoLoaded && status !== 'error' && gpuList.length > 0;
  const error =
    status === 'error'
      ? 'GPU telemetry unavailable. Refresh to try again.'
      : isGPUInfoLoaded && !gpuData
        ? 'No GPU telemetry is available for this worker.'
        : gpuData && !gpuData.hasNvidiaSmi && !gpuData.isMac
          ? 'No NVIDIA GPUs detected. Check that nvidia-smi is available.'
          : isGPUInfoLoaded && gpuList.length === 0
            ? 'No GPUs found on this worker.'
            : null;

  return (
    <section className="studio-hardware" aria-labelledby="hardware-title">
      <div className="studio-section-heading">
        <h2 id="hardware-title">Local hardware</h2>
        <span className="studio-online">
          <span className={`studio-status-dot ${ready && connected ? '' : 'studio-status-dot-muted'}`} />
          {ready && connected ? 'Online' : ready ? 'Reconnecting' : !isGPUInfoLoaded ? 'Checking' : 'Unavailable'}
        </span>
      </div>
      {!isGPUInfoLoaded && (
        <p className="studio-hardware-notice" role="status">
          Checking local hardware…
        </p>
      )}
      {error && (
        <p className="studio-hardware-notice" role="status">
          {error}
        </p>
      )}
      {ready && !connected && (
        <p className="studio-hardware-notice" role="status">
          Live connection interrupted. Showing the last available readings.
        </p>
      )}
      {ready && gpuList.map(gpu => <HardwareDevice key={gpu.index} gpu={gpu} />)}
      <div className="studio-hardware-footer">
        <span>
          {lastUpdated
            ? `Updated ${lastUpdated.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`
            : 'Live telemetry'}
        </span>
        <button
          type="button"
          onClick={() => void refreshGpuInfo()}
          disabled={status === 'loading'}
          className="studio-icon-button"
          aria-label="Refresh GPU telemetry"
          title="Refresh GPU telemetry"
        >
          <RefreshCw size={15} className={status === 'loading' ? 'animate-spin' : ''} />
        </button>
      </div>
    </section>
  );
}
