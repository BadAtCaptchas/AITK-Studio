'use client';

import { useMemo, useRef, useState, useEffect } from 'react';
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from 'recharts';
import {
  Activity,
  Cpu,
  Download,
  HardDrive,
  ListFilter,
  MemoryStick,
  Save,
  Thermometer,
  Trash2,
  Zap,
} from 'lucide-react';
import type { Job, SystemMetricSample } from '@/types';
import useSystemTelemetry from '@/hooks/useSystemTelemetry';

type Props = {
  job: Job;
  dashboard: any;
};

function formatBytes(bytes: number | null | undefined) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

function formatMB(mb: number) {
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(0)} MB`;
}

function numericSpeed(speed: string) {
  const match = speed.match(/([\d.]+)/);
  return match ? Number(match[1]) : null;
}

function latest(samples: SystemMetricSample[], scope: string, metric: string, device = 'host') {
  for (let i = samples.length - 1; i >= 0; i--) {
    const sample = samples[i];
    if (sample.scope === scope && sample.metric === metric && (device === '*' || sample.device_id === device)) {
      return sample;
    }
  }
  return null;
}

function metricSeries(samples: SystemMetricSample[], scope: string, metric: string, device = '*') {
  return samples
    .filter(sample => sample.scope === scope && sample.metric === metric && (device === '*' || sample.device_id === device))
    .slice(-60)
    .map(sample => sample.value);
}

function Sparkline({ values, color = '#2563eb' }: { values: number[]; color?: string }) {
  const points = useMemo(() => {
    if (values.length < 2) return '';
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    return values
      .map((value, index) => {
        const x = (index / (values.length - 1)) * 96;
        const y = 28 - ((value - min) / span) * 24;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }, [values]);
  return (
    <svg className="h-8 w-24" viewBox="0 0 96 32" aria-hidden="true">
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.7" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
  series,
  color,
}: {
  icon: any;
  label: string;
  value: string;
  detail: string;
  series: number[];
  color: string;
}) {
  return (
    <div className="flex min-w-[190px] items-center gap-3 border-r border-white/10 px-5 py-3">
      <Icon className="h-4 w-4 shrink-0" style={{ color }} />
      <div className="min-w-0 flex-1">
        <div className="text-[11px] uppercase tracking-wide text-gray-500">{label}</div>
        <div className="mt-1 text-lg leading-none text-white">{value}</div>
        <div className="mt-1 truncate text-[11px] text-gray-500">{detail}</div>
      </div>
      <Sparkline values={series} color={color} />
    </div>
  );
}

function LogLine({ line }: { line: string }) {
  const color = /error|traceback/i.test(line)
    ? 'text-red-400'
    : /warn/i.test(line)
      ? 'text-yellow-400'
      : /\btrain\b/i.test(line)
        ? 'text-green-400'
        : /\binfo\b/i.test(line)
          ? 'text-blue-400'
          : 'text-gray-300';
  return <pre className={`whitespace-pre-wrap break-words leading-relaxed ${color}`}>{line}</pre>;
}

function ConsoleLogs({ lines }: { lines: string[] }) {
  const [tab, setTab] = useState<'logs' | 'events' | 'metrics' | 'system'>('logs');
  const [autoScroll, setAutoScroll] = useState(true);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoScroll && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines, autoScroll]);

  const shownLines =
    tab === 'logs'
      ? lines
      : tab === 'events'
        ? lines.filter(line => /started|stopped|saved|checkpoint|error|warn/i.test(line))
        : tab === 'metrics'
          ? lines.filter(line => /loss|lr|tok|gpu|vram|step/i.test(line))
          : lines.filter(line => /cpu|gpu|vram|ram|cuda|device/i.test(line));

  return (
    <section className="flex min-h-[520px] flex-col border-r border-white/10">
      <div className="flex h-11 items-center gap-6 border-b border-white/10 px-4 text-xs uppercase tracking-wide">
        {(['logs', 'events', 'metrics', 'system'] as const).map(item => (
          <button
            key={item}
            type="button"
            onClick={() => setTab(item)}
            className={item === tab ? 'h-full border-b-2 border-blue-500 text-white' : 'text-gray-500 hover:text-gray-200'}
          >
            {item}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-4 normal-case tracking-normal text-gray-400">
          <button type="button" onClick={() => setAutoScroll(v => !v)} className="flex items-center gap-2 hover:text-white">
            <span className={autoScroll ? 'h-2 w-2 rounded-full bg-green-500' : 'h-2 w-2 rounded-full bg-gray-600'} />
            Auto-scroll
          </button>
          <button type="button" disabled className="flex cursor-not-allowed items-center gap-2 opacity-50">
            <Trash2 className="h-3.5 w-3.5" />
            Clear
          </button>
          <button type="button" disabled className="flex cursor-not-allowed items-center gap-2 opacity-50">
            <Download className="h-3.5 w-3.5" />
            Download
          </button>
        </div>
      </div>
      <div className="flex items-center gap-2 px-4 py-3 text-xs text-gray-400">
        <span className="h-2 w-2 rounded-full bg-green-500" />
        Streaming live logs...
      </div>
      <div ref={ref} className="min-h-0 flex-1 overflow-auto px-4 pb-4 font-mono text-[12px]">
        {shownLines.length ? shownLines.map((line, index) => <LogLine key={`${index}-${line}`} line={line} />) : <div className="text-gray-500">No matching log lines.</div>}
      </div>
      <div className="flex h-10 items-center border-t border-white/10 px-4 text-xs text-gray-500">
        <span>Log level: INFO</span>
        <span className="ml-auto">Lines: {lines.length.toLocaleString()}</span>
      </div>
    </section>
  );
}

function LossPanel({ dashboard }: { dashboard: any }) {
  const data = useMemo(() => {
    const rows = new Map<number, any>();
    for (const series of dashboard.lossSeries ?? []) {
      for (const point of series.points ?? []) {
        const row = rows.get(point.step) ?? { step: point.step };
        row[series.key] = point.value;
        rows.set(point.step, row);
      }
    }
    return Array.from(rows.values()).sort((a, b) => a.step - b.step).slice(-500);
  }, [dashboard.lossSeries]);

  const keys = (dashboard.lossSeries ?? []).map((series: any) => series.key);
  const latestRows = keys.map((key: string) => {
    const points = dashboard.lossSeries.find((series: any) => series.key === key)?.points ?? [];
    const values = points.map((point: any) => point.value).filter((value: any) => typeof value === 'number');
    return { key, current: values[values.length - 1], min: Math.min(...values), max: Math.max(...values), avg: values.reduce((a: number, b: number) => a + b, 0) / Math.max(1, values.length) };
  });

  return (
    <section className="border-b border-white/10 p-4">
      <div className="mb-3 flex items-center">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-white">Loss Graph</h2>
        <div className="ml-auto flex gap-1 text-[11px]">
          {['1H', '6H', '24H', 'ALL'].map(label => (
            <button key={label} className={`rounded border border-white/10 px-2 py-1 ${label === '6H' ? 'bg-white/10 text-white' : 'text-gray-500'}`}>
              {label}
            </button>
          ))}
        </div>
      </div>
      <div className="h-52">
        {data.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 12, right: 12, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.08)" />
              <XAxis dataKey="step" tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={{ stroke: 'rgba(255,255,255,0.12)' }} />
              <YAxis tick={{ fill: '#8a8a8a', fontSize: 11 }} axisLine={{ stroke: 'rgba(255,255,255,0.12)' }} width={42} />
              <Tooltip contentStyle={{ background: '#070707', border: '1px solid rgba(255,255,255,.15)', color: '#fff' }} />
              {keys.map((key: string, index: number) => (
                <Line key={key} dataKey={key} stroke={index === 0 ? '#2563eb' : '#60a5fa'} strokeWidth={index === 0 ? 2 : 1.5} dot={false} type="monotone" isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center border border-white/10 text-sm text-gray-500">Waiting for loss data...</div>
        )}
      </div>
      <div className="mt-3 grid grid-cols-5 gap-2 border-t border-white/10 pt-3 text-xs">
        <div className="text-gray-500">Series</div>
        <div className="text-gray-500">Current</div>
        <div className="text-gray-500">Min</div>
        <div className="text-gray-500">Max</div>
        <div className="text-gray-500">Avg</div>
        {latestRows.map(row => (
          <div key={row.key} className="contents">
            <div className="text-white">{row.key}</div>
            <div>{Number.isFinite(row.current) ? row.current.toFixed(4) : '-'}</div>
            <div>{Number.isFinite(row.min) ? row.min.toFixed(4) : '-'}</div>
            <div>{Number.isFinite(row.max) ? row.max.toFixed(4) : '-'}</div>
            <div>{Number.isFinite(row.avg) ? row.avg.toFixed(4) : '-'}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

function CheckpointsPanel({ dashboard }: { dashboard: any }) {
  return (
    <section className="border-b border-white/10 p-4">
      <div className="mb-3 flex items-center">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-white">Checkpoints</h2>
        <span className="ml-auto rounded-full bg-green-500/10 px-2 py-0.5 text-[11px] text-green-400">Auto-save: ON</span>
        <button className="ml-3 flex items-center gap-1 text-xs text-gray-300">
          <Save className="h-3.5 w-3.5" />
          Save Now
        </button>
      </div>
      <div className="grid grid-cols-[1fr_70px_74px_76px_90px_24px] gap-2 text-[11px] uppercase text-gray-500">
        <div>Name</div>
        <div>Step</div>
        <div>Time</div>
        <div>Size</div>
        <div>Metrics</div>
        <div />
      </div>
      <div className="mt-2 space-y-1">
        {(dashboard.checkpoints ?? []).slice(0, 6).map((checkpoint: any) => (
          <a
            key={checkpoint.path}
            href={`/api/files/${encodeURIComponent(checkpoint.path)}`}
            className={`grid grid-cols-[1fr_70px_74px_76px_90px_24px] items-center gap-2 rounded px-1.5 py-1.5 text-xs hover:bg-white/5 ${checkpoint.is_latest ? 'bg-white/5 text-white' : 'text-gray-300'}`}
          >
            <div className="truncate">
              <span className={`mr-2 inline-block h-2 w-2 rounded-full ${checkpoint.is_latest ? 'bg-green-500' : 'bg-gray-500'}`} />
              {checkpoint.name}
            </div>
            <div>{checkpoint.step?.toLocaleString() ?? '-'}</div>
            <div>{new Date(checkpoint.modified_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
            <div>{formatBytes(checkpoint.size)}</div>
            <div>Loss {dashboard.lossSeries?.[0]?.points?.at(-1)?.value?.toFixed?.(4) ?? '-'}</div>
            <div className="text-gray-500">⋮</div>
          </a>
        ))}
        {!dashboard.checkpoints?.length && <div className="py-6 text-center text-sm text-gray-500">No checkpoints found.</div>}
      </div>
    </section>
  );
}

function ConfigSummary({ dashboard }: { dashboard: any }) {
  const config = dashboard.configSummary ?? {};
  const rows = [
    ['Model', config.model],
    ['Quantization', config.quantization],
    ['Precision', config.precision],
    ['Sequence Length', config.sequence_length ?? 'n/a'],
    ['Global Batch Size', config.batch_size],
    ['Gradient Accumulation', config.gradient_accumulation],
    ['Optimizer', config.optimizer],
    ['Scheduler', config.scheduler],
    ['Dataset', config.dataset],
    ['Train / Val Split', config.train_val_split],
    ['Total Samples', config.total_samples ?? 'n/a'],
    ['Dataloader Workers', config.dataloader_workers ?? 'n/a'],
    ['Max Steps', config.max_steps],
    ['Save Steps', config.save_steps],
    ['Log Steps', config.log_steps],
    ['Seed', config.seed],
  ];
  return (
    <section className="p-4">
      <div className="mb-3 flex items-center">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-white">Configuration</h2>
        <button className="ml-auto text-xs text-gray-400">Edit Config</button>
      </div>
      <div className="grid grid-cols-1 gap-x-8 gap-y-2 text-xs md:grid-cols-2">
        {rows.map(([label, value]) => (
          <div key={String(label)} className="grid grid-cols-[130px_1fr] gap-3">
            <div className="text-gray-500">{label}</div>
            <div className="truncate text-gray-300">{String(value ?? 'n/a')}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function JobConsole({ job, dashboard }: Props) {
  const { samples } = useSystemTelemetry('6h', 5000);
  const gpuID = job.gpu_ids === 'mps' ? '0' : job.gpu_ids.split(',')[0] || '0';
  const cpu = latest(samples, 'cpu', 'cpu_utilization');
  const gpu = latest(samples, 'gpu', 'gpu_utilization', gpuID);
  const vram = latest(samples, 'gpu', 'vram_used_mb', gpuID);
  const temp = latest(samples, 'gpu', 'temperature', gpuID);
  const power = latest(samples, 'gpu', 'power_draw', gpuID);
  const ram = latest(samples, 'ram', 'ram_used_mb');
  const ramTotal = ram ? JSON.parse(ram.metadata || '{}').total_mb : null;
  const vramTotal = vram ? JSON.parse(vram.metadata || '{}').total_mb : null;
  const speed = numericSpeed(job.speed_string);

  return (
    <div className="h-full min-h-0 bg-black text-gray-300">
      <div className="flex overflow-x-auto border-b border-white/10">
        <MetricCard icon={Cpu} label="CPU" value={`${(cpu?.value ?? 0).toFixed(1)}%`} detail={cpu ? JSON.parse(cpu.metadata || '{}').name || 'CPU' : 'No CPU data'} series={metricSeries(samples, 'cpu', 'cpu_utilization')} color="#3b82f6" />
        <MetricCard icon={Activity} label="GPU" value={`${(gpu?.value ?? 0).toFixed(1)}%`} detail={gpu ? JSON.parse(gpu.metadata || '{}').name || `GPU ${gpuID}` : `GPU ${gpuID}`} series={metricSeries(samples, 'gpu', 'gpu_utilization', gpuID)} color="#22c55e" />
        <MetricCard icon={HardDrive} label="VRAM" value={`${formatMB(vram?.value ?? 0)}`} detail={vramTotal ? `/ ${formatMB(vramTotal)}` : 'No VRAM total'} series={metricSeries(samples, 'gpu', 'vram_used_mb', gpuID)} color="#22c55e" />
        <MetricCard icon={Thermometer} label="Temp" value={`${(temp?.value ?? 0).toFixed(0)} C`} detail="GPU" series={metricSeries(samples, 'gpu', 'temperature', gpuID)} color="#10b981" />
        <MetricCard icon={Zap} label="Power" value={`${(power?.value ?? 0).toFixed(0)} W`} detail="GPU" series={metricSeries(samples, 'gpu', 'power_draw', gpuID)} color="#eab308" />
        <MetricCard icon={MemoryStick} label="RAM" value={`${formatMB(ram?.value ?? 0)}`} detail={ramTotal ? `/ ${formatMB(ramTotal)}` : 'System memory'} series={metricSeries(samples, 'ram', 'ram_used_mb')} color="#3b82f6" />
        <MetricCard icon={ListFilter} label="Training Speed" value={speed == null ? '?' : speed.toFixed(2)} detail={job.speed_string || 'Tokens/sec'} series={speed == null ? [] : [speed * 0.9, speed, speed * 1.02, speed * 0.98, speed]} color="#22c55e" />
      </div>

      <div className="grid min-h-0 grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(420px,40vw)]">
        <ConsoleLogs lines={dashboard.logLines ?? []} />
        <aside className="min-w-0 border-l border-white/10">
          <LossPanel dashboard={dashboard} />
          <CheckpointsPanel dashboard={dashboard} />
          <ConfigSummary dashboard={dashboard} />
        </aside>
      </div>
    </div>
  );
}
