'use client';

import GpuMonitor from '@/components/GPUMonitor';
import JobsTable from '@/components/JobsTable';
import TensorBoardLink from '@/components/TensorBoardLink';
import { TopBar, MainContent } from '@/components/layout';
import useGPUInfo from '@/hooks/useGPUInfo';
import useJobsList from '@/hooks/useJobsList';
import useQueueList from '@/hooks/useQueueList';
import Link from 'next/link';
import {
  Activity,
  Cpu,
  Database,
  Flame,
  Gauge,
  ImagePlus,
  ListOrdered,
  MemoryStick,
  Plus,
} from 'lucide-react';

const activeStatuses = new Set(['queued', 'running', 'stopping']);

function formatMemory(mb: number) {
  if (!Number.isFinite(mb) || mb <= 0) return '0 MB';
  return mb >= 1024 ? `${(mb / 1024).toFixed(mb >= 10240 ? 0 : 1)} GB` : `${Math.round(mb)} MB`;
}

function clampPercent(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(100, Math.max(0, Math.round(value)));
}

export default function Dashboard() {
  const { gpuList, isGPUInfoLoaded, status: gpuStatus } = useGPUInfo();
  const { jobs } = useJobsList({ onlyActive: true, reloadInterval: 5000 });
  const { queues } = useQueueList();

  const runningJobs = jobs.filter(job => ['running', 'stopping'].includes(job.status)).length;
  const queuedJobs = jobs.filter(job => job.status === 'queued').length;
  const activeJobs = jobs.filter(job => activeStatuses.has(job.status)).length;
  const runningQueues = queues.filter(queue => queue.is_running).length;
  const queueCount = queues.length || gpuList.length;
  const totalMemory = gpuList.reduce((sum, gpu) => sum + (gpu.memory.total || 0), 0);
  const freeMemory = gpuList.reduce((sum, gpu) => sum + (gpu.memory.free || 0), 0);
  const usedMemory = Math.max(0, totalMemory - freeMemory);
  const maxGpuLoad = gpuList.reduce((max, gpu) => Math.max(max, gpu.utilization.gpu || 0), 0);
  const hottestGpu = gpuList.reduce((max, gpu) => Math.max(max, gpu.temperature || 0), 0);
  const primaryGpuName = gpuList[0]?.name || 'local GPU';
  const queuesAreRunning = runningQueues > 0;
  const gpuReady = isGPUInfoLoaded && gpuStatus !== 'error' && gpuList.length > 0;

  const vramUsedPercent = totalMemory > 0 ? clampPercent((usedMemory / totalMemory) * 100) : 0;
  const loadPercent = clampPercent(maxGpuLoad);
  const queuePercent = queueCount > 0 ? clampPercent((runningQueues / queueCount) * 100) : 0;

  const heroTitle =
    activeJobs > 0
      ? runningJobs > 0
        ? 'Training is underway.'
        : 'Your next run is queued.'
      : gpuReady
        ? 'Your AI workspace is ready.'
        : 'Bring your workspace online.';

  const heroDetail =
    activeJobs > 0
      ? `${runningJobs} running, ${queuedJobs} queued. Keep an eye on throughput and queue position below.`
      : gpuReady
        ? `${primaryGpuName} has ${formatMemory(freeMemory)} free VRAM waiting for the next idea.`
        : gpuStatus === 'error'
          ? 'GPU telemetry needs attention before the dashboard can show readiness.'
          : 'AI Toolkit is checking the local worker and GPU telemetry.';

  const statusTone = gpuReady
    ? 'text-emerald-300'
    : gpuStatus === 'error'
      ? 'text-rose-300'
      : 'text-amber-300';

  const statusDotTone = gpuReady
    ? 'bg-emerald-400'
    : gpuStatus === 'error'
      ? 'bg-rose-400'
      : 'bg-amber-400';

  const statCards = [
    {
      label: 'Accelerators',
      icon: Cpu,
      iconTone: 'text-cyan-300',
      iconBg: 'bg-cyan-500/10 ring-cyan-500/20',
      value: gpuReady ? `${gpuList.length} GPU${gpuList.length === 1 ? '' : 's'}` : '--',
      detail: gpuReady ? primaryGpuName : 'scanning hardware',
      barPercent: gpuReady ? 100 : 0,
      barTone: 'bg-cyan-400/70',
    },
    {
      label: 'VRAM',
      icon: MemoryStick,
      iconTone: 'text-violet-300',
      iconBg: 'bg-violet-500/10 ring-violet-500/20',
      value: formatMemory(freeMemory),
      detail: `free of ${formatMemory(totalMemory)}`,
      barPercent: vramUsedPercent,
      barTone: vramUsedPercent > 90 ? 'bg-rose-400/80' : 'bg-violet-400/70',
    },
    {
      label: 'Load',
      icon: Gauge,
      iconTone: 'text-amber-300',
      iconBg: 'bg-amber-500/10 ring-amber-500/20',
      value: gpuReady ? `${maxGpuLoad}%` : '--',
      detail: gpuReady ? `${hottestGpu}\u00B0C peak` : 'pending telemetry',
      barPercent: loadPercent,
      barTone: loadPercent > 90 ? 'bg-rose-400/80' : 'bg-amber-400/70',
    },
    {
      label: 'Queue',
      icon: ListOrdered,
      iconTone: 'text-emerald-300',
      iconBg: 'bg-emerald-500/10 ring-emerald-500/20',
      value: queueCount ? `${runningQueues}/${queueCount}` : '--',
      detail: queueCount ? (queuesAreRunning ? 'queues running' : 'queue stopped') : 'no queues',
      barPercent: queuePercent,
      barTone: 'bg-emerald-400/70',
    },
  ];

  return (
    <>
      <TopBar>
        <div className="flex shrink-0 items-center gap-2">
          <Activity className="h-4 w-4 text-cyan-300" />
          <h1 className="text-base font-semibold">Dashboard</h1>
        </div>
        <div className="flex-1"></div>
        <Link href="/jobs/new" className="operator-button py-1 text-xs">
          <Plus className="h-3.5 w-3.5" />
          New Job
        </Link>
        <TensorBoardLink />
      </TopBar>
      <MainContent className="bg-gray-950 px-0 pt-12 sm:px-0">
        <section className="relative overflow-hidden border-b border-gray-900 bg-gray-950">
          {/* Decorative background accents */}
          <div className="pointer-events-none absolute inset-0" aria-hidden="true">
            <div className="absolute -top-32 left-1/4 h-72 w-72 rounded-full bg-cyan-500/[0.07] blur-3xl" />
            <div className="absolute -top-24 right-1/4 h-64 w-64 rounded-full bg-violet-500/[0.06] blur-3xl" />
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-500/40 to-transparent" />
          </div>

          <div className="relative mx-auto max-w-[1680px] px-3 py-8 sm:px-4 lg:px-5">
            <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
              <div className="min-w-0">
                <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-gray-500">
                  <span className={`inline-flex items-center gap-2 font-medium ${statusTone}`}>
                    <span className="relative flex h-2 w-2">
                      {gpuReady && (
                        <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${statusDotTone}`} />
                      )}
                      <span className={`relative inline-flex h-2 w-2 rounded-full ${statusDotTone}`} />
                    </span>
                    {gpuReady ? `${gpuList.length} GPU${gpuList.length === 1 ? '' : 's'} online` : 'Checking GPU'}
                  </span>
                  <span className="hidden h-3 w-px bg-gray-800 sm:inline-block" />
                  <span>{formatMemory(freeMemory)} free VRAM</span>
                  <span className="hidden h-3 w-px bg-gray-800 sm:inline-block" />
                  <span>
                    {queueCount
                      ? queuesAreRunning
                        ? `${runningQueues}/${queueCount} queues running`
                        : 'Queue stopped'
                      : 'No queues'}
                  </span>
                  <span className="hidden h-3 w-px bg-gray-800 sm:inline-block" />
                  <span>
                    {activeJobs} active job{activeJobs === 1 ? '' : 's'}
                  </span>
                </div>
                <h2 className="max-w-3xl bg-gradient-to-r from-gray-50 via-gray-100 to-gray-400 bg-clip-text text-3xl font-semibold tracking-tight text-transparent sm:text-4xl">
                  {heroTitle}
                </h2>
                <p className="mt-3 max-w-3xl text-sm leading-6 text-gray-400">{heroDetail}</p>
              </div>

              <div className="flex flex-wrap items-center gap-2 lg:justify-end">
                <Link
                  href="/jobs/new"
                  className="operator-button border-emerald-700/70 bg-gradient-to-b from-emerald-900/70 to-emerald-950/70 px-4 py-2 text-sm font-medium text-emerald-100 shadow-[0_0_18px_-6px_rgba(16,185,129,0.45)] transition-shadow hover:border-emerald-600 hover:from-emerald-800/80 hover:to-emerald-900/80 hover:shadow-[0_0_24px_-6px_rgba(16,185,129,0.6)]"
                >
                  <Plus className="h-4 w-4" />
                  Start Training
                </Link>
                <Link href="/generate" className="operator-button px-3 py-2 text-sm transition-colors">
                  <ImagePlus className="h-4 w-4 text-cyan-300" />
                  Generate
                </Link>
                <Link href="/datasets" className="operator-button px-3 py-2 text-sm transition-colors">
                  <Database className="h-4 w-4 text-violet-300" />
                  Datasets
                </Link>
              </div>
            </div>

            <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {statCards.map(card => {
                const Icon = card.icon;
                return (
                  <div
                    key={card.label}
                    className="group relative min-w-0 overflow-hidden rounded-md border border-gray-800/80 bg-gray-900/50 p-4 transition-colors hover:border-gray-700 hover:bg-gray-900/70"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="text-[10px] font-semibold uppercase tracking-widest text-gray-500">
                          {card.label}
                        </div>
                        <div className="mt-1.5 truncate text-xl font-semibold text-gray-100">{card.value}</div>
                        <div className="mt-0.5 truncate text-xs text-gray-500">{card.detail}</div>
                      </div>
                      <div
                        className={`flex h-9 w-9 flex-none items-center justify-center rounded-md ring-1 ${card.iconBg}`}
                      >
                        <Icon className={`h-[18px] w-[18px] ${card.iconTone}`} />
                      </div>
                    </div>
                    <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-gray-800/80">
                      <div
                        className={`h-full rounded-full transition-all duration-700 ease-out ${card.barTone}`}
                        style={{ width: `${card.barPercent}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        <div className="mx-auto max-w-[1680px] space-y-8 px-3 py-6 sm:px-4 lg:px-5">
          <GpuMonitor />
          <section className="w-full">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-2.5">
                <div className="flex h-8 w-8 flex-none items-center justify-center rounded-md bg-cyan-500/10 ring-1 ring-cyan-500/20">
                  <Flame className="h-4 w-4 text-cyan-300" />
                </div>
                <div className="min-w-0">
                  <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-300">Active queues</h2>
                  <p className="mt-0.5 truncate text-xs text-gray-600">
                    Live queue position, worker state, and job progress
                  </p>
                </div>
              </div>
              <Link
                href="/jobs"
                className="shrink-0 rounded-sm border border-transparent px-2 py-1 text-xs text-cyan-300 transition-colors hover:border-gray-800 hover:bg-gray-900 hover:text-cyan-200"
              >
                View all
              </Link>
            </div>
            <JobsTable onlyActive />
          </section>
        </div>
      </MainContent>
    </>
  );
}