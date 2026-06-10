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
  CheckCircle2,
  Database,
  ImagePlus,
  ListOrdered,
  Plus,
} from 'lucide-react';

const activeStatuses = new Set(['queued', 'running', 'stopping']);

function formatMemory(mb: number) {
  if (!Number.isFinite(mb) || mb <= 0) return '0 MB';
  return mb >= 1024 ? `${(mb / 1024).toFixed(mb >= 10240 ? 0 : 1)} GB` : `${Math.round(mb)} MB`;
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
  const maxGpuLoad = gpuList.reduce((max, gpu) => Math.max(max, gpu.utilization.gpu || 0), 0);
  const hottestGpu = gpuList.reduce((max, gpu) => Math.max(max, gpu.temperature || 0), 0);
  const primaryGpuName = gpuList[0]?.name || 'local GPU';
  const queuesAreRunning = runningQueues > 0;
  const gpuReady = isGPUInfoLoaded && gpuStatus !== 'error' && gpuList.length > 0;

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
        <section className="border-b border-gray-900 bg-gray-950">
          <div className="mx-auto max-w-[1680px] px-3 py-6 sm:px-4 lg:px-5">
            <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
              <div className="min-w-0">
                <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-gray-500">
                  <span
                    className={`inline-flex items-center gap-1.5 font-medium ${
                      gpuReady ? 'text-emerald-300' : gpuStatus === 'error' ? 'text-rose-300' : 'text-amber-300'
                    }`}
                  >
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    {gpuReady ? `${gpuList.length} GPU${gpuList.length === 1 ? '' : 's'} online` : 'Checking GPU'}
                  </span>
                  <span>{formatMemory(freeMemory)} free VRAM</span>
                  <span>
                    {queueCount
                      ? queuesAreRunning
                        ? `${runningQueues}/${queueCount} queues running`
                        : 'Queue stopped'
                      : 'No queues'}
                  </span>
                  <span>
                    {activeJobs} active job{activeJobs === 1 ? '' : 's'}
                  </span>
                </div>
                <h2 className="max-w-3xl text-2xl font-semibold text-gray-100 sm:text-3xl">{heroTitle}</h2>
                <p className="mt-2 max-w-3xl text-sm leading-6 text-gray-400">{heroDetail}</p>
              </div>

              <div className="flex flex-wrap items-center gap-2 lg:justify-end">
                <Link
                  href="/jobs/new"
                  className="operator-button border-emerald-800 bg-emerald-950/55 px-4 py-2 text-sm font-medium text-emerald-100 hover:bg-emerald-900"
                >
                  <Plus className="h-4 w-4" />
                  Start Training
                </Link>
                <Link href="/generate" className="operator-button px-3 py-2 text-sm">
                  <ImagePlus className="h-4 w-4" />
                  Generate
                </Link>
                <Link href="/datasets" className="operator-button px-3 py-2 text-sm">
                  <Database className="h-4 w-4" />
                  Datasets
                </Link>
              </div>
            </div>

            <div className="mt-6 grid gap-3 border-t border-gray-900 pt-4 text-sm sm:grid-cols-2 lg:grid-cols-4">
              <div className="min-w-0">
                <div className="text-[10px] font-semibold uppercase tracking-wide text-gray-600">Accelerators</div>
                <div className="mt-1 truncate text-gray-200">
                  {gpuReady ? `${gpuList.length} GPU${gpuList.length === 1 ? '' : 's'}` : '--'}
                  <span className="ml-2 text-xs text-gray-500">{gpuReady ? 'online' : 'scanning'}</span>
                </div>
              </div>
              <div className="min-w-0">
                <div className="text-[10px] font-semibold uppercase tracking-wide text-gray-600">VRAM</div>
                <div className="mt-1 truncate text-gray-200">
                  {formatMemory(freeMemory)}
                  <span className="ml-2 text-xs text-gray-500">of {formatMemory(totalMemory)}</span>
                </div>
              </div>
              <div className="min-w-0">
                <div className="text-[10px] font-semibold uppercase tracking-wide text-gray-600">Load</div>
                <div className="mt-1 truncate text-gray-200">
                  {gpuReady ? `${maxGpuLoad}%` : '--'}
                  <span className="ml-2 text-xs text-gray-500">{gpuReady ? `${hottestGpu}C peak` : 'pending'}</span>
                </div>
              </div>
              <div className="min-w-0">
                <div className="text-[10px] font-semibold uppercase tracking-wide text-gray-600">Queue</div>
                <div className="mt-1 truncate text-gray-200">
                  {queueCount ? `${runningQueues}/${queueCount}` : '--'}
                  <span className="ml-2 text-xs text-gray-500">{queuesAreRunning ? 'running' : 'stopped'}</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <div className="mx-auto max-w-[1680px] space-y-6 px-3 py-5 sm:px-4 lg:px-5">
          <GpuMonitor />
          <section className="w-full">
            <div className="mb-2 flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-2">
                <ListOrdered className="h-4 w-4 flex-none text-gray-500" />
                <div className="min-w-0">
                  <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-400">Active queues</h2>
                  <p className="mt-0.5 truncate text-xs text-gray-600">Live queue position, worker state, and job progress</p>
                </div>
              </div>
              <Link href="/jobs" className="shrink-0 text-xs text-cyan-300 hover:text-cyan-200">
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
