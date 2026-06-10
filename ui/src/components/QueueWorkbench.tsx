'use client';

import { useEffect, useMemo, useState } from 'react';
import type { DragEvent } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import classNames from 'classnames';
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  CircleDashed,
  Clock3,
  Cpu,
  GripVertical,
  History,
  Info,
  ListFilter,
  Loader2,
  MoreVertical,
  PauseCircle,
  Play,
  PlayCircle,
  RefreshCcw,
  Rocket,
  Settings,
  SlidersHorizontal,
  X,
  XCircle,
} from 'lucide-react';
import JobActionBar from '@/components/JobActionBar';
import { PageNotice, ProgressBar, QueueStateBadge, StatusBadge } from '@/components/OperatorPrimitives';
import { HFDownloadProgressInline } from '@/components/HFDownloadProgress';
import { ComfyInstallProgressInline } from '@/components/ComfyInstallProgress';
import useGPUInfo from '@/hooks/useGPUInfo';
import useJobLog from '@/hooks/useJobLog';
import useJobsList from '@/hooks/useJobsList';
import useQueueList from '@/hooks/useQueueList';
import useWorkers from '@/hooks/useWorkers';
import type { Job, Queue } from '@/types';
import { getAvaliableJobActions, getTotalSteps } from '@/utils/jobs';
import { reorderQueue, startQueue, stopQueue } from '@/utils/queue';

type QueueWorkbenchProps = {
  filterText: string;
  focusGpuIDs?: string | null;
};

type TabKey = 'active' | 'history' | 'failed' | 'all';
type SortKey = 'newest' | 'oldest' | 'name';

type JobGroup = {
  name: string;
  jobs: Job[];
  workerID: string;
  gpuIDs: string | null;
};

type PendingReorder = {
  laneKey: string;
  workerID: string;
  gpuIDs: string;
  jobIDs: string[];
};

const activeJobStatuses = new Set(['queued', 'running', 'stopping']);

function getLaneKey(workerID: string, gpuIDs: string | null) {
  return `${workerID}:${gpuIDs || ''}`;
}

function isQueuedJob(job: Job) {
  return job.status === 'queued';
}

function hasSameIDs(a: string[], b: string[]) {
  if (a.length !== b.length) return false;
  const set = new Set(a);
  return b.every(id => set.has(id));
}

function applyPendingOrder(group: JobGroup, pending: PendingReorder | null): JobGroup {
  if (!pending || pending.laneKey !== getLaneKey(group.workerID, group.gpuIDs)) return group;
  const queuedJobs = group.jobs.filter(isQueuedJob);
  const queuedIDs = queuedJobs.map(job => job.id);
  if (!hasSameIDs(queuedIDs, pending.jobIDs)) return group;

  const queuedByID = new Map(queuedJobs.map(job => [job.id, job]));
  const orderedQueuedJobs = pending.jobIDs.map(id => queuedByID.get(id)).filter(Boolean) as Job[];
  let queuedIndex = 0;

  return {
    ...group,
    jobs: group.jobs.map(job => (isQueuedJob(job) ? orderedQueuedJobs[queuedIndex++] || job : job)),
  };
}

function isRemoteWorker(workerID: string) {
  return workerID !== 'local';
}

function jobDisplayTitle(row: Job) {
  if (row.job_type === 'caption') {
    const splits = (row.job_ref || '').split(/[/\\]/);
    return { prefix: 'Caption', title: splits[splits.length - 1] || row.name };
  }
  if (row.job_type === 'generate') {
    return { prefix: 'Generate', title: row.name };
  }
  return { prefix: 'Train', title: row.name };
}

function safeTotalSteps(job: Job) {
  if (job.job_type !== 'train') return null;
  try {
    return getTotalSteps(job);
  } catch {
    return null;
  }
}

function getProgress(job: Job) {
  const totalSteps = safeTotalSteps(job);
  if (!totalSteps) return 0;
  return Math.max(0, Math.min(100, (job.step / totalSteps) * 100));
}

function formatDate(value: Job['updated_at']) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return 'Unknown';
  return date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function formatRelative(value: Job['updated_at']) {
  const date = new Date(value);
  const time = date.getTime();
  if (!Number.isFinite(time)) return 'Updated recently';
  const minutes = Math.max(0, Math.round((Date.now() - time) / 60000));
  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

function shortID(value: string) {
  return value.length > 10 ? `${value.slice(0, 10)}...` : value;
}

function sortHistoryJobs(jobs: Job[], sort: SortKey) {
  const sorted = [...jobs];
  if (sort === 'name') {
    return sorted.sort((a, b) => a.name.localeCompare(b.name));
  }
  return sorted.sort((a, b) => {
    const aTime = new Date(a.updated_at).getTime();
    const bTime = new Date(b.updated_at).getTime();
    return sort === 'oldest' ? aTime - bTime : bTime - aTime;
  });
}

function statusAccent(job: Job) {
  if (job.status === 'error' || job.status === 'failed') {
    return 'border-l-rose-500 bg-rose-950/10';
  }
  if (job.status === 'running') {
    return 'border-l-blue-500 bg-blue-950/10';
  }
  if (job.status === 'queued') {
    return 'border-l-amber-500 bg-amber-950/10';
  }
  return 'border-l-gray-800 bg-transparent';
}

function WorkerHero({
  group,
  queue,
  isBusy,
  onRefresh,
}: {
  group: JobGroup;
  queue: Queue | undefined;
  isBusy: boolean;
  onRefresh: () => void;
}) {
  const queueRunning = queue?.is_running === true;
  const running = group.jobs.filter(job => job.status === 'running').length;
  const waiting = group.jobs.filter(job => job.status === 'queued').length;
  const total = group.jobs.length;

  const toggleQueue = async () => {
    if (!group.gpuIDs || isBusy) return;
    if (queueRunning) {
      await stopQueue(group.gpuIDs, group.workerID);
    } else {
      await startQueue(group.gpuIDs, group.workerID);
    }
    onRefresh();
  };

  return (
    <section className="overflow-hidden border border-gray-800 bg-[#0b1118]">
      <div className="flex flex-col gap-4 p-4 lg:flex-row lg:items-center">
        <div className="flex min-w-0 flex-1 items-start gap-3">
          <div className="flex h-11 w-11 flex-none items-center justify-center rounded-md border border-gray-700 bg-gray-950/80 text-blue-200">
            <Cpu className="h-5 w-5" />
          </div>
          <div className="min-w-0">
            <div className="flex min-w-0 flex-wrap items-center gap-2">
              <h2 className="truncate text-base font-semibold text-gray-100">{group.name}</h2>
              {group.gpuIDs && (
                <span className="rounded-md border border-gray-700 bg-gray-950 px-2 py-0.5 text-xs text-gray-300">
                  GPU {group.gpuIDs}
                </span>
              )}
              <MoreVertical className="h-4 w-4 text-gray-500" />
            </div>
            <div className="mt-3 flex items-center gap-2">
              <QueueStateBadge running={queueRunning} />
              <span className="text-sm text-gray-400">
                {queueRunning ? 'Jobs run in queue order on this worker.' : 'Start the queue to process waiting jobs.'}
              </span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2 text-center sm:min-w-[260px]">
          <div className="border-r border-gray-800 px-3">
            <div className="text-lg font-semibold text-gray-100">{running}</div>
            <div className="text-xs text-gray-500">Running</div>
          </div>
          <div className="border-r border-gray-800 px-3">
            <div className="text-lg font-semibold text-gray-100">{waiting}</div>
            <div className="text-xs text-gray-500">Waiting</div>
          </div>
          <div className="px-3">
            <div className="text-lg font-semibold text-gray-100">{total}</div>
            <div className="text-xs text-gray-500">Total Jobs</div>
          </div>
        </div>

        <div className="grid gap-2 sm:min-w-[180px]">
          <button
            type="button"
            disabled={!group.gpuIDs || isBusy}
            onClick={() => void toggleQueue()}
            className={classNames(
              'inline-flex h-11 items-center justify-center gap-2 rounded-md px-4 text-sm font-semibold text-white transition-colors disabled:cursor-not-allowed disabled:opacity-50',
              queueRunning ? 'bg-rose-700 hover:bg-rose-600' : 'bg-blue-600 hover:bg-blue-500',
            )}
          >
            {isBusy ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : queueRunning ? (
              <PauseCircle className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            {queueRunning ? 'Stop Queue' : 'Start Queue'}
          </button>
          <button
            type="button"
            onClick={onRefresh}
            className="inline-flex h-9 items-center justify-center gap-2 rounded-md border border-gray-800 bg-gray-950/80 text-sm text-gray-200 hover:bg-gray-900"
          >
            <RefreshCcw className="h-4 w-4" />
            Refresh
          </button>
        </div>
      </div>
    </section>
  );
}

function QueueEmptyState({ queueRunning }: { queueRunning: boolean }) {
  return (
    <div className="border-t border-gray-800 px-4 py-5 sm:pl-14">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-full border border-gray-700 text-gray-500">
          <CircleDashed className="h-5 w-5" />
        </div>
        <div className="min-w-0">
          <div className="font-medium text-gray-100">No jobs waiting on this GPU</div>
          <div className="mt-1 text-sm text-gray-400">
            {queueRunning ? 'Add a training job to keep this worker busy.' : 'Start the queue or add a training job.'}
          </div>
        </div>
        <Rocket className="ml-auto hidden h-6 w-6 text-gray-700 sm:block" />
      </div>
    </div>
  );
}

function QueueJobCard({
  job,
  index,
  showTimeline = true,
  selected,
  dragTarget,
  reorderBusy,
  onSelect,
  onDragStart,
  onDragOver,
  onDrop,
  onDragEnd,
  onRefresh,
}: {
  job: Job;
  index: number;
  showTimeline?: boolean;
  selected: boolean;
  dragTarget: boolean;
  reorderBusy: boolean;
  onSelect: (jobID: string) => void;
  onDragStart: (jobID: string) => void;
  onDragOver: (jobID: string, event: DragEvent<HTMLDivElement>) => void;
  onDrop: (jobID: string) => void;
  onDragEnd: () => void;
  onRefresh: () => void;
}) {
  const { prefix, title } = jobDisplayTitle(job);
  const totalSteps = safeTotalSteps(job);
  const canDrag = job.status === 'queued' && !reorderBusy;

  return (
    <div className={classNames('relative', showTimeline && 'pl-10')}>
      {showTimeline && (
        <>
          <div className="absolute left-[0.95rem] top-0 h-full w-px bg-gray-800" />
          <div
            className={classNames(
              'absolute left-0 top-6 z-[1] flex h-8 w-8 items-center justify-center rounded-full border bg-[#070b10] text-sm font-semibold',
              job.status === 'error' || job.status === 'failed'
                ? 'border-rose-500 text-rose-100'
                : selected
                  ? 'border-blue-400 text-blue-100'
                  : 'border-gray-700 text-gray-300',
            )}
          >
            {index + 1}
          </div>
        </>
      )}
      <div
        role="button"
        tabIndex={0}
        draggable={canDrag}
        onClick={() => onSelect(job.id)}
        onKeyDown={event => {
          if (event.key === 'Enter' || event.key === ' ') onSelect(job.id);
        }}
        onDragStart={event => {
          if (!canDrag) {
            event.preventDefault();
            return;
          }
          event.dataTransfer.effectAllowed = 'move';
          event.dataTransfer.setData('text/plain', job.id);
          onDragStart(job.id);
        }}
        onDragOver={event => onDragOver(job.id, event)}
        onDrop={event => {
          event.preventDefault();
          event.stopPropagation();
          onDrop(job.id);
        }}
        onDragEnd={onDragEnd}
        className={classNames(
          'group select-none overflow-hidden border-b border-l-2 border-r-0 border-t-0 border-gray-800 px-4 outline-none transition-colors hover:bg-gray-900/45',
          showTimeline ? 'py-4' : 'py-3',
          statusAccent(job),
          selected && (showTimeline ? 'border-l-blue-500 bg-blue-950/20' : 'bg-blue-950/10'),
          dragTarget && 'bg-blue-950/25',
          canDrag ? 'cursor-grab active:cursor-grabbing' : 'cursor-pointer',
        )}
      >
        <div className="flex min-w-0 items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex min-w-0 items-center gap-2">
              {!showTimeline && (
                <span className="w-5 flex-none text-right text-xs tabular-nums text-gray-600">{index + 1}</span>
              )}
              {canDrag && (
                <GripVertical
                  aria-label="Drag to reorder"
                  className="h-4 w-4 flex-none text-gray-600 group-hover:text-gray-400"
                />
              )}
              <span className="truncate text-base font-semibold text-gray-100">{title}</span>
              <span className="rounded-md border border-gray-700 bg-gray-950 px-2 py-0.5 text-[10px] uppercase tracking-wide text-gray-400">
                {prefix}
              </span>
            </div>
            <div className="mt-2 text-xs text-gray-500">Added {formatRelative(job.created_at)}</div>
          </div>
          <div className="flex flex-none items-center gap-2">
            <span className="text-xs text-gray-500">{formatRelative(job.updated_at)}</span>
            <ArrowRight className="h-4 w-4 text-gray-500 transition-transform group-hover:translate-x-0.5 group-hover:text-gray-300" />
          </div>
        </div>

        <div
          className={classNames(
            'grid gap-4 text-sm md:grid-cols-[1fr_0.55fr_0.45fr_0.75fr_1.8fr] md:items-start',
            showTimeline ? 'mt-4' : 'mt-3 pl-7',
          )}
        >
          <div className="min-w-0">
            <div className="text-xs text-gray-500">Steps</div>
            <div className="mt-1 text-gray-100">{totalSteps ? `${job.step} / ${totalSteps}` : `Step ${job.step}`}</div>
            {totalSteps ? <ProgressBar value={getProgress(job)} className="mt-2" /> : null}
          </div>
          <div className="min-w-0 border-gray-800 md:border-l md:pl-4">
            <div className="text-xs text-gray-500">Worker</div>
            <div className="mt-1 truncate text-gray-100">{job.worker_id === 'local' ? 'Local' : 'Remote'}</div>
          </div>
          <div className="border-gray-800 md:border-l md:pl-4">
            <div className="text-xs text-gray-500">GPU</div>
            <div className="mt-1 text-gray-100">{job.gpu_ids || '-'}</div>
          </div>
          <div className="border-gray-800 md:border-l md:pl-4">
            <div className="text-xs text-gray-500">Status</div>
            <div className="mt-1">
              <StatusBadge status={job.status} />
            </div>
          </div>
          <div className="min-w-0 border-gray-800 md:border-l md:pl-4">
            <div className="text-xs text-gray-500">Info</div>
            <div className="mt-1 min-w-0 text-gray-300">
              {job.comfy_install_progress ? (
                <ComfyInstallProgressInline progress={job.comfy_install_progress} fallback={job.info} />
              ) : (
                <HFDownloadProgressInline progress={job.hf_download_progress} fallback={job.info || 'No details yet'} />
              )}
            </div>
          </div>
        </div>

        <div
          className={classNames(
            'mt-4 flex items-center justify-between gap-3 border-t border-gray-800/70 pt-3',
            !showTimeline && 'pl-7',
          )}
        >
          <div className="min-w-0 truncate text-xs text-gray-500">
            Position {index + 1} - ID {shortID(job.id)}
          </div>
          <div
            className="flex-none"
            onClick={event => event.stopPropagation()}
            onKeyDown={event => event.stopPropagation()}
          >
            <JobActionBar job={job} onRefresh={onRefresh} autoStartQueue={false} />
          </div>
        </div>
      </div>
    </div>
  );
}

function JobListLane({
  title,
  description,
  jobs,
  selectedJobID,
  sort,
  reorderBusy,
  onSelect,
  onRefresh,
  onReorder,
}: {
  title: string;
  description?: string;
  jobs: Job[];
  selectedJobID: string | null;
  sort: SortKey;
  reorderBusy: string | null;
  onSelect: (jobID: string) => void;
  onRefresh: () => void;
  onReorder?: (targetJobID: string) => void;
}) {
  const sortedJobs = sortHistoryJobs(jobs, sort);

  if (sortedJobs.length === 0) {
    return (
      <section className="border border-gray-800 bg-[#080d12] p-4 text-sm text-gray-400">
        <div className="font-medium text-gray-200">{title}</div>
        {description && <div className="mt-1">{description}</div>}
      </section>
    );
  }

  return (
    <section className="border border-gray-800 bg-[#080d12]">
      <div className="flex min-w-0 items-center justify-between gap-3 border-b border-gray-800 px-4 py-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold text-gray-100">{title}</h2>
          {description && <div className="mt-0.5 truncate text-xs text-gray-500">{description}</div>}
        </div>
        <span className="rounded-md border border-gray-800 bg-gray-950 px-2 py-0.5 text-xs text-gray-400">
          {sortedJobs.length} jobs
        </span>
      </div>
      <div>
        {sortedJobs.map((job, index) => (
          <QueueJobCard
            key={job.id}
            job={job}
            index={index}
            showTimeline={false}
            selected={selectedJobID === job.id}
            dragTarget={false}
            reorderBusy={Boolean(reorderBusy)}
            onSelect={onSelect}
            onDragStart={() => undefined}
            onDragOver={(_jobID, event) => event.preventDefault()}
            onDrop={() => onReorder?.(job.id)}
            onDragEnd={() => undefined}
            onRefresh={onRefresh}
          />
        ))}
      </div>
    </section>
  );
}

function ActiveQueueLane({
  group,
  queue,
  selectedJobID,
  dragJobID,
  dragOverJobID,
  reorderBusy,
  pendingReorder,
  onSelect,
  onRefresh,
  onDragStart,
  onDragOver,
  onDrop,
  onDragEnd,
  onApplyReorder,
  onCancelReorder,
}: {
  group: JobGroup;
  queue: Queue | undefined;
  selectedJobID: string | null;
  dragJobID: string | null;
  dragOverJobID: string | null;
  reorderBusy: string | null;
  pendingReorder: PendingReorder | null;
  onSelect: (jobID: string) => void;
  onRefresh: () => void;
  onDragStart: (jobID: string) => void;
  onDragOver: (jobID: string, event: DragEvent<HTMLDivElement>) => void;
  onDrop: (jobID: string, group: JobGroup) => void;
  onDragEnd: () => void;
  onApplyReorder: (group: JobGroup) => void;
  onCancelReorder: (laneKey: string) => void;
}) {
  const queueRunning = queue?.is_running === true;
  const laneKey = getLaneKey(group.workerID, group.gpuIDs);
  const isSavingOrder = reorderBusy === laneKey;
  const hasPendingOrder = pendingReorder?.laneKey === laneKey;

  return (
    <div className="space-y-3">
      <WorkerHero group={group} queue={queue} isBusy={isSavingOrder} onRefresh={onRefresh} />
      <section className="border border-gray-800 bg-[#080d12]">
        <div className="flex min-w-0 flex-col gap-3 border-b border-gray-800 px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex min-w-0 items-center gap-2">
            <ListFilter className="h-4 w-4 text-blue-300" />
            <h2 className="truncate text-sm font-semibold text-gray-100">Queue order</h2>
            <span className="rounded-full border border-gray-800 bg-gray-950 px-2 py-0.5 text-xs text-gray-400">
              {group.jobs.length}
            </span>
          </div>
          <div className="flex flex-none items-center gap-2">
            {isSavingOrder && (
              <span className="inline-flex items-center gap-1 text-xs text-blue-300">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Saving order
              </span>
            )}
            {hasPendingOrder && !isSavingOrder && (
              <>
                <span className="text-xs font-medium text-amber-300">Unsaved order</span>
                <button
                  type="button"
                  onClick={() => onCancelReorder(laneKey)}
                  className="inline-flex h-8 items-center border border-gray-700 bg-gray-950 px-3 text-xs font-medium text-gray-300 transition-colors hover:border-gray-500 hover:text-gray-100"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={() => onApplyReorder(group)}
                  className="inline-flex h-8 items-center border border-blue-500 bg-blue-600 px-3 text-xs font-semibold text-white transition-colors hover:bg-blue-500"
                >
                  Apply order
                </button>
              </>
            )}
          </div>
        </div>
        <div>
          {group.jobs.length === 0 ? (
            <QueueEmptyState queueRunning={queueRunning} />
          ) : (
            group.jobs.map((job, index) => (
              <QueueJobCard
                key={job.id}
                job={job}
                index={index}
                selected={selectedJobID === job.id}
                dragTarget={dragOverJobID === job.id && dragJobID !== job.id}
                reorderBusy={Boolean(reorderBusy)}
                onSelect={onSelect}
                onDragStart={onDragStart}
                onDragOver={onDragOver}
                onDrop={jobID => onDrop(jobID, group)}
                onDragEnd={onDragEnd}
                onRefresh={onRefresh}
              />
            ))
          )}
        </div>
      </section>
    </div>
  );
}

function EventIcon({ line }: { line: string }) {
  if (/error|failed|exit code/i.test(line)) return <XCircle className="h-4 w-4 text-rose-400" />;
  if (/start|queued|running/i.test(line)) return <Info className="h-4 w-4 text-blue-400" />;
  if (/complete|saved|done/i.test(line)) return <CheckCircle2 className="h-4 w-4 text-emerald-400" />;
  return <Clock3 className="h-4 w-4 text-gray-500" />;
}

function SelectedJobInspector({ job, onRefresh }: { job: Job; onRefresh: () => void }) {
  const { log, status } = useJobLog(job.id, ['running', 'stopping'].includes(job.status) ? 5000 : null);
  const { prefix, title } = jobDisplayTitle(job);
  const totalSteps = safeTotalSteps(job);
  const logLines = log
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)
    .slice(-4)
    .reverse();
  const events = logLines.length > 0 ? logLines : [job.info || 'No recent events available.'];
  const actions = getAvaliableJobActions(job);

  return (
    <aside className="sticky top-3 flex max-h-[calc(100dvh-4.5rem)] min-h-[520px] flex-col overflow-hidden rounded-md border border-gray-800 bg-[#0a1017]">
      <div className="flex min-w-0 items-start justify-between gap-3 border-b border-gray-800 px-4 py-4">
        <div className="min-w-0">
          <div className="flex min-w-0 items-center gap-2">
            <h2 className="truncate text-base font-semibold text-gray-100">{title}</h2>
            <StatusBadge status={job.status} />
          </div>
          <div className="mt-1 text-xs uppercase tracking-wide text-gray-500">{prefix}</div>
        </div>
        <Link href={`/jobs/${job.id}`} className="operator-icon-button h-8 w-8" title="Open job" aria-label="Open job">
          <ArrowRight className="h-4 w-4" />
        </Link>
      </div>

      <div className="operator-scrollbar-none min-h-0 flex-1 overflow-y-auto">
        <section className="space-y-3 border-b border-gray-800 px-4 py-4 text-sm">
          <div className="grid grid-cols-[6rem_1fr] gap-y-3">
            <div className="text-gray-500">Worker</div>
            <div className="truncate text-gray-100">{job.worker_id === 'local' ? 'Local' : job.worker_id}</div>
            <div className="text-gray-500">GPU</div>
            <div className="text-gray-100">{job.gpu_ids || '-'}</div>
            <div className="text-gray-500">Added</div>
            <div className="text-gray-100">{formatDate(job.created_at)}</div>
            <div className="text-gray-500">ID</div>
            <div className="truncate font-mono text-xs text-gray-300">{job.id}</div>
          </div>
          <div>
            <div className="mb-2 flex items-center justify-between text-xs text-gray-500">
              <span>Steps</span>
              <span>{Math.round(getProgress(job))}%</span>
            </div>
            <ProgressBar value={getProgress(job)} />
            <div className="mt-1 text-xs text-gray-400">{totalSteps ? `${job.step} / ${totalSteps}` : `Step ${job.step}`}</div>
          </div>
        </section>

        <section className="border-b border-gray-800 px-4 py-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h3 className="text-sm font-semibold text-gray-100">Recent Events</h3>
            <Link href={`/jobs/${job.id}`} className="text-xs text-blue-300 hover:text-blue-200">
              View full log
            </Link>
          </div>
          <div className="rounded-md border border-gray-800 bg-gray-950/45">
            {status === 'loading' ? (
              <div className="flex items-center gap-2 px-3 py-3 text-sm text-gray-400">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading log
              </div>
            ) : (
              events.map((line, index) => (
                <div
                  key={`${line}-${index}`}
                  className="flex min-w-0 items-start gap-2 border-b border-gray-800 px-3 py-2 text-sm last:border-b-0"
                >
                  <EventIcon line={line} />
                  <span className="min-w-0 flex-1 truncate text-gray-300">{line}</span>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="px-4 py-4">
          <h3 className="mb-3 text-sm font-semibold text-gray-100">Quick Actions</h3>
          <div className="border-y border-gray-800 py-2">
            <JobActionBar job={job} onRefresh={onRefresh} className="flex-wrap justify-start" autoStartQueue={false} />
          </div>
          <div className="mt-3 flex flex-wrap gap-2 text-sm">
            <Link
              href={`/jobs/${job.id}`}
              className="inline-flex h-9 items-center gap-2 border border-blue-500/40 bg-blue-600/10 px-3 text-blue-100 hover:bg-blue-600/20"
            >
              <PlayCircle className="h-4 w-4" />
              View Logs
            </Link>
            {job.job_type === 'train' && actions.canEdit ? (
              <Link
                href={`/jobs/new?id=${job.id}`}
                className="inline-flex h-9 items-center gap-2 border border-gray-800 bg-gray-950 px-3 text-gray-200 hover:bg-gray-900"
              >
                <Settings className="h-4 w-4" />
                Edit Job
              </Link>
            ) : (
              <div className="inline-flex h-9 items-center gap-2 border border-gray-800 bg-gray-950 px-3 text-gray-500">
                <Settings className="h-4 w-4" />
                Edit Job
              </div>
            )}
          </div>
          <div className="mt-5 border-t border-gray-800 pt-3 text-sm text-gray-300">
            <div className="flex items-center gap-2 font-medium text-gray-100">
              <Info className="h-4 w-4 text-blue-300" />
              Having issues?
            </div>
            <Link href={`/jobs/${job.id}`} className="mt-2 inline-flex items-center gap-2 text-xs text-blue-300 hover:text-blue-200">
              Check job details
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </section>
      </div>
    </aside>
  );
}

function EmptyInspector() {
  return (
    <aside className="sticky top-3 flex min-h-[360px] items-center justify-center rounded-md border border-gray-800 bg-[#0a1017] p-6 text-center text-sm text-gray-400">
      <div>
        <CircleDashed className="mx-auto mb-3 h-8 w-8 text-gray-600" />
        <div className="font-medium text-gray-200">Select a job</div>
        <div className="mt-1">Job details, logs, and actions will appear here.</div>
      </div>
    </aside>
  );
}

export default function QueueWorkbench({ filterText, focusGpuIDs }: QueueWorkbenchProps) {
  const router = useRouter();
  const { jobs, status, refreshJobs } = useJobsList({ reloadInterval: 5000 });
  const { queues, status: queueStatus, refreshQueues } = useQueueList();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const { workers, status: workerStatus } = useWorkers();
  const [activeTab, setActiveTab] = useState<TabKey>('active');
  const [sort, setSort] = useState<SortKey>('newest');
  const [selectedJobID, setSelectedJobID] = useState<string | null>(null);
  const [dragJobID, setDragJobID] = useState<string | null>(null);
  const [dragOverJobID, setDragOverJobID] = useState<string | null>(null);
  const [reorderBusy, setReorderBusy] = useState<string | null>(null);
  const [reorderError, setReorderError] = useState<string | null>(null);
  const [pendingReorder, setPendingReorder] = useState<PendingReorder | null>(null);
  const [hasInlineInspector, setHasInlineInspector] = useState(false);

  const refresh = () => {
    refreshJobs();
    refreshQueues();
  };

  useEffect(() => {
    const query = window.matchMedia('(min-width: 1280px)');
    const update = () => setHasInlineInspector(query.matches);
    update();
    query.addEventListener('change', update);
    return () => query.removeEventListener('change', update);
  }, []);

  const activateJob = (jobID: string) => {
    if (hasInlineInspector) {
      setSelectedJobID(jobID);
      return;
    }
    router.push(`/jobs/${jobID}`);
  };

  const filteredJobs = useMemo(() => {
    const query = filterText.trim().toLowerCase();
    if (!query) return jobs;
    return jobs.filter(job =>
      [job.name, job.status, job.info, job.job_type, job.job_ref, job.gpu_ids, job.worker_id]
        .filter(Boolean)
        .some(value => `${value}`.toLowerCase().includes(query)),
    );
  }, [filterText, jobs]);

  const jobsDict = useMemo<Record<string, JobGroup>>(() => {
    if (!isGPUInfoLoaded) return {};
    if (filteredJobs.length === 0 && gpuList.length === 0 && queues.length === 0) return {};

    const groups: Record<string, JobGroup> = {};
    const workerName = (workerID: string) => {
      if (workerID === 'local') return 'Local';
      return workers.find(worker => worker.id === workerID)?.name || 'Remote';
    };
    const gpuName = (workerID: string, gpuID: string) => {
      if (workerID === 'local') {
        return gpuList.find(gpu => `${gpu.index}` === gpuID)?.name || `GPU #${gpuID}`;
      }
      const worker = workers.find(candidate => candidate.id === workerID);
      try {
        const gpus = JSON.parse(worker?.gpus || '[]') as Array<{ index: number; name: string }>;
        return gpus.find(gpu => `${gpu.index}` === gpuID)?.name || `GPU #${gpuID}`;
      } catch {
        return `GPU #${gpuID}`;
      }
    };
    const ensureGroup = (workerID: string, gpuIDs: string) => {
      const key = `${workerID}:${gpuIDs}`;
      if (!groups[key]) {
        groups[key] = {
          name: `${workerName(workerID)} / ${gpuName(workerID, gpuIDs)}`,
          jobs: [],
          workerID,
          gpuIDs,
        };
      }
      return groups[key];
    };

    gpuList.forEach(gpu => {
      groups[`local:${gpu.index}`] = {
        name: `Local / ${gpu.name}`,
        jobs: [],
        workerID: 'local',
        gpuIDs: `${gpu.index}`,
      };
    });
    queues.forEach(queue => ensureGroup(queue.worker_id, queue.gpu_ids));
    groups.idle = { name: 'Idle / history', jobs: [], workerID: 'local', gpuIDs: null };

    filteredJobs.forEach(job => {
      const workerID = job.worker_id || 'local';
      const gpuIDs = job.gpu_ids || '0';
      const key = `${workerID}:${gpuIDs}`;
      if (isRemoteWorker(workerID)) {
        const group = ensureGroup(workerID, gpuIDs);
        if (activeJobStatuses.has(job.status)) group.jobs.push(job);
        else groups.idle.jobs.push(job);
      } else if (activeJobStatuses.has(job.status) && key in groups) {
        groups[key].jobs.push(job);
      } else {
        groups.idle.jobs.push(job);
      }
    });

    Object.keys(groups).forEach(key => {
      if (key === 'idle') {
        groups[key].jobs = sortHistoryJobs(groups[key].jobs, sort);
      } else {
        groups[key].jobs.sort((a, b) => {
          if (a.queue_position === null) return 1;
          if (b.queue_position === null) return -1;
          return a.queue_position - b.queue_position;
        });
      }
    });

    return groups;
  }, [filteredJobs, gpuList, isGPUInfoLoaded, queues, sort, workers]);

  const activeGroups = useMemo(() => {
    const keys = Object.keys(jobsDict).filter(key => key !== 'idle');
    return keys
      .sort((a, b) => {
        const aFocused = a === `local:${focusGpuIDs}`;
        const bFocused = b === `local:${focusGpuIDs}`;
        if (aFocused !== bFocused) return aFocused ? -1 : 1;
        return a.localeCompare(b);
      })
      .map(key => jobsDict[key]);
  }, [focusGpuIDs, jobsDict]);

  useEffect(() => {
    if (!pendingReorder) return;
    const group = jobsDict[pendingReorder.laneKey];
    if (!group) {
      setPendingReorder(null);
      return;
    }
    const queuedIDs = group.jobs.filter(isQueuedJob).map(job => job.id);
    if (!hasSameIDs(queuedIDs, pendingReorder.jobIDs)) setPendingReorder(null);
  }, [jobsDict, pendingReorder]);

  const idleJobs = jobsDict.idle?.jobs || [];
  const failedJobs = filteredJobs.filter(job => job.status === 'error' || job.status === 'failed');
  const activeJobs = filteredJobs.filter(job => activeJobStatuses.has(job.status));
  const visibleJobs = useMemo(() => {
    if (activeTab === 'history') return idleJobs;
    if (activeTab === 'failed') return sortHistoryJobs(failedJobs, sort);
    if (activeTab === 'all') return [...activeGroups.flatMap(group => group.jobs), ...idleJobs];
    return [...activeGroups.flatMap(group => group.jobs), ...idleJobs.slice(0, 4)];
  }, [activeGroups, activeTab, failedJobs, idleJobs, sort]);
  const selectedJob =
    filteredJobs.find(job => job.id === selectedJobID) ||
    (hasInlineInspector ? visibleJobs[0] || filteredJobs[0] || null : null);

  useEffect(() => {
    if (!selectedJob) {
      setSelectedJobID(null);
      return;
    }
    if (selectedJobID !== selectedJob.id) setSelectedJobID(selectedJob.id);
  }, [selectedJob, selectedJobID]);

  const tableError =
    status === 'error'
      ? 'Jobs could not be loaded.'
      : queueStatus === 'error'
        ? 'Queues could not be loaded.'
        : workerStatus === 'error'
          ? 'Workers could not be loaded.'
          : null;

  let isLoading = status === 'loading' || queueStatus === 'loading' || workerStatus === 'loading' || !isGPUInfoLoaded;
  if (Object.keys(jobsDict).length > 0) isLoading = false;

  const tabs: Array<{ key: TabKey; label: string; count: number; icon: typeof PlayCircle }> = [
    { key: 'active', label: 'Active', count: activeJobs.length, icon: PlayCircle },
    { key: 'history', label: 'History', count: idleJobs.length, icon: History },
    { key: 'failed', label: 'Failed', count: failedJobs.length, icon: AlertTriangle },
    { key: 'all', label: 'All', count: filteredJobs.length, icon: CircleDashed },
  ];

  const handleDragStart = (jobID: string) => {
    setReorderError(null);
    setDragJobID(jobID);
  };

  const handleDragOver = (jobID: string, event: DragEvent<HTMLDivElement>) => {
    if (!dragJobID || dragJobID === jobID) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
    setDragOverJobID(jobID);
  };

  const handleDrop = (targetJobID: string, group: JobGroup) => {
    if (!dragJobID || dragJobID === targetJobID || !group.gpuIDs) {
      setDragJobID(null);
      setDragOverJobID(null);
      return;
    }

    const queuedJobs = group.jobs.filter(job => job.status === 'queued');
    const fromIndex = queuedJobs.findIndex(job => job.id === dragJobID);
    const toIndex = queuedJobs.findIndex(job => job.id === targetJobID);
    if (fromIndex < 0 || toIndex < 0) {
      setDragJobID(null);
      setDragOverJobID(null);
      return;
    }

    const nextJobs = [...queuedJobs];
    const [moved] = nextJobs.splice(fromIndex, 1);
    nextJobs.splice(toIndex, 0, moved);
    setPendingReorder({
      laneKey: getLaneKey(group.workerID, group.gpuIDs),
      workerID: group.workerID,
      gpuIDs: group.gpuIDs,
      jobIDs: nextJobs.map(job => job.id),
    });
    setReorderError(null);
    setDragJobID(null);
    setDragOverJobID(null);
  };

  const handleApplyReorder = async (group: JobGroup) => {
    if (!group.gpuIDs) return;
    const laneKey = getLaneKey(group.workerID, group.gpuIDs);
    if (!pendingReorder || pendingReorder.laneKey !== laneKey) return;

    setReorderBusy(laneKey);
    setReorderError(null);
    try {
      await reorderQueue(group.gpuIDs, pendingReorder.jobIDs, group.workerID);
      setPendingReorder(null);
      refresh();
    } catch (error: any) {
      setReorderError(error?.response?.data?.error || error?.message || 'Failed to reorder queue.');
    } finally {
      setReorderBusy(null);
      setDragJobID(null);
      setDragOverJobID(null);
    }
  };

  const handleCancelReorder = (laneKey: string) => {
    setPendingReorder(current => (current?.laneKey === laneKey ? null : current));
    setDragJobID(null);
    setDragOverJobID(null);
  };

  const handleDragEnd = () => {
    setDragJobID(null);
    setDragOverJobID(null);
  };

  if (isLoading && Object.keys(jobsDict).length === 0) {
    return (
      <div className="p-3">
        <PageNotice tone="neutral" title="Loading queues and jobs">
          Fetching queue state, workers, and GPU telemetry.
        </PageNotice>
      </div>
    );
  }

  return (
    <div className="grid min-h-full gap-3 bg-[#02060a] p-3 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="min-w-0 space-y-3">
        <section className="overflow-hidden rounded-md border border-gray-800 bg-[#080d12]">
          <div className="flex min-w-0 flex-col gap-3 border-b border-gray-800 px-3 py-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="operator-scrollbar-none flex min-w-0 gap-1 overflow-x-auto">
              {tabs.map(tab => {
                const Icon = tab.icon;
                const active = activeTab === tab.key;
                return (
                  <button
                    key={tab.key}
                    type="button"
                    onClick={() => setActiveTab(tab.key)}
                    className={classNames(
                      'inline-flex h-10 flex-none items-center gap-2 border-b-2 px-3 text-sm transition-colors',
                      active
                        ? 'border-blue-500 text-gray-100'
                        : 'border-transparent text-gray-400 hover:text-gray-200',
                    )}
                  >
                    <Icon className="h-4 w-4" />
                    {tab.label}
                    <span className="rounded-full border border-gray-800 bg-gray-950 px-2 py-0.5 text-xs text-gray-400">
                      {tab.count}
                    </span>
                  </button>
                );
              })}
            </div>
            <div className="flex items-center justify-between gap-3 text-sm text-gray-400">
              <span>{filteredJobs.length} jobs</span>
              <label className="inline-flex h-9 items-center gap-2 rounded-md border border-gray-800 bg-gray-950 px-3">
                <SlidersHorizontal className="h-4 w-4 text-gray-500" />
                <span className="text-xs text-gray-500">Sort</span>
                <select
                  value={sort}
                  onChange={event => setSort(event.target.value as SortKey)}
                  className="bg-gray-950 text-sm text-gray-100 outline-none [color-scheme:dark]"
                >
                  <option value="newest" className="bg-gray-950 text-gray-100">
                    Newest
                  </option>
                  <option value="oldest" className="bg-gray-950 text-gray-100">
                    Oldest
                  </option>
                  <option value="name" className="bg-gray-950 text-gray-100">
                    Name
                  </option>
                </select>
              </label>
            </div>
          </div>
        </section>

        {tableError && (
          <PageNotice tone="danger" title="Queue data is incomplete">
            {tableError}
          </PageNotice>
        )}

        {reorderError && (
          <PageNotice tone="danger" title="Queue order was not saved" action={<button className="operator-icon-button h-7 w-7" onClick={() => setReorderError(null)}><X className="h-4 w-4" /></button>}>
            {reorderError}
          </PageNotice>
        )}

        {(activeTab === 'active' || activeTab === 'all') &&
          activeGroups.map(group => {
            const queue = queues.find(q => q.worker_id === group.workerID && q.gpu_ids === group.gpuIDs);
            const displayGroup = applyPendingOrder(group, pendingReorder);
            return (
              <ActiveQueueLane
                key={`${group.workerID}:${group.gpuIDs}`}
                group={displayGroup}
                queue={queue}
                selectedJobID={selectedJobID}
                dragJobID={dragJobID}
                dragOverJobID={dragOverJobID}
                reorderBusy={reorderBusy}
                pendingReorder={pendingReorder}
                onSelect={activateJob}
                onRefresh={refresh}
                onDragStart={handleDragStart}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onDragEnd={handleDragEnd}
                onApplyReorder={handleApplyReorder}
                onCancelReorder={handleCancelReorder}
              />
            );
          })}

        {activeTab === 'active' && idleJobs.length > 0 && (
          <JobListLane
            title="Recent history"
            description="Stopped, completed, and failed jobs remain available here."
            jobs={idleJobs.slice(0, 6)}
            selectedJobID={selectedJobID}
            sort={sort}
            reorderBusy={reorderBusy}
            onSelect={activateJob}
            onRefresh={refresh}
          />
        )}

        {activeTab === 'history' && (
          <JobListLane
            title="History"
            description="Completed, stopped, and failed jobs."
            jobs={idleJobs}
            selectedJobID={selectedJobID}
            sort={sort}
            reorderBusy={reorderBusy}
            onSelect={activateJob}
            onRefresh={refresh}
          />
        )}

        {activeTab === 'failed' && (
          <JobListLane
            title="Failed jobs"
            description="Jobs that need attention."
            jobs={failedJobs}
            selectedJobID={selectedJobID}
            sort={sort}
            reorderBusy={reorderBusy}
            onSelect={activateJob}
            onRefresh={refresh}
          />
        )}

        {activeTab === 'all' && idleJobs.length > 0 && (
          <JobListLane
            title="Idle / history"
            description="All non-active jobs."
            jobs={idleJobs}
            selectedJobID={selectedJobID}
            sort={sort}
            reorderBusy={reorderBusy}
            onSelect={activateJob}
            onRefresh={refresh}
          />
        )}

      </div>

      {hasInlineInspector && (
        <div className="min-w-0">
          {selectedJob ? <SelectedJobInspector job={selectedJob} onRefresh={refresh} /> : <EmptyInspector />}
        </div>
      )}
    </div>
  );
}
