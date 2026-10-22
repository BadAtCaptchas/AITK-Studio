'use client';

import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import type { DragEvent, KeyboardEvent, ReactNode } from 'react';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import Link from 'next/link';
import classNames from 'classnames';
import { ArrowDown, ArrowUp, ArrowUpRight, GripVertical, Loader2, Play, RefreshCcw, Square, X } from 'lucide-react';
import JobActionBar from '@/components/JobActionBar';
import JobLossGraph from '@/components/JobLossGraph';
import QueueSyncStatus from '@/components/QueueSyncStatus';
import { PageNotice, ProgressBar, StatusBadge } from '@/components/OperatorPrimitives';
import { HFDownloadProgressInline } from '@/components/HFDownloadProgress';
import { ComfyInstallProgressInline } from '@/components/ComfyInstallProgress';
import useGPUInfo from '@/hooks/useGPUInfo';
import useJobLog from '@/hooks/useJobLog';
import useJob from '@/hooks/useJob';
import useJobsList from '@/hooks/useJobsList';
import useQueueList from '@/hooks/useQueueList';
import useWorkers from '@/hooks/useWorkers';
import useHydrated from '@/hooks/useHydrated';
import type { Job, Queue } from '@/types';
import { getTotalSteps } from '@/utils/jobs';
import { ACTIVE_JOB_STATUSES } from '@/utils/jobStatus';
import { reorderQueue, startQueue, stopQueue } from '@/utils/queue';
import {
  clampInspectorWidth,
  hasCompleteQueuedJobs,
  matchesQueuedLaneOrder,
  moveQueuedJob,
  readQueueWorkspace,
  rememberQueueWorkspace,
  QUEUE_SORTS,
  type QueueSort,
  type QueueTab,
} from '@/utils/queueWorkspace';

type QueueWorkbenchProps = {
  filterText: string;
  emptyAction?: ReactNode;
  focusGpuIDs?: string | null;
  onRestoreView?: () => void;
};
type JobGroup = { name: string; jobs: Job[]; workerID: string; gpuIDs: string };
type PendingReorder = { laneKey: string; workerID: string; gpuIDs: string; baseJobIDs: string[]; jobIDs: string[] };
const activeStatuses = new Set(ACTIVE_JOB_STATUSES);
const laneKey = (workerID: string, gpuIDs: string) => `${workerID}:${gpuIDs}`;
const jobHref = (job: Job) => `/jobs/${encodeURIComponent(job.id)}`;
const isQueued = (job: Job) => job.status === 'queued';
const inspectorQuery = '(min-width: 1280px)';
const subscribeInspectorViewport = (listener: () => void) => {
  const query = window.matchMedia(inspectorQuery);
  query.addEventListener('change', listener);
  return () => query.removeEventListener('change', listener);
};
const inlineInspectorSnapshot = () => window.matchMedia(inspectorQuery).matches;
const serverInspectorSnapshot = () => false;

function savedInspectorWidth(): number {
  try {
    const value = typeof window === 'undefined' ? null : window.localStorage.getItem('aitk.queue.inspectorWidth');
    return value === null ? 360 : clampInspectorWidth(Number(value));
  } catch {
    return 360;
  }
}

function totalSteps(job: Job): number | null {
  if (job.job_type !== 'train') return null;
  try {
    return getTotalSteps(job);
  } catch {
    return null;
  }
}
function progress(job: Job): number {
  const total = totalSteps(job);
  return total ? Math.max(0, Math.min(100, (job.step / total) * 100)) : 0;
}
function displayName(job: Job): string {
  return job.job_type === 'caption' ? job.job_ref?.split(/[/\\]/).at(-1) || job.name : job.name;
}
function formatDate(value: Job['updated_at']): string {
  const date = new Date(value);
  return Number.isFinite(date.getTime())
    ? date.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
    : 'Unavailable';
}
function sortJobs(jobs: Job[], sort: QueueSort): Job[] {
  return [...jobs].sort((a, b) =>
    sort === 'name'
      ? a.name.localeCompare(b.name)
      : (new Date(a.updated_at).getTime() - new Date(b.updated_at).getTime()) * (sort === 'oldest' ? 1 : -1),
  );
}
function orderedGroup(group: JobGroup, pending: PendingReorder | null): JobGroup {
  if (!pending || pending.laneKey !== laneKey(group.workerID, group.gpuIDs)) return group;
  const queued = new Map(group.jobs.filter(isQueued).map(job => [job.id, job]));
  if (queued.size !== pending.jobIDs.length || pending.jobIDs.some(id => !queued.has(id))) return group;
  const ordered = pending.jobIDs.map(id => queued.get(id)!);
  let index = 0;
  return { ...group, jobs: group.jobs.map(job => (isQueued(job) ? ordered[index++] : job)) };
}
function errorMessage(error: unknown): string {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = error.response;
    if (response && typeof response === 'object' && 'data' in response) {
      const data = response.data;
      if (data && typeof data === 'object' && 'error' in data && typeof data.error === 'string') return data.error;
    }
  }
  return error instanceof Error ? error.message : 'Could not save queue order. Try again.';
}

function WorkerToolbar({
  group,
  queue,
  unavailable,
  onRefresh,
}: {
  group: JobGroup;
  queue?: Queue;
  unavailable: boolean;
  onRefresh: () => void;
}) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const running = queue?.is_running === true;
  const toggle = async () => {
    if (pending || unavailable) return;
    setPending(true);
    setError(null);
    try {
      if (running) await stopQueue(group.gpuIDs, group.workerID);
      else await startQueue(group.gpuIDs, group.workerID);
      onRefresh();
    } catch (error: unknown) {
      setError(errorMessage(error));
    } finally {
      setPending(false);
    }
  };
  return (
    <header className="border-y border-gray-800 bg-gray-900/60 px-3 py-2">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <h2 className="min-w-0 truncate text-sm font-semibold text-gray-100">{group.name}</h2>
        <span className="text-xs text-gray-400">
          {unavailable ? 'Queue unavailable' : running ? 'Queue running' : 'Queue stopped'}
        </span>
        <span className="text-xs tabular-nums text-gray-500">{group.jobs.filter(isQueued).length} queued shown</span>
        <button
          type="button"
          disabled={pending || unavailable}
          onClick={() => void toggle()}
          className="operator-button ml-auto h-8 text-xs"
        >
          {pending ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : running ? (
            <Square className="h-3 w-3" />
          ) : (
            <Play className="h-3 w-3" />
          )}
          {running ? 'Stop queue' : 'Start queue'}
        </button>
      </div>
      {error && (
        <p role="alert" className="mt-2 text-xs text-rose-700 dark:text-rose-300">
          {error}
        </p>
      )}
    </header>
  );
}

function ColumnHeadings() {
  return (
    <div
      aria-hidden="true"
      className="hidden grid-cols-[minmax(12rem,1.4fr)_minmax(7rem,.7fr)_minmax(9rem,1fr)_7rem_11rem] gap-3 border-b border-gray-800 px-3 py-2 text-xs text-gray-500 lg:grid"
    >
      <span>Name / position</span>
      <span>State</span>
      <span data-queue-steps-heading>Steps / progress</span>
      <span>Updated</span>
      <span>Actions</span>
    </div>
  );
}

function JobRow({
  job,
  selected,
  queuePosition,
  canDrag,
  dragTarget,
  tabIndex,
  onSelect,
  onRefresh,
  onDragStart,
  onDragOver,
  onDrop,
  onDragEnd,
}: {
  job: Job;
  selected: boolean;
  queuePosition?: number;
  canDrag?: boolean;
  dragTarget?: boolean;
  tabIndex: number;
  onSelect: (id: string) => void;
  onRefresh: () => void;
  onDragStart?: (id: string) => void;
  onDragOver?: (id: string, event: DragEvent<HTMLDivElement>) => void;
  onDrop?: (id: string) => void;
  onDragEnd?: () => void;
}) {
  const total = totalSteps(job);
  return (
    <div
      data-queue-row={job.id}
      role="group"
      aria-label={`Job ${displayName(job)}`}
      draggable={canDrag}
      onClick={() => onSelect(job.id)}
      onDragStart={event => {
        if (!canDrag) {
          event.preventDefault();
          return;
        }
        event.dataTransfer.effectAllowed = 'move';
        event.dataTransfer.setData('text/plain', job.id);
        onDragStart?.(job.id);
      }}
      onDragOver={event => onDragOver?.(job.id, event)}
      onDrop={event => {
        event.preventDefault();
        onDrop?.(job.id);
      }}
      onDragEnd={onDragEnd}
      className={classNames(
        'grid cursor-pointer grid-cols-2 items-center gap-x-3 gap-y-2 border-b border-l-2 border-gray-800 px-3 py-2.5 text-sm outline-none hover:bg-gray-900/60 focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-brand-400 lg:grid-cols-[minmax(12rem,1.4fr)_minmax(7rem,.7fr)_minmax(9rem,1fr)_7rem_11rem]',
        selected ? 'border-l-brand-400 bg-brand-500/10' : 'border-l-transparent',
        dragTarget && 'bg-brand-500/20',
      )}
    >
      <div className="min-w-0">
        <button
          type="button"
          data-queue-select={job.id}
          aria-label={`Inspect ${displayName(job)}`}
          aria-pressed={selected}
          tabIndex={tabIndex}
          onClick={event => {
            event.stopPropagation();
            onSelect(job.id);
          }}
          className="flex max-w-full items-center gap-2 rounded-sm text-left outline-none focus-visible:ring-2 focus-visible:ring-brand-400"
        >
          {canDrag && <GripVertical aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-gray-500" />}
          <span className="truncate font-medium text-gray-100">{displayName(job)}</span>
        </button>
        <div className="mt-1 truncate text-xs text-gray-500">
          {job.job_type === 'train' ? 'Training' : job.job_type === 'caption' ? 'Caption' : 'Generation'}
          {queuePosition !== undefined && ` · Queue ${queuePosition}`}
          {job.speed_string && ` · ${job.speed_string}`}
        </div>
      </div>
      <div className="min-w-0">
        <StatusBadge status={job.status} />
        <div className="mt-1 truncate text-xs text-gray-500" title={job.info || undefined}>
          {job.comfy_install_progress ? (
            <ComfyInstallProgressInline progress={job.comfy_install_progress} fallback={job.info} />
          ) : job.hf_download_progress ? (
            <HFDownloadProgressInline progress={job.hf_download_progress} fallback={job.info || ''} />
          ) : (
            job.info || `${job.worker_id === 'local' ? 'Local' : 'Remote'} / GPU ${job.gpu_ids}`
          )}
        </div>
      </div>
      <div data-queue-steps className="min-w-0 text-xs tabular-nums text-gray-300">
        {total ? `${job.step.toLocaleString()} / ${total.toLocaleString()}` : `Step ${job.step.toLocaleString()}`}
        {total ? <ProgressBar value={progress(job)} className="mt-1.5" /> : null}
      </div>
      <span className="text-xs tabular-nums text-gray-500">{formatDate(job.updated_at)}</span>
      <div
        className="col-span-2 flex items-center gap-1 lg:col-span-1"
        onClick={event => event.stopPropagation()}
        onKeyDown={event => event.stopPropagation()}
      >
        <JobActionBar job={job} onRefresh={onRefresh} hideView autoStartQueue={false} />
        <Link
          className="operator-icon-button h-8 w-8"
          href={jobHref(job)}
          title="Open job"
          aria-label={`Open ${displayName(job)}`}
        >
          <ArrowUpRight className="h-4 w-4" />
        </Link>
      </div>
    </div>
  );
}

function SelectedJobInspector({ job, onClose, onRefresh }: { job: Job; onClose: () => void; onRefresh: () => void }) {
  const { log, status } = useJobLog(job.id, activeStatuses.has(job.status) ? 5000 : null);
  const { job: fullJob, status: detailStatus, refreshJob } = useJob(job.id);
  const actionJob = fullJob ? { ...job, job_config: fullJob.job_config } : null;
  const total = totalSteps(job);
  return (
    <aside id="queue-job-inspector" className="flex h-full min-h-0 flex-col bg-gray-950">
      <header className="flex items-start justify-between gap-3 border-b border-gray-800 px-4 py-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold text-gray-100">{displayName(job)}</h2>
          <div className="mt-2">
            <StatusBadge status={job.status} />
          </div>
        </div>
        <button type="button" onClick={onClose} className="operator-icon-button h-8 w-8" aria-label="Close inspector">
          <X className="h-4 w-4" />
        </button>
      </header>
      <div className="min-h-0 flex-1 overflow-auto">
        <dl className="grid grid-cols-[5rem_minmax(0,1fr)] gap-x-3 gap-y-3 border-b border-gray-800 px-4 py-4 text-xs">
          <dt className="text-gray-500">Worker</dt>
          <dd className="truncate">{job.worker_id === 'local' ? 'Local' : job.worker_id}</dd>
          <dt className="text-gray-500">GPU</dt>
          <dd>{job.gpu_ids || 'Unavailable'}</dd>
          <dt className="text-gray-500">Steps</dt>
          <dd className="tabular-nums">
            {total ? `${job.step.toLocaleString()} / ${total.toLocaleString()}` : job.step.toLocaleString()}
          </dd>
          <dt className="text-gray-500">Speed</dt>
          <dd>{job.speed_string || 'Unavailable'}</dd>
          <dt className="text-gray-500">Updated</dt>
          <dd>{formatDate(job.updated_at)}</dd>
          <dt className="text-gray-500">Created</dt>
          <dd>{formatDate(job.created_at)}</dd>
        </dl>
        {total ? (
          <div className="border-b border-gray-800 px-4 py-3">
            <div className="mb-2 flex justify-between text-xs text-gray-400">
              <span>Progress</span>
              <span>{Math.round(progress(job))}%</span>
            </div>
            <ProgressBar value={progress(job)} />
          </div>
        ) : null}
        {job.remote_error && (
          <div className="p-3">
            <PageNotice tone="warning" title="Remote worker error">
              {job.remote_error}
            </PageNotice>
          </div>
        )}
        {job.job_type === 'train' && (
          <div className="min-h-[380px] border-b border-gray-800">
            <JobLossGraph job={job} compact />
          </div>
        )}
        <section className="px-4 py-3">
          <div className="mb-3 flex items-center justify-between gap-2">
            <h3 className="text-xs font-semibold">Output</h3>
            <Link className="text-xs text-brand-300" href={`${jobHref(job)}?tab=overview`}>
              Full log
            </Link>
          </div>
          {status === 'loading' ? (
            <p className="text-xs text-gray-500">Loading log…</p>
          ) : status === 'error' ? (
            <p role="status" className="text-xs text-amber-800 dark:text-amber-300">
              Log unavailable. Open the job to retry.
            </p>
          ) : (
            <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-gray-400">
              {log.trim().split(/\r?\n/).slice(-15).join('\n') || job.info || 'No output yet.'}
            </pre>
          )}
          {job.job_type === 'train' && (
            <Link
              className="mt-4 inline-flex items-center gap-1 text-xs text-brand-300"
              href={`${jobHref(job)}?tab=loss_log`}
            >
              Training metrics
              <ArrowUpRight className="h-3 w-3" />
            </Link>
          )}
        </section>
      </div>
      <footer className="flex flex-wrap items-center gap-2 border-t border-gray-800 p-3">
        <Link className="operator-button-primary h-8 text-xs" href={jobHref(job)}>
          <ArrowUpRight className="h-3.5 w-3.5" />
          Open job
        </Link>
        {actionJob ? (
          <JobActionBar
            job={actionJob}
            onRefresh={() => {
              onRefresh();
              refreshJob();
            }}
            hideView
            autoStartQueue={false}
            className="flex-wrap"
          />
        ) : detailStatus === 'error' ? (
          <button type="button" className="operator-button h-8 text-xs" onClick={() => refreshJob()}>
            Retry actions
          </button>
        ) : (
          <span className="text-xs text-gray-500">Loading actions…</span>
        )}
      </footer>
    </aside>
  );
}

export default function QueueWorkbench({ filterText, emptyAction, focusGpuIDs, onRestoreView }: QueueWorkbenchProps) {
  const [saved] = useState(readQueueWorkspace);
  const [activeTab, setActiveTab] = useState<QueueTab>(saved.tab);
  const [sort, setSort] = useState<QueueSort>(saved.sort);
  const [selectedJobID, setSelectedJobID] = useState<string | null>(saved.selectedJobID);
  const [focusedJobID, setFocusedJobID] = useState<string | null>(saved.selectedJobID);
  const { jobs, status, refreshJobs, hasMore, loadMore, isRefreshing, freshness, statusCounts } = useJobsList({
    view: activeTab,
    reloadInterval: 5000,
  });
  const { queues, status: queueStatus, freshness: queueFreshness, refreshQueues } = useQueueList(5000);
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const { workers, status: workerStatus, refreshWorkers } = useWorkers();
  const hasInlineInspector = useSyncExternalStore(
    subscribeInspectorViewport,
    inlineInspectorSnapshot,
    serverInspectorSnapshot,
  );
  const [inspectorWidth, setInspectorWidth] = useState(savedInspectorWidth);
  const widthLoaded = useHydrated();
  const resizeStart = useRef<{ x: number; width: number } | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const [dragJobID, setDragJobID] = useState<string | null>(null);
  const [dragOverJobID, setDragOverJobID] = useState<string | null>(null);
  const [pendingReorder, setPendingReorder] = useState<PendingReorder | null>(null);
  const [reorderBusy, setReorderBusy] = useState(false);
  const [reorderError, setReorderError] = useState<string | null>(null);
  const restored = useRef(false);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!widthLoaded) return;
    try {
      window.localStorage.setItem('aitk.queue.inspectorWidth', String(inspectorWidth));
    } catch {
      /* Storage is optional. */
    }
  }, [inspectorWidth, widthLoaded]);
  useEffect(() => {
    rememberQueueWorkspace({ tab: activeTab, sort, selectedJobID });
  }, [activeTab, sort, selectedJobID]);
  useEffect(() => {
    if (restored.current) {
      if (status === 'success') rememberQueueWorkspace({ loadedCount: jobs.length });
      return;
    }
    // A failed page is recoverable. Keep the saved selection/scroll target until retry finishes.
    if (status !== 'success') return;
    if (jobs.length < saved.loadedCount && hasMore) {
      if (!isRefreshing) void loadMore();
      return;
    }
    restored.current = true;
    onRestoreView?.();
  }, [jobs.length, status, hasMore, isRefreshing, loadMore, onRestoreView, saved.loadedCount]);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 5000);
    return () => window.clearInterval(timer);
  }, []);

  const refresh = () => {
    refreshJobs();
    refreshQueues();
    refreshWorkers();
  };
  const filteredJobs = useMemo(() => {
    const query = filterText.trim().toLowerCase();
    return query
      ? jobs.filter(job =>
          [job.name, job.status, job.info, job.job_type, job.job_ref, job.gpu_ids, job.worker_id].some(value =>
            value?.toLowerCase().includes(query),
          ),
        )
      : jobs;
  }, [filterText, jobs]);
  const groups = useMemo(() => {
    const lanes = new Map<string, JobGroup>();
    const ensure = (workerID: string, gpuIDs: string) => {
      const key = laneKey(workerID, gpuIDs);
      let group = lanes.get(key);
      if (!group) {
        const name = workerID === 'local' ? 'Local' : workers.find(worker => worker.id === workerID)?.name || workerID;
        group = { name: `${name} / GPU ${gpuIDs}`, workerID, gpuIDs, jobs: [] };
        lanes.set(key, group);
      }
      return group;
    };
    gpuList.forEach(gpu => ensure('local', `${gpu.index}`));
    queues.forEach(queue => ensure(queue.worker_id, queue.gpu_ids));
    filteredJobs
      .filter(job => activeStatuses.has(job.status))
      .forEach(job => ensure(job.worker_id || 'local', job.gpu_ids || '0').jobs.push(job));
    return [...lanes.values()]
      .map(group => ({
        ...group,
        jobs: group.jobs.sort(
          (a, b) => (a.queue_position ?? Number.MAX_SAFE_INTEGER) - (b.queue_position ?? Number.MAX_SAFE_INTEGER),
        ),
      }))
      .sort(
        (a, b) =>
          Number(b.workerID === 'local' && b.gpuIDs === focusGpuIDs) -
            Number(a.workerID === 'local' && a.gpuIDs === focusGpuIDs) || a.name.localeCompare(b.name),
      );
  }, [filteredJobs, queues, gpuList, workers, focusGpuIDs]);
  if (
    pendingReorder &&
    !matchesQueuedLaneOrder(jobs, pendingReorder.baseJobIDs, pendingReorder.workerID, pendingReorder.gpuIDs)
  ) {
    setPendingReorder(null);
    setReorderError('Queue changed while you were arranging jobs. Review the current order and try again.');
  }
  const selectedJob = filteredJobs.find(job => job.id === selectedJobID) || null;
  const history = sortJobs(
    filteredJobs.filter(job => !activeStatuses.has(job.status)),
    sort,
  );
  const visibleJobs =
    activeTab === 'history' || activeTab === 'failed'
      ? history
      : [...groups.flatMap(group => group.jobs), ...(activeTab === 'all' ? history : [])];
  const tabStopID = visibleJobs.some(job => job.id === focusedJobID) ? focusedJobID : visibleJobs[0]?.id;
  const completeQueueSnapshot = hasCompleteQueuedJobs(jobs, statusCounts?.queued ?? (statusCounts ? 0 : null));
  const canReorder =
    !filterText.trim() && !hasMore && completeQueueSnapshot && status === 'success' && !isRefreshing && !reorderBusy;
  const totalFor = (values: string[]) =>
    statusCounts ? values.reduce((sum, status) => sum + (statusCounts[status] || 0), 0) : '—';
  const tabs: { key: QueueTab; label: string; count: number | string }[] = [
    { key: 'active', label: 'Queue', count: totalFor(ACTIVE_JOB_STATUSES) },
    { key: 'history', label: 'History', count: totalFor(['stopped', 'completed', 'error', 'failed']) },
    { key: 'failed', label: 'Failed', count: totalFor(['error', 'failed']) },
    {
      key: 'all',
      label: 'All',
      count: statusCounts ? Object.values(statusCounts).reduce((sum, count) => sum + count, 0) : '—',
    },
  ];
  const stageMove = (group: JobGroup, from: string, to: string) => {
    if (!canReorder) return;
    const ids = moveQueuedJob(group.jobs, from, to, group.workerID, group.gpuIDs);
    if (ids) {
      const key = laneKey(group.workerID, group.gpuIDs);
      setPendingReorder({
        laneKey: key,
        workerID: group.workerID,
        gpuIDs: group.gpuIDs,
        baseJobIDs:
          pendingReorder?.laneKey === key ? pendingReorder.baseJobIDs : group.jobs.filter(isQueued).map(job => job.id),
        jobIDs: ids,
      });
      setReorderError(null);
    }
  };
  const applyOrder = async (group: JobGroup) => {
    if (!pendingReorder || !canReorder || pendingReorder.laneKey !== laneKey(group.workerID, group.gpuIDs)) return;
    setReorderBusy(true);
    setReorderError(null);
    try {
      await reorderQueue(group.gpuIDs, pendingReorder.jobIDs, group.workerID);
      setPendingReorder(null);
      refresh();
    } catch (error: unknown) {
      setReorderError(errorMessage(error));
    } finally {
      setReorderBusy(false);
    }
  };
  const closeInspector = () => {
    const row = listRef.current?.querySelector<HTMLElement>(`[data-queue-select="${CSS.escape(selectedJobID || '')}"]`);
    setSelectedJobID(null);
    row?.focus({ preventScroll: true });
  };
  const handleRowKeys = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!(event.target instanceof HTMLElement) || !event.target.hasAttribute('data-queue-select')) return;
    const rows = [...event.currentTarget.querySelectorAll<HTMLElement>('[data-queue-select]')];
    const index = rows.indexOf(event.target);
    const next =
      event.key === 'ArrowDown'
        ? Math.min(rows.length - 1, index + 1)
        : event.key === 'ArrowUp'
          ? Math.max(0, index - 1)
          : event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? rows.length - 1
              : -1;
    if (next < 0) return;
    event.preventDefault();
    rows[next]?.focus();
    if (hasInlineInspector) setSelectedJobID(rows[next]?.dataset.queueSelect || null);
  };
  return (
    <div className="min-w-0">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-gray-800 pb-3 text-xs tabular-nums text-gray-400">
        <span>
          <strong className="font-medium text-gray-100">{totalFor(['running'])}</strong> running
        </span>
        <span>
          <strong className="font-medium text-gray-100">{totalFor(['queued'])}</strong> queued
        </span>
        <span>
          <strong className="font-medium text-gray-100">{totalFor(['completed'])}</strong> completed
        </span>
        <span className="ml-auto">
          {isGPUInfoLoaded ? `${gpuList.length} local GPUs` : 'GPU telemetry unavailable'}
        </span>
      </div>
      <div className="flex min-w-0 items-start">
        <div
          ref={listRef}
          onKeyDown={handleRowKeys}
          onFocus={event => {
            if (event.target instanceof HTMLElement && event.target.dataset.queueSelect)
              setFocusedJobID(event.target.dataset.queueSelect);
          }}
          className="min-w-0 flex-1"
        >
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-gray-800 py-2">
            <div className="flex overflow-x-auto" aria-label="Job views">
              {tabs.map(tab => (
                <button
                  key={tab.key}
                  type="button"
                  aria-pressed={activeTab === tab.key}
                  onClick={() => {
                    setActiveTab(tab.key);
                    setPendingReorder(null);
                    restored.current = true;
                    rememberQueueWorkspace({ loadedCount: 50, scrollTop: 0 });
                  }}
                  className={classNames(
                    'inline-flex h-8 items-center gap-2 border-b-2 px-3 text-xs',
                    activeTab === tab.key
                      ? 'border-brand-400 text-brand-300'
                      : 'border-transparent text-gray-400 hover:text-gray-100',
                  )}
                >
                  {tab.label}
                  <span className="tabular-nums text-gray-500">{tab.count}</span>
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2 pr-2">
              <label className="flex items-center gap-2 text-xs text-gray-500">
                Sort
                <select
                  value={sort}
                  onChange={event => {
                    const value = event.target.value;
                    if (QUEUE_SORTS.some(sort => sort === value)) setSort(value as QueueSort);
                  }}
                  className="h-8 rounded border border-gray-700 bg-gray-950 px-2 text-xs text-gray-200"
                  aria-label="Sort history"
                >
                  <option value="newest">Newest</option>
                  <option value="oldest">Oldest</option>
                  <option value="name">Name</option>
                </select>
              </label>
              <button
                type="button"
                className="operator-icon-button h-8 w-8"
                onClick={refresh}
                aria-label="Refresh queue"
              >
                <RefreshCcw className={classNames('h-3.5 w-3.5', isRefreshing && 'animate-spin')} />
              </button>
            </div>
          </div>
          {(status === 'loading' || status === 'idle') && jobs.length === 0 && (
            <div role="status" className="px-3 py-5 text-sm text-gray-500">
              Loading jobs…
            </div>
          )}
          {(status === 'error' || queueStatus === 'error' || workerStatus === 'error') && (
            <PageNotice tone="warning" title="Queue data is incomplete">
              {status === 'error' && 'Jobs could not be refreshed. '}
              {queueStatus === 'error' && 'Queue state could not be refreshed. '}
              {workerStatus === 'error' && 'Worker details could not be refreshed. '}
              <button type="button" className="underline" onClick={refresh}>
                Retry
              </button>
            </PageNotice>
          )}
          {reorderError && (
            <PageNotice tone="danger" title="Queue order was not saved">
              {reorderError}
            </PageNotice>
          )}
          {(activeTab === 'active' || activeTab === 'all') &&
            groups.map(original => {
              const group = orderedGroup(original, pendingReorder);
              const key = laneKey(group.workerID, group.gpuIDs);
              const queued = group.jobs.filter(isQueued);
              const selectedIndex = queued.findIndex(job => job.id === selectedJobID);
              const changed = pendingReorder?.laneKey === key;
              return (
                <section key={key} className="mb-4 overflow-x-auto">
                  <WorkerToolbar
                    group={group}
                    queue={queues.find(queue => queue.worker_id === group.workerID && queue.gpu_ids === group.gpuIDs)}
                    unavailable={queueStatus === 'error' || queueStatus === 'loading'}
                    onRefresh={refresh}
                  />
                  {group.jobs.length > 0 ? (
                    <>
                      <ColumnHeadings />
                      {group.jobs.map(job => (
                        <JobRow
                          key={job.id}
                          job={job}
                          selected={job.id === selectedJobID}
                          tabIndex={tabStopID === job.id ? 0 : -1}
                          queuePosition={
                            !filterText.trim() && !hasMore && isQueued(job) ? queued.indexOf(job) + 1 : undefined
                          }
                          canDrag={isQueued(job) && canReorder}
                          dragTarget={dragOverJobID === job.id}
                          onSelect={setSelectedJobID}
                          onRefresh={refresh}
                          onDragStart={setDragJobID}
                          onDragOver={(id, event) => {
                            if (canReorder && dragJobID && isQueued(job)) {
                              event.preventDefault();
                              setDragOverJobID(id);
                            }
                          }}
                          onDrop={id => {
                            if (dragJobID) stageMove(group, dragJobID, id);
                            setDragJobID(null);
                            setDragOverJobID(null);
                          }}
                          onDragEnd={() => {
                            setDragJobID(null);
                            setDragOverJobID(null);
                          }}
                        />
                      ))}
                    </>
                  ) : (
                    <div className="flex flex-wrap items-center gap-3 px-3 py-5 text-xs text-gray-500">
                      <span>{filterText.trim() ? 'No matching jobs in this lane.' : 'No active jobs.'}</span>
                      {!filterText.trim() && emptyAction}
                    </div>
                  )}
                  {queued.length > 1 && (
                    <div className="flex flex-wrap items-center gap-2 border-b border-gray-800 px-3 py-2">
                      <button
                        className="operator-button h-8 text-xs"
                        type="button"
                        disabled={!canReorder || selectedIndex <= 0}
                        onClick={() => stageMove(group, queued[selectedIndex].id, queued[selectedIndex - 1].id)}
                      >
                        <ArrowUp className="h-3.5 w-3.5" />
                        Move up
                      </button>
                      <button
                        className="operator-button h-8 text-xs"
                        type="button"
                        disabled={!canReorder || selectedIndex < 0 || selectedIndex >= queued.length - 1}
                        onClick={() => stageMove(group, queued[selectedIndex].id, queued[selectedIndex + 1].id)}
                      >
                        <ArrowDown className="h-3.5 w-3.5" />
                        Move down
                      </button>
                      {(filterText.trim() || hasMore) && (
                        <span className="text-xs text-gray-500">Clear search and load all jobs to reorder.</span>
                      )}
                      {!filterText.trim() && !hasMore && !completeQueueSnapshot && status === 'success' && (
                        <span className="text-xs text-gray-500">
                          Queue data is still synchronizing. Refresh before reordering.
                        </span>
                      )}
                      {changed && (
                        <>
                          <span className="ml-auto text-xs text-amber-800 dark:text-amber-300">Unsaved order</span>
                          <button
                            className="operator-button h-8 text-xs"
                            type="button"
                            disabled={reorderBusy}
                            onClick={() => setPendingReorder(null)}
                          >
                            Cancel
                          </button>
                          <button
                            className="operator-button-primary h-8 text-xs"
                            type="button"
                            disabled={!canReorder}
                            onClick={() => void applyOrder(group)}
                          >
                            {reorderBusy ? 'Saving…' : 'Apply order'}
                          </button>
                        </>
                      )}
                    </div>
                  )}
                </section>
              );
            })}
          {(activeTab === 'history' || activeTab === 'failed' || (activeTab === 'all' && history.length > 0)) && (
            <section className="overflow-x-auto">
              <ColumnHeadings />
              {history.map(job => (
                <JobRow
                  key={job.id}
                  job={job}
                  selected={job.id === selectedJobID}
                  tabIndex={tabStopID === job.id ? 0 : -1}
                  onSelect={setSelectedJobID}
                  onRefresh={refresh}
                />
              ))}
            </section>
          )}
          {status === 'success' && filteredJobs.length === 0 && (activeTab !== 'active' || groups.length === 0) && (
            <div className="px-3 py-6 text-sm text-gray-500">
              {filterText.trim() ? 'No matching jobs.' : 'No jobs in this view.'}
            </div>
          )}
          {hasMore && (
            <button
              className="operator-button m-3"
              type="button"
              disabled={isRefreshing}
              onClick={() => void loadMore()}
            >
              {isRefreshing ? 'Loading…' : 'Load more jobs'}
            </button>
          )}
        </div>
        {hasInlineInspector && selectedJob && (
          <div
            className="sticky top-[calc(var(--studio-page-toolbar-height,48px)+.75rem)] flex h-[calc(100dvh-12rem)] shrink-0"
            style={{ width: inspectorWidth }}
          >
            <div
              role="separator"
              tabIndex={0}
              aria-label="Resize job inspector"
              aria-orientation="vertical"
              aria-controls="queue-job-inspector"
              aria-valuemin={280}
              aria-valuemax={520}
              aria-valuenow={inspectorWidth}
              className="w-2 shrink-0 cursor-col-resize touch-none border-l border-gray-800 bg-gray-900/50 hover:bg-brand-500/30 focus-visible:bg-brand-500/30 focus-visible:outline-none"
              onPointerDown={event => {
                resizeStart.current = { x: event.clientX, width: inspectorWidth };
                event.currentTarget.setPointerCapture(event.pointerId);
              }}
              onPointerMove={event => {
                if (resizeStart.current)
                  setInspectorWidth(
                    clampInspectorWidth(resizeStart.current.width + resizeStart.current.x - event.clientX),
                  );
              }}
              onPointerUp={event => {
                resizeStart.current = null;
                event.currentTarget.releasePointerCapture(event.pointerId);
              }}
              onPointerCancel={() => {
                resizeStart.current = null;
              }}
              onKeyDown={event => {
                if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
                  event.preventDefault();
                  setInspectorWidth(width =>
                    clampInspectorWidth(
                      event.key === 'Home'
                        ? 280
                        : event.key === 'End'
                          ? 520
                          : width + (event.key === 'ArrowLeft' ? 20 : -20),
                    ),
                  );
                }
              }}
            />
            <div className="min-w-0 flex-1">
              <SelectedJobInspector job={selectedJob} onRefresh={refresh} onClose={closeInspector} />
            </div>
          </div>
        )}
      </div>
      <footer className="space-y-1 border-t border-gray-800 px-3 py-2 text-xs text-gray-500">
        <QueueSyncStatus freshness={queueFreshness} workers={workers} />
        {freshness && (freshness.failed > 0 || now - Date.parse(freshness.updatedAt) > 30000) && (
          <p role="status" className="text-amber-800 dark:text-amber-300">
            Job synchronization delayed. Last checked {new Date(freshness.updatedAt).toLocaleTimeString()}.
          </p>
        )}
      </footer>
      <Dialog
        open={widthLoaded && !hasInlineInspector && selectedJob !== null}
        onClose={closeInspector}
        className="fixed inset-0 z-50"
      >
        <DialogBackdrop className="fixed inset-0 bg-black/50" />
        <div className="fixed inset-0 flex justify-end">
          <DialogPanel className="h-full w-full max-w-md border-l border-gray-700 bg-gray-950 shadow-xl">
            <DialogTitle className="sr-only">Job inspector</DialogTitle>
            {selectedJob && <SelectedJobInspector job={selectedJob} onRefresh={refresh} onClose={closeInspector} />}
          </DialogPanel>
        </div>
      </Dialog>
    </div>
  );
}
