'use client';

import { useMemo, useState } from 'react';
import Link from 'next/link';
import classNames from 'classnames';
import { Activity, ListFilter, Loader2, Play, RefreshCcw, Search } from 'lucide-react';
import JobActionBar from '@/components/JobActionBar';
import { PageNotice, ProgressBar, StatusBadge } from '@/components/OperatorPrimitives';
import useJobsList from '@/hooks/useJobsList';
import useWorkers from '@/hooks/useWorkers';
import { getTotalSteps } from '@/utils/jobs';
import { formatProjectTime } from './ProjectWorkspaceShell';
import type { Job } from '@/types';

type StatusFilter = 'all' | 'active' | 'completed' | 'failed' | 'stopped';
type TypeFilter = 'all' | 'train' | 'generate' | 'caption';

function matchesStatus(job: Job, filter: StatusFilter) {
  if (filter === 'all') return true;
  if (filter === 'active') return ['queued', 'starting', 'running', 'stopping'].includes(job.status);
  if (filter === 'failed') return ['failed', 'error'].includes(job.status);
  return job.status === filter;
}

function runLabel(job: Job) {
  if (job.job_type === 'generate') return 'Generate';
  if (job.job_type === 'caption') return 'Caption';
  return 'Train';
}

function RunProgress({ job }: { job: Job }) {
  if (job.job_type !== 'train') return <span className="text-xs text-gray-600">Not step-based</span>;
  const total = getTotalSteps(job);
  return (
    <div className="min-w-28">
      <div className="flex items-center justify-between text-[10px] text-gray-500">
        <span>{job.step.toLocaleString()}</span>
        <span>{total ? total.toLocaleString() : '—'}</span>
      </div>
      <ProgressBar value={total ? (job.step / total) * 100 : 0} className="mt-1" />
    </div>
  );
}

export default function ProjectRunsTable({ projectID }: { projectID: string }) {
  const { jobs, status, refreshJobs } = useJobsList({ projectID, reloadInterval: 5000 });
  const { workers } = useWorkers();
  const [query, setQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [typeFilter, setTypeFilter] = useState<TypeFilter>('all');

  const filteredJobs = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return [...jobs]
      .filter(job => matchesStatus(job, statusFilter))
      .filter(job => typeFilter === 'all' || job.job_type === typeFilter)
      .filter(job => {
        if (!normalizedQuery) return true;
        const workerName = workers.find(worker => worker.id === job.worker_id)?.name || '';
        return [job.name, job.info, job.status, job.job_type, job.job_ref || '', job.worker_id, workerName].some(
          value => value.toLowerCase().includes(normalizedQuery),
        );
      })
      .sort((a, b) => {
        const activeA = ['queued', 'starting', 'running', 'stopping'].includes(a.status) ? 1 : 0;
        const activeB = ['queued', 'starting', 'running', 'stopping'].includes(b.status) ? 1 : 0;
        if (activeA !== activeB) return activeB - activeA;
        if (activeA && a.queue_position !== b.queue_position) return a.queue_position - b.queue_position;
        return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
      });
  }, [jobs, query, statusFilter, typeFilter, workers]);

  const counts = useMemo(
    () => ({
      active: jobs.filter(job => ['queued', 'starting', 'running', 'stopping'].includes(job.status)).length,
      completed: jobs.filter(job => job.status === 'completed').length,
      failed: jobs.filter(job => ['error', 'failed'].includes(job.status)).length,
    }),
    [jobs],
  );

  if (status === 'loading' && jobs.length === 0) {
    return (
      <div className="flex min-h-64 items-center justify-center text-sm text-gray-500">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading project runs
      </div>
    );
  }

  if (status === 'error' && jobs.length === 0) {
    return (
      <PageNotice tone="danger" title="Project runs could not be loaded">
        The project job list is temporarily unavailable.
      </PageNotice>
    );
  }

  return (
    <div className="space-y-3">
      <div className="grid gap-2 sm:grid-cols-3">
        {[
          { label: 'Active', value: counts.active, className: 'text-brand-300' },
          { label: 'Completed', value: counts.completed, className: 'text-emerald-300' },
          { label: 'Needs attention', value: counts.failed, className: 'text-amber-300' },
        ].map(item => (
          <div key={item.label} className="rounded-sm border border-gray-800 bg-gray-900/35 px-3 py-2">
            <div className="text-[10px] font-semibold uppercase tracking-wide text-gray-600">{item.label}</div>
            <div className={classNames('mt-1 text-lg font-semibold', item.className)}>{item.value}</div>
          </div>
        ))}
      </div>

      <div className="flex flex-col gap-2 rounded-sm border border-gray-800 bg-gray-900/30 p-2 sm:flex-row sm:items-center">
        <label className="relative min-w-0 flex-1">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-600" />
          <input
            value={query}
            onChange={event => setQuery(event.target.value)}
            placeholder="Search run name, worker, or status"
            className="h-9 w-full rounded-sm border border-gray-800 bg-gray-950 pl-9 pr-3 text-sm text-gray-100 outline-none placeholder:text-gray-600 focus:border-brand-700"
          />
        </label>
        <div className="operator-scrollbar-none flex min-w-0 items-center gap-2 overflow-x-auto">
          <ListFilter className="h-4 w-4 flex-none text-gray-600" />
          <select
            value={statusFilter}
            onChange={event => setStatusFilter(event.target.value as StatusFilter)}
            className="h-9 flex-none rounded-sm border border-gray-800 bg-gray-950 px-2.5 text-xs text-gray-300 outline-none focus:border-brand-700"
          >
            <option value="all">All statuses</option>
            <option value="active">Active</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="stopped">Stopped</option>
          </select>
          <select
            value={typeFilter}
            onChange={event => setTypeFilter(event.target.value as TypeFilter)}
            className="h-9 flex-none rounded-sm border border-gray-800 bg-gray-950 px-2.5 text-xs text-gray-300 outline-none focus:border-brand-700"
          >
            <option value="all">All run types</option>
            <option value="train">Training</option>
            <option value="generate">Generation</option>
            <option value="caption">Captioning</option>
          </select>
          <button
            type="button"
            onClick={refreshJobs}
            className="operator-icon-button h-9 w-9 flex-none"
            title="Refresh runs"
          >
            <RefreshCcw className={classNames('h-4 w-4', status === 'loading' ? 'animate-spin' : '')} />
          </button>
        </div>
      </div>

      <div className="text-[11px] text-gray-600">
        {filteredJobs.length} of {jobs.length} runs
      </div>

      {filteredJobs.length === 0 ? (
        <div className="flex min-h-56 items-center justify-center rounded-md border border-dashed border-gray-800 bg-gray-900/20 p-5 text-center">
          <div>
            <Play className="mx-auto h-8 w-8 text-gray-700" />
            <div className="mt-3 text-sm font-semibold text-gray-300">No runs match these filters</div>
            <p className="mt-1 text-xs text-gray-600">Clear the search or choose another status.</p>
          </div>
        </div>
      ) : (
        <>
          <div className="hidden overflow-hidden rounded-md border border-gray-800 bg-gray-900/30 md:block">
            <div className="grid grid-cols-[minmax(220px,1fr)_130px_130px_150px_120px_170px] gap-3 border-b border-gray-800 px-3 py-2 text-[10px] font-semibold uppercase tracking-wide text-gray-600">
              <span>Run</span>
              <span>Status</span>
              <span>Progress</span>
              <span>Worker</span>
              <span>Updated</span>
              <span className="text-right">Actions</span>
            </div>
            <div className="divide-y divide-gray-800/80">
              {filteredJobs.map(job => {
                const workerName =
                  job.worker_id === 'local'
                    ? 'Local'
                    : workers.find(worker => worker.id === job.worker_id)?.name || 'Remote';
                return (
                  <div
                    key={job.id}
                    className="grid min-w-[980px] grid-cols-[minmax(220px,1fr)_130px_130px_150px_120px_170px] items-center gap-3 px-3 py-2.5 hover:bg-gray-900/70"
                  >
                    <Link
                      href={`/projects/${encodeURIComponent(projectID)}/runs/${encodeURIComponent(job.id)}`}
                      className="flex min-w-0 items-center gap-2"
                    >
                      {['running', 'stopping'].includes(job.status) ? (
                        <Activity className="h-4 w-4 flex-none animate-pulse text-brand-400" />
                      ) : (
                        <Play className="h-4 w-4 flex-none text-gray-600" />
                      )}
                      <span className="min-w-0">
                        <span className="flex min-w-0 items-center gap-2">
                          <span className="rounded-sm border border-gray-800 bg-gray-950 px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-gray-500">
                            {runLabel(job)}
                          </span>
                          <span className="truncate text-sm font-medium text-gray-200">{job.name}</span>
                        </span>
                        <span className="mt-0.5 block truncate text-[11px] text-gray-600">
                          {job.info || job.job_ref || job.id}
                        </span>
                      </span>
                    </Link>
                    <StatusBadge status={job.status} className="w-fit" />
                    <RunProgress job={job} />
                    <div className="min-w-0">
                      <div className="truncate text-xs text-gray-300">{workerName}</div>
                      <div className="mt-0.5 truncate text-[10px] text-gray-600">GPU {job.gpu_ids || 'default'}</div>
                    </div>
                    <span className="text-xs text-gray-500">{formatProjectTime(job.updated_at)}</span>
                    <div className="flex justify-end">
                      <JobActionBar
                        job={job}
                        onRefresh={refreshJobs}
                        afterDelete={refreshJobs}
                        autoStartQueue={false}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="space-y-2 md:hidden">
            {filteredJobs.map(job => {
              const workerName =
                job.worker_id === 'local'
                  ? 'Local'
                  : workers.find(worker => worker.id === job.worker_id)?.name || 'Remote';
              return (
                <article key={job.id} className="rounded-md border border-gray-800 bg-gray-900/35 p-3">
                  <div className="flex min-w-0 items-start justify-between gap-3">
                    <Link
                      href={`/projects/${encodeURIComponent(projectID)}/runs/${encodeURIComponent(job.id)}`}
                      className="min-w-0 flex-1"
                    >
                      <div className="flex min-w-0 items-center gap-2">
                        <span className="rounded-sm border border-gray-800 bg-gray-950 px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-gray-500">
                          {runLabel(job)}
                        </span>
                        <span className="truncate text-sm font-semibold text-gray-200">{job.name}</span>
                      </div>
                      <div className="mt-1 truncate text-xs text-gray-600">{job.info || job.job_ref || job.id}</div>
                    </Link>
                    <StatusBadge status={job.status} />
                  </div>
                  <div className="mt-3">
                    <RunProgress job={job} />
                  </div>
                  <div className="mt-3 flex items-end justify-between gap-3 border-t border-gray-800 pt-2">
                    <div className="text-[11px] text-gray-600">
                      <div>
                        {workerName} / GPU {job.gpu_ids || 'default'}
                      </div>
                      <div className="mt-0.5">{formatProjectTime(job.updated_at)}</div>
                    </div>
                    <JobActionBar job={job} onRefresh={refreshJobs} afterDelete={refreshJobs} autoStartQueue={false} />
                  </div>
                </article>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
