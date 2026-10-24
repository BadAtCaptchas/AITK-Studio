'use client';
import { useMemo, useState } from 'react';
import Link from '@/navigation/Link';
import { Cpu, RefreshCw, Search, ExternalLink, Server } from 'lucide-react';
import useGPUInfo from '@/hooks/useGPUInfo';
import useCPUInfo from '@/hooks/useCPUInfo';
import useMonitorStream from '@/hooks/useMonitorStream';
import useJobsList from '@/hooks/useJobsList';
import useQueueList from '@/hooks/useQueueList';
import useWorkers from '@/hooks/useWorkers';
import useJob from '@/hooks/useJob';
import useJobLog from '@/hooks/useJobLog';
import JobLossGraph from '@/components/JobLossGraph';
import JobActionBar from '@/components/JobActionBar';
import JobConfigViewer from '@/components/JobConfigViewer';
import SampleImages from '@/components/SampleImages';
import { ProgressBar } from '@/components/OperatorPrimitives';
import { getTotalSteps } from '@/utils/jobs';
import { isActiveJobStatus } from '@/utils/jobStatus';
import type { Job } from '@/types';

const tabs = ['Metrics', 'Samples', 'Configuration', 'Logs'] as const;
type MonitorTab = (typeof tabs)[number];
type RunView = 'all' | 'active' | 'history' | 'failed';
const number = (value: number | null | undefined, unit = '') =>
  typeof value === 'number' && Number.isFinite(value) ? `${Math.round(value)}${unit}` : '—';
const memory = (value: number) => (Number.isFinite(value) ? `${(value / 1024).toFixed(1)} GB` : '—');
function totalSteps(job: Job) {
  try {
    return getTotalSteps(job) || 0;
  } catch {
    return 0;
  }
}
function dateLabel(value: Job['updated_at']) {
  const date = new Date(value);
  return Number.isFinite(date.getTime())
    ? date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    : '—';
}

function HardwareStrip() {
  const gpu = useGPUInfo();
  const cpu = useCPUInfo();
  const monitor = useMonitorStream();
  return (
    <section className="studio-hardware" aria-label="Local hardware">
      {gpu.gpuList.map(device => (
        <div className="studio-hardware-item" key={device.index}>
          <Cpu size={15} className="text-gray-400" />
          <span className="font-medium">{device.name}</span>
          <span className="studio-hardware-meter" aria-hidden="true">
            <i style={{ width: `${Math.min(100, Math.max(0, device.utilization.gpu || 0))}%` }} />
          </span>
          <span>{number(device.utilization.gpu, '%')}</span>
          <span className="text-gray-400">
            VRAM {memory(device.memory.used)} / {memory(device.memory.total)}
          </span>
          <span className={device.temperature >= 80 ? 'text-amber-800 dark:text-amber-400' : 'text-gray-400'}>
            {number(device.temperature, '°C')}
          </span>
        </div>
      ))}
      {!gpu.gpuList.length && (
        <span>
          {gpu.status === 'error' ? 'GPU unavailable' : gpu.isGPUInfoLoaded ? 'No GPU detected' : 'Loading GPU…'}
        </span>
      )}
      {gpu.status === 'error' && (
        <button type="button" className="text-amber-800 dark:text-amber-400 underline" onClick={() => void gpu.refreshGpuInfo()}>
          Retry GPU telemetry
        </button>
      )}
      <span className="studio-hardware-item text-gray-400">
        <Cpu size={14} /> CPU {number(cpu.cpuInfo?.currentLoad, '%')}
        {cpu.cpuInfo && (
          <span>
            RAM {memory(cpu.cpuInfo.totalMemory - cpu.cpuInfo.availableMemory)} / {memory(cpu.cpuInfo.totalMemory)}
          </span>
        )}
      </span>
      {cpu.status === 'error' && (
        <button type="button" className="text-amber-800 dark:text-amber-400 underline" onClick={() => void cpu.refreshCpuInfo()}>
          Retry CPU telemetry
        </button>
      )}
      <span className="ml-auto text-xs text-gray-500">
        {monitor.connected
          ? 'Live telemetry'
          : gpu.lastUpdated
            ? `Polled ${gpu.lastUpdated.toLocaleTimeString()}`
            : 'Connecting…'}
      </span>
    </section>
  );
}

function RunOutput({ job, expanded = true }: { job: Job; expanded?: boolean }) {
  const { log, status, refresh } = useJobLog(job.id, isActiveJobStatus(job.status) ? 3000 : null, job.attempt_id ?? null);
  return (
    <details className="studio-output" open={expanded}>
      <summary>
        Output{' '}
        <span className="ml-3 text-gray-500">
          {status === 'error' ? 'Unavailable' : status === 'loading' ? 'Loading…' : 'Training log'}
        </span>
      </summary>
      {status === 'error' && (
        <div className="studio-inline-notice" role="status">
          Could not refresh output. {log ? 'Showing the last received log.' : ''}
          <button onClick={() => void refresh()}>Retry</button>
        </div>
      )}
      <pre aria-label="Training output">
        {log ? log.split('\n').slice(-180).join('\n') : status === 'loading' ? 'Loading output…' : 'No output yet.'}
      </pre>
    </details>
  );
}

function RunMonitor({ id, workerName }: { id: string; workerName: (id: string) => string }) {
  const { job, status, refreshJob } = useJob(id, 5000);
  const [tab, setTab] = useState<MonitorTab>('Metrics');
  if (!job)
    return (
      <div className="studio-inline-notice" role="status">
        {status === 'error' ? 'Selected run unavailable.' : 'Loading selected run…'}
        {status === 'error' && <button onClick={() => void refreshJob()}>Retry</button>}
      </div>
    );
  const steps = totalSteps(job);
  return (
    <>
      <div className="studio-monitor-tabs" role="tablist" aria-label="Selected run views">
        {tabs.map(value => (
          <button
            key={value}
            type="button"
            role="tab"
            id={`overview-tab-${value}`}
            aria-controls={`overview-panel-${value}`}
            aria-selected={tab === value}
            tabIndex={tab === value ? 0 : -1}
            onClick={() => setTab(value)}
            onKeyDown={event => {
              const index = tabs.indexOf(value);
              const next =
                event.key === 'ArrowRight'
                  ? (index + 1) % tabs.length
                  : event.key === 'ArrowLeft'
                    ? (index + tabs.length - 1) % tabs.length
                    : event.key === 'Home'
                      ? 0
                      : event.key === 'End'
                        ? tabs.length - 1
                        : null;
              if (next !== null) {
                event.preventDefault();
                setTab(tabs[next]);
                document.getElementById(`overview-tab-${tabs[next]}`)?.focus();
              }
            }}
          >
            {value}
          </button>
        ))}
        <Link
          href={`/jobs/${job.id}?tab=${tab === 'Metrics' ? 'metrics' : tab === 'Configuration' ? 'config' : tab.toLowerCase()}`}
          className="ml-auto flex items-center gap-2 whitespace-nowrap px-3 text-xs text-gray-400"
        >
          Open run <ExternalLink size={13} />
        </Link>
      </div>
      {status === 'error' && (
        <div className="studio-inline-notice text-amber-800 dark:text-amber-400" role="status">
          Run details could not refresh; showing last received state.
          <button onClick={() => void refreshJob()}>Retry</button>
        </div>
      )}
      {job.remote_error && (
        <div className="studio-inline-notice text-amber-800 dark:text-amber-400" role="status">
          Worker update failed: {job.remote_error}. Last sync:{' '}
          {job.remote_sync_at ? dateLabel(job.remote_sync_at) : 'never'}.
        </div>
      )}
      <div className="studio-monitor-workspace">
        <section
          className="studio-monitor-main"
          role="tabpanel"
          id={`overview-panel-${tab}`}
          aria-labelledby={`overview-tab-${tab}`}
        >
          {tab === 'Metrics' && <JobLossGraph job={job} compact />}
          {tab === 'Samples' && (
            <div className="h-[430px]">
              <SampleImages job={job} layout="panel" />
            </div>
          )}
          {tab === 'Configuration' && (
            <div className="h-[430px]">
              <JobConfigViewer job={job} />
            </div>
          )}
          {tab === 'Logs' && <RunOutput key={job.id} job={job} />}
        </section>
        <aside className="studio-monitor-inspector" aria-label="Run inspector">
          <h2 className="mb-3 break-words text-sm font-semibold">{job.name}</h2>
          <span className="studio-state text-xs capitalize" data-state={job.status}>
            {job.status}
          </span>
          <dl className="mt-5 grid grid-cols-[80px_minmax(0,1fr)] gap-x-3 gap-y-3">
            <dt>Target</dt>
            <dd>
              {workerName(job.worker_id)} / GPU {job.gpu_ids || '—'}
            </dd>
            <dt>Steps</dt>
            <dd>
              {job.step.toLocaleString()} / {steps ? steps.toLocaleString() : '—'}
            </dd>
            <dt>Progress</dt>
            <dd>{steps ? `${Math.round(Math.min(100, (job.step / steps) * 100))}%` : '—'}</dd>
            <dt>Speed</dt>
            <dd>{job.speed_string || '—'}</dd>
            <dt>Updated</dt>
            <dd>{dateLabel(job.updated_at)}</dd>
            <dt>Created</dt>
            <dd>{dateLabel(job.created_at)}</dd>
          </dl>
          {job.info && (
            <p className="mt-5 break-words border-t border-gray-800 pt-3 text-xs text-gray-400">{job.info}</p>
          )}
          <div className="mt-5 border-t border-gray-800 pt-3">
            <JobActionBar job={job} onRefresh={refreshJob} hideView />
          </div>
          <Link href={`/jobs/${job.id}?tab=metrics`} className="mt-4 block text-xs text-brand-400">
            Full monitoring and chart settings →
          </Link>
        </aside>
      </div>
      {tab !== 'Logs' && <RunOutput key={job.id} job={job} />}
    </>
  );
}

export default function Dashboard() {
  const active = useJobsList({ job_type: 'train', view: 'active', reloadInterval: 5000 });
  const history = useJobsList({ job_type: 'train', view: 'history', reloadInterval: 15000 });
  const queues = useQueueList(5000);
  const workers = useWorkers();
  const [view, setView] = useState<RunView>('all');
  const [search, setSearch] = useState('');
  const [selectedID, setSelectedID] = useState<string | null>(null);
  const allJobs = useMemo(() => {
    // Independent polls can briefly contain the same run across a state change.
    const byID = new Map<string, Job>();
    for (const job of [...active.jobs, ...history.jobs]) {
      const previous = byID.get(job.id);
      if (!previous || new Date(job.updated_at).getTime() > new Date(previous.updated_at).getTime())
        byID.set(job.id, job);
    }
    return [...byID.values()].sort((a, b) => {
      const priority = (job: Job) =>
        isActiveJobStatus(job.status) && job.status !== 'queued' ? 0 : job.status === 'queued' ? 1 : 2;
      return priority(a) - priority(b) || new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
    });
  }, [active.jobs, history.jobs]);
  const filtered = allJobs.filter(
    job =>
      (view === 'all' ||
        (view === 'active'
          ? isActiveJobStatus(job.status)
          : view === 'history'
            ? !isActiveJobStatus(job.status)
            : ['error', 'failed'].includes(job.status))) &&
      job.name.toLowerCase().includes(search.toLowerCase()),
  );
  const selected = filtered.find(job => job.id === selectedID) || filtered[0];
  const counts = active.statusCounts || history.statusCounts;
  const failedWorkerUpdates = Math.max(active.freshness?.failed ?? 0, history.freshness?.failed ?? 0);
  const workerName = (id: string) =>
    !id || id === 'local' ? 'Local' : workers.workers.find(worker => worker.id === id)?.name || id;
  const refresh = () => {
    void active.refreshJobs();
    void history.refreshJobs();
    void queues.refreshQueues();
    workers.refreshWorkers();
  };
  const loading =
    !allJobs.length && [active.status, history.status].some(status => status === 'idle' || status === 'loading');
  return (
    <div className="studio-workbench">
      <HardwareStrip />
      <header className="studio-workbench-toolbar">
        <h1>Runs</h1>
        <div className="studio-filter-tabs" role="group" aria-label="Run status">
          {(
            [
              ['all', 'All runs'],
              ['active', 'Active'],
              ['history', 'Recent'],
              ['failed', 'Failed'],
            ] as const
          ).map(([value, label]) => (
            <button key={value} type="button" aria-pressed={view === value} onClick={() => setView(value)}>
              {label}
            </button>
          ))}
        </div>
        <label className="ml-auto flex items-center gap-2 rounded border border-gray-700 bg-gray-950 px-2 py-1.5 text-gray-400">
          <Search size={14} />
          <input
            aria-label="Search runs"
            placeholder="Search runs…"
            className="w-40 min-w-0 bg-transparent text-xs text-gray-100 outline-none"
            value={search}
            onChange={event => setSearch(event.target.value)}
          />
        </label>
        <button
          type="button"
          className="operator-icon-button"
          onClick={refresh}
          aria-label="Refresh overview"
          title="Refresh"
        >
          <RefreshCw size={15} />
        </button>
        <Link href="/jobs" className="operator-button">
          Queue
        </Link>
      </header>
      {(active.status === 'error' || history.status === 'error' || failedWorkerUpdates > 0) && (
        <div className="studio-inline-notice text-amber-800 dark:text-amber-400" role="status">
          {active.status === 'error' ? 'Active runs could not refresh. ' : ''}
          {history.status === 'error' ? 'Recent runs could not refresh. ' : ''}
          {failedWorkerUpdates > 0 ? 'Some worker updates failed. ' : ''}Available data remains visible.
          <button onClick={refresh}>Retry</button>
        </div>
      )}
      <div className="max-h-[350px] min-h-[150px] overflow-auto" aria-label="Run list">
        <table className="studio-run-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>State</th>
              <th>Target</th>
              <th>Steps</th>
              <th>Progress</th>
              <th>Speed</th>
              <th>Updated</th>
              <th>
                <span className="sr-only">Open</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((job, index) => {
              const steps = totalSteps(job);
              const progress = steps ? Math.min(100, (job.step / steps) * 100) : 0;
              return (
                <tr key={job.id} data-selected={selected?.id === job.id}>
                  <td>
                    <button
                      type="button"
                      className="studio-run-name"
                      aria-pressed={selected?.id === job.id}
                      onClick={() => setSelectedID(job.id)}
                      onKeyDown={event => {
                        const next =
                          event.key === 'ArrowDown'
                            ? Math.min(filtered.length - 1, index + 1)
                            : event.key === 'ArrowUp'
                              ? Math.max(0, index - 1)
                              : event.key === 'Home'
                                ? 0
                                : event.key === 'End'
                                  ? filtered.length - 1
                                  : null;
                        if (next !== null) {
                          event.preventDefault();
                          setSelectedID(filtered[next].id);
                          const table = event.currentTarget.closest('tbody');
                          table?.querySelectorAll<HTMLButtonElement>('.studio-run-name')[next]?.focus();
                        }
                      }}
                    >
                      {job.name}
                    </button>
                  </td>
                  <td>
                    <span className="studio-state capitalize" data-state={job.status}>
                      {job.status}
                    </span>
                  </td>
                  <td className="text-gray-400">
                    {workerName(job.worker_id)} / GPU {job.gpu_ids || '—'}
                  </td>
                  <td>
                    {job.step.toLocaleString()} / {steps ? steps.toLocaleString() : '—'}
                  </td>
                  <td>
                    <div className="flex items-center gap-3">
                      <ProgressBar className="w-24" value={progress} />
                      <span className="w-8 text-right text-gray-400">{Math.round(progress)}%</span>
                    </div>
                  </td>
                  <td className="text-gray-400">{job.speed_string || '—'}</td>
                  <td className="text-gray-400">{dateLabel(job.updated_at)}</td>
                  <td>
                    <Link href={`/jobs/${job.id}`} className="operator-icon-button" aria-label={`Open ${job.name}`}>
                      <ExternalLink size={14} />
                    </Link>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {!filtered.length && (
          <p role="status" className="p-6 text-sm text-gray-400">
            {loading
              ? 'Loading runs…'
              : allJobs.length
                ? 'No runs match this filter.'
                : active.status === 'error' || history.status === 'error'
                  ? 'Run data is unavailable. Retry above.'
                  : 'No training runs. Create a training job to begin.'}
          </p>
        )}
        {(active.hasMore || history.hasMore) && (
          <div className="flex gap-3 px-4 py-2">
            {active.hasMore && (
              <button
                type="button"
                className="operator-button"
                disabled={active.isRefreshing}
                onClick={() => void active.loadMore()}
              >
                Load more active runs
              </button>
            )}
            {history.hasMore && (
              <button
                type="button"
                className="operator-button"
                disabled={history.isRefreshing}
                onClick={() => void history.loadMore()}
              >
                Load more recent runs
              </button>
            )}
          </div>
        )}
      </div>
      <div className="studio-workbench-status" aria-label="Queue state">
        <Server size={13} />
        {queues.queues.map(queue => (
          <span key={queue.id}>
            {workerName(queue.worker_id)} / GPU {queue.gpu_ids}:{' '}
            <strong className="font-medium">{queue.is_running ? 'queue running' : 'queue stopped'}</strong>
          </span>
        ))}
        {!queues.queues.length && (
          <span>
            {queues.status === 'error'
              ? 'Queue state unavailable'
              : queues.status === 'success'
                ? 'No queues configured'
                : 'Loading queues…'}
          </span>
        )}
        {queues.status === 'error' && (
          <button className="underline text-amber-800 dark:text-amber-400" onClick={() => void queues.refreshQueues()}>
            Retry queue state
          </button>
        )}
        {queues.freshness
          .filter(item => item.stale || item.error)
          .map(item => (
            <span key={item.workerId} className="text-amber-800 dark:text-amber-400">
              {workerName(item.workerId)} stale · last sync {item.updatedAt ? dateLabel(item.updatedAt) : 'never'}
            </span>
          ))}
        {workers.status === 'error' && (
          <button className="underline text-amber-800 dark:text-amber-400" onClick={workers.refreshWorkers}>
            Worker names unavailable · retry
          </button>
        )}
      </div>
      {selected ? (
        <RunMonitor key={selected.id} id={selected.id} workerName={workerName} />
      ) : (
        <div className="flex min-h-[280px] flex-1 items-center justify-center text-sm text-gray-500">
          Select a run to inspect metrics, samples and output.
        </div>
      )}
      <footer className="studio-workbench-status">
        <span>{counts ? `${counts.running || 0} running` : 'Run counts unavailable'}</span>
        {counts && (
          <>
            <span>{counts.queued || 0} queued</span>
            <span>{counts.completed || 0} completed</span>
            <span>{(counts.error || 0) + (counts.failed || 0)} failed</span>
          </>
        )}
        <span className="ml-auto">
          {allJobs.length} loaded
          {active.freshness?.updatedAt ? ` · synced ${dateLabel(active.freshness.updatedAt)}` : ''}
        </span>
      </footer>
    </div>
  );
}
