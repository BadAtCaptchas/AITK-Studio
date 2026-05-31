import { useMemo, useState } from 'react';
import Link from 'next/link';
import { CgSpinner } from 'react-icons/cg';
import { Search } from 'lucide-react';
import useJobsList from '@/hooks/useJobsList';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import type { Job, Queue } from '@/types';
import JobActionBar from './JobActionBar';
import useQueueList from '@/hooks/useQueueList';
import { startQueue, stopQueue } from '@/utils/queue';
import useGPUInfo from '@/hooks/useGPUInfo';
import { HFDownloadProgressInline } from '@/components/HFDownloadProgress';
import useWorkers from '@/hooks/useWorkers';
import { getTotalSteps } from '@/utils/jobs';
import { PageNotice, ProgressBar, QueueStateBadge, StatusBadge } from '@/components/OperatorPrimitives';

interface JobsTableProps {
  autoStartQueue?: boolean;
  onlyActive?: boolean;
  job_type?: string | null;
}

type JobGroup = {
  name: string;
  jobs: Job[];
  workerID: string;
  gpuIDs: string | null;
};

const activeJobStatuses = new Set(['queued', 'running', 'stopping']);

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

export default function JobsTable({ onlyActive = false, job_type = null }: JobsTableProps) {
  const { jobs, status, refreshJobs } = useJobsList({ onlyActive, reloadInterval: 5000, job_type });
  const { queues, status: queueStatus, refreshQueues } = useQueueList();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const { workers, status: workerStatus } = useWorkers();
  const [filterText, setFilterText] = useState('');

  const refresh = () => {
    refreshJobs();
    refreshQueues();
  };

  const columns: TableColumn[] = [
    {
      title: 'Name',
      key: 'name',
      render: row => {
        const { prefix, title } = jobDisplayTitle(row);
        return (
          <Link href={`/jobs/${row.id}`} className="flex min-w-0 items-center gap-2 font-medium text-gray-100">
            {['running', 'stopping'].includes(row.status) ? (
              <CgSpinner className="h-4 w-4 flex-none animate-spin text-cyan-400" />
            ) : null}
            <span className="rounded-sm border border-gray-800 bg-gray-950 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-gray-500">
              {prefix}
            </span>
            <span className="truncate">{title}</span>
          </Link>
        );
      },
    },
    {
      title: 'Steps',
      key: 'steps',
      className: 'w-40',
      render: row => {
        if (row.job_type !== 'train') return <span className="text-gray-600">-</span>;
        const totalSteps = getTotalSteps(row);

        return (
          <div>
            <div className="text-xs text-gray-400">
              {totalSteps ? `${row.step} / ${totalSteps}` : `Step ${row.step}`}
            </div>
            {totalSteps ? <ProgressBar value={(row.step / totalSteps) * 100} className="mt-1" /> : null}
          </div>
        );
      },
    },
    {
      title: 'Worker',
      key: 'worker_id',
      className: 'whitespace-nowrap',
      render: row => {
        if (row.worker_id === 'local') return <span>Local</span>;
        return <span>{workers.find(worker => worker.id === row.worker_id)?.name || 'Remote'}</span>;
      },
    },
    {
      title: 'GPU',
      key: 'gpu_ids',
      className: 'whitespace-nowrap',
    },
    {
      title: 'Status',
      key: 'status',
      className: 'whitespace-nowrap',
      render: row => <StatusBadge status={row.status} />,
    },
    {
      title: 'Info',
      key: 'info',
      className: 'max-w-xs truncate',
      render: row => <HFDownloadProgressInline progress={row.hf_download_progress} fallback={row.info} />,
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'whitespace-nowrap text-right',
      render: row => <JobActionBar job={row} onRefresh={refreshJobs} autoStartQueue={false} />,
    },
  ];

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

    const jd: { [key: string]: JobGroup } = {};
    const workerName = (workerID: string) => {
      if (workerID === 'local') return 'Local';
      return workers.find(worker => worker.id === workerID)?.name || 'Remote';
    };
    const gpuName = (workerID: string, gpuID: string) => {
      if (workerID === 'local') {
        return gpuList.find(gpu => `${gpu.index}` === gpuID)?.name || `GPU #${gpuID}`;
      }
      const worker = workers.find(worker => worker.id === workerID);
      try {
        const gpus = JSON.parse(worker?.gpus || '[]') as Array<{ index: number; name: string }>;
        return gpus.find(gpu => `${gpu.index}` === gpuID)?.name || `GPU #${gpuID}`;
      } catch {
        return `GPU #${gpuID}`;
      }
    };
    const ensureWorkerGpuGroup = (workerID: string, gpuIDs: string) => {
      const key = `${workerID}:${gpuIDs}`;
      if (!jd[key]) {
        jd[key] = {
          name: `${workerName(workerID)} / ${gpuName(workerID, gpuIDs)}`,
          jobs: [],
          workerID,
          gpuIDs,
        };
      }
      return jd[key];
    };

    gpuList.forEach(gpu => {
      jd[`local:${gpu.index}`] = {
        name: `Local / ${gpu.name}`,
        jobs: [],
        workerID: 'local',
        gpuIDs: `${gpu.index}`,
      };
    });
    queues.forEach(queue => {
      ensureWorkerGpuGroup(queue.worker_id, queue.gpu_ids);
    });
    jd.idle = { name: 'Idle / history', jobs: [], workerID: 'local', gpuIDs: null };
    filteredJobs.forEach(job => {
      const workerID = job.worker_id || 'local';
      const gpuIDs = job.gpu_ids || '0';
      const key = `${workerID}:${gpuIDs}`;
      if (isRemoteWorker(workerID)) {
        ensureWorkerGpuGroup(workerID, gpuIDs).jobs.push(job);
      } else if (activeJobStatuses.has(job.status) && key in jd) {
        jd[key].jobs.push(job);
      } else {
        jd.idle.jobs.push(job);
      }
    });

    Object.keys(jd).forEach(key => {
      if (key === 'idle') {
        jd[key].jobs.sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime());
      } else {
        jd[key].jobs.sort((a, b) => {
          if (a.queue_position === null) return 1;
          if (b.queue_position === null) return -1;
          return a.queue_position - b.queue_position;
        });
      }
    });
    return jd;
  }, [filteredJobs, gpuList, isGPUInfoLoaded, queues, workers]);

  let isLoading = status === 'loading' || queueStatus === 'loading' || workerStatus === 'loading' || !isGPUInfoLoaded;
  if (Object.keys(jobsDict).length > 0) isLoading = false;

  const tableError =
    status === 'error'
      ? 'Jobs could not be loaded.'
      : queueStatus === 'error'
        ? 'Queues could not be loaded.'
        : workerStatus === 'error'
          ? 'Workers could not be loaded.'
          : null;

  if (isLoading && Object.keys(jobsDict).length === 0) {
    return (
      <div className="operator-surface p-3">
        <PageNotice tone="neutral" title="Loading queues and jobs">
          Fetching queue state, workers, and GPU telemetry.
        </PageNotice>
      </div>
    );
  }

  const activeGroups = Object.keys(jobsDict)
    .sort()
    .filter(key => key !== 'idle');

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="text-xs text-gray-500">
          {filteredJobs.length} of {jobs.length} jobs shown
          {onlyActive ? ' (active only)' : ''}
        </div>
        <label className="relative block w-full sm:w-72">
          <Search className="pointer-events-none absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
          <input
            value={filterText}
            onChange={event => setFilterText(event.target.value)}
            placeholder="Filter jobs, status, GPU"
            className="h-8 w-full border border-gray-800 bg-gray-950 pl-8 pr-3 text-sm text-gray-100 placeholder:text-gray-500 focus:border-cyan-700 focus:outline-none"
          />
        </label>
      </div>

      {tableError && (
        <PageNotice tone="danger" title="Queue data is incomplete">
          {tableError}
        </PageNotice>
      )}

      {activeGroups.map(groupKey => {
        const group = jobsDict[groupKey];
        const queue = queues.find(q => q.worker_id === group.workerID && q.gpu_ids === group.gpuIDs) as Queue;
        const queueRunning = queue?.is_running === true;
        return (
          <div key={groupKey}>
            <div className="flex flex-col gap-2 border border-b-0 border-gray-800 bg-gray-900 px-3 py-2 text-sm sm:flex-row sm:items-center">
              <div className="flex min-w-0 flex-1 items-center gap-2">
                <h2 className="truncate font-semibold text-gray-100">{group.name}</h2>
                <span className="rounded-sm border border-gray-700 bg-gray-950 px-2 py-0.5 text-xs text-gray-300">
                  GPU {group.gpuIDs}
                </span>
                <span className="text-xs text-gray-500">{group.jobs.length} jobs</span>
              </div>
              <div className="flex flex-wrap items-center gap-2 text-sm text-gray-300">
                <QueueStateBadge running={queueRunning} />
                {queueRunning ? (
                  <button
                    onClick={async () => {
                      await stopQueue(queue.gpu_ids as string, queue.worker_id);
                      refresh();
                    }}
                    className="operator-button border-rose-800 bg-rose-950/60 py-1 text-xs text-rose-100 hover:bg-rose-900"
                  >
                    Stop
                  </button>
                ) : (
                  <button
                    onClick={async () => {
                      await startQueue(group.gpuIDs as string, group.workerID);
                      refresh();
                    }}
                    className="operator-button border-emerald-800 bg-emerald-950/60 py-1 text-xs text-emerald-100 hover:bg-emerald-900"
                    disabled={!group.gpuIDs}
                  >
                    Start
                  </button>
                )}
              </div>
            </div>
            <UniversalTable
              columns={columns}
              rows={group.jobs}
              isLoading={isLoading}
              onRefresh={refresh}
              theadClassName={queueRunning ? 'text-emerald-300' : 'text-rose-300'}
              emptyTitle={queueRunning ? 'Queue is running with no jobs' : 'Queue is stopped with no jobs'}
              emptyDescription="Jobs assigned to this worker/GPU will appear here by queue position."
              errorMessage={tableError}
            />
          </div>
        );
      })}

      {!onlyActive && Object.keys(jobsDict).includes('idle') && (
        <div>
          <div className="flex border border-b-0 border-gray-800 bg-gray-900 px-3 py-2 text-sm">
            <div className="flex flex-1 items-center gap-2">
              <h2 className="font-semibold text-gray-100">Idle / history</h2>
              <span className="text-xs text-gray-500">{jobsDict['idle'].jobs.length} jobs</span>
            </div>
          </div>
          <UniversalTable
            columns={columns}
            rows={jobsDict['idle'].jobs}
            isLoading={isLoading}
            onRefresh={refresh}
            emptyTitle={filterText ? 'No idle jobs match the filter' : 'No idle jobs'}
            emptyDescription="Completed, stopped, and failed jobs will appear here."
            errorMessage={tableError}
          />
        </div>
      )}
    </div>
  );
}
