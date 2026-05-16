import { useMemo } from 'react';
import useJobsList from '@/hooks/useJobsList';
import Link from 'next/link';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import type { Job, JobConfig, Queue } from '@/types';
import JobActionBar from './JobActionBar';
import useQueueList from '@/hooks/useQueueList';
import classNames from 'classnames';
import { startQueue, stopQueue } from '@/utils/queue';
import { CgSpinner } from 'react-icons/cg';
import useGPUInfo from '@/hooks/useGPUInfo';
import { HFDownloadProgressInline } from '@/components/HFDownloadProgress';
import useWorkers from '@/hooks/useWorkers';

interface JobsTableProps {
  autoStartQueue?: boolean;
  onlyActive?: boolean;
  job_type?: string | null;
}

export default function JobsTable({ onlyActive = false, job_type = null }: JobsTableProps) {
  const { jobs, status, refreshJobs } = useJobsList({ onlyActive, reloadInterval: 5000, job_type });
  const { queues, status: queueStatus, refreshQueues } = useQueueList();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const { workers, status: workerStatus } = useWorkers();

  const refresh = () => {
    refreshJobs();
    refreshQueues();
  };

  const columns: TableColumn[] = [
    {
      title: 'Name',
      key: 'name',
      render: row => {
        let title = row.name;
        // if (row.job_type === 'train') title = `Train: ${title}`;
        if (row.job_type === 'caption') {
          let splits = row.job_ref.split(/[/\\]/);
          const datasetPath = `${splits[splits.length - 1]}`;
          title = (
            <>
              <small className="opacity-50">CAPTION: </small> {datasetPath}
            </>
          );
        }
        if (row.job_type === 'generate') {
          title = (
            <>
              <small className="opacity-50">GENERATE: </small> {title}
            </>
          );
        }
        return (
          <Link href={`/jobs/${row.id}`} className="font-medium whitespace-nowrap">
            {['running', 'stopping'].includes(row.status) ? (
              <CgSpinner className="inline animate-spin mr-2 text-blue-400" />
            ) : null}
            {title}
          </Link>
        );
      },
    },
    {
      title: 'Steps',
      key: 'steps',
      render: row => {
        const jobConfig: JobConfig = JSON.parse(row.job_config);
        if (row.job_type !== 'train') {
          return <></>;
        }
        const totalSteps = jobConfig.config.process[0].train?.steps;

        return (
          <div>
            <div className="text-xs text-gray-400">
              {row.step} / {totalSteps}
            </div>
            <div className="bg-gray-700 rounded-full h-1.5">
              <div
                className="bg-blue-500 h-1.5 rounded-full"
                style={{ width: `${(row.step / totalSteps) * 100}%` }}
              ></div>
            </div>
          </div>
        );
      },
    },
    {
      title: 'Worker',
      key: 'worker_id',
      render: row => {
        if (row.worker_id === 'local') return <span>Local</span>;
        return <span>{workers.find(worker => worker.id === row.worker_id)?.name || 'Remote'}</span>;
      },
    },
    {
      title: 'GPU',
      key: 'gpu_ids',
    },
    {
      title: 'Status',
      key: 'status',
      render: row => {
        let statusClass = 'text-gray-400';
        if (row.status === 'completed') statusClass = 'text-green-400';
        if (row.status === 'failed') statusClass = 'text-red-400';
        if (row.status === 'running') statusClass = 'text-blue-400';

        return <span className={statusClass}>{row.status}</span>;
      },
    },
    {
      title: 'Info',
      key: 'info',
      className: 'truncate max-w-xs',
      render: row => <HFDownloadProgressInline progress={row.hf_download_progress} fallback={row.info} />,
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'text-right',
      render: row => {
        return <JobActionBar job={row} onRefresh={refreshJobs} autoStartQueue={false} />;
      },
    },
  ];

  const jobsDict = useMemo(() => {
    if (!isGPUInfoLoaded) return {};
    if (jobs.length === 0) return {};
    let jd: { [key: string]: { name: string; jobs: Job[]; workerID: string; gpuIDs: string | null } } = {};
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

    gpuList.forEach(gpu => {
      jd[`local:${gpu.index}`] = {
        name: `Local / ${gpu.name}`,
        jobs: [],
        workerID: 'local',
        gpuIDs: `${gpu.index}`,
      };
    });
    queues.forEach(queue => {
      const key = `${queue.worker_id}:${queue.gpu_ids}`;
      if (!jd[key]) {
        jd[key] = {
          name: `${workerName(queue.worker_id)} / ${gpuName(queue.worker_id, queue.gpu_ids)}`,
          jobs: [],
          workerID: queue.worker_id,
          gpuIDs: queue.gpu_ids,
        };
      }
    });
    jd['idle'] = { name: 'Idle', jobs: [], workerID: 'local', gpuIDs: null };
    jobs.forEach(job => {
      const key = `${job.worker_id || 'local'}:${job.gpu_ids || '0'}`;
      if (['queued', 'running', 'stopping'].includes(job.status) && key in jd) {
        jd[key].jobs.push(job);
      } else {
        jd['idle'].jobs.push(job);
      }
    });
    // sort the queued/running jobs by queue position
    Object.keys(jd).forEach(key => {
      if (key === 'idle') {
        jd[key].jobs.sort((a, b) => {
          // sort by updated_at, newest first
          return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
        });
      } else {
        jd[key].jobs.sort((a, b) => {
          if (a.queue_position === null) return 1;
          if (b.queue_position === null) return -1;
          return a.queue_position - b.queue_position;
        });
      }
    });
    return jd;
  }, [jobs, queues, workers, gpuList, isGPUInfoLoaded]);

  let isLoading = status === 'loading' || queueStatus === 'loading' || workerStatus === 'loading' || !isGPUInfoLoaded;

  // if job dict is populated, we are always loaded
  if (Object.keys(jobsDict).length > 0) isLoading = false;

  return (
    <div>
      {Object.keys(jobsDict)
        .sort()
        .filter(key => key !== 'idle')
        .map(groupKey => {
          const group = jobsDict[groupKey];
          const queue = queues.find(q => q.worker_id === group.workerID && q.gpu_ids === group.gpuIDs) as Queue;
          return (
            <div key={groupKey} className="mb-6">
              <div
                className={classNames(
                  'text-md flex px-4 py-1 rounded-t-lg',
                  { 'bg-green-600 dark:bg-green-900': queue?.is_running },
                  { 'bg-red-600 dark:bg-red-900': !queue?.is_running },
                )}
              >
                <div className="flex items-center space-x-2 flex-1 py-2">
                  <h2 className="font-semibold text-white">{group.name}</h2>
                  <span className="px-2 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300"># {queue?.gpu_ids}</span>
                </div>
                <div className="text-sm text-gray-300 italic flex items-center">
                  {queue?.is_running ? (
                    <>
                      <span className="text-green-100 dark:text-green-400 mr-2">Queue Running</span>
                      <button
                        onClick={async () => {
                          await stopQueue(queue.gpu_ids as string, queue.worker_id);
                          refresh();
                        }}
                        className="ml-4 text-xs text-white bg-red-600 hover:bg-red-700 px-2 py-1 rounded"
                      >
                        STOP
                      </button>
                    </>
                  ) : (
                    <>
                      <span className="text-red-100 dark:text-red-400 mr-2">Queue Stopped</span>
                      <button
                        onClick={async () => {
                          await startQueue(group.gpuIDs as string, group.workerID);
                          refresh();
                        }}
                        className="ml-4 text-xs text-white bg-green-600 hover:bg-green-700 px-2 py-1 rounded"
                      >
                        START
                      </button>
                    </>
                  )}
                </div>
              </div>
              <UniversalTable
                columns={columns}
                rows={group.jobs}
                isLoading={isLoading}
                onRefresh={refresh}
                theadClassName={
                  queue?.is_running
                    ? 'bg-green-700 dark:bg-green-950 text-white dark:text-gray-400'
                    : 'bg-red-700 dark:bg-red-950 text-white dark:text-gray-400'
                }
              />
            </div>
          );
        })}
      {!onlyActive && Object.keys(jobsDict).includes('idle') && (
        <div className="mb-6 opacity-50">
          <div className="text-md flex px-4 py-1 rounded-t-lg bg-slate-600">
            <div className="flex items-center space-x-2 flex-1 py-2">
              <h2 className="font-semibold text-gray-100">Idle</h2>
            </div>
          </div>
          <UniversalTable columns={columns} rows={jobsDict['idle'].jobs} isLoading={isLoading} onRefresh={refresh} />
        </div>
      )}
    </div>
  );
}
