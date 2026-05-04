'use client';

import { use, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@headlessui/react';
import { ChevronRight, FileText, Pause, Square, Gauge, Code2, Image as ImageIcon, BarChart3 } from 'lucide-react';
import useJob from '@/hooks/useJob';
import useJobDashboard from '@/hooks/useJobDashboard';
import SampleImages from '@/components/SampleImages';
import JobConfigViewer from '@/components/JobConfigViewer';
import JobActionBar from '@/components/JobActionBar';
import JobConsole from '@/components/JobConsole';

type PageKey = 'console' | 'samples' | 'config';

const pages: Array<{ key: PageKey; label: string; icon: any; trainOnly?: boolean }> = [
  { key: 'console', label: 'Console', icon: BarChart3 },
  { key: 'samples', label: 'Samples', icon: ImageIcon, trainOnly: true },
  { key: 'config', label: 'Config', icon: Code2 },
];

function statusColor(status: string) {
  if (status === 'running') return 'text-green-400';
  if (status === 'error') return 'text-red-400';
  if (status === 'queued') return 'text-yellow-400';
  if (status === 'completed') return 'text-blue-400';
  return 'text-gray-400';
}

export default function JobPage({ params }: { params: { jobID: string } }) {
  const usableParams = use(params as any) as { jobID: string };
  const router = useRouter();
  const { job, status, refreshJob } = useJob(usableParams.jobID, 5000);
  const { dashboard, status: dashboardStatus } = useJobDashboard(usableParams.jobID, 5000);
  const [pageKey, setPageKey] = useState<PageKey>('console');
  const jobType = job?.job_type || 'unknown';

  return (
    <div className="absolute inset-0 overflow-hidden bg-black">
      <header className="absolute left-0 top-0 z-20 flex h-14 w-full items-center border-b border-white/10 bg-black px-5">
        <div className="flex min-w-0 items-center gap-2 text-sm">
          <button type="button" onClick={() => router.push('/jobs')} className="text-gray-500 hover:text-white">
            Job
          </button>
          <ChevronRight className="h-4 w-4 text-gray-600" />
          <h1 className="truncate text-base text-white">{job?.name || 'Loading...'}</h1>
          {job && (
            <span className={`ml-3 flex items-center gap-2 text-xs font-semibold uppercase ${statusColor(job.status)}`}>
              <span className={`h-2 w-2 rounded-full ${job.status === 'running' ? 'bg-green-500' : 'bg-current'}`} />
              {job.stop && job.status === 'running' ? 'stopping' : job.status}
            </span>
          )}
        </div>

        {dashboard && (
          <div className="ml-8 hidden min-w-[420px] flex-1 items-center gap-4 lg:flex">
            <div className="text-[11px] uppercase text-gray-500">Progress</div>
            <div className="h-1.5 flex-1 rounded-full bg-white/10">
              <div className="h-full rounded-full bg-blue-600" style={{ width: `${dashboard.progress ?? 0}%` }} />
            </div>
            <div className="w-14 text-right text-xs text-white">{(dashboard.progress ?? 0).toFixed(1)}%</div>
            <div className="border-l border-white/10 pl-5 text-xs">
              <div className="text-gray-500">Runtime</div>
              <div className="font-mono text-white">{dashboard.timing?.runtime || '--:--:--'}</div>
            </div>
            <div className="text-xs">
              <div className="text-gray-500">ETA</div>
              <div className="font-mono text-white">{dashboard.timing?.eta || '--:--:--'}</div>
            </div>
          </div>
        )}

        <div className="ml-auto flex items-center gap-1 text-xs text-gray-400">
          <Button disabled className="job-console-tool opacity-50" title="Pause unavailable">
            <Pause className="h-4 w-4" />
            <span>Pause</span>
          </Button>
          <Button disabled className="job-console-tool opacity-50" title="Stop from the action menu">
            <Square className="h-4 w-4" />
            <span>Stop</span>
          </Button>
          <Button disabled className="job-console-tool opacity-50" title="Logs are shown in the console">
            <FileText className="h-4 w-4" />
            <span>Logs</span>
          </Button>
          <Button disabled className="job-console-tool opacity-50" title="TensorBoard integration is not available yet">
            <Gauge className="h-4 w-4" />
            <span>TensorBoard</span>
          </Button>
          {job && (
            <JobActionBar
              job={job}
              onRefresh={refreshJob}
              hideView
              autoStartQueue
              afterDelete={() => router.push('/jobs')}
              className="console-actions"
            />
          )}
        </div>
      </header>

      <div className="absolute left-0 top-14 z-10 flex h-10 w-full items-center border-b border-white/10 bg-black px-4">
        {pages.map(page => {
          if (page.trainOnly && jobType !== 'train') return null;
          const Icon = page.icon;
          return (
            <button
              key={page.key}
              type="button"
              onClick={() => setPageKey(page.key)}
              className={`mr-2 flex h-10 items-center gap-2 border-b-2 px-3 text-xs uppercase tracking-wide ${
                pageKey === page.key ? 'border-blue-500 text-white' : 'border-transparent text-gray-500 hover:text-gray-200'
              }`}
            >
              <Icon className="h-4 w-4" />
              {page.label}
            </button>
          );
        })}
      </div>

      <main className={`absolute inset-x-0 bottom-0 top-24 ${pageKey === 'console' ? 'overflow-hidden' : 'overflow-auto'}`}>
        {(status === 'loading' || dashboardStatus === 'loading') && !job && <div className="p-6 text-gray-400">Loading...</div>}
        {status === 'error' && !job && <div className="p-6 text-red-400">Error fetching job</div>}
        {job && pageKey === 'console' && dashboard && <JobConsole job={job} dashboard={dashboard} />}
        {job && pageKey === 'console' && !dashboard && <div className="p-6 text-gray-500">Loading console...</div>}
        {job && pageKey === 'samples' && <div className="p-4"><SampleImages job={job} /></div>}
        {job && pageKey === 'config' && <JobConfigViewer job={job} />}
      </main>
    </div>
  );
}
